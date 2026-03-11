import os
import cv2
import json
import asyncio
import numpy as np
import datetime
import firebase_admin
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from firebase_admin import credentials, firestore

from severity_model  import analyze_image
from hospital_logic  import get_hospital_tier
from ambulance_logic import fleet, find_nearest, format_distance, get_ambulance, get_ambulance_by_name

# ── Firebase Init ─────────────────────────────────
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db  = firestore.client()
app = FastAPI()

# ── CORS ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SSE clients ───────────────────────────────────
sse_clients: list[asyncio.Queue] = []

# ── In-memory cache of latest assignments ─────────
# Stores { ambulance_name: payload } so when browser
# connects late it still gets the assignment
latest_assignments: dict = {}

async def broadcast(event: str, data: dict):
    msg  = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    dead = []
    for q in sse_clients:
        try:
            await q.put(msg)
        except Exception:
            dead.append(q)
    for q in dead:
        sse_clients.remove(q)


# ══════════════════════════════════════════════════
#  /analyze
# ══════════════════════════════════════════════════
@app.post("/analyze")
async def analyze_patient(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_arr      = np.frombuffer(image_bytes, np.uint8)
        img         = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        severity = analyze_image(img)
        hospital = get_hospital_tier(severity)

        report_data = {
            "case_id":              datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            "severity_score":       severity,
            "recommended_facility": hospital,
            "timestamp":            datetime.datetime.now(datetime.timezone.utc),
        }

        doc_ref = db.collection("triage_reports").document()
        doc_ref.set(report_data)

        return {
            "success":        True,
            "severity_score": severity,
            "hospital_tier":  hospital,
            "firebase_id":    doc_ref.id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════
#  SSE — /dispatch/events
#  Sends cached assignments immediately on connect
# ══════════════════════════════════════════════════
@app.get("/dispatch/events")
async def dispatch_events(request: Request):
    queue: asyncio.Queue = asyncio.Queue()
    sse_clients.append(queue)

    async def event_stream():
        try:
            yield "event: connected\ndata: {\"message\": \"connected\"}\n\n"
            print(f"[SSE] New client connected. Cached assignments: {list(latest_assignments.keys())}")

            # ── Send ALL cached assignments immediately ──
            # This handles the case where browser connects after Firestore fired
            for amb_name, payload in latest_assignments.items():
                print(f"[SSE] Sending cached assignment for {amb_name}")
                yield f"event: assignment\ndata: {json.dumps(payload)}\n\n"

            # ── Then keep listening for new ones ──
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30)
                    yield msg
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            if queue in sse_clients:
                sse_clients.remove(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ══════════════════════════════════════════════════
#  GET /dispatch/fleet
# ══════════════════════════════════════════════════
@app.get("/dispatch/fleet")
async def get_fleet():
    return {"success": True, "fleet": fleet}


# ══════════════════════════════════════════════════
#  POST /dispatch/nearest
# ══════════════════════════════════════════════════
class LocationBody(BaseModel):
    lat: float
    lng: float

@app.post("/dispatch/nearest")
async def nearest_ambulance(body: LocationBody):
    ambulance, dist = find_nearest(body.lat, body.lng)
    if not ambulance:
        raise HTTPException(status_code=404, detail="No available ambulance")
    return {
        "success":        True,
        "ambulance":      ambulance,
        "distanceMetres": dist,
        "distanceText":   format_distance(dist)
    }


# ══════════════════════════════════════════════════
#  POST /dispatch/send
# ══════════════════════════════════════════════════
class DispatchBody(BaseModel):
    ambulance_id:  str
    accident_lat:  float
    accident_lng:  float
    hazard_id:     str | None = None

@app.post("/dispatch/send")
async def dispatch_ambulance(body: DispatchBody):
    amb = get_ambulance(body.ambulance_id)

    if not amb:
        raise HTTPException(status_code=404, detail="Ambulance not found")
    if amb["status"] != "available":
        raise HTTPException(status_code=409, detail=f"Ambulance is {amb['status']}")

    amb["status"] = "dispatched"

    dispatch_data = {
        "ambulance_id":   body.ambulance_id,
        "ambulance_name": amb["name"],
        "driver":         amb["driver"],
        "accident_lat":   body.accident_lat,
        "accident_lng":   body.accident_lng,
        "hazard_id":      body.hazard_id,
        "dispatched_at":  datetime.datetime.now(datetime.timezone.utc),
        "status":         "dispatched"
    }

    dispatch_ref = db.collection("dispatches").document()
    dispatch_ref.set(dispatch_data)

    if body.hazard_id:
        try:
            db.collection("hazards").document(body.hazard_id).update({"status": "responded"})
        except Exception:
            pass

    await broadcast("dispatched", {
        "ambulance_id":   body.ambulance_id,
        "ambulance_name": amb["name"],
        "driver":         amb["driver"],
        "accident_lat":   body.accident_lat,
        "accident_lng":   body.accident_lng,
        "dispatch_id":    dispatch_ref.id
    })

    return {
        "success":     True,
        "dispatch_id": dispatch_ref.id,
        "ambulance":   amb,
        "message":     f"{amb['name']} dispatched successfully"
    }


# ══════════════════════════════════════════════════
#  POST /dispatch/arrived
# ══════════════════════════════════════════════════
class ArrivalBody(BaseModel):
    ambulance_id: str

@app.post("/dispatch/arrived")
async def ambulance_arrived(body: ArrivalBody):
    amb = get_ambulance(body.ambulance_id)
    if not amb:
        raise HTTPException(status_code=404, detail="Ambulance not found")

    amb["status"] = "available"
    await broadcast("arrived", {
        "ambulance_id":   body.ambulance_id,
        "ambulance_name": amb["name"]
    })
    return {"success": True, "message": f"{amb['name']} is now available"}


# ══════════════════════════════════════════════════
#  PATCH /dispatch/location
# ══════════════════════════════════════════════════
class LocationUpdateBody(BaseModel):
    ambulance_id: str
    lat:          float
    lng:          float

@app.patch("/dispatch/location")
async def update_location(body: LocationUpdateBody):
    amb = get_ambulance(body.ambulance_id)
    if not amb:
        raise HTTPException(status_code=404, detail="Ambulance not found")

    amb["position"] = {"lat": body.lat, "lng": body.lng}
    await broadcast("locationUpdate", {
        "ambulance_id": body.ambulance_id,
        "lat":          body.lat,
        "lng":          body.lng
    })
    return {"success": True}


# ══════════════════════════════════════════════════
#  FIRESTORE LISTENER — watches "allocations"
# ══════════════════════════════════════════════════
@app.on_event("startup")
async def start_firestore_listener():
    loop = asyncio.get_event_loop()

    def on_snapshot(col_snapshot, changes, read_time):
        print(f"[FIRESTORE] Snapshot fired! Total changes: {len(changes)}")

        for change in changes:
            print(f"[FIRESTORE] Change type: {change.type.name}")
            data = change.document.to_dict()
            print(f"[FIRESTORE] Data: {data}")

            if change.type.name == "ADDED" and data.get("status") == "assigned":
                loc = data.get("location")
                lat = loc.latitude  if loc else 0
                lng = loc.longitude if loc else 0

                payload = {
                    "id":        change.document.id,
                    "ambulance": data.get("ambulance"),
                    "driver":    data.get("driver"),
                    "phone":     str(data.get("phone", "")),
                    "distance":  data.get("distance"),
                    "location":  {"lat": lat, "lng": lng},
                    "status":    data.get("status"),
                    "timestamp": data.get("timestamp")
                }

                # ── Cache it so late-connecting browsers get it too ──
                amb_name = data.get("ambulance")
                latest_assignments[amb_name] = payload
                print(f"[FIRESTORE] ✅ Cached + Broadcasting for {amb_name}")

                asyncio.run_coroutine_threadsafe(
                    broadcast("assignment", payload), loop
                )

    db.collection("allocations").on_snapshot(on_snapshot)
    print("[FIRESTORE] ✅ Listening to allocations collection...")
