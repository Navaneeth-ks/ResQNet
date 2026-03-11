# backend/api.py
# Run: python api.py
# Exposes:
#   POST /api/accident/detect
#   POST /api/pothole/detect
#   GET  /api/ambulance/route
#   POST /api/ambulance/dispatch
#   GET  /api/output/<filename>
#   GET  /api/health

import os
import uuid
import time
import cv2
import numpy as np
import requests as http_req
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# ─── Paths ────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
UPLOAD_DIR     = BASE_DIR / "uploads"
OUTPUT_DIR     = BASE_DIR / "outputs"
ACCIDENT_MODEL = BASE_DIR / "accident" / "yolov8n.pt"    
POTHOLE_MODEL  = BASE_DIR / "pothole"  / "models" / "best.pt"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VID = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_ALL = ALLOWED_IMG | ALLOWED_VID

MAX_FILE_SIZE = 200 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# ✅ Global video processing limits
MAX_SECONDS = 20     # only process first 20 seconds
FRAME_SKIP  = 2      # process every 3rd frame (3x faster)


# ══════════════════════════════════════════════════════════════
#  HELPER: save uploaded file
# ══════════════════════════════════════════════════════════════
def save_upload(file_obj) -> Path:
    ext      = Path(secure_filename(file_obj.filename)).suffix.lower()
    uid      = uuid.uuid4().hex[:10]
    filename = f"{uid}{ext}"
    dest     = UPLOAD_DIR / filename
    file_obj.save(str(dest))
    return dest


# ══════════════════════════════════════════════════════════════
#  ACCIDENT DETECTION  — POST /api/accident/detect
#  Scenario A: best.pt detects vehicles → IoU overlap = accident
# ══════════════════════════════════════════════════════════════
def _compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-5)


def run_accident_detection(source_path: Path, iou_thresh=0.15):
    model        = YOLO(str(ACCIDENT_MODEL))
    ext          = source_path.suffix.lower()
    out_filename = f"accident_{source_path.stem}_out{ext}"
    out_path     = OUTPUT_DIR / out_filename

    # ── IMAGE ──────────────────────────────────────────────────
    if ext in ALLOWED_IMG:
        frame    = cv2.imread(str(source_path))
        results  = model(frame)[0]
        vehicles = []

        for box in results.boxes:
            # ✅ NO class filter — use ALL detections from custom best.pt
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicles.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        accident_detected = False
        accident_pairs    = []
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                if _compute_iou(vehicles[i], vehicles[j]) > iou_thresh:
                    accident_detected = True
                    accident_pairs.append([i, j])
                    b1, b2 = vehicles[i], vehicles[j]
                    cv2.rectangle(frame, (b1[0], b1[1]), (b1[2], b1[3]), (0, 0, 255), 3)
                    cv2.rectangle(frame, (b2[0], b2[1]), (b2[2], b2[3]), (0, 0, 255), 3)

        if accident_detected:
            cv2.putText(frame, "ACCIDENT DETECTED", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        cv2.imwrite(str(out_path), frame)
        return {
            "detected":         accident_detected,
            "vehicles_found":   len(vehicles),
            "accident_pairs":   accident_pairs,
            "confidence":       float(np.mean([float(b.conf[0]) for b in results.boxes])
                                      if results.boxes else 0),
            "frames_processed": 1,
            "output_file":      out_filename,
            "output_url":       f"/api/output/{out_filename}"
        }

    # ── VIDEO ──────────────────────────────────────────────────
    cap    = cv2.VideoCapture(str(source_path))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ✅ Only process first MAX_SECONDS
    max_frames = int(fps_in * MAX_SECONDS)

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)
    )

    frame_id          = 0
    accident_detected = False
    total_accidents   = 0
    all_confs         = []
    last_frame        = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # ✅ Stop after MAX_SECONDS
        if frame_id > max_frames:
            break

        # ✅ Skip frames for speed — reuse last annotated frame
        if frame_id % (FRAME_SKIP + 1) != 0:
            writer.write(last_frame if last_frame is not None else frame)
            continue

        results  = model(frame)[0]
        vehicles = []

        for box in results.boxes:
            # ✅ NO class filter — use ALL detections from custom best.pt
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            vehicles.append((x1, y1, x2, y2, cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            all_confs.append(float(box.conf[0]))

        frame_accident = False
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                if _compute_iou(vehicles[i][:4], vehicles[j][:4]) > iou_thresh:
                    frame_accident    = True
                    accident_detected = True
                    total_accidents  += 1
                    b1 = vehicles[i][:4]
                    b2 = vehicles[j][:4]
                    cv2.rectangle(frame, (b1[0], b1[1]), (b1[2], b1[3]), (0, 0, 255), 3)
                    cv2.rectangle(frame, (b2[0], b2[1]), (b2[2], b2[3]), (0, 0, 255), 3)

        if frame_accident:
            cv2.putText(frame, "ACCIDENT DETECTED", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        last_frame = frame
        writer.write(frame)

    cap.release()
    writer.release()

    return {
        "detected":          accident_detected,
        "accident_frames":   total_accidents,
        "frames_processed":  frame_id,
        "frames_analysed":   frame_id // (FRAME_SKIP + 1) or 1,
        "seconds_analysed":  round(frame_id / fps_in, 1),
        "confidence":        float(np.mean(all_confs)) if all_confs else 0,
        "output_file":       out_filename,
        "output_url":        f"/api/output/{out_filename}"
    }


@app.route("/api/accident/detect", methods=["POST"])
def accident_detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'file'"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_ALL:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 415

    try:
        source_path = save_upload(file)
        iou_thresh  = float(request.form.get("iou", 0.15))
        result      = run_accident_detection(source_path, iou_thresh)
        source_path.unlink(missing_ok=True)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
#  POTHOLE DETECTION  — POST /api/pothole/detect
# ══════════════════════════════════════════════════════════════
def _severity(bbox_area_norm, y_center_norm):
    score = 0.6 * bbox_area_norm + 0.4 * (1 - y_center_norm)
    if score < 0.002:
        return "Low",    (0, 255, 0)
    if score < 0.01:
        return "Medium", (0, 255, 255)
    return               "High",   (0, 0, 255)


def run_pothole_detection(source_path: Path, conf=0.4, iou=0.5):
    model     = YOLO(str(POTHOLE_MODEL))
    ext       = source_path.suffix.lower()
    out_name  = f"pothole_{source_path.stem}_out{ext}"
    out_path  = OUTPUT_DIR / out_name

    BOX_COLOR  = (0, 60, 255)
    TEXT_COLOR = (255, 255, 255)

    def _annotate(frame):
        h        = frame.shape[0]
        results  = model.predict(frame, conf=conf, iou=iou, imgsz=640, verbose=False)
        count    = 0
        sev_cnt  = {"Low": 0, "Medium": 0, "High": 0}
        det_list = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cf        = float(box.conf[0])
                area_norm = (x2 - x1) * (y2 - y1) / (h ** 2)
                y_norm    = (y1 + y2) / 2 / h
                sev, sev_col = _severity(area_norm, y_norm)
                sev_cnt[sev] += 1
                count += 1
                label = f"pothole {cf:.2f} | {sev}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), BOX_COLOR, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), sev_col, 2)
                cv2.putText(frame, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
                det_list.append({
                    "bbox":     [x1, y1, x2, y2],
                    "conf":     round(cf, 3),
                    "severity": sev
                })

        return frame, count, sev_cnt, det_list

    # ── IMAGE ──────────────────────────────────────────────────
    if ext in ALLOWED_IMG:
        frame = cv2.imread(str(source_path))
        frame, count, sev_cnt, det_list = _annotate(frame)
        cv2.imwrite(str(out_path), frame)
        return {
            "potholes_count":     count,
            "severity":           "High" if sev_cnt["High"] else
                                  ("Medium" if sev_cnt["Medium"] else "Low"),
            "severity_breakdown": sev_cnt,
            "detections":         det_list,
            "output_file":        out_name,
            "output_url":         f"/api/output/{out_name}"
        }

    # ── VIDEO ──────────────────────────────────────────────────
    cap    = cv2.VideoCapture(str(source_path))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ✅ Only process first MAX_SECONDS
    max_frames = int(fps_in * MAX_SECONDS)

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)
    )

    frame_id   = 0
    peak_count = 0
    peak_sev   = {"Low": 0, "Medium": 0, "High": 0}
    last_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # ✅ Stop after MAX_SECONDS
        if frame_id > max_frames:
            break

        if frame_id % (FRAME_SKIP + 1) == 0:
            frame, c, s, _ = _annotate(frame)

            # ✅ Track PEAK frame only
            if c > peak_count:
                peak_count = c
                peak_sev   = s.copy()

            last_frame = frame
            writer.write(frame)
        else:
            writer.write(last_frame if last_frame is not None else frame)

    cap.release()
    writer.release()

    dominant       = max(peak_sev, key=peak_sev.get) if any(peak_sev.values()) else "Low"
    sampled_frames = frame_id // (FRAME_SKIP + 1) or 1

    return {
        "potholes_count":     peak_count,
        "severity":           dominant,
        "severity_breakdown": peak_sev,
        "avg_per_frame":      round(peak_count / 1, 1),
        "frames_processed":   frame_id,
        "frames_analysed":    sampled_frames,
        "seconds_analysed":   round(frame_id / fps_in, 1),
        "output_file":        out_name,
        "output_url":         f"/api/output/{out_name}"
    }


@app.route("/api/pothole/detect", methods=["POST"])
def pothole_detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'file'"}), 400

    file = request.files["file"]
    ext  = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_ALL:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 415

    try:
        source_path = save_upload(file)
        conf        = float(request.form.get("conf", 0.4))
        iou         = float(request.form.get("iou",  0.5))
        result      = run_pothole_detection(source_path, conf, iou)
        source_path.unlink(missing_ok=True)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
#  AMBULANCE ROUTING  — GET /api/ambulance/route
# ══════════════════════════════════════════════════════════════
@app.route("/api/ambulance/route")
def ambulance_route():
    try:
        olat = float(request.args.get("olat"))
        olng = float(request.args.get("olng"))
        dlat = float(request.args.get("dlat"))
        dlng = float(request.args.get("dlng"))
    except (TypeError, ValueError):
        return jsonify({
            "error":   "Provide olat, olng, dlat, dlng as query params",
            "example": "/api/ambulance/route?olat=10.0159&olng=76.3419&dlat=10.0261&dlng=76.3082"
        }), 400

    osrm_url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{olng},{olat};{dlng},{dlat}"
        f"?overview=full&geometries=geojson&steps=false"
    )

    try:
        resp = http_req.get(osrm_url, timeout=10)
        resp.raise_for_status()
        osrm_data = resp.json()

        if osrm_data.get("code") != "Ok" or not osrm_data.get("routes"):
            return jsonify({"error": "OSRM returned no routes"}), 502

        route     = osrm_data["routes"][0]
        coords    = route["geometry"]["coordinates"]
        waypoints = [[c[1], c[0]] for c in coords]   # [lng,lat] → [lat,lng]

        return jsonify({
            "waypoints":   waypoints,
            "eta_minutes": round(route["duration"]  / 60,   1),
            "distance_km": round(route["distance"]  / 1000, 2),
            "source":      "osrm",
            "origin":      [olat, olng],
            "destination": [dlat, dlng]
        }), 200

    except http_req.exceptions.Timeout:
        return jsonify({"error": "OSRM request timed out"}), 504
    except http_req.exceptions.ConnectionError:
        return jsonify({"error": "Could not connect to OSRM routing server"}), 502
    except Exception as e:
        return jsonify({"error": f"Routing failed: {str(e)}"}), 500


# ══════════════════════════════════════════════════════════════
#  AMBULANCE DISPATCH  — POST /api/ambulance/dispatch
# ══════════════════════════════════════════════════════════════
dispatch_log = {}

@app.route("/api/ambulance/dispatch", methods=["POST"])
def ambulance_dispatch():
    body    = request.get_json(silent=True) or {}
    unit_id = body.get("unit_id", "").strip()

    if not unit_id:
        return jsonify({"error": "Provide unit_id in JSON body"}), 400

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    dispatch_log[unit_id] = {
        "unit_id":   unit_id,
        "status":    "dispatched",
        "timestamp": timestamp
    }

    return jsonify({
        "status":    "dispatched",
        "unit_id":   unit_id,
        "message":   f"{unit_id} has been dispatched successfully",
        "timestamp": timestamp
    }), 200


@app.route("/api/ambulance/dispatch/log")
def dispatch_log_view():
    return jsonify(list(dispatch_log.values())), 200


# ══════════════════════════════════════════════════════════════
#  SERVE OUTPUT FILES  — GET /api/output/<filename>
# ══════════════════════════════════════════════════════════════
@app.route("/api/output/<filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


# ══════════════════════════════════════════════════════════════
#  HEALTH CHECK  — GET /api/health
# ══════════════════════════════════════════════════════════════
@app.route("/api/health")
def health():
    # Print model class names on health check — useful for debugging
    try:
        acc_model   = YOLO(str(ACCIDENT_MODEL))
        acc_classes = acc_model.names
    except Exception:
        acc_classes = "model load error"

    return jsonify({
        "status":                "ok",
        "accident_model":        str(ACCIDENT_MODEL),
        "pothole_model":         str(POTHOLE_MODEL),
        "accident_model_exists": ACCIDENT_MODEL.exists(),
        "pothole_model_exists":  POTHOLE_MODEL.exists(),
        "accident_classes":      acc_classes,   # ✅ shows your model's class IDs
        "upload_dir":            str(UPLOAD_DIR),
        "output_dir":            str(OUTPUT_DIR),
        "max_seconds":           MAX_SECONDS,
        "frame_skip":            FRAME_SKIP,
        "server_time":           time.strftime("%Y-%m-%d %H:%M:%S")
    })


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  SafeNet API  —  http://localhost:5000")
    print("="*55)
    print(f"  Video limit  : first {MAX_SECONDS} seconds only")
    print(f"  Frame skip   : every {FRAME_SKIP+1} frames ({FRAME_SKIP+1}x faster)")
    print(f"  Class filter : NONE (uses all detections from best.pt)")
    print("  POST /api/accident/detect")
    print("  POST /api/pothole/detect")
    print("  GET  /api/ambulance/route?olat=&olng=&dlat=&dlng=")
    print("  POST /api/ambulance/dispatch")
    print("  GET  /api/ambulance/dispatch/log")
    print("  GET  /api/output/<filename>")
    print("  GET  /api/health")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
