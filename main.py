import os
import cv2
import numpy as np
import datetime
import firebase_admin
from fastapi import FastAPI, UploadFile, File, HTTPException
from firebase_admin import credentials, firestore

# Import your existing logic
from severity_model import analyze_image
from hospital_logic import get_hospital_tier

# Initialize Firebase using the path to your JSON file
# Ensure 'serviceAccountKey.json' is in the same directory as this file
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
app = FastAPI()

@app.post("/analyze")
async def analyze_patient(file: UploadFile = File(...)):
    try:
        # 1. Read the image sent from Flutter
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # 2. Run your Computer Graphics/Image Processing logic
        severity = analyze_image(img)
        hospital = get_hospital_tier(severity)

        # 3. Automatic Upload to Firebase
        report_data = {
            "case_id": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            "severity_score": severity,
            "recommended_facility": hospital,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "processed_by": "Ambulance-AI-v1"
        }

        # Add to 'triage_reports' collection
        doc_ref = db.collection("triage_reports").document()
        doc_ref.set(report_data)

        # 4. Return response to Flutter app
        return {
            "success": True,
            "firebase_id": doc_ref.id,
            "data": report_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))