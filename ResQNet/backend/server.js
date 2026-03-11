const express       = require("express");
const multer        = require("multer");
const axios         = require("axios");
const FormData      = require("form-data");
const path          = require("path");
const fs            = require("fs");
const { addHazard } = require("./hazards");

const app    = express();
const upload = multer({ dest: "uploads/" });

app.use(express.json());
app.use(express.static(path.join(__dirname, "../frontend")));

const FLASK_URL = "http://localhost:5000";

// ─── Helper: forward file to Flask ───────────────────────────
async function forwardToFlask(endpoint, file, extraFields = {}) {
  const form = new FormData();
  form.append("file", fs.createReadStream(file.path), file.originalname);
  Object.entries(extraFields).forEach(([k, v]) => form.append(k, String(v)));

  const resp = await axios.post(`${FLASK_URL}${endpoint}`, form, {
    headers:          form.getHeaders(),
    maxBodyLength:    Infinity,
    maxContentLength: Infinity,
    timeout:          300000
  });

  fs.unlink(file.path, () => {});
  return resp.data;
}


// ══════════════════════════════════════════════════════════════
//  ACCIDENT DETECTION  — POST /api/accident/detect
// ══════════════════════════════════════════════════════════════
app.post("/api/accident/detect", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });

  try {
    const result = await forwardToFlask("/api/accident/detect", req.file, {
      iou: req.body.iou || 0.15
    });

    if (result.detected) {
      // lat/lng sent from frontend, fallback to Ernakulam center for demo
       const lat = parseFloat(req.body.lat) || 9.9908649;
       const lng = parseFloat(req.body.lng) || 76.3021516;

      const firestoreId = await addHazard("accident", lat, lng, {
        confidence:       result.confidence       || 0,
        frames_detected:  result.accident_frames  || 0,
        frames_total:     result.frames_processed || 1,
        seconds_analysed: result.seconds_analysed || 0,
        source:           "ai_detection"
      });

      result.firestore_written = true;
      result.firestore_id      = firestoreId;
    } else {
      result.firestore_written = false;
    }

    res.json(result);
  } catch (err) {
    console.error("❌ Accident detect error:", err.message);
    res.status(500).json({ error: err.message });
  }
});


// ══════════════════════════════════════════════════════════════
//  POTHOLE DETECTION  — POST /api/pothole/detect
// ══════════════════════════════════════════════════════════════
app.post("/api/pothole/detect", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });

  try {
    const result = await forwardToFlask("/api/pothole/detect", req.file, {
      conf: req.body.conf || 0.4,
      iou:  req.body.iou  || 0.5
    });

    if ((result.potholes_count || 0) > 0) {
      const lat = parseFloat(req.body.lat) || 9.9930419;
      const lng = parseFloat(req.body.lng) || 76.3017048;

      const firestoreId = await addHazard("pothole", lat, lng, {
        count:            result.potholes_count,
        severity:         result.severity                    || "Low",
        severity_low:     result.severity_breakdown?.Low    || 0,
        severity_medium:  result.severity_breakdown?.Medium || 0,
        severity_high:    result.severity_breakdown?.High   || 0,
        seconds_analysed: result.seconds_analysed           || 0,
        source:           "ai_detection"
      });

      result.firestore_written = true;
      result.firestore_id      = firestoreId;
    } else {
      result.firestore_written = false;
    }

    res.json(result);
  } catch (err) {
    console.error("❌ Pothole detect error:", err.message);
    res.status(500).json({ error: err.message });
  }
});


// ══════════════════════════════════════════════════════════════
//  AMBULANCE ROUTING  — GET /api/ambulance/route
// ══════════════════════════════════════════════════════════════
app.get("/api/ambulance/route", async (req, res) => {
  try {
    const { olat, olng, dlat, dlng } = req.query;
    const resp = await axios.get(`${FLASK_URL}/api/ambulance/route`, {
      params: { olat, olng, dlat, dlng }
    });
    res.json(resp.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// ══════════════════════════════════════════════════════════════
//  AMBULANCE DISPATCH  — POST /api/ambulance/dispatch
// ══════════════════════════════════════════════════════════════
app.post("/api/ambulance/dispatch", async (req, res) => {
  try {
    const resp = await axios.post(`${FLASK_URL}/api/ambulance/dispatch`, req.body);
    res.json(resp.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// ══════════════════════════════════════════════════════════════
//  SERVE FLASK OUTPUT FILES  — GET /api/output/:filename
// ══════════════════════════════════════════════════════════════
app.get("/api/output/:filename", async (req, res) => {
  try {
    const resp = await axios.get(
      `${FLASK_URL}/api/output/${req.params.filename}`,
      { responseType: "stream" }
    );
    resp.data.pipe(res);
  } catch {
    res.status(404).json({ error: "Output file not found" });
  }
});


// ══════════════════════════════════════════════════════════════
//  HEALTH CHECK  — GET /api/health
// ══════════════════════════════════════════════════════════════
app.get("/api/health", async (req, res) => {
  try {
    const resp = await axios.get(`${FLASK_URL}/api/health`);
    res.json({ node: "ok", flask: resp.data, project: "safenet-9e51e" });
  } catch {
    res.json({ node: "ok", flask: "unreachable", project: "safenet-9e51e" });
  }
});


// ══════════════════════════════════════════════════════════════
//  START
// ══════════════════════════════════════════════════════════════
const PORT = 3000;
app.listen(PORT, () => {
  console.log("\n" + "=".repeat(52));
  console.log(`  SafeNet Node Server — http://localhost:${PORT}`);
  console.log(`  Firebase Project    — safenet-9e51e`);
  console.log("=".repeat(52));
  console.log("  POST /api/accident/detect  → Firestore if detected");
  console.log("  POST /api/pothole/detect   → Firestore if found");
  console.log("  GET  /api/ambulance/route");
  console.log("  POST /api/ambulance/dispatch");
  console.log("  GET  /api/output/:filename");
  console.log("  GET  /api/health");
  console.log("=".repeat(52) + "\n");
});
