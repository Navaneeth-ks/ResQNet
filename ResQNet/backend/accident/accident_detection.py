import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import argparse

# Accept input file from terminal
parser = argparse.ArgumentParser(description="AI Accident Detection")
parser.add_argument("video", help="Path to input video file (e.g., cars.mp4)")
parser.add_argument("--iou", type=float, default=0.25, help="IoU threshold for accident detection")
args = parser.parse_args()

# Load YOLOv8 model
#model = YOLO("yolov8n.pt")
model = YOLO("yolov8n.pt")

# Open video using terminal argument
cap = cv2.VideoCapture(args.video)

if not cap.isOpened():
    print(f"❌ Error: Cannot open video file '{args.video}'")
    exit(1)

# Store previous positions
prev_positions = {}
speeds = defaultdict(list)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    vehicles = []

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in [2, 3, 5, 7]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            vehicles.append((x1, y1, x2, y2, cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    for i, v in enumerate(vehicles):
        cx, cy = v[4], v[5]
        if i in prev_positions:
            px, py = prev_positions[i]
            speed = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            speeds[i].append(speed)
        prev_positions[i] = (cx, cy)

    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            box1 = vehicles[i][:4]
            box2 = vehicles[j][:4]
            iou = compute_iou(box1, box2)
            if iou > args.iou:
                cv2.putText(frame, "ACCIDENT DETECTED", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                cv2.rectangle(frame, (box1[0], box1[1]), (box1[2], box1[3]), (0, 0, 255), 3)
                cv2.rectangle(frame, (box2[0], box2[1]), (box2[2], box2[3]), (0, 0, 255), 3)

    cv2.imshow("AI Accident Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or Q to quit
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
print("✅ Video processing complete.")

