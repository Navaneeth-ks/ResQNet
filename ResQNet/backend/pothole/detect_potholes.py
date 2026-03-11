
"""
Pothole Detection using YOLOv12
CGIP Project - Computer Graphics & Image Processing
Run: python detect_potholes.py --source <image/video/webcam/youtube>
"""
'''
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
import os
try:
    import yt_dlp
except ImportError:
    print("ERROR: Install yt-dlp first: pip install yt-dlp")
    exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CONFIG = {
    "model_path": "models/yolov12n.pt",   # Change to your trained weights
    "conf_threshold": 0.4,                # Confidence threshold
    "iou_threshold": 0.5,                 # NMS IoU threshold
    "img_size": 640,                      # Inference image size
    "class_names": ["pothole"],           # Class labels
    "save_output": True,                  # Save results
    "output_dir": "output",               # Output folder
}

# Bounding box color (BGR)
BOX_COLOR = (0, 60, 255)        # Red-orange
TEXT_COLOR = (255, 255, 255)    # White
LABEL_BG   = (0, 60, 255)

def draw_boxes(frame, results, class_names):
    """Draw bounding boxes and labels on frame."""
    pothole_count = 0

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            label = class_names[cls] if cls < len(class_names) else f"class_{cls}"

            pothole_count += 1

            # Draw filled rect for label background
            label_text = f"{label}  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), LABEL_BG, -1)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            # Draw label text
            cv2.putText(frame, label_text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

    return frame, pothole_count

def draw_hud(frame, pothole_count, fps, frame_num):
    """Draw HUD overlay with stats."""
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    title   = "CGIP | Pothole Detection [YOLOv12]"
    fps_txt = f"FPS: {fps:.1f}"
    cnt_txt = f"Potholes Detected: {pothole_count}"
    frm_txt = f"Frame: {frame_num}"

    cv2.putText(frame, title,   (10,  26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, fps_txt, (w - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1, cv2.LINE_AA)

    # Bottom info bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 35), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)
    cv2.putText(frame, cnt_txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, frm_txt, (w - 140, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (160, 160, 160), 1, cv2.LINE_AA)

    return frame

def run_detection(source, model_path, conf, iou, img_size, save_output, output_dir):
    """Main detection loop."""
    print(f"\n{'='*55}")
    print(f"  CGIP Pothole Detection — YOLOv12")
    print(f"  Source  : {source}")
    print(f"  Model   : {model_path}")
    print(f"  Conf    : {conf}  |  IoU : {iou}")
    print(f"{'='*55}\n")

    # Load model
    model = YOLO(model_path)
    class_names = CONFIG["class_names"]

    # Open source
    if source == "0" or source.isdigit():
        cap = cv2.VideoCapture(int(source))
        source_name = "webcam"
    else:
        # Check if YouTube URL
        if "youtube.com" in source.lower() or "youtu.be" in source.lower():
            print("  Processing YouTube video...")
            ydl_opts = {
                'format': 'best[height<=720]',  # Limit resolution for speed
                'quiet': True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(source, download=False)
                    stream_url = info['url']
                
                cap = cv2.VideoCapture(stream_url)
                source_name = info.get('title', 'youtube_video')[:50]  # Truncate long titles
                print(f"  Stream ready: {source_name}")
            except Exception as e:
                print(f"  YouTube error: {e}")
                print("  Try downloading manually: yt-dlp -f 'best[height<=720]' <URL>")
                return
        else:
            cap = cv2.VideoCapture(source)
            source_name = Path(source).stem

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # Video writer setup
    writer = None
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{source_name}_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, fps_out, (w, h))
        print(f"  Saving output to: {out_path}\n")

    frame_num     = 0
    total_potholes = 0
    prev_time     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Inference
        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            verbose=False
        )

        # Draw detections
        frame, pothole_count = draw_boxes(frame, results, class_names)
        total_potholes += pothole_count

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # HUD
        frame = draw_hud(frame, pothole_count, fps, frame_num)

        # Show
        cv2.imshow("Pothole Detection — CGIP [q to quit]", frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit by user.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*55}")
    print(f"  Done!  Frames processed : {frame_num}")
    print(f"  Total pothole detections: {total_potholes}")
    if save_output and writer:
        print(f"  Output saved to        : {out_path}")
    print(f"{'='*55}\n")

def run_on_image(image_path, model_path, conf, iou, img_size, output_dir):
    """Run detection on a single image and save result."""
    model = YOLO(model_path)
    class_names = CONFIG["class_names"]

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(frame, conf=conf, iou=iou, imgsz=img_size, verbose=False)
    frame, pothole_count = draw_boxes(frame, results, class_names)

    # Add simple banner
    cv2.putText(frame, f"CGIP | YOLOv12 | Potholes: {pothole_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOX_COLOR, 2, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    stem    = Path(image_path).stem
    out_img = os.path.join(output_dir, f"{stem}_detected.jpg")
    cv2.imwrite(out_img, frame)
    print(f"[INFO] Image saved: {out_img}  | Potholes found: {pothole_count}")

    cv2.imshow("Pothole Detection — CGIP", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CGIP Pothole Detection — YOLOv12")
    parser.add_argument("--source",  type=str,  default="0",
                        help="Source: '0' webcam, path to image/video, or YouTube URL")
    parser.add_argument("--model",   type=str,  default=CONFIG["model_path"],
                        help="Path to YOLOv12 .pt weights")
    parser.add_argument("--conf",    type=float, default=CONFIG["conf_threshold"])
    parser.add_argument("--iou",     type=float, default=CONFIG["iou_threshold"])
    parser.add_argument("--imgsz",   type=int,   default=CONFIG["img_size"])
    parser.add_argument("--output",  type=str,   default=CONFIG["output_dir"])
    parser.add_argument("--no-save", action="store_true", help="Disable saving output")
    args = parser.parse_args()

    src = args.source
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if Path(src).suffix.lower() in img_exts:
        run_on_image(src, args.model, args.conf, args.iou, args.imgsz, args.output)
    else:
        run_detection(src, args.model, args.conf, args.iou,
                      args.imgsz, not args.no_save, args.output)
'''
"""
Pothole Detection using YOLOv12 with Severity Estimation
CGIP Project - Computer Graphics & Image Processing
Run: python detect_potholes.py --source <image/video/webcam/youtube>
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
import os

try:
    import yt_dlp
except ImportError:
    print("ERROR: Install yt-dlp first: pip install yt-dlp")
    exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CONFIG = {
    "model_path": "models/yolov12n.pt",   # Change to your trained weights
    "conf_threshold": 0.4,                # Confidence threshold
    "iou_threshold": 0.5,                 # NMS IoU threshold
    "img_size": 640,                      # Inference image size
    "class_names": ["pothole"],           # Class labels
    "save_output": True,                  # Save results
    "output_dir": "output",               # Output folder
}

# Default colors (BGR)
TEXT_COLOR = (255, 255, 255)    # White
LABEL_BG   = (0, 60, 255)       # Red-orange background

def draw_boxes(frame, results, class_names, frame_height):
    """Draw bounding boxes, labels, and severity on frame."""
    pothole_count = 0
    severity_counts = {"Low": 0, "Medium": 0, "High": 0}

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = class_names[cls] if cls < len(class_names) else f"class_{cls}"

            # Severity estimation: hybrid score (normalized area + y-position for depth proxy)
            bbox_area = (x2 - x1) * (y2 - y1) / (frame_height ** 2)
            y_center_norm = (y1 + y2) / 2 / frame_height  # Lower y = closer (higher severity)
            severity_score = 0.6 * bbox_area + 0.4 * (1 - y_center_norm)

            if severity_score < 0.002:
                severity = "Low"
                severity_color = (0, 255, 0)    # Green
            elif severity_score < 0.01:
                severity = "Medium"
                severity_color = (0, 255, 255)  # Yellow
            else:
                severity = "High"
                severity_color = (0, 0, 255)    # Red

            severity_counts[severity] += 1
            pothole_count += 1

            # Updated label with severity
            label_text = f"{label} {conf:.2f} | {severity}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), LABEL_BG, -1)

            # Severity-colored bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), severity_color, 2)

            # Label text
            cv2.putText(frame, label_text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

    return frame, pothole_count, severity_counts

def draw_hud(frame, pothole_count, severity_counts, fps, frame_num):
    """Draw HUD overlay with stats including severity breakdown."""
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    title = "CGIP | Pothole Detection [YOLOv12 + Severity]"
    fps_txt = f"FPS: {fps:.1f}"
    cv2.putText(frame, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, fps_txt, (w - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1, cv2.LINE_AA)

    # Bottom info bar (taller for severity)
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 70), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)

    cnt_txt = f"Potholes: {pothole_count}"
    sev_txt = f"L:{severity_counts['Low']} M:{severity_counts['Medium']} H:{severity_counts['High']}"
    frm_txt = f"Frame: {frame_num}"

    cv2.putText(frame, cnt_txt, (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, sev_txt, (10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, frm_txt, (w - 140, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    return frame

def run_detection(source, model_path, conf, iou, img_size, save_output, output_dir):
    """Main detection loop."""
    print(f"\n{'='*60}")
    print(f"  CGIP Pothole Detection — YOLOv12 + Severity Estimation")
    print(f"  Source  : {source}")
    print(f"  Model   : {model_path}")
    print(f"  Conf    : {conf}  |  IoU : {iou}")
    print(f"{'='*60}\n")

    # Load model
    model = YOLO(model_path)
    class_names = CONFIG["class_names"]

    # Open source
    if source == "0" or source.isdigit():
        cap = cv2.VideoCapture(int(source))
        source_name = "webcam"
    else:
        # Check if YouTube URL
        if "youtube.com" in source.lower() or "youtu.be" in source.lower():
            print("  Processing YouTube video...")
            ydl_opts = {
                'format': 'best[height<=720]',  # Limit resolution for speed
                'quiet': True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(source, download=False)
                    stream_url = info['url']
                
                cap = cv2.VideoCapture(stream_url)
                source_name = info.get('title', 'youtube_video')[:50]  # Truncate long titles
                print(f"  Stream ready: {source_name}")
            except Exception as e:
                print(f"  YouTube error: {e}")
                print("  Try downloading manually: yt-dlp -f 'best[height<=720]' <URL>")
                return
        else:
            cap = cv2.VideoCapture(source)
            source_name = Path(source).stem

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # Video writer setup
    writer = None
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{source_name}_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, fps_out, (w, h))
        print(f"  Saving output to: {out_path}\n")

    frame_num = 0
    total_potholes = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Inference
        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            verbose=False
        )

        # Draw detections with severity
        frame, pothole_count, severity_counts = draw_boxes(frame, results, class_names, frame.shape[0])
        total_potholes += pothole_count

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # HUD with severity
        frame = draw_hud(frame, pothole_count, severity_counts, fps, frame_num)

        # Show
        cv2.imshow("Pothole Detection — CGIP [q to quit]", frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit by user.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"  Done!  Frames processed : {frame_num}")
    print(f"  Total pothole detections: {total_potholes}")
    if save_output and writer:
        print(f"  Output saved to        : {out_path}")
    print(f"{'='*60}\n")

def run_on_image(image_path, model_path, conf, iou, img_size, output_dir):
    """Run detection on a single image and save result."""
    model = YOLO(model_path)
    class_names = CONFIG["class_names"]

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(frame, conf=conf, iou=iou, imgsz=img_size, verbose=False)
    frame, pothole_count, severity_counts = draw_boxes(frame, results, class_names, frame.shape[0])

    # Add banner with severity summary
    banner_text = f"CGIP | YOLOv12 | Potholes: {pothole_count} (L:{severity_counts['Low']} M:{severity_counts['Medium']} H:{severity_counts['High']})"
    cv2.putText(frame, banner_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem
    out_img = os.path.join(output_dir, f"{stem}_detected.jpg")
    cv2.imwrite(out_img, frame)
    print(f"[INFO] Image saved: {out_img}  | Potholes: {pothole_count} | Severity: L{severity_counts['Low']} M{severity_counts['Medium']} H{severity_counts['High']}")

    cv2.imshow("Pothole Detection — CGIP", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CGIP Pothole Detection — YOLOv12 + Severity")
    parser.add_argument("--source",  type=str,  default="0",
                        help="Source: '0' webcam, path to image/video, or YouTube URL")
    parser.add_argument("--model",   type=str,  default=CONFIG["model_path"],
                        help="Path to YOLOv12 .pt weights")
    parser.add_argument("--conf",    type=float, default=CONFIG["conf_threshold"])
    parser.add_argument("--iou",     type=float, default=CONFIG["iou_threshold"])
    parser.add_argument("--imgsz",   type=int,   default=CONFIG["img_size"])
    parser.add_argument("--output",  type=str,   default=CONFIG["output_dir"])
    parser.add_argument("--no-save", action="store_true", help="Disable saving output")
    args = parser.parse_args()

    src = args.source
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if Path(src).suffix.lower() in img_exts:
        run_on_image(src, args.model, args.conf, args.iou, args.imgsz, args.output)
    else:
        run_detection(src, args.model, args.conf, args.iou,
                      args.imgsz, not args.no_save, args.output)
