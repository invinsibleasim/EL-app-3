# el_yolov5_infer.py
"""
YOLOv5 object detection for PV electroluminescence (EL) defects using a custom best.pt.
- Supports images and videos.
- Adjustable confidence/IoU thresholds.
- Saves annotated outputs and CSV of detections.
- Optional CLAHE contrast enhancement for EL grayscale images.

Usage examples:
  python el_yolov5_infer.py --weights best.pt --source path/to/el_image.jpg --conf 0.30 --iou 0.45 --clahe
  python el_yolov5_infer.py --weights best.pt --source path/to/el_video.mp4 --conf 0.30 --iou 0.45 --classes 0 1 2

Notes:
- This script uses PyTorch Hub to load Ultralytics YOLOv5 with custom weights.
- Make sure your best.pt was trained in the ultralytics/yolov5 repo.
"""

import argparse
import os
from pathlib import Path
import json

import cv2
import numpy as np
import torch

# ---------------------------
# Utilities
# ---------------------------

def ensure_rgb(img_bgr: np.ndarray, apply_clahe: bool = False) -> np.ndarray:
    """Ensure 3-channel RGB input; apply CLAHE if requested.
    EL images are often grayscale; YOLOv5 expects 3-channel RGB.
    """
    if img_bgr is None:
        raise ValueError("Failed to read image")
    # Detect grayscale
    if len(img_bgr.shape) == 2 or img_bgr.shape[2] == 1:
        gray = img_bgr if len(img_bgr.shape) == 2 else img_bgr[..., 0]
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return rgb
    else:
        # BGR -> RGB
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if apply_clahe:
            # Convert to HSV, boost V with CLAHE
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            v = hsv[..., 2]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            hsv[..., 2] = clahe.apply(v)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb


def draw_boxes(img_rgb: np.ndarray, df) -> np.ndarray:
    """Draw bounding boxes from results.pandas().xyxy[0] dataframe."""
    img = img_rgb.copy()
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        conf = row.get('confidence', 0.0)
        name = str(row.get('name', row.get('class', 'obj')))
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - bl), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def save_results(out_dir: Path, stem: str, df, annotated_rgb: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save annotated image
    out_img = out_dir / f"{stem}_annotated.jpg"
    cv2.imwrite(str(out_img), cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
    # Save CSV
    out_csv = out_dir / f"{stem}_detections.csv"
    df.to_csv(out_csv, index=False)
    # Save JSON
    out_json = out_dir / f"{stem}_detections.json"
    with open(out_json, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2)


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True, help='Path to best.pt')
    ap.add_argument('--source', type=str, required=True, help='Image or video path')
    ap.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    ap.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    ap.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    ap.add_argument('--classes', nargs='*', type=int, default=None, help='Optional class indices to filter')
    ap.add_argument('--clahe', action='store_true', help='Apply CLAHE contrast enhancement')
    ap.add_argument('--out', type=str, default='runs/el_detect', help='Output directory')
    args = ap.parse_args()

    weights = args.weights
    source = args.source
    out_dir = Path(args.out)

    # Load custom YOLOv5 model via PyTorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=False)
    model.conf = args.conf  # confidence
    model.iou = args.iou    # nms iou
    model.classes = args.classes  # class filter

    if not os.path.exists(source):
        raise FileNotFoundError(f"Source not found: {source}")

    # Image path
    if any(source.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']):
        bgr = cv2.imread(source, cv2.IMREAD_UNCHANGED)
        rgb = ensure_rgb(bgr, apply_clahe=args.clahe)
        # Inference (accepts numpy RGB)
        results = model(rgb, size=args.imgsz)
        df = results.pandas().xyxy[0]
        annotated = draw_boxes(rgb, df)
        save_results(out_dir, Path(source).stem, df, annotated)
        print(f"Saved to {out_dir}")
        return

    # Video path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {source}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = out_dir / f"{Path(source).stem}_annotated.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

    frame_idx = 0
    all_rows = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        rgb = ensure_rgb(frame_bgr, apply_clahe=args.clahe)
        results = model(rgb, size=args.imgsz)
        df = results.pandas().xyxy[0]
        df['frame'] = frame_idx
        all_rows.append(df)
        annotated = draw_boxes(rgb, df)
        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    cap.release()
    writer.release()

    if all_rows:
        df_all = all_rows[0].iloc[0:0].copy()
        df_all = json.loads(json.dumps(pd.concat(all_rows, ignore_index=True).to_dict(orient='records')))
        # Save as CSV and JSON
        import pandas as pd
        df_cat = pd.concat(all_rows, ignore_index=True)
        df_cat.to_csv(out_dir / f"{Path(source).stem}_detections.csv", index=False)
        with open(out_dir / f"{Path(source).stem}_detections.json", 'w') as f:
            json.dump(df_cat.to_dict(orient='records'), f, indent=2)
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    main()

