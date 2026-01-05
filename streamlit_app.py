
import os
import io
import glob
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="Object Recognition Dashboard", layout="wide")
st.title("🔍 Object Recognition Dashboard (YOLO)")

# Ensure dirs exist
for d in ["models", "data/uploaded_data", "data/sample_images", "data/sample_videos", "output", "tmp"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ---------------------------
# Try Ultralytics (preferred)
# ---------------------------
ULTRA_AVAILABLE = False
try:
    from ultralytics import YOLO  # supports v5/v8/v11 .pt models
    ULTRA_AVAILABLE = True
except Exception as e:
    st.warning(f"Ultralytics not available: {e}. Upload a .pt requires Ultralytics. "
               "If you're on Streamlit Cloud and this fails, add runtime.txt (python-3.11.9).")

# ---------------------------
# Utilities
# ---------------------------
def pil_to_rgb(img: Image.Image) -> np.ndarray:
    """PIL -> RGB numpy uint8"""
    return np.array(img.convert("RGB"))

def draw_boxes(img_bgr: np.ndarray,
               boxes: List[List[int]],
               classes: List[int],
               scores: List[float],
               class_names: List[str]) -> np.ndarray:
    """Overlay rectangles + labels."""
    vis = img_bgr.copy()
    for (x1, y1, x2, y2), cls, score in zip(boxes, classes, scores):
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        color = (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{name} {score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis

def parse_ultra_result(res) -> Dict[str, Any]:
    """
    Parse one Ultralytics result object → {'boxes','classes','scores'}.
    Compatible with YOLOv8/11 Results.
    """
    out = {"boxes": [], "classes": [], "scores": []}
    if res is None or getattr(res, "boxes", None) is None:
        return out
    b = res.boxes
    xyxy = b.xyxy.cpu().numpy().astype(int).tolist()
    cls = b.cls.cpu().numpy().astype(int).tolist()
    conf = b.conf.cpu().numpy().astype(float).tolist()
    out["boxes"], out["classes"], out["scores"] = xyxy, cls, conf
    return out

def zip_in_memory(file_tuples: List[Tuple[str, bytes]]) -> io.BytesIO:
    """Create and return a ZIP (in memory)."""
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for zpath, data in file_tuples:
            zf.writestr(zpath, data)
    buf.seek(0)
    return buf

# ---------------------------
# Sidebar: Settings
# ---------------------------
st.sidebar.title("Settings")

# Model source
model_choice = st.sidebar.radio("Select YOLO weight file", ["Use demo model (models/best.pt)", "Upload your model (.pt)"])

# Confidence
confidence = st.sidebar.slider("Confidence", min_value=0.05, max_value=1.0, value=0.45, step=0.05)
iou_thres  = st.sidebar.slider("IoU (NMS)", min_value=0.10, max_value=0.95, value=0.45, step=0.05)
imgsz      = st.sidebar.selectbox("imgsz (inference size)", [640, 512, 416], index=0)

# Device (Ultralytics will pick CPU if CUDA unavailable)
device = "cuda" if ULTRA_AVAILABLE and hasattr(cv2, "cuda") else "cpu"
device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0 if device == "cpu" else 1)

# Class names override (optional)
class_names_str = st.sidebar.text_area("Class names (comma-separated) — optional", "hotspot, crack, delamination")
user_class_names = [s.strip() for s in class_names_str.split(",") if s.strip()]

# Class filter
use_class_filter = st.sidebar.checkbox("Filter classes for display", False)
filter_classes = []
if use_class_filter and user_class_names:
    filter_classes = st.sidebar.multiselect("Select classes to show", user_class_names, default=user_class_names)

# Input type & source
input_option = st.sidebar.radio("Select input type", ["image", "video"])
data_src     = st.sidebar.radio("Select input source", ["Sample data", "Upload your own data"])

# ---------------------------
# Load model
# ---------------------------
ultra_model = None
model_path = None

if model_choice == "Upload your model (.pt)":
    if not ULTRA_AVAILABLE:
        st.error("Ultralytics not installed. You cannot load .pt here. Add runtime.txt (python-3.11.9) and ultralytics in requirements, or use ONNX path.")
    else:
        model_bytes = st.sidebar.file_uploader("Upload a YOLO .pt model", type=["pt"])
        if model_bytes:
            model_path = Path("models") / model_bytes.name
            model_path.write_bytes(model_bytes.read())
            try:
                ultra_model = YOLO(str(model_path))
                st.sidebar.success(f"Loaded model: {model_path.name}")
            except Exception as e:
                st.sidebar.error(f"Failed to load .pt: {e}")
else:
    # Demo model path
    model_path = Path("models/best.pt")
    if model_path.exists() and ULTRA_AVAILABLE:
        try:
            ultra_model = YOLO(str(model_path))
            st.sidebar.success(f"Loaded demo model: {model_path.name}")
        except Exception as e:
            st.sidebar.error(f"Failed to load demo model: {e}")
    else:
        st.sidebar.info("Place your demo weight at 'models/best.pt' to use this option.")

# Determine class names
class_names: List[str] = []
if ultra_model is not None:
    # Ultralytics model.names is a dict {id: name}
    names_dict = getattr(ultra_model, "names", None) or {}
    class_names = [names_dict[i] for i in sorted(names_dict.keys())] if names_dict else []
# Override with user-specified names if provided
if user_class_names:
    class_names = user_class_names

if ultra_model is not None:
    st.info(f"Model classes: {len(class_names) if class_names else '(unknown)'} — {class_names if class_names else '(no names provided)'}")

# ---------------------------
# Inference helpers
# ---------------------------
def run_ultralytics_image(rgb_arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Run Ultralytics on a single RGB image → overlay RGB + per-class counts."""
    results = ultra_model.predict(
        source=rgb_arr,
        imgsz=int(imgsz),
        conf=confidence,
        iou=iou_thres,
        device=device,
        verbose=False
    )
    res = results[0]
    parsed = parse_ultra_result(res)

    # Optional class filter
    if filter_classes and class_names:
        keep = []
        for k, cid in enumerate(parsed["classes"]):
            cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            if cname in filter_classes:
                keep.append(k)
        parsed["boxes"]   = [parsed["boxes"][k]   for k in keep]
        parsed["classes"] = [parsed["classes"][k] for k in keep]
        parsed["scores"]  = [parsed["scores"][k]  for k in keep]

    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    vis_bgr = draw_boxes(bgr, parsed["boxes"], parsed["classes"], parsed["scores"], class_names)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    # Counts
    counts: Dict[str, int] = {}
    for cid in parsed["classes"]:
        cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        counts[cname] = counts.get(cname, 0) + 1
    return vis_rgb, counts

# ---------------------------
# Image input handler
# ---------------------------
def image_input_handler():
    # Get image
    if data_src == "Sample data":
        img_paths = sorted(glob.glob("data/sample_images/*"))
        if not img_paths:
            st.warning("No sample images found in 'data/sample_images'. Please upload your own.")
            return
        idx = st.slider("Select a test image", min_value=1, max_value=len(img_paths), value=1, step=1)
        img_file = img_paths[idx - 1]
        rgb = pil_to_rgb(Image.open(img_file))
    else:
        img_bytes = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg', 'bmp', 'tif', 'tiff'])
        if not img_bytes:
            st.info("Upload an image to run inference.")
            return
        ext = img_bytes.name.split('.')[-1]
        img_file = f"data/uploaded_data/upload.{ext}"
        Image.open(img_bytes).save(img_file)
        rgb = pil_to_rgb(Image.open(img_file))

    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb, caption="Selected Image", use_column_width=True)

    with col2:
        if ultra_model is None:
            st.error("Model not loaded. Please upload/select a YOLO .pt model.")
            return
        vis_rgb, counts = run_ultralytics_image(rgb)
        st.image(vis_rgb, caption="Model Prediction", use_column_width=True)
        st.json({"detections_per_class": counts})

# ---------------------------
# Video input handler
# ---------------------------
def video_input_handler():
    # Get video
    if data_src == "Sample data":
        sample_vid = "data/sample_videos/sample.mp4"
        if not Path(sample_vid).exists():
            st.warning("No sample video found in 'data/sample_videos/sample.mp4'. Please upload your own.")
            return
        vid_file = sample_vid
    else:
        vid_bytes = st.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi', 'mov', 'mkv'])
        if not vid_bytes:
            st.info("Upload a video to run inference.")
            return
        ext = vid_bytes.name.split('.')[-1]
        vid_file = f"data/uploaded_data/upload.{ext}"
        with open(vid_file, 'wb') as out:
            out.write(vid_bytes.read())

    cap = cv2.VideoCapture(vid_file)
    if not cap.isOpened():
        st.error("Cannot open video.")
        return

    # Custom size (optional)
    custom_size = st.sidebar.checkbox("Custom frame size")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if custom_size:
        width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
        height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

    fps_live = 0.0
    st1, st2, st3 = st.columns(3)
    with st1:
        st.markdown("## Height"); st1_text = st.markdown(f"{height}")
    with st2:
        st.markdown("## Width");  st2_text = st.markdown(f"{width}")
    with st3:
        st.markdown("## FPS");    st3_text = st.markdown(f"{fps_live:.2f}")

    st.markdown("---")
    output = st.empty()
    prev_time = time.time()

    # Writer
    out_dir = Path("output") / Path(vid_file).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_out = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps_out, (int(width), int(height)))

    processed_frames = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            st.write("Stream ended.")
            break
        frame_bgr = cv2.resize(frame_bgr, (int(width), int(height)))

        if ultra_model is None:
            st.error("Model not loaded. Please upload/select a YOLO .pt model.")
            break

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        vis_rgb, _counts = run_ultralytics_image(rgb)
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

        output.image(vis_rgb, use_column_width=True)
        writer.write(vis_bgr)

        curr_time = time.time()
        fps_live = 1.0 / max(1e-6, (curr_time - prev_time))
        prev_time = curr_time

        st1_text.markdown(f"**{int(height)}**")
        st2_text.markdown(f"**{int(width)}**")
        st3_text.markdown(f"**{fps_live:.2f}**")

        processed_frames += 1

    cap.release()
    writer.release()

    st.success(f"Processed {processed_frames} frames")
    st.video(str(out_mp4))

# ---------------------------
# Router
# ---------------------------
if input_option == "image":
    image_input_handler()
else:
    video_input_handler()
