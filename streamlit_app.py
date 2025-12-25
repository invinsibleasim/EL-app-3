# app_el_yolov5_streamlit.py
"""
Streamlit front-end for YOLOv5 object detection on PV Electroluminescence (EL) images and videos
using your custom `best.pt`.

Features
- Upload or path-select your YOLOv5 `best.pt` weights
- Image & video inference
- Adjustable confidence & IoU thresholds
- Optional class filter
- Optional CLAHE contrast enhancement for EL images
- Download annotated PNG/MP4 and detections CSV/JSON

Note
- This app loads YOLOv5 via PyTorch Hub (`ultralytics/yolov5`) and then your custom weights.
- Ensure your `best.pt` was trained with the original Ultralytics YOLOv5 repository.
"""

import io
import os
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch

# ------------------------------
# Helpers
# ------------------------------

@st.cache_resource(show_spinner=False)
def load_yolov5_model(weights_path: str):
    """Load custom YOLOv5 model via PyTorch Hub and cache it.
    weights_path: path to best.pt
    """
    # Load model from Ultralytics YOLOv5 hub with custom weights
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
    return model


def ensure_rgb(img_bgr: np.ndarray, apply_clahe: bool = False) -> np.ndarray:
    """Ensure 3-channel RGB; apply CLAHE for EL contrast enhancement if requested."""
    if img_bgr is None:
        raise ValueError("Failed to read image")
    if len(img_bgr.shape) == 2 or (img_bgr.ndim == 3 and img_bgr.shape[2] == 1):
        gray = img_bgr if img_bgr.ndim == 2 else img_bgr[..., 0]
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return rgb
    # BGR -> RGB
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if apply_clahe:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v = hsv[..., 2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[..., 2] = clahe.apply(v)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def draw_boxes(img_rgb: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Draw bounding boxes from YOLOv5 results dataframe."""
    img = img_rgb.copy()
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        conf = float(row.get('confidence', 0.0))
        name = str(row.get('name', row.get('class', 'obj')))
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - bl), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def df_download_buttons(df: pd.DataFrame, stem: str):
    csv = df.to_csv(index=False).encode('utf-8')
    json_str = df.to_json(orient='records', indent=2)
    st.download_button(
        label=f"⬇️ Download detections CSV ({stem})",
        data=csv,
        file_name=f"{stem}_detections.csv",
        mime="text/csv",
    )
    st.download_button(
        label=f"⬇️ Download detections JSON ({stem})",
        data=json_str,
        file_name=f"{stem}_detections.json",
        mime="application/json",
    )


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="YOLOv5 EL Defect Detection", page_icon="🔧", layout="wide")
st.title("🛠️ YOLOv5 Object Detection for PV EL Defects")
st.caption("Upload your `best.pt` and EL images/videos. Configure thresholds, class filters, and CLAHE for contrast.")

st.sidebar.header("⚙️ Configuration")
# Weights: upload or path
w_choice = st.sidebar.radio("Select weights source", ["Upload best.pt", "Use local path"], index=0)
weights_path = None
if w_choice == "Upload best.pt":
    w_file = st.sidebar.file_uploader("Upload best.pt", type=["pt"], accept_multiple_files=False)
    if w_file:
        tmp_w = Path("best_uploaded.pt")
        tmp_w.write_bytes(w_file.read())
        weights_path = str(tmp_w)
else:
    weights_path = st.sidebar.text_input("Local path to best.pt", value="best.pt")

conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.30, 0.01)
iou = st.sidebar.slider("IoU threshold (NMS)", 0.1, 1.0, 0.45, 0.01)
imgsz = st.sidebar.number_input("Image size (inference)", min_value=320, max_value=1280, value=640, step=64)
clahe = st.sidebar.checkbox("Apply CLAHE (contrast boost for EL)", value=True)
class_filter = st.sidebar.text_input("Filter by class indices (comma-separated, optional)", value="")

# Load model
if weights_path:
    with st.spinner("Loading YOLOv5 model…"):
        try:
            model = load_yolov5_model(weights_path)
            model.conf = conf
            model.iou = iou
            if class_filter.strip():
                model.classes = [int(x.strip()) for x in class_filter.split(',') if x.strip().isdigit()]
            else:
                model.classes = None
            names = getattr(model, 'names', None)
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()
else:
    st.info("Upload or enter path to your `best.pt` to continue.")
    st.stop()

# Tabs for image and video
img_tab, vid_tab = st.tabs(["🖼️ Image", "🎥 Video"])

# ------------------------------
# Image tab
# ------------------------------
with img_tab:
    st.subheader("Image Inference")
    img_file = st.file_uploader("Upload EL image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
    cam_img = st.camera_input("Or capture from webcam (optional)")

    source_image = None
    stem = None
    if img_file is not None:
        source_image = Image.open(img_file).convert("RGB")  # PIL
        stem = Path(img_file.name).stem
    elif cam_img is not None:
        source_image = Image.open(cam_img).convert("RGB")
        stem = "webcam_capture"

    if source_image is not None:
        st.image(source_image, caption="Input", use_column_width=True)
        # Convert PIL to numpy BGR for preprocessing then back to RGB
        bgr = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
        rgb = ensure_rgb(bgr, apply_clahe=clahe)

        with st.spinner("Running detection…"):
            results = model(rgb, size=int(imgsz))
            df = results.pandas().xyxy[0]
        annotated = draw_boxes(rgb, df)
        st.image(annotated, caption="Annotated", use_column_width=True)

        st.write("### Detections")
        st.dataframe(df)
        df_download_buttons(df, stem or "image")

        # Download annotated image
        ann_pil = Image.fromarray(annotated)
        buf = io.BytesIO()
        ann_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download annotated image (PNG)",
            data=buf.getvalue(),
            file_name=f"{(stem or 'image')}_annotated.png",
            mime="image/png",
        )

# ------------------------------
# Video tab
# ------------------------------
with vid_tab:
    st.subheader("Video Inference")
    vid_file = st.file_uploader("Upload EL video", type=["mp4", "avi", "mov", "mkv"])
    frame_skip = st.number_input("Frame skip (process every Nth frame)", min_value=1, max_value=20, value=1)

    if vid_file is not None:
        temp_video_path = Path("temp_input_el_video")
        temp_video_path.write_bytes(vid_file.read())
        cap = cv2.VideoCapture(str(temp_video_path))
        if not cap.isOpened():
            st.error("Failed to open video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = Path("el_annotated_output.mp4")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0.0, text="Processing video…")
            all_rows: List[pd.DataFrame] = []
            frame_idx = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % int(frame_skip) != 0:
                    continue
                rgb = ensure_rgb(frame_bgr, apply_clahe=clahe)
                results = model(rgb, size=int(imgsz))
                df = results.pandas().xyxy[0]
                df['frame'] = frame_idx
                all_rows.append(df)
                annotated = draw_boxes(rgb, df)
                writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                if total_frames:
                    progress.progress(min(frame_idx / total_frames, 1.0), text=f"Frame {frame_idx}/{total_frames}")

            cap.release()
            writer.release()
            st.success("Video processed.")
            st.video(str(out_path))

            # Aggregate detections
            if all_rows:
                df_cat = pd.concat(all_rows, ignore_index=True)
                st.write("### Detections (aggregated)")
                st.dataframe(df_cat.head(500))  # show preview
                df_download_buttons(df_cat, Path(vid_file.name).stem)
                # Download annotated video
                with open(out_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download annotated video (MP4)",
                        data=f.read(),
                        file_name=f"{Path(vid_file.name).stem}_annotated.mp4",
                        mime="video/mp4",
                    )

st.markdown(
    """
    ---
    **Notes**
    - This app uses PyTorch Hub to load Ultralytics YOLOv5 and your custom weights.
    - CLAHE improves visibility of faint EL defects; disable if your training already includes contrast normalization.
    - Class filter expects indices (e.g., `0,2,3`). Model class names come from your checkpoint.
    """
)
