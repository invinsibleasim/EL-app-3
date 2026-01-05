
import os
import io
import time
import json
import cv2
import glob
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ---------------------------
# App setup
# ---------------------------
st.set_page_config(page_title="YOLO EL Defect Detection", layout="wide")
st.title("🔍 Object/Defect Recognition Dashboard (Streamlit)")

# Ensure working dirs
for d in ["models", "data/uploaded_data", "output", "tmp"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ---------------------------
# Try Ultralytics (preferred for .pt)
# ---------------------------
ULTRA_AVAILABLE = False
try:
    from ultralytics import YOLO  # supports YOLOv5/8/11
    ULTRA_AVAILABLE = True
except Exception:
    ULTRA_AVAILABLE = False

# ---------------------------
# Utils
# ---------------------------
def pil_to_rgb(img: Image.Image) -> np.ndarray:
    """PIL Image -> NumPy RGB uint8"""
    return np.array(img.convert("RGB"))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def draw_boxes(img_bgr: np.ndarray,
               boxes: List[List[int]],
               classes: List[int],
               scores: List[float],
               class_names: List[str]) -> np.ndarray:
    """Draw rectangles & labels."""
    vis = img_bgr.copy()
    for (x1, y1, x2, y2), cls, score in zip(boxes, classes, scores):
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        color = (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{name} {score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis

def parse_ultralytics_result(res) -> Dict[str, Any]:
    """
    Parse one Ultralytics result object (YOLOv8/11).
    Returns dict with xyxy boxes (int), classes, scores.
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
    """Create a ZIP in memory. file_tuples: [(zip_path_inside, bytes), ...]"""
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for zpath, data in file_tuples:
            zf.writestr(zpath, data)
    buf.seek(0)
    return buf

# ---------------------------
# Optional ONNX (OpenCV DNN) fallback
# ---------------------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms_xyxy(boxes, scores, iou_thres=0.45):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        idxs = rest[iou < iou_thres]
    return keep

def run_onnx(net, img_bgr, input_size=640, conf_thres=0.25, iou_thres=0.45):
    """Run OpenCV DNN on ONNX with decoded heads (expects [N, 5+C])."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized, r, dwdh = letterbox(img_rgb, (input_size, input_size))
    blob = resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    net.setInput(blob)
    pred = net.forward()
    pred = np.squeeze(pred)
    if pred.ndim != 2 or pred.shape[1] < 6:
        return {"boxes": [], "scores": [], "classes": []}
    boxes = pred[:, :4]
    obj = pred[:, 4]
    cls_scores = pred[:, 5:]
    cls_ids = cls_scores.argmax(axis=1)
    cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
    conf = obj * cls_conf
    mask = conf >= conf_thres
    boxes, conf, cls_ids = boxes[mask], conf[mask], cls_ids[mask]
    if boxes.size == 0:
        return {"boxes": [], "scores": [], "classes": []}
    boxes = xywh2xyxy(boxes)
    boxes[:, [0, 2]] -= dwdh[0]
    boxes[:, [1, 3]] -= dwdh[1]
    boxes /= r
    H, W = img_rgb.shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, W - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, H - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, W - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, H - 1)
    keep = nms_xyxy(boxes, conf, iou_thres)
    return {
        "boxes": boxes[keep].astype(int).tolist(),
        "scores": [float(s) for s in conf[keep]],
        "classes": [int(c) for c in cls_ids[keep]]
    }

# ---------------------------
# Sidebar - Model & settings
# ---------------------------
st.sidebar.header("🧠 Model")
model_source = st.sidebar.radio("Choose inference backend", ["Ultralytics (.pt)", "OpenCV DNN (.onnx)"], index=0 if ULTRA_AVAILABLE else 1)

confidence = st.sidebar.slider("Confidence", 0.05, 1.0, 0.45, 0.05)
iou_thres  = st.sidebar.slider("IoU (NMS)", 0.10, 0.95, 0.45, 0.05)
imgsz      = st.sidebar.selectbox("imgsz (inference size)", [640, 512, 416], index=0)

# Upload model
model_pt_path = None
model_onnx_path = None
net = None
ultra_model = None
class_names: List[str] = []

if model_source == "Ultralytics (.pt)":
    if not ULTRA_AVAILABLE:
        st.sidebar.error("Ultralytics not installed. Please switch to ONNX (OpenCV DNN) or install ultralytics.")
    else:
        model_bytes = st.sidebar.file_uploader("Upload YOLO .pt", type=["pt"])
        if model_bytes:
            model_pt_path = Path("models") / model_bytes.name
            model_pt_path.write_bytes(model_bytes.read())
            try:
                ultra_model = YOLO(str(model_pt_path))
                # Names extraction
                names_dict = getattr(ultra_model, "names", None) or {}
                class_names = [names_dict[i] for i in sorted(names_dict.keys())] if names_dict else [f"class_{i}" for i in range(1)]
                st.sidebar.success(f"Loaded model: {model_pt_path.name}")
                st.sidebar.info(f"Model classes: {len(class_names)} — {class_names}")
            except Exception as e:
                st.sidebar.error(f"Failed to load .pt with Ultralytics: {e}")
else:
    onnx_bytes = st.sidebar.file_uploader("Upload YOLO ONNX (.onnx)", type=["onnx"])
    if onnx_bytes:
        model_onnx_path = Path("models") / onnx_bytes.name
        model_onnx_path.write_bytes(onnx_bytes.read())
        try:
            net = cv2.dnn.readNetFromONNX(str(model_onnx_path))
            st.sidebar.success(f"Loaded ONNX: {model_onnx_path.name}")
        except Exception as e:
            st.sidebar.error(f"Failed to load ONNX: {e}")

# Optional class names (for ONNX or override)
class_names_str = st.sidebar.text_area("Class names (comma-separated)", "hotspot, crack, delamination")
user_class_names = [s.strip() for s in class_names_str.split(",") if s.strip()]
if user_class_names:
    class_names = user_class_names

# Class filter
use_class_filter = st.sidebar.checkbox("Filter classes for display", False)
filter_classes = []
if use_class_filter and class_names:
    filter_classes = st.sidebar.multiselect("Select classes to show", class_names, default=class_names)

# Input choice
st.sidebar.header("📥 Input")
input_option = st.sidebar.radio("Select input type", ["image", "video"])
data_src     = st.sidebar.radio("Select input source", ["Upload your own data", "Sample data"])

# ---------------------------
# Inference helpers
# ---------------------------
def run_ultralytics_image(model, rgb_arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Run Ultralytics on a single RGB array. Returns RGB overlay and counts."""
    results = model.predict(
        source=rgb_arr,
        imgsz=int(imgsz),
        conf=confidence,
        iou=iou_thres,
        device="cpu",
        verbose=False
    )
    res = results[0]
    parsed = parse_ultralytics_result(res)
    # Class filter
    if filter_classes and class_names:
        keep = []
        for k, cid in enumerate(parsed["classes"]):
            cname = class_names[cid] if cid < len(class_names) else str(cid)
            if cname in filter_classes:
                keep.append(k)
        parsed["boxes"]   = [parsed["boxes"][k]   for k in keep]
        parsed["classes"] = [parsed["classes"][k] for k in keep]
        parsed["scores"]  = [parsed["scores"][k]  for k in keep]
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    vis_bgr = draw_boxes(bgr, parsed["boxes"], parsed["classes"], parsed["scores"], class_names)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    # counts
    counts = {}
    for cid in parsed["classes"]:
        cname = class_names[cid] if cid < len(class_names) else str(cid)
        counts[cname] = counts.get(cname, 0) + 1
    return vis_rgb, counts

def run_onnx_image(net, bgr_arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    det = run_onnx(net, bgr_arr, input_size=int(imgsz), conf_thres=confidence, iou_thres=iou_thres)
    # Class filter
    if filter_classes and class_names:
        keep = []
        for k, cid in enumerate(det["classes"]):
            cname = class_names[cid] if cid < len(class_names) else str(cid)
            if cname in filter_classes:
                keep.append(k)
        det["boxes"]   = [det["boxes"][k]   for k in keep]
        det["classes"] = [det["classes"][k] for k in keep]
        det["scores"]  = [det["scores"][k]  for k in keep]
    vis_bgr = draw_boxes(bgr_arr, det["boxes"], det["classes"], det["scores"], class_names)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    counts = {}
    for cid in det["classes"]:
        cname = class_names[cid] if cid < len(class_names) else str(cid)
        counts[cname] = counts.get(cname, 0) + 1
    return vis_rgb, counts

# ---------------------------
# Image input
# ---------------------------
def image_input_handler():
    img_file = None
    if data_src == "Sample data":
        img_paths = sorted(glob.glob("data/sample_images/*"))
        if not img_paths:
            st.warning("No sample images found in 'data/sample_images'. Please upload your own.")
            return
        idx = st.slider("Select a test image", 1, len(img_paths), 1)
        img_file = img_paths[idx - 1]
        img = Image.open(img_file).convert("RGB")
        rgb = pil_to_rgb(img)
    else:
        img_bytes = st.file_uploader("Upload an image", type=["png", "jpeg", "jpg", "bmp", "tif", "tiff"])
        if not img_bytes:
            st.info("Upload an image to run inference.")
            return
        ext = img_bytes.name.split(".")[-1]
        img_file = f"data/uploaded_data/upload.{ext}"
        Image.open(img_bytes).save(img_file)
        rgb = pil_to_rgb(Image.open(img_file))

    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb, caption="Selected Image", use_column_width=True)

    with col2:
        if model_source == "Ultralytics (.pt)" and ULTRA_AVAILABLE and ultra_model is not None:
            vis_rgb, counts = run_ultralytics_image(ultra_model, rgb)
        elif model_source == "OpenCV DNN (.onnx)" and net is not None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            vis_rgb, counts = run_onnx_image(net, bgr)
        else:
            st.error("Model not loaded. Please upload the correct model for the selected backend.")
            return
        st.image(vis_rgb, caption="Model Prediction", use_column_width=True)
        st.json({"detections_per_class": counts})

# ---------------------------
# Video input
# ---------------------------
def video_input_handler():
    vid_file = None
    if data_src == "Sample data":
        sample_vid = "data/sample_videos/sample.mp4"
        if not Path(sample_vid).exists():
            st.warning("No sample video found at 'data/sample_videos/sample.mp4'. Please upload your own.")
            return
        vid_file = sample_vid
    else:
        vid_bytes = st.file_uploader("Upload a video", type=["mp4", "mpv", "avi", "mov", "mkv"])
        if not vid_bytes:
            st.info("Upload a video to run inference.")
            return
        ext = vid_bytes.name.split(".")[-1]
        vid_file = f"data/uploaded_data/upload.{ext}"
        with open(vid_file, "wb") as f:
            f.write(vid_bytes.read())

    cap = cv2.VideoCapture(vid_file)
    if not cap.isOpened():
        st.error("Cannot open video.")
        return

    # Custom size controls
    custom_size = st.checkbox("Custom frame size")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if custom_size:
        width = st.number_input("Width", min_value=120, step=20, value=width)
        height = st.number_input("Height", min_value=120, step=20, value=height)

    fps_live = 0.0
    st1, st2, st3 = st.columns(3)
    with st1:
        st.markdown("## Height")
        st1_text = st.markdown(f"{height}")
    with st2:
        st.markdown("## Width")
        st2_text = st.markdown(f"{width}")
    with st3:
        st.markdown("## FPS")
        st3_text = st.markdown(f"{fps_live:.2f}")

    st.markdown("---")
    output = st.empty()
    prev_time = time.time()

    # Write output video to disk
    out_dir = Path("output") / Path(vid_file).stem
    ensure_dir(out_dir)
    out_mp4 = out_dir / "overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, cap.get(cv2.CAP_PROP_FPS) or 25, (int(width), int(height)))

    processed_frames = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            st.write("Stream ended.")
            break
        frame_bgr = cv2.resize(frame_bgr, (int(width), int(height)))

        if model_source == "Ultralytics (.pt)" and ULTRA_AVAILABLE and ultra_model is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            vis_rgb, _counts = run_ultralytics_image(ultra_model, rgb)
            vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        elif model_source == "OpenCV DNN (.onnx)" and net is not None:
            vis_rgb, _counts = run_onnx_image(net, frame_bgr)
            vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        else:
            st.error("Model not loaded. Please upload the correct model for the selected backend.")
            break

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
# Main routing
# ---------------------------
input_option = input_option = input_option  # already defined above
if input_option == "image":
    image_input_handler()
else:
    video_input_handler()
``

