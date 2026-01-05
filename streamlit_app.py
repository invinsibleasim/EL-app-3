
import os
import io
import time
from pathlib import Path
import tempfile
import json
import warnings

import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image

#############################
# Utility & Model Loading   #
#############################

def save_uploaded_file(uploaded_file, suffix=""):
    """Save an uploaded file to a temporary location and return the path."""
    tmp_dir = tempfile.mkdtemp()
    filename = uploaded_file.name
    path = os.path.join(tmp_dir, f"{Path(filename).stem}{suffix}{Path(filename).suffix}")
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def parse_selected_classes(selected_names, model_names):
    """Map selected class names to indices."""
    if selected_names is None or len(selected_names) == 0:
        return None
    # Build name->index map
    if isinstance(model_names, dict):
        name_to_idx = {v: k for k, v in model_names.items()}
    else:
        name_to_idx = {v: i for i, v in enumerate(model_names)}
    indices = []
    for name in selected_names:
        if name in name_to_idx:
            indices.append(name_to_idx[name])
        else:
            st.warning(f"Class name '{name}' not found in model.names")
    return sorted(set(indices)) if indices else None


@st.cache_resource(show_spinner=True)
def load_yolov5_model(weights_path, repo_dir=None, device=None):
    """Load YOLOv5 custom model. Uses local repo if provided; else tries torch.hub."""
    dev = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if repo_dir and os.path.isdir(repo_dir):
            model = torch.hub.load(repo_dir, "custom", path=weights_path, source="local", trust_repo=True)
        else:
            model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, trust_repo=True)
    except Exception as e:
        raise RuntimeError(
            "Failed to load YOLOv5 model. If offline, set 'YOLOv5 repo (local)' to a local clone."
            f"Original error: {e}"
        )
    model.to(dev)
    model.eval()
    return model, dev


def run_inference_on_image(model, img_path, out_dir, imgsz):
    """Run inference and save annotated image + CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = model(str(img_path), size=int(imgsz))

    # Save annotated image
    try:
        results.save(save_dir=str(out_dir))
    except TypeError:
        # Fallback manual render
        results.render()
        im = results.ims[0]  # BGR np array
        stem = Path(img_path).stem
        out_path = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), im)

    # Save CSV
    try:
        df = results.pandas().xyxy[0]
        stem = Path(img_path).stem
        (out_dir / f"{stem}.csv").write_text(df.to_csv(index=False))
    except Exception as e:
        warnings.warn(f"Could not save detection CSV: {e}")

    return results


#############################
# Streamlit UI              #
#############################

st.set_page_config(page_title="EL Defect Detection (YOLOv5)", layout="wide")

st.title("EL Defect Detection using YOLOv5 (best.pt)")
st.write(
    "Upload EL images and a trained YOLOv5 weights file (best.pt). Configure confidence, IOU, and classes, "
    "then run detection. Annotated images and CSVs will be saved to an output folder."
)

with st.sidebar:
    st.header("Model & Settings")
    weights_upload = st.file_uploader("Upload YOLOv5 weights (.pt)", type=["pt"], accept_multiple_files=False)
    repo_dir = st.text_input(
        "YOLOv5 repo (local)",
        help=(
            "Optional: local path to cloned YOLOv5 repo (e.g., ./yolov5). "
            "Use this if you're in an offline environment."
        ),
        value=""
    )
    device_choice = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)

    conf_thres = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou_thres = st.slider("NMS IoU threshold", 0.0, 1.0, 0.45, 0.01)
    imgsz = st.number_input("Image size (pixels)", min_value=320, max_value=4096, value=1280, step=64)
    max_det = st.number_input("Max detections", min_value=1, max_value=10000, value=1000, step=10)
    agnostic_nms = st.checkbox("Agnostic NMS", value=False)

    out_root = st.text_input("Output root folder", value="./runs/el_streamlit")

st.divider()

st.subheader("Upload EL images")
uploaded_images = st.file_uploader(
    "Upload one or more EL images", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True
)

run_btn = st.button("Run Detection", type="primary")

model = None
model_names = None
model_device = None
weights_path = None

if weights_upload is not None:
    weights_path = save_uploaded_file(weights_upload)

if run_btn:
    if weights_path is None:
        st.error("Please upload a YOLOv5 weights file (best.pt) to proceed.")
        st.stop()
    if uploaded_images is None or len(uploaded_images) == 0:
        st.error("Please upload at least one EL image.")
        st.stop()

    # Load model
    try:
        repo_arg = repo_dir if repo_dir.strip() else None
        device_arg = None if device_choice == "auto" else device_choice
        with st.spinner("Loading YOLOv5 model..."):
            model, model_device = load_yolov5_model(weights_path, repo_arg, device_arg)
    except Exception as e:
        st.error(str(e))
        st.stop()

    model.conf = float(conf_thres)
    model.iou = float(iou_thres)
    model.max_det = int(max_det)
    model.agnostic = bool(agnostic_nms)

    st.success(f"Model loaded on device: {model_device}")

    # Class selection UI
    model_names = model.names
    if isinstance(model_names, dict):
        cls_display = [model_names[i] for i in sorted(model_names.keys())]
    else:
        cls_display = list(model_names)
    selected_cls_names = st.multiselect("Select classes to detect (optional)", options=cls_display)
    selected_cls_indices = parse_selected_classes(selected_cls_names, model_names)

    # Prepare output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    cfg = {
        "weights": weights_upload.name if weights_upload else None,
        "repo_dir": repo_dir,
        "device": model_device,
        "conf": conf_thres,
        "iou": iou_thres,
        "imgsz": imgsz,
        "max_det": max_det,
        "agnostic_nms": agnostic_nms,
        "selected_classes": selected_cls_names,
        "out_dir": str(out_dir),
    }
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))

    # Apply class filter to model
    model.classes = selected_cls_indices if selected_cls_indices is not None else None

    # Process images
    cols = st.columns(2)
    left, right = cols

    results_summary = []
    for i, upl in enumerate(uploaded_images):
        st.write("")
        with st.spinner(f"Processing image {i+1}/{len(uploaded_images)}: {upl.name}"):
            img_path = save_uploaded_file(upl, suffix="")
            res = run_inference_on_image(model, img_path, out_dir, imgsz=int(imgsz))

            # Display annotated image if present
            stem = Path(img_path).stem
            annotated_path = None
            # YOLOv5 saves annotated image under out_dir / original_filename
            candidate = Path(out_dir) / upl.name
            if candidate.exists():
                annotated_path = candidate
            else:
                # fallback: stem.jpg
                alt = Path(out_dir) / f"{stem}.jpg"
                if alt.exists():
                    annotated_path = alt

            if annotated_path and annotated_path.exists():
                img = Image.open(annotated_path)
                left.image(img, caption=f"Annotated: {upl.name}", use_column_width=True)
            else:
                st.warning("Annotated image not found; check output folder.")

            # Show table of detections
            try:
                df = res.pandas().xyxy[0]
                right.dataframe(df, use_container_width=True)
                results_summary.append({"image": upl.name, "detections": len(df)})
            except Exception as e:
                st.warning(f"No detections table available: {e}")

    st.success(f"Completed. Outputs saved to: {out_dir}")

    # Zip download
    try:
        import shutil
        zip_path = str(out_dir) + ".zip"
        shutil.make_archive(str(out_dir), "zip", str(out_dir))
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Download all outputs (ZIP)",
                data=f,
                file_name=os.path.basename(zip_path),
                mime="application/zip",
            )
    except Exception as e:
        st.warning(f"Could not create ZIP: {e}")

    # Summary
    if results_summary:
        st.subheader("Detection summary")
        st.table(pd.DataFrame(results_summary))

