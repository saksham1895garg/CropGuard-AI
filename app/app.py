"""
CropGuard AI - Streamlit Web Application.
Combines ResNet50 (38-class foliar disease classification) and
U-Net (foliar lesion segmentation) into a unified, beginner-friendly diagnostic tool.
"""

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st

# Ensure repository root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropguard.inference import (
    load_and_orient_image,
    preprocess_classification_input,
    preprocess_segmentation_input,
    load_classification_model,
    load_segmentation_model,
    predict_classification,
    predict_segmentation,
    create_segmentation_overlay,
    load_pest_model,
    predict_pest_detection,
    render_pest_overlay,
    PLANTVILLAGE_CLASSES,
)
from src.cropguard.detection.models import get_detector_metadata

# Page configuration
st.set_page_config(
    page_title="CropGuard AI — Plant Disease & Lesion Diagnostic System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling: clean light theme with dark green accents (#1b4d3e)
st.markdown(
    """
    <style>
    :root {
        --primary-color: #1b4d3e;
        --primary-light: #e8f5e9;
        --accent-color: #2e7d32;
        --border-color: #e0e0e0;
        --text-color: #212529;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4d3e;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #495057;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .diag-card-healthy {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 16px;
        border-radius: 6px;
        margin-bottom: 16px;
    }
    .diag-card-diseased {
        background-color: #fff3e0;
        border-left: 5px solid #e65100;
        padding: 16px;
        border-radius: 6px;
        margin-bottom: 16px;
    }
    .disclaimer-box {
        background-color: #f1f8e9;
        border: 1px solid #c8e6c9;
        border-radius: 8px;
        padding: 16px;
        font-size: 0.9rem;
        color: #2e7d32;
        margin-top: 24px;
    }
    .stProgress > div > div > div > div {
        background-color: #1b4d3e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading ResNet50 Classification Model (24.1M parameters)...")
def cached_classification_model():
    """Loads and caches the production ResNet50 model."""
    return load_classification_model()


@st.cache_resource(show_spinner="Loading U-Net Lesion Segmentation Model (7.77M parameters)...")
def cached_segmentation_model():
    """Loads and caches the production U-Net model."""
    return load_segmentation_model()


@st.cache_resource(show_spinner="Loading YOLO26n Pest Detection Model (3.15M parameters)...")
def cached_pest_model():
    """Loads and caches the YOLO pest detection model."""
    return load_pest_model()


def main():
    # Sidebar
    st.sidebar.image(
        "https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f33f.svg",
        width=48,
    )
    st.sidebar.title("CropGuard AI")
    st.sidebar.markdown(
        "**Multi-Task Deep Learning System: Classification, Segmentation & Pest Detection**"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Diagnostic Controls")

    threshold = st.sidebar.slider(
        "Segmentation Threshold (τ)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability threshold for binarizing U-Net output map. Benchmark default is 0.50.",
    )

    overlay_alpha = st.sidebar.slider(
        "Overlay Transparency (α)",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Opacity of the highlighted lesion overlay over the original leaf.",
    )

    pest_conf = st.sidebar.slider(
        "Pest Detection Confidence (τ_pest)",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Minimum confidence threshold for reporting detected insect pests. Default is 0.25.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Curated Demo Samples")
    sample_options = {
        "None (Upload my own image)": None,
        "Tomato — Late Blight (Lesions)": "app/sample_images/Tomato_Late_Blight.jpg",
        "Apple — Black Rot (Lesions)": "app/sample_images/Apple_Black_Rot.jpg",
        "Corn — Common Rust (Fustules)": "app/sample_images/Corn_Common_Rust.jpg",
        "Tomato — Healthy Leaf": "app/sample_images/Tomato_Healthy.jpg",
        "Rice — Leaf Roller (Insect Pest)": "app/sample_images/Rice_Leaf_Roller_Pest.jpg",
    }
    selected_sample_label = st.sidebar.selectbox("Choose demo preset:", list(sample_options.keys()))
    sample_file_path = sample_options[selected_sample_label]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Production Architecture")
    st.sidebar.markdown(
        """
        - **Classifier:** ResNet50 (24.13M params)
          - *Classes:* 38 PlantVillage categories
          - *Test Accuracy:* **98.42%**
          - *Preproc:* Caffe (RGB-to-BGR + mean sub)
        - **Segmenter:** U-Net (7.77M params)
          - *Dataset:* PlantSeg (2,295 test pairs)
          - *Test Dice:* **63.48%** | *IoU:* **46.49%**
          - *Preproc:* [0.0, 1.0] float32
        - **Detector:** YOLO26n (2.58M params)
          - *Dataset:* IP102 (102 pest categories)
          - *Test mAP@50:* **34.33%** | *mAP@50-95:* **20.79%**
          - *Latency:* **2.03 ms** / image (RTX 5060)
          - *Status:* Checkpoint frozen (`best.pt`)
        """
    )

    # Main header
    st.markdown('<div class="main-title">🌿 CropGuard AI — Agricultural Pathology & Pest Diagnosis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Unified foliar disease classification (ResNet50), lesion segmentation (U-Net), and insect pest detection (YOLO26n).'
        '</div>',
        unsafe_allow_html=True,
    )

    # Step indicator
    cols_step = st.columns(5)
    with cols_step[0]:
        st.info("**Step 1:** Image Input")
    with cols_step[1]:
        st.info("**Step 2:** ResNet50")
    with cols_step[2]:
        st.info("**Step 3:** U-Net")
    with cols_step[3]:
        st.info("**Step 4:** YOLO Detection")
    with cols_step[4]:
        st.info("**Step 5:** Diagnostic Export")

    st.markdown("---")

    # Image upload section
    st.subheader("1. Leaf Image Input")
    uploaded_file = st.file_uploader(
        "Upload a leaf image (JPG, JPEG, or PNG):",
        type=["jpg", "jpeg", "png"],
        help="Upload an RGB leaf image. Mobile camera EXIF orientation is automatically corrected.",
    )

    image_source = None
    if uploaded_file is not None:
        image_source = uploaded_file
        source_name = uploaded_file.name
    elif sample_file_path is not None and os.path.exists(sample_file_path):
        image_source = sample_file_path
        source_name = selected_sample_label
    else:
        st.info("👆 Please upload a leaf image above, or select a curated demo preset in the left sidebar to get started.")
        st.stop()

    # Load and normalize image
    try:
        pil_image = load_and_orient_image(image_source)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # Preview uploaded image
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(pil_image, caption=f"Input Image ({pil_image.width} × {pil_image.height} px)", use_container_width=True)
    with col_info:
        st.markdown(f"**Source:** `{source_name}`")
        st.markdown(f"**Dimensions:** `{pil_image.width} × {pil_image.height}` pixels (RGB)")
        st.markdown("**Status:** Ready for multi-model inference pipeline.")

    # Load models
    try:
        cls_model = cached_classification_model()
        seg_model = cached_segmentation_model()
        pest_model = cached_pest_model()
    except Exception as e:
        st.error(f"Failed to load production models: {e}")
        st.stop()

    st.markdown("---")

    # Run Multi-Model Inference Pipeline
    with st.spinner("Executing ResNet50 classification, U-Net lesion segmentation, and YOLO pest detection..."):
        # Classification inference
        cls_input = preprocess_classification_input(pil_image)
        cls_result = predict_classification(cls_model, cls_input)

        # Segmentation inference
        seg_input = preprocess_segmentation_input(pil_image)
        seg_result = predict_segmentation(seg_model, seg_input, threshold=threshold)

        # Pest detection inference
        pest_result = predict_pest_detection(pest_model, pil_image, conf_threshold=pest_conf)
        pest_metadata = get_detector_metadata(pest_model)

        # Generate overlays
        overlay_image = create_segmentation_overlay(
            original_image=pil_image,
            binary_mask=seg_result["binary_mask"],
            color=(220, 38, 38),
            alpha=overlay_alpha,
        )
        pest_overlay_image = render_pest_overlay(
            image=pil_image,
            detections=pest_result["detections"],
            box_color=(220, 38, 38),
        )

    # 2. Classification Section
    st.subheader("2. Disease Classification (ResNet50)")
    top_pred = cls_result["top_prediction"]
    is_healthy = top_pred["is_healthy"]

    col_cls_top, col_cls_bars = st.columns([1, 1])
    with col_cls_top:
        card_class = "diag-card-healthy" if is_healthy else "diag-card-diseased"
        badge_text = "HEALTHY CROP" if is_healthy else "DISEASE DETECTED"
        st.markdown(
            f"""
            <div class="{card_class}">
                <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                    {badge_text}
                </div>
                <div style="font-size: 1.6rem; font-weight: 700; margin-top: 4px;">
                    {top_pred['full_display']}
                </div>
                <div style="font-size: 1.1rem; margin-top: 6px;">
                    Confidence: <strong>{top_pred['confidence_pct']:.2f}%</strong>
                </div>
                <div style="font-size: 0.85rem; color: #6c757d; margin-top: 4px;">
                    Model: ResNet50 | Latency: {cls_result['latency_ms']:.1f} ms
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_cls_bars:
        st.markdown("**Top-3 Class Candidates:**")
        for cand in cls_result["top_k"]:
            st.write(f"**{cand['rank']}. {cand['full_display']}** ({cand['confidence_pct']:.2f}%)")
            st.progress(min(max(cand["confidence"], 0.0), 1.0))

    st.markdown("---")

    # 3. Segmentation Section
    st.subheader("3. Foliar Lesion Segmentation (U-Net)")
    cov_pct = seg_result["lesion_coverage_pct"]

    # Severity interpretation
    if cov_pct < 1.0:
        severity_label = "Negligible / None (<1%)"
        sev_color = "#2e7d32"
    elif cov_pct < 5.0:
        severity_label = "Mild Lesion Coverage (1–5%)"
        sev_color = "#fbc02d"
    elif cov_pct < 20.0:
        severity_label = "Moderate Foliar Damage (5–20%)"
        sev_color = "#f57c00"
    else:
        severity_label = "Severe Foliar Necrosis (>20%)"
        sev_color = "#d32f2f"

    # Display 3-panel visualization
    seg_cols = st.columns(3)
    with seg_cols[0]:
        st.image(pil_image, caption="Original Leaf Image", use_container_width=True)
    with seg_cols[1]:
        # Binary mask as black & white image
        mask_display = (seg_result["binary_mask"] * 255).astype(np.uint8)
        st.image(
            mask_display,
            caption=f"Binary Lesion Mask (τ={threshold:.2f})",
            use_container_width=True,
        )
    with seg_cols[2]:
        st.image(
            overlay_image,
            caption=f"Lesion Overlay (α={overlay_alpha:.2f})",
            use_container_width=True,
        )

    # Severity metrics cards
    m_cols = st.columns(4)
    with m_cols[0]:
        st.metric(label="Lesion Coverage", value=f"{cov_pct:.2f}%")
    with m_cols[1]:
        st.metric(label="Damage Assessment", value=severity_label)
    with m_cols[2]:
        st.metric(label="Active Threshold", value=f"τ = {threshold:.2f}")
    with m_cols[3]:
        st.metric(label="U-Net Latency", value=f"{seg_result['latency_ms']:.1f} ms")

    st.markdown("---")

    st.markdown("---")

    # 4. Insect Pest Detection Section
    st.subheader("4. Insect Pest Detection — YOLO26n")
    num_pests = pest_result["num_detections"]

    # Metrics row
    pest_metric_cols = st.columns(4)
    with pest_metric_cols[0]:
        st.metric(
            label="Total Detected Pests",
            value=f"{num_pests} pest{'s' if num_pests != 1 else ''}",
        )
    with pest_metric_cols[1]:
        top_name = pest_result["detections"][0]["display_name"] if num_pests > 0 else "None Detected"
        st.metric(label="Top Prediction", value=top_name)
    with pest_metric_cols[2]:
        st.metric(label="Active Threshold", value=f"τ = {pest_conf:.2f}")
    with pest_metric_cols[3]:
        st.metric(label="YOLO26n Latency", value=f"{pest_result['latency_ms']:.1f} ms")

    # Side-by-side display of original uploaded image and annotated bounding-box image
    pest_display_cols = st.columns(2)
    with pest_display_cols[0]:
        st.image(
            pil_image,
            caption="Original Uploaded Image",
            use_container_width=True,
        )
    with pest_display_cols[1]:
        if num_pests > 0:
            st.image(
                pest_overlay_image,
                caption=f"Annotated Image with Bounding Boxes ({num_pests} detected, τ={pest_conf:.2f})",
                use_container_width=True,
            )
        else:
            st.image(
                pil_image,
                caption=f"Annotated Image (0 Bounding Boxes at τ={pest_conf:.2f})",
                use_container_width=True,
            )

    # Detailed detections or graceful zero-detection handling
    if num_pests > 0:
        st.markdown(f"**Identified Insect Pests ({num_pests}):**")
        for det in pest_result["detections"]:
            coords = det["box_xyxy"]
            st.markdown(
                f"- **Box {det['box_index']}: {det['display_name']}** — "
                f"Confidence: **{det['confidence_pct']:.1f}%** | "
                f"Bounding Box Coordinates: `[{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]`"
            )
    else:
        st.markdown(
            f"""
            <div class="diag-card-healthy">
                <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #2e7d32;">
                    🟢 NO INSECT PESTS DETECTED
                </div>
                <div style="font-size: 1.15rem; font-weight: 600; margin-top: 4px; color: #1b4d3e;">
                    0 insect pests identified at confidence threshold τ ≥ {pest_conf:.2f}
                </div>
                <div style="font-size: 0.9rem; color: #495057; margin-top: 4px;">
                    The foliar surface shows no detectable insect bodies or mobile pests under active detection sensitivity. Clean foliar pathology images accurately report zero pests.
                </div>
                <div style="font-size: 0.8rem; color: #6c757d; margin-top: 6px;">
                    Model: YOLO26n (2.58M parameters) | Latency: {pest_result['latency_ms']:.1f} ms | Held-Out Test mAP@50: 34.33%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Technical note on YOLO26n & IP102
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 12px; font-size: 0.85rem; color: #495057;">
            <strong>ℹ️ Module Architecture & Training Status:</strong><br>
            Object detection powered by fine-tuned Ultralytics <strong>YOLO26n</strong> ({pest_metadata['parameter_count']:,} parameters).
            Trained on <strong>IP102</strong> benchmark ({pest_metadata['num_classes']} canonical pest categories).
            Evaluated on official held-out test split (3,798 images): <strong>34.33% mAP@50</strong>, <strong>20.79% mAP@50-95</strong>, <strong>2.03 ms/image latency</strong>. Checkpoint frozen.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # 5. Results & Export Section
    st.subheader("5. Combined Diagnostic Summary & Export")

    summary_data = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_image": source_name,
        "classification": {
            "model": "ResNet50 (24.13M parameters)",
            "top_prediction": top_pred["full_display"],
            "crop": top_pred["crop"],
            "condition": top_pred["condition"],
            "confidence_pct": round(top_pred["confidence_pct"], 2),
            "is_healthy": top_pred["is_healthy"],
            "latency_ms": round(cls_result["latency_ms"], 2),
            "top_3_candidates": [
                {
                    "rank": c["rank"],
                    "class": c["full_display"],
                    "confidence_pct": round(c["confidence_pct"], 2),
                }
                for c in cls_result["top_k"]
            ],
        },
        "segmentation": {
            "model": "U-Net (7.77M parameters)",
            "lesion_coverage_pct": round(cov_pct, 2),
            "severity_assessment": severity_label,
            "threshold": threshold,
            "total_pixels": seg_result["total_pixels"],
            "lesion_pixels": seg_result["lesion_pixels"],
            "latency_ms": round(seg_result["latency_ms"], 2),
        },
        "pest_detection": {
            "model": f"YOLO26n ({pest_metadata['parameter_count']:,} parameters)",
            "num_detections": num_pests,
            "confidence_threshold": pest_conf,
            "status": "No pests detected" if num_pests == 0 else f"{num_pests} pests detected",
            "ontology": pest_result["ontology"],
            "detections": pest_result["detections"],
            "latency_ms": round(pest_result["latency_ms"], 2),
            "test_map50": 0.3433,
            "test_map50_95": 0.2079,
        },
        "system_info": {
            "pipeline": "CropGuard AI Production Tri-Model Pipeline",
            "framework": "TensorFlow / PyTorch Dual Runtime",
        },
    }

    col_sum_text, col_downloads = st.columns([2, 1])

    with col_sum_text:
        st.markdown(
            f"""
            | Diagnostic Dimension | Outcome | Details |
            | :--- | :--- | :--- |
            | **Pathology Diagnosis** | `{top_pred['full_display']}` | Confidence: **{top_pred['confidence_pct']:.2f}%** |
            | **Condition Type** | `{'Healthy Foliage' if is_healthy else 'Infected / Symptomatic'}` | Top-3 Margin: {cls_result['top_k'][0]['confidence_pct'] - cls_result['top_k'][1]['confidence_pct']:.2f}% |
            | **Foliar Lesion Severity** | `{cov_pct:.2f}%` affected area | Classification: `{severity_label}` |
            | **Insect Pest Detection** | `{'0 pests detected' if num_pests == 0 else f'{num_pests} pests detected'}` | Active Threshold: τ = {pest_conf:.2f} \\| Model: YOLO26n |
            | **Total Pipeline Latency** | `{cls_result['latency_ms'] + seg_result['latency_ms'] + pest_result['latency_ms']:.1f} ms` | ResNet50: {cls_result['latency_ms']:.1f}ms \\| U-Net: {seg_result['latency_ms']:.1f}ms \\| YOLO: {pest_result['latency_ms']:.1f}ms |
            """
        )

    with col_downloads:
        st.markdown("**Export Diagnosis:**")

        # JSON download
        json_str = json.dumps(summary_data, indent=2)
        st.download_button(
            label="📄 Download Diagnostic Report (JSON)",
            data=json_str,
            file_name=f"cropguard_diagnosis_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True,
        )

        # PNG download (lesion overlay)
        buf = io.BytesIO()
        overlay_image.save(buf, format="PNG")
        overlay_bytes = buf.getvalue()

        st.download_button(
            label="🖼️ Download Lesion Overlay (PNG)",
            data=overlay_bytes,
            file_name=f"cropguard_lesion_overlay_{int(time.time())}.png",
            mime="image/png",
            use_container_width=True,
        )

        # PNG download (pest overlay if detected)
        if num_pests > 0:
            pest_buf = io.BytesIO()
            pest_overlay_image.save(pest_buf, format="PNG")
            st.download_button(
                label="🐛 Download Pest Detection Overlay (PNG)",
                data=pest_buf.getvalue(),
                file_name=f"cropguard_pest_detection_{int(time.time())}.png",
                mime="image/png",
                use_container_width=True,
            )

    # 6. Scientific Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Scientific & Agricultural Disclaimer:</strong><br>
            <ul>
                <li><strong>Complementary Datasets:</strong> ResNet50 was trained on <em>PlantVillage</em> (38 lab-controlled classes), while U-Net was trained on <em>PlantSeg</em> (pixel-level foliar lesions across 115 crop pathologies).</li>
                <li><strong>Distinct Diagnostic Dimensions:</strong> Classification predicts the probable disease entity based on morphological leaf features; segmentation quantifies visible necrotic/chlorotic surface area.</li>
                <li><strong>Correlation ≠ Causation:</strong> Segmented lesion tissue indicates symptomatic leaf surface area, but does not definitively prove specific fungal, bacterial, or viral causation (abiotic stress, spray injury, and nutritional deficiencies can produce similar necrosis).</li>
                <li><strong>Insect Pest Detection:</strong> Powered by fine-tuned Ultralytics YOLO26n (2.58M parameters) across the 102 canonical IP102 pest classes (CVPR 2019). Evaluated on 3,798 held-out test images (34.33% mAP@50, 20.79% mAP@50-95, 2.03 ms latency). The model reliably reports zero detections on clean foliar pathology images without hallucinating bounding boxes.</li>
                <li><strong>Field Generalization:</strong> Benchmark performance (ResNet50 98.42% accuracy, U-Net 63.48% Dice) was established on curated test partitions. Complex field lighting, canopy overlap, and soil backgrounds may introduce diagnostic variance.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
