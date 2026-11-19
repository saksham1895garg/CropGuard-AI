"""
CropGuard AI - Pest Detection Inference Engine & Bounding Box Visualizer.
Runs inference on YOLO26n and renders high-contrast bounding boxes and diagnostic badges.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from src.cropguard.detection.classes import format_pest_name
from src.cropguard.detection.models import is_ip102_finetuned


def predict_pests(
    model: YOLO,
    image_input: Union[Image.Image, np.ndarray, str, Path],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Performs YOLO object detection on input image and extracts bounding boxes,
    class labels, confidence scores, and inference latency.

    Args:
        model: Loaded YOLO model.
        image_input: RGB PIL Image, numpy array, or file path.
        conf_threshold: Minimum detection confidence threshold (default 0.25).
        iou_threshold: Non-maximum suppression IoU threshold (default 0.45).

    Returns:
        Structured dictionary containing detections, metrics, and model metadata.
    """
    if isinstance(image_input, (str, Path)):
        pil_image = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_image = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    img_w, img_h = pil_image.size
    finetuned = is_ip102_finetuned(model)

    t0 = time.perf_counter()
    results = model.predict(
        source=pil_image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    detections: List[Dict[str, Any]] = []

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                raw_name = model.names.get(cls_id, f"class_{cls_id}") if hasattr(model, "names") else f"class_{cls_id}"
                formatted = format_pest_name(raw_name)

                x1, y1, x2, y2 = xyxy
                # Clip to image boundary
                x1_c = max(0.0, min(float(x1), float(img_w)))
                y1_c = max(0.0, min(float(y1), float(img_h)))
                x2_c = max(0.0, min(float(x2), float(img_w)))
                y2_c = max(0.0, min(float(y2), float(img_h)))

                bw = x2_c - x1_c
                bh = y2_c - y1_c

                if bw <= 0.0 or bh <= 0.0:
                    continue

                xc_norm = ((x1_c + x2_c) / 2.0) / img_w
                yc_norm = ((y1_c + y2_c) / 2.0) / img_h
                w_norm = bw / img_w
                h_norm = bh / img_h

                detections.append({
                    "box_index": i + 1,
                    "box_xyxy": [round(x1_c, 1), round(y1_c, 1), round(x2_c, 1), round(y2_c, 1)],
                    "box_yolo_normalized": [round(xc_norm, 4), round(yc_norm, 4), round(w_norm, 4), round(h_norm, 4)],
                    "confidence": conf,
                    "confidence_pct": conf * 100.0,
                    "class_id": cls_id,
                    "raw_class": raw_name,
                    "display_name": formatted,
                    "is_pest_class": finetuned,
                })

    num_det = len(detections)
    if num_det == 0:
        status_msg = f"No insect pests detected (0 bounding boxes found at confidence ≥ {conf_threshold:.2f})."
    else:
        status_msg = f"Detected {num_det} object{'s' if num_det > 1 else ''} at confidence ≥ {conf_threshold:.2f}."

    return {
        "num_detections": num_det,
        "detections": detections,
        "latency_ms": latency_ms,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "is_ip102_finetuned": finetuned,
        "ontology": "IP102 Agricultural Pests (102 classes)" if finetuned else "COCO Benchmark (80 classes)",
        "status_message": status_msg,
    }


def render_pest_detections(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (220, 38, 38),  # Crimson red
    text_color: Tuple[int, int, int] = (255, 255, 255),  # White
    line_width: int = 3,
) -> Image.Image:
    """
    Renders high-contrast bounding boxes and label badges onto a PIL Image.

    Args:
        image: Original RGB PIL image.
        detections: List of detection dictionaries from predict_pests().
        box_color: RGB tuple for bounding box and badge fill.
        text_color: RGB tuple for label text.
        line_width: Pixel width of bounding box edges.

    Returns:
        Annotated PIL Image.
    """
    canvas = image.copy().convert("RGB")
    if not detections:
        return canvas

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x1, y1, x2, y2 = det["box_xyxy"]
        label = f"{det['display_name']} {det['confidence_pct']:.1f}%"

        # Draw box outline
        for offset in range(line_width):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=box_color,
            )

        # Draw text badge
        if font is not None:
            bbox = font.getbbox(label)
            tw = bbox[2] - bbox[0] + 8
            th = bbox[3] - bbox[1] + 6
        else:
            tw = len(label) * 7 + 8
            th = 16

        badge_y1 = max(0, y1 - th - 2)
        badge_y2 = badge_y1 + th
        badge_x2 = min(canvas.width, x1 + tw)

        draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=box_color)
        draw.text((x1 + 4, badge_y1 + 2), label, fill=text_color, font=font)

    return canvas
