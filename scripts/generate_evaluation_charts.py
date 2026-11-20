"""
Generate high-resolution evaluation charts for CropGuard AI presentation.
All figures are saved in presentation/assets/
Authoritative data source: reports/final_evaluation/*.json
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Ensure output directory exists
OUT_DIR = "presentation/assets"
os.makedirs(OUT_DIR, exist_ok=True)

# Set global matplotlib style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = False

# Consistent Palette
C_PRIMARY = "#1B4D3E"      # Deep Forest Green
C_ACCENT = "#2E7D32"       # Plant Green
C_AMBER = "#C27D00"        # Earthy Gold / Amber
C_AMBER_LIGHT = "#FEF3C7"  # Soft Amber Tint
C_BLUE = "#0284C7"         # Clean Sky / Teal Blue
C_BLUE_LIGHT = "#E0F2FE"   # Soft Blue Tint
C_GREEN_LIGHT = "#E8F5E9"  # Soft Green Tint
C_DARK = "#1E293B"         # Dark Slate Text
C_MUTED = "#64748B"        # Slate Muted Text
C_CARD_BG = "#FFFFFF"      # White Card Fill
C_BORDER = "#E2E8F0"       # Subtle Gray Border
C_CANVAS_BG = "#F8F9FA"    # Light Neutral Canvas


# -----------------------------------------------------------------------------
# Chart 1: dataset_splits.png
# -----------------------------------------------------------------------------
def generate_dataset_splits():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), facecolor=C_CANVAS_BG)
    fig.suptitle("Dataset Partitions & Empirical Split Integrity (Zero-Leakage)", 
                 fontsize=18, fontweight='bold', color=C_PRIMARY, y=0.98)

    datasets = [
        {
            "name": "PlantVillage",
            "task": "Foliar Disease Classification",
            "total": 54305,
            "classes": "38 Classes (14 Crops)",
            "splits": [("Train", 35221, 64.86, "#1B4D3E"),
                       ("Val", 8085, 14.89, "#D97706"),
                       ("Test", 10999, 20.25, "#0284C7")],
            "note": "Group-Aware Partition\n0% Physical Leaf Leakage"
        },
        {
            "name": "PlantSeg",
            "task": "Lesion Semantic Segmentation",
            "total": 11458,
            "classes": "115 Disease Categories",
            "splits": [("Train", 7916, 69.09, "#1B4D3E"),
                       ("Val", 1247, 10.88, "#D97706"),
                       ("Test", 2295, 20.03, "#0284C7")],
            "note": "Official Benchmark Partition\nZero Cross-Split Contamination"
        },
        {
            "name": "IP102 Benchmark",
            "task": "Insect Pest Detection",
            "total": 18976,
            "classes": "102 Insect Taxa (22,283 Boxes)",
            "splits": [("Train", 12898, 67.97, "#1B4D3E"),
                       ("Val", 2280, 12.02, "#D97706"),
                       ("Test", 3798, 20.01, "#0284C7")],
            "note": "15,178 trainval (85/15 Split)\n3,798 Official Test Quarantined"
        }
    ]

    for ax, d in zip(axes, datasets):
        ax.set_facecolor(C_CANVAS_BG)
        labels = [s[0] for s in d["splits"]]
        counts = [s[1] for s in d["splits"]]
        pcts = [s[2] for s in d["splits"]]
        colors = [s[3] for s in d["splits"]]

        # Donut chart
        wedges, _ = ax.pie(
            counts, 
            colors=colors, 
            startangle=90, 
            wedgeprops=dict(width=0.38, edgecolor=C_CANVAS_BG, linewidth=2.5)
        )

        # Center text: Total Count & Title
        ax.text(0, 0.12, f"{d['total']:,}", ha='center', va='center', 
                fontsize=16, fontweight='bold', color=C_DARK)
        ax.text(0, -0.08, "Total Samples", ha='center', va='center', 
                fontsize=10, color=C_MUTED, fontweight='semibold')
        ax.text(0, -0.22, d["classes"], ha='center', va='center', 
                fontsize=8.5, color=C_PRIMARY, fontweight='bold')

        # Subplot Titles
        ax.set_title(f"{d['name']}\n{d['task']}", fontsize=13, fontweight='bold', 
                     color=C_PRIMARY, pad=12)

        # Bottom annotation box
        ax.text(0, -1.35, d["note"], ha='center', va='center', fontsize=9.5, 
                fontweight='semibold', color=C_DARK,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFFF', 
                          edgecolor=C_BORDER, linewidth=1.2))

        # Direct Legend inside each panel with counts and pcts
        legend_labels = [f"{s[0]}: {s[1]:,} ({s[2]:.1f}%)" for s in d["splits"]]
        ax.legend(wedges, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  frameon=False, fontsize=9.5)

    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.94])
    out_file = os.path.join(OUT_DIR, "dataset_splits.png")
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_file}")


# -----------------------------------------------------------------------------
# Chart 2: resnet50_results.png
# -----------------------------------------------------------------------------
def generate_resnet50_results():
    fig = plt.figure(figsize=(13.5, 5.2), facecolor=C_CANVAS_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_facecolor(C_CANVAS_BG)

    # Title Banner
    ax.text(0.5, 0.91, "ResNet50 — Foliar Disease Classification", 
            ha='center', va='center', fontsize=18, fontweight='bold', color=C_PRIMARY)
    ax.text(0.5, 0.83, "Evaluated on 10,999 Held-Out Test Images  |  Group-Aware Partition (0% Leakage)  |  38 Classes", 
            ha='center', va='center', fontsize=11, fontweight='semibold', color=C_MUTED)

    # 3 Large Metric Cards
    cards = [
        {"metric": "98.42%", "name": "Test Accuracy", "desc": "Overall classification accuracy across all 10,999 test samples", "color": C_PRIMARY},
        {"metric": "97.79%", "name": "Macro F1-Score", "desc": "Balanced harmonic mean across all 38 classes regardless of support", "color": C_ACCENT},
        {"metric": "99.98%", "name": "Macro ROC-AUC", "desc": "Near-perfect probabilistic discriminative ranking power across all taxa", "color": C_AMBER}
    ]

    card_w = 0.28
    card_h = 0.52
    gap = 0.04
    start_x = 0.05
    y_card = 0.24

    for i, c in enumerate(cards):
        x = start_x + i * (card_w + gap)
        # Card Background
        rect = patches.FancyBboxPatch(
            (x, y_card), card_w, card_h,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=C_CARD_BG, edgecolor=C_BORDER, linewidth=1.5
        )
        ax.add_patch(rect)

        # Top Accent Ribbon
        ribbon = patches.FancyBboxPatch(
            (x + 0.005, y_card + card_h - 0.02), card_w - 0.01, 0.015,
            boxstyle="round,pad=0.005,rounding_size=0.01",
            facecolor=c["color"], edgecolor="none"
        )
        ax.add_patch(ribbon)

        # Big Number
        ax.text(x + card_w/2, y_card + 0.35, c["metric"], 
                ha='center', va='center', fontsize=34, fontweight='bold', color=c["color"])
        # Metric Title
        ax.text(x + card_w/2, y_card + 0.22, c["name"], 
                ha='center', va='center', fontsize=14, fontweight='bold', color=C_DARK)
        # Metric Description
        ax.text(x + card_w/2, y_card + 0.10, c["desc"], 
                ha='center', va='center', fontsize=9.5, color=C_MUTED, wrap=True, multialignment='center')

    # Bottom Metadata Pill
    pill = patches.FancyBboxPatch(
        (0.05, 0.06), 0.90, 0.12,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#FFFFFF", edgecolor=C_BORDER, linewidth=1.2
    )
    ax.add_patch(pill)
    ax.text(0.5, 0.12, 
            "Macro Precision: 97.82%  |  Macro Recall: 97.81%  |  Inference Latency: 81.76 ms/img (Batch Size 1, GPU)  |  Parameters: 24,130,214", 
            ha='center', va='center', fontsize=10.5, fontweight='bold', color=C_PRIMARY)

    out_file = os.path.join(OUT_DIR, "resnet50_results.png")
    plt.savefig(out_file, dpi=300, facecolor=C_CANVAS_BG, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_file}")


# -----------------------------------------------------------------------------
# Chart 3: unet_results.png
# -----------------------------------------------------------------------------
def generate_unet_results():
    fig = plt.figure(figsize=(15, 5.8), facecolor=C_CANVAS_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_facecolor(C_CANVAS_BG)

    # Title Banner
    ax.text(0.5, 0.92, "U-Net — Pixel-Level Lesion Semantic Segmentation", 
            ha='center', va='center', fontsize=18, fontweight='bold', color=C_PRIMARY)
    ax.text(0.5, 0.84, "Evaluated on 2,295 Held-Out Test Image-Mask Pairs  |  PlantSeg Official Partition  |  115 Disease Categories", 
            ha='center', va='center', fontsize=11, fontweight='semibold', color=C_MUTED)

    # 5 Metric Cards (Layout 5 columns)
    cards = [
        {"metric": "63.48%", "name": "Dice Coeff", "desc": "Sørensen-Dice harmonic overlap index", "color": C_PRIMARY},
        {"metric": "46.49%", "name": "Mean IoU", "desc": "Jaccard spatial boundary overlap index", "color": C_ACCENT},
        {"metric": "94.04%", "name": "Specificity", "desc": "Healthy tissue preservation rate", "color": C_PRIMARY},
        {"metric": "71.79%", "name": "Precision", "desc": "True positive segmented lesion pixels", "color": C_AMBER},
        {"metric": "56.89%", "name": "Recall", "desc": "Sensitivity to diffuse chlorotic borders", "color": "#0284C7"}
    ]

    card_w = 0.17
    card_h = 0.50
    gap = 0.022
    start_x = 0.04
    y_card = 0.25

    for i, c in enumerate(cards):
        x = start_x + i * (card_w + gap)
        # Card Background
        rect = patches.FancyBboxPatch(
            (x, y_card), card_w, card_h,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=C_CARD_BG, edgecolor=C_BORDER, linewidth=1.5
        )
        ax.add_patch(rect)

        # Top Accent Ribbon
        ribbon = patches.FancyBboxPatch(
            (x + 0.005, y_card + card_h - 0.018), card_w - 0.01, 0.013,
            boxstyle="round,pad=0.005,rounding_size=0.01",
            facecolor=c["color"], edgecolor="none"
        )
        ax.add_patch(ribbon)

        # Big Number
        ax.text(x + card_w/2, y_card + 0.33, c["metric"], 
                ha='center', va='center', fontsize=26, fontweight='bold', color=c["color"])
        # Metric Title
        ax.text(x + card_w/2, y_card + 0.20, c["name"], 
                ha='center', va='center', fontsize=12.5, fontweight='bold', color=C_DARK)
        # Metric Description
        ax.text(x + card_w/2, y_card + 0.09, c["desc"], 
                ha='center', va='center', fontsize=8.5, color=C_MUTED, multialignment='center')

    # Bottom Metadata Pill
    pill = patches.FancyBboxPatch(
        (0.04, 0.06), 0.92, 0.12,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#FFFFFF", edgecolor=C_BORDER, linewidth=1.2
    )
    ax.add_patch(pill)
    ax.text(0.5, 0.12, 
            "Ground Truth Severity: 21.04%  |  Predicted Severity: 16.67%  |  Inference Latency: 20.71 ms/img (Batch Size 8, 48.3 FPS)  |  Parameters: 7,771,873", 
            ha='center', va='center', fontsize=10.5, fontweight='bold', color=C_PRIMARY)

    out_file = os.path.join(OUT_DIR, "unet_results.png")
    plt.savefig(out_file, dpi=300, facecolor=C_CANVAS_BG, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_file}")


# -----------------------------------------------------------------------------
# Chart 4: yolo_results.png
# -----------------------------------------------------------------------------
def generate_yolo_results():
    fig = plt.figure(figsize=(14, 5.2), facecolor=C_CANVAS_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_facecolor(C_CANVAS_BG)

    # Title Banner
    ax.text(0.5, 0.91, "YOLO26n — Insect Pest Detection", 
            ha='center', va='center', fontsize=18, fontweight='bold', color=C_PRIMARY)
    ax.text(0.5, 0.83, "Evaluated on 3,798 Official Held-Out Test Images  |  IP102 Benchmark  |  96 Evaluated Classes (102 Canonical)", 
            ha='center', va='center', fontsize=11, fontweight='semibold', color=C_MUTED)

    # 4 Large Metric Cards
    cards = [
        {"metric": "34.33%", "name": "mAP @ 0.50", "desc": "Standard object detection metric at 50% IoU threshold", "color": C_PRIMARY},
        {"metric": "20.79%", "name": "mAP @ 0.50:0.95", "desc": "Stringent multi-IoU threshold average (COCO standard)", "color": C_BLUE},
        {"metric": "44.31%", "name": "Precision", "desc": "Detection validity across 102 pest taxa", "color": C_ACCENT},
        {"metric": "38.64%", "name": "Recall", "desc": "Pest retrieval rate in complex outdoor field foliage", "color": C_AMBER}
    ]

    card_w = 0.21
    card_h = 0.52
    gap = 0.03
    start_x = 0.045
    y_card = 0.24

    for i, c in enumerate(cards):
        x = start_x + i * (card_w + gap)
        # Card Background
        rect = patches.FancyBboxPatch(
            (x, y_card), card_w, card_h,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=C_CARD_BG, edgecolor=C_BORDER, linewidth=1.5
        )
        ax.add_patch(rect)

        # Top Accent Ribbon
        ribbon = patches.FancyBboxPatch(
            (x + 0.005, y_card + card_h - 0.02), card_w - 0.01, 0.015,
            boxstyle="round,pad=0.005,rounding_size=0.01",
            facecolor=c["color"], edgecolor="none"
        )
        ax.add_patch(ribbon)

        # Big Number
        ax.text(x + card_w/2, y_card + 0.35, c["metric"], 
                ha='center', va='center', fontsize=30, fontweight='bold', color=c["color"])
        # Metric Title
        ax.text(x + card_w/2, y_card + 0.22, c["name"], 
                ha='center', va='center', fontsize=13.5, fontweight='bold', color=C_DARK)
        # Metric Description
        ax.text(x + card_w/2, y_card + 0.10, c["desc"], 
                ha='center', va='center', fontsize=9, color=C_MUTED, multialignment='center')

    # Bottom Metadata Pill
    pill = patches.FancyBboxPatch(
        (0.045, 0.06), 0.91, 0.12,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#FFFFFF", edgecolor=C_BORDER, linewidth=1.2
    )
    ax.add_patch(pill)
    ax.text(0.5, 0.12, 
            "Model Inference: 2.03 ms/img  |  Total Latency: 3.16 ms/img (316.5 FPS)  |  Parameters: 2,582,972 (Edge-Ready Architecture)", 
            ha='center', va='center', fontsize=10.5, fontweight='bold', color=C_PRIMARY)

    out_file = os.path.join(OUT_DIR, "yolo_results.png")
    plt.savefig(out_file, dpi=300, facecolor=C_CANVAS_BG, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_file}")


# -----------------------------------------------------------------------------
# Chart 5: model_latency.png
# -----------------------------------------------------------------------------
def generate_model_latency():
    fig, (ax_bar, ax_detail) = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=C_CANVAS_BG, 
                                            gridspec_kw={'width_ratios': [1.3, 1]})
    
    fig.suptitle("Operational Inference Latency Profile (GPU Execution)", 
                 fontsize=17, fontweight='bold', color=C_PRIMARY, y=0.98)

    # Left: Horizontal Bar Chart
    models = ["ResNet50\n(Batch Size 1)", "U-Net\n(Batch Size 8)", "YOLO26n Total\n(Letterbox Stream)", "YOLO26n Model\n(Inference Core)"]
    latencies = [81.76, 20.71, 3.16, 2.03]
    colors = ["#1B4D3E", "#2E7D32", "#0284C7", "#38BDF8"]

    y_pos = np.arange(len(models))
    bars = ax_bar.barh(y_pos, latencies, color=colors, height=0.55, edgecolor=C_BORDER, linewidth=1)
    ax_bar.set_facecolor(C_CARD_BG)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(models, fontsize=10.5, fontweight='semibold', color=C_DARK)
    ax_bar.invert_yaxis()  # Labels read top-to-bottom
    ax_bar.set_xlabel("Latency per Image (milliseconds)", fontsize=11, fontweight='bold', color=C_DARK)
    ax_bar.grid(axis='x', linestyle='--', alpha=0.5)

    # Add data labels
    for bar, lat in zip(bars, latencies):
        width = bar.get_width()
        fps = 1000.0 / lat
        ax_bar.text(width + 1.8, bar.get_y() + bar.get_height()/2, 
                    f"{lat:.2f} ms  ({fps:.1f} FPS)", 
                    va='center', ha='left', fontsize=10.5, fontweight='bold', color=C_DARK)

    ax_bar.set_xlim(0, 105)

    # Right: YOLO Pipeline Breakdown + Engineering Disclosure Box
    ax_detail.axis('off')
    ax_detail.set_facecolor(C_CANVAS_BG)

    # Card for YOLO breakdown
    rect_breakdown = patches.FancyBboxPatch(
        (0.05, 0.48), 0.90, 0.48,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=C_CARD_BG, edgecolor=C_BORDER, linewidth=1.5
    )
    ax_detail.add_patch(rect_breakdown)
    ax_detail.text(0.5, 0.90, "YOLO26n Frame Latency Breakdown", 
                   ha='center', va='center', fontsize=12, fontweight='bold', color=C_PRIMARY)

    breakdown_text = (
        "• Preprocessing: 0.32 ms (Letterbox & tensor normalization)\n"
        "• Model Inference Core: 2.03 ms (Forward pass on RTX 5060)\n"
        "• Postprocessing & NMS: 0.81 ms (Bounding box decoding)\n"
        "• Total Frame Latency: 3.16 ms  →  Throughput: 316.5 FPS"
    )
    ax_detail.text(0.10, 0.68, breakdown_text, 
                   ha='left', va='center', fontsize=9.5, color=C_DARK, linespacing=1.6)

    # Disclosure Warning Box
    rect_disc = patches.FancyBboxPatch(
        (0.05, 0.05), 0.90, 0.38,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=C_AMBER_LIGHT, edgecolor=C_AMBER, linewidth=1.2
    )
    ax_detail.add_patch(rect_disc)
    ax_detail.text(0.5, 0.37, "ENGINEERING BENCHMARK DISCLOSURE", 
                   ha='center', va='center', fontsize=10.5, fontweight='bold', color=C_AMBER)
    disc_text = (
        "Notice: Differing batch sizes are disclosed transparently:\n"
        "• ResNet50: Batch size 1 (on-demand single-image query).\n"
        "• U-Net: Batch size 8 (batched tensor pipeline).\n"
        "• YOLO26n: Single frame streaming on CUDA.\n"
        "These metrics represent operational production latency, not a\n"
        "synthetically matched speed benchmark."
    )
    ax_detail.text(0.09, 0.19, disc_text, 
                   ha='left', va='center', fontsize=8.5, color=C_DARK, linespacing=1.4)

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])
    out_file = os.path.join(OUT_DIR, "model_latency.png")
    plt.savefig(out_file, dpi=300, facecolor=C_CANVAS_BG, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_file}")


# -----------------------------------------------------------------------------
# Chart 6: evaluation_summary.png
# -----------------------------------------------------------------------------
def generate_evaluation_summary():
    # 3-column comprehensive executive summary combining real predictions
    fig = plt.figure(figsize=(16, 9.2), facecolor=C_CANVAS_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_facecolor(C_CANVAS_BG)

    # Main Header
    ax.text(0.5, 0.965, "CropGuard AI — Multi-Model System Evaluation Summary", 
            ha='center', va='center', fontsize=20, fontweight='bold', color=C_PRIMARY)
    ax.text(0.5, 0.935, "Rigorous empirical evaluation of three decoupled deep architectures on strictly quarantined test partitions", 
            ha='center', va='center', fontsize=11.5, fontweight='semibold', color=C_MUTED)

    col_w = 0.295
    col_h = 0.89
    gap = 0.027
    start_x = 0.03
    y_col = 0.03

    columns_data = [
        {
            "branch": "BRANCH 1: PATHOLOGY CLASSIFICATION",
            "model": "ResNet50 (24.1M params)",
            "task": "38-Class Foliar Disease Classification",
            "dataset": "PlantVillage (Group-Aware Split)",
            "test_count": "10,999 held-out test images",
            "metrics": [
                ("Test Accuracy", "98.42%"),
                ("Macro F1-Score", "97.79%"),
                ("Macro ROC-AUC", "99.98%"),
                ("Inference Latency", "81.8 ms (12.2 FPS)")
            ],
            "pred_label": "Actual Sample Prediction:",
            "pred_image": "test_samples/classification_and_lesions/01_Tomato_Late_Blight.jpg",
            "pred_caption": "Tomato Late Blight (Conf: 99.4%)",
            "color": C_PRIMARY
        },
        {
            "branch": "BRANCH 2: LESION SEGMENTATION",
            "model": "U-Net (7.8M params)",
            "task": "Pixel-Level Binary Lesion Delineation",
            "dataset": "PlantSeg (Official Benchmark Split)",
            "test_count": "2,295 held-out image-mask pairs",
            "metrics": [
                ("Sørensen-Dice Coeff", "63.48%"),
                ("Mean IoU Score", "46.49%"),
                ("Healthy Specificity", "94.04%"),
                ("Inference Latency", "20.7 ms (48.3 FPS)")
            ],
            "pred_label": "Actual Held-Out Test Prediction:",
            "pred_image": "reports/final_evaluation/unet_sample_predictions/sample_01.png",
            "pred_caption": "Input  |  Ground Truth  |  Predicted Mask",
            "color": C_ACCENT
        },
        {
            "branch": "BRANCH 3: INSECT PEST DETECTION",
            "model": "YOLO26n (2.6M params)",
            "task": "Single-Stage 102-Class Pest Localization",
            "dataset": "IP102 Benchmark (VOC XML Converted)",
            "test_count": "3,798 held-out test images",
            "metrics": [
                ("mAP @ 0.50", "34.33%"),
                ("mAP @ 0.50:0.95", "20.79%"),
                ("Class Precision", "44.31%"),
                ("Inference Latency", "2.03 ms (316.5 FPS)")
            ],
            "pred_label": "Actual Held-Out Test Prediction:",
            "pred_image": "reports/final_evaluation/yolo26n_sample_predictions/test_pred_IP000000064.png",
            "pred_caption": "Predicted Bounding Boxes & Confidence",
            "color": C_BLUE
        }
    ]

    for i, col in enumerate(columns_data):
        x = start_x + i * (col_w + gap)
        # Card Background
        card = patches.FancyBboxPatch(
            (x, y_col), col_w, col_h,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=C_CARD_BG, edgecolor=C_BORDER, linewidth=1.5
        )
        ax.add_patch(card)

        # Header Pill
        header_pill = patches.FancyBboxPatch(
            (x + 0.01, y_col + col_h - 0.055), col_w - 0.02, 0.045,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor=col["color"], edgecolor="none"
        )
        ax.add_patch(header_pill)
        ax.text(x + col_w/2, y_col + col_h - 0.033, col["branch"], 
                ha='center', va='center', fontsize=9.5, fontweight='bold', color="#FFFFFF")

        # Model & Task Spec
        ax.text(x + col_w/2, y_col + col_h - 0.085, col["model"], 
                ha='center', va='center', fontsize=13, fontweight='bold', color=C_DARK)
        ax.text(x + col_w/2, y_col + col_h - 0.115, col["task"], 
                ha='center', va='center', fontsize=9.5, fontweight='semibold', color=C_MUTED)

        # Dataset & Partition
        ax.text(x + col_w/2, y_col + col_h - 0.155, f"Dataset: {col['dataset']}", 
                ha='center', va='center', fontsize=9.5, fontweight='bold', color=col["color"])
        ax.text(x + col_w/2, y_col + col_h - 0.180, f"Test Split: {col['test_count']}", 
                ha='center', va='center', fontsize=9, color=C_DARK)

        # Divider line
        ax.plot([x + 0.02, x + col_w - 0.02], [y_col + col_h - 0.205, y_col + col_h - 0.205], 
                color=C_BORDER, linewidth=1.2)

        # Metric Badges (4 boxes)
        y_metric_start = y_col + col_h - 0.23
        for m_idx, (m_name, m_val) in enumerate(col["metrics"]):
            ym = y_metric_start - m_idx * 0.065
            m_box = patches.FancyBboxPatch(
                (x + 0.02, ym - 0.045), col_w - 0.04, 0.052,
                boxstyle="round,pad=0.005,rounding_size=0.012",
                facecolor="#F8FAFC", edgecolor=C_BORDER, linewidth=1.0
            )
            ax.add_patch(m_box)
            ax.text(x + 0.035, ym - 0.019, m_name, ha='left', va='center', 
                    fontsize=9.5, fontweight='semibold', color=C_DARK)
            ax.text(x + col_w - 0.035, ym - 0.019, m_val, ha='right', va='center', 
                    fontsize=11.5, fontweight='bold', color=col["color"])

        # Divider line above sample prediction
        y_img_divider = y_col + 0.355
        ax.plot([x + 0.02, x + col_w - 0.02], [y_img_divider, y_img_divider], 
                color=C_BORDER, linewidth=1.2)

        # Prediction Title
        ax.text(x + col_w/2, y_img_divider - 0.025, col["pred_label"], 
                ha='center', va='center', fontsize=9.5, fontweight='bold', color=C_DARK)

        # Embed Actual Image
        img_path = col["pred_image"]
        if os.path.exists(img_path):
            img_pil = Image.open(img_path)
            # Create sub-axes for image
            # Bounding box coordinates in normalized figure coordinates:
            img_x = x + 0.02
            img_y = y_col + 0.065
            img_w = col_w - 0.04
            img_h = 0.25
            sub_ax = fig.add_axes([img_x, img_y, img_w, img_h])
            sub_ax.imshow(img_pil)
            sub_ax.axis('off')

        # Caption under image
        ax.text(x + col_w/2, y_col + 0.035, col["pred_caption"], 
                ha='center', va='center', fontsize=8.5, fontweight='semibold', color=C_MUTED)

    out_file = os.path.join(OUT_DIR, "evaluation_summary.png")
    plt.savefig(out_file, dpi=300, facecolor=C_CANVAS_BG, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_file}")


if __name__ == "__main__":
    generate_dataset_splits()
    generate_resnet50_results()
    generate_unet_results()
    generate_yolo_results()
    generate_model_latency()
    generate_evaluation_summary()
    print("All evaluation visual assets generated successfully!")
