"""
CropGuard AI - Agricultural Sample Visualizer
Generates high-resolution diagnostic plots for leaf disease classification,
lesion segmentation masks, and pest detection bounding boxes.
Outputs are saved to docs/images/ for inclusion in documentation and reports.
"""

import argparse
import csv
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_IMG_DIR = PROJECT_ROOT / "docs" / "images"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "classification"


def plot_class_distribution(dist_csv: Path, out_path: Path):
    """
    Plots horizontal bar chart of dataset class sample counts.
    """
    if not dist_csv.exists():
        print(f"Distribution CSV not found: {dist_csv}")
        return

    classes = []
    counts = []
    with open(dist_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes.append(row["class_name"].replace("___", " - "))
            counts.append(int(row["count"]))

    if not classes:
        return

    plt.figure(figsize=(12, 14))
    y_pos = np.arange(len(classes))
    colors = ["#2e7d32" if "healthy" in c.lower() else "#d32f2f" for c in classes]

    plt.barh(y_pos, counts, color=colors, align="center", alpha=0.85, edgecolor="#333333", linewidth=0.5)
    plt.yticks(y_pos, classes, fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel("Sample Count", fontsize=11, fontweight="bold")
    plt.title(f"PlantVillage Class Distribution ({len(classes)} Classes, {sum(counts):,} Total Samples)",
              fontsize=13, fontweight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    for i, v in enumerate(counts):
        plt.text(v + (max(counts) * 0.005), i + 0.2, f"{v:,}", color="#222222", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved class distribution plot to: {out_path}")


def plot_sample_grid(manifest_csv: Path, out_path: Path, num_samples=16):
    """
    Plots a 4x4 grid of diverse crop disease samples from the manifest.
    """
    if not manifest_csv.exists():
        print(f"Manifest not found: {manifest_csv}")
        return

    records = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    if not records:
        return

    # Sample diverse classes
    by_class = {}
    for r in records:
        by_class.setdefault(r["full_class"], []).append(r)

    selected = []
    classes = sorted(by_class.keys())
    rng = random.Random(42)
    selected_classes = rng.sample(classes, min(num_samples, len(classes)))
    for c in selected_classes:
        selected.append(rng.choice(by_class[c]))

    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
    axes = axes.flatten()

    for idx, r in enumerate(selected[:16]):
        img_path = PROJECT_ROOT / r["local_path"]
        ax = axes[idx]
        try:
            with Image.open(img_path) as img:
                ax.imshow(img)
            cond = r['condition'].replace('_', ' ')
            status = "[Healthy]" if r['is_healthy'] == 'True' else "[Diseased]"
            ax.set_title(f"{r['crop']}\n{cond}\n{status}", fontsize=8, pad=4)
        except Exception as e:
            ax.text(0.5, 0.5, "Image Error", ha="center")
        ax.axis("off")

    # Hide any unused axes
    for idx in range(len(selected), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("CropGuard AI - Representative Foliar Disease Samples", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved sample grid plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="CropGuard AI - Sample Visualization")
    parser.add_argument("--dist", action="store_true", help="Plot class distribution bar chart")
    parser.add_argument("--grid", action="store_true", help="Plot representative leaf grid")
    args = parser.parse_args()

    dist_csv = METADATA_DIR / "class_distribution.csv"
    manifest_csv = METADATA_DIR / "dataset_manifest.csv"

    if args.dist or (not args.dist and not args.grid):
        plot_class_distribution(dist_csv, DOCS_IMG_DIR / "plantvillage_distribution.png")

    if args.grid or (not args.dist and not args.grid):
        plot_sample_grid(manifest_csv, DOCS_IMG_DIR / "plantvillage_samples_grid.png")


if __name__ == "__main__":
    main()
