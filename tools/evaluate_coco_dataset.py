"""
Fancy COCO dataset audit script.

It analyzes a COCO-format instance segmentation / detection dataset without
requiring generation-time metadata or depth.

Outputs:
  - category_counts.png
  - bbox_area_ratio.png
  - bbox_width_ratio.png
  - bbox_height_ratio.png
  - bbox_center_heatmap.png
  - visible_ratio_proxy.png
  - bbox_clip_ratio.png
  - occlusion_level_proxy.png
  - objects_per_image.png
  - dataset_score.csv
  - per_category_summary.csv
  - instance_audit.csv

Usage:
  python audit_coco_dataset_fancy.py --coco_json path/to/instances_train.json --save_dir audit_results
"""

import argparse
import csv
import json
from pycocotools import mask as coco_mask
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from tools.logger import logging, logging_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging_handler)

DATASET_SCORE_WARNING_THRESHOLD = 60.0

# -----------------------------
# Global plotting style
# -----------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfbfb",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "axes.titleweight": "bold",
    "axes.titlesize": 18,
    "axes.labelweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.size": 12,
    "legend.fontsize": 11,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight"
})


def load_coco_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def bbox_area(bbox: List[float]) -> float:
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    return float(bbox[0]) + float(bbox[2]) / 2.0, float(bbox[1]) + float(bbox[3]) / 2.0


def bbox_clip_ratio(bbox: List[float], image_w: float, image_h: float) -> float:
    x, y, w, h = map(float, bbox)
    raw_area = max(0.0, w) * max(0.0, h)
    if raw_area <= 1e-6:
        return 0.0

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(image_w, x + w)
    y2 = min(image_h, y + h)

    clipped_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(clipped_area / raw_area)


def polygon_area(poly: List[float]) -> float:
    if len(poly) < 6:
        return 0.0

    xs = np.asarray(poly[0::2], dtype=np.float64)
    ys = np.asarray(poly[1::2], dtype=np.float64)
    return float(0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def segmentation_area_approx(annotation: Dict[str, Any]) -> Optional[float]:
    seg = annotation.get("segmentation")
    if seg is None:
        return None
    if isinstance(seg, dict) and 'counts' in seg:
        return float(coco_mask.area(seg))   # RLE #type:ignore
    elif isinstance(seg, list):
        # polygon
        return float(sum(polygon_area(poly) for poly in seg if isinstance(poly, list)))
    return None


def approximate_visible_ratio(annotation: Dict[str, Any], image_w: float, image_h: float) -> float:
    bbox = annotation["bbox"]
    box_area = bbox_area(bbox)

    if box_area <= 1e-6:
        return 0.0

    mask_area = segmentation_area_approx(annotation)
    if mask_area is not None:
        return float(np.clip(mask_area / box_area, 0.0, 1.0))

    return bbox_clip_ratio(bbox, image_w, image_h)


def occlusion_level_from_proxy(visible_ratio_proxy: float, bbox_area_ratio: float, clip_ratio: float) -> str:
    if clip_ratio < 0.75:
        return "high_or_truncated"
    if visible_ratio_proxy < 0.25:
        return "high"
    if visible_ratio_proxy < 0.45:
        return "medium"
    if bbox_area_ratio < 0.002:
        return "tiny_far"
    return "low"


def _score_high_is_good(value: float, good_at: float, bad_at: float) -> float:
    if good_at == bad_at:
        return 100.0
    return float(np.clip((value - bad_at) / (good_at - bad_at), 0.0, 1.0) * 100.0)


def _score_low_is_good(value: float, good_at: float, bad_at: float) -> float:
    if good_at == bad_at:
        return 100.0
    return float(np.clip((bad_at - value) / (bad_at - good_at), 0.0, 1.0) * 100.0)


def _rating_from_score(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def compute_dataset_score(total_images: int, total_annotations: int,
                          category_counts: Counter, bbox_area_ratio: List[float],
                          visible_ratio_proxy: List[float], clip_ratios: List[float]) -> Dict[str, float]:
    """Return a 0-100 heuristic score for dataset health."""
    if total_images <= 0 or total_annotations <= 0 or not category_counts:
        return {
            "overall": 0.0,
            "coverage": 0.0,
            "category_balance": 0.0,
            "object_scale": 0.0,
            "visibility": 0.0,
            "clipping": 0.0
        }

    counts = np.asarray(list(category_counts.values()), dtype=np.float64)
    areas = np.asarray(bbox_area_ratio, dtype=np.float64)
    visible = np.asarray(visible_ratio_proxy, dtype=np.float64)
    clips = np.asarray(clip_ratios, dtype=np.float64)

    annotations_per_image = total_annotations / total_images
    coverage_score = _score_high_is_good(annotations_per_image, good_at=3.0, bad_at=0.25)

    if counts.size == 1:
        balance_score = 100.0
    else:
        imbalance_ratio = float(np.max(counts) / max(1.0, np.min(counts)))
        balance_score = _score_low_is_good(imbalance_ratio, good_at=1.5, bad_at=10.0)

    if areas.size:
        tiny_ratio = float(np.mean(areas < 0.002))
        huge_ratio = float(np.mean(areas > 0.75))
        scale_score = _score_low_is_good(tiny_ratio + huge_ratio, good_at=0.05, bad_at=0.45)
    else:
        scale_score = 0.0

    visibility_score = _score_high_is_good(float(np.mean(visible)) if visible.size else 0.0, good_at=0.70, bad_at=0.25)
    clipping_score = _score_high_is_good(float(np.mean(clips)) if clips.size else 0.0, good_at=0.98, bad_at=0.75)

    weights = {"coverage": 0.20, "category_balance": 0.20, "object_scale": 0.20, "visibility": 0.25, "clipping": 0.15}
    overall = (
        coverage_score * weights["coverage"]
        + balance_score * weights["category_balance"]
        + scale_score * weights["object_scale"]
        + visibility_score * weights["visibility"]
        + clipping_score * weights["clipping"]
    )

    return {
        "overall": float(overall),
        "coverage": float(coverage_score),
        "category_balance": float(balance_score),
        "object_scale": float(scale_score),
        "visibility": float(visibility_score),
        "clipping": float(clipping_score)
    }


# -----------------------------
# Fancy plotting helpers
# -----------------------------
def _add_card_style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

def _auto_bins(values: List[float], bins: int) -> int:
    if len(values) < 2:
        return bins
    return max(10, min(bins, int(np.sqrt(len(values))) + 5))

def save_hist(
    values: List[float],
    path: Path,
    title: str,
    xlabel: str,
    bins: int = 40,
    subtitle: Optional[str] = None,
) -> None:
    if not values:
        return

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    n_bins = _auto_bins(arr.tolist(), bins)

    counts, bin_edges, patches = ax.hist(
        arr,
        bins=n_bins,
        edgecolor="#1f2937",
        linewidth=0.7,
        alpha=0.88,
        label=f"N = {arr.size:,}",
    )

    mean_v = float(np.mean(arr))
    median_v = float(np.median(arr))
    p90_v = float(np.percentile(arr, 90))

    ax.axvline(mean_v, linestyle="--", color="#BD0202", linewidth=1.0, label=f"Mean: {mean_v:.4f}")
    ax.axvline(median_v, linestyle="--", color="#810077", linewidth=1.0, label=f"Median: {median_v:.4f}")
    ax.axvline(p90_v, linestyle="--", color="#007714", linewidth=1.0, label=f"P90: {p90_v:.4f}")

    ax.set_title(title, fontweight="bold", pad=16)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, 
                ha="center", va="bottom", fontsize=11, color="#666666")

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    _add_card_style(ax)

    leg = ax.legend(title="Summary",        frameon=True,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.8,
        loc="best",
    )
    leg.get_title().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(path, dpi=360)   # type:ignore
    plt.close(fig)

def save_bar(counter: Counter, labels: Dict[Any, str], path: Path, title: str, 
             ylabel: str = "Count", subtitle: Optional[str] = None) -> None:
    if not counter:
        return

    # Sort by count descending
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    keys = [k for k, _ in items]
    names = [labels.get(k, str(k)) for k in keys]
    values = [counter[k] for k in keys]
    total = sum(values)

    fig_width = max(9, len(names) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bars = ax.bar(
        names,
        values,
        edgecolor="#1f2937",
        linewidth=0.8,
        alpha=0.9,
        label=f"Total = {total:,}",
    )

    ax.set_title(title, fontweight="bold", pad=16)
    if subtitle:
        ax.text(
            0.5,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            color="#666666",
        )

    ax.set_ylabel(ylabel, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
        tick.set_fontweight("bold")

    _add_card_style(ax)

    max_v = max(values) if values else 1
    for bar, value in zip(bars, values):
        pct = value / total * 100 if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_v * 0.012,
            f"{value:,}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )

    ax.set_ylim(0, max_v * 1.16)

    leg = ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="best")
    for text in leg.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(path, dpi=220)   # type:ignore
    plt.close(fig)

def save_heatmap(
    x_values: List[float],
    y_values: List[float],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    bins: Tuple[int, int] = (50, 50),
    subtitle: Optional[str] = None
) -> None:
    if not x_values or not y_values:
        return

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    h = ax.hist2d(x, y, bins=bins, range=[[0.0, 1.0], [0.0, 1.0]], cmap="magma")
    cbar = fig.colorbar(h[3], ax=ax, pad=0.02)
    cbar.set_label("Object Count", fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(title, fontweight="bold", pad=16)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, 
                ha="center", va="bottom", fontsize=11, color="#666666")

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.8)

    # Helpful guide lines
    ax.axvline(0.5, color="white", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axhline(0.5, color="white", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(0.505, 0.515, "image center", color="white", fontsize=10, fontweight="bold", alpha=0.85)

    fig.tight_layout()
    fig.savefig(path, dpi=360) # type:ignore
    plt.close(fig)


def audit_coco(coco_json: str) -> None:
    save_path = Path(coco_json).parent / "dataset_evaluation"
    save_path.mkdir(parents=True, exist_ok=True)

    coco = load_coco_json(coco_json)

    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    images = {img["id"]: img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])
    total_images = len(images)
    total_annotations = len(annotations)

    category_counts = Counter()
    objects_per_image = Counter()

    bbox_area_px = []
    bbox_area_ratio = []
    bbox_width_ratio = []
    bbox_height_ratio = []
    bbox_center_x_norm = []
    bbox_center_y_norm = []
    visible_ratio_proxy = []
    clip_ratios = []
    occlusion_counter = Counter()

    per_category_rows = defaultdict(list)

    for ann in annotations:
        img = images.get(ann["image_id"])
        if img is None:
            continue

        image_w = float(img["width"])
        image_h = float(img["height"])
        image_area = max(1.0, image_w * image_h)

        cid = ann["category_id"]
        category_counts[cid] += 1
        objects_per_image[ann["image_id"]] += 1

        bbox = ann["bbox"]
        box_area = bbox_area(bbox)
        cx, cy = bbox_center(bbox)

        area_ratio = box_area / image_area
        width_ratio = float(bbox[2]) / image_w
        height_ratio = float(bbox[3]) / image_h
        cx_norm = cx / image_w
        cy_norm = cy / image_h

        clip_ratio = bbox_clip_ratio(bbox, image_w, image_h)
        vis_ratio = approximate_visible_ratio(ann, image_w, image_h)
        occ = occlusion_level_from_proxy(vis_ratio, area_ratio, clip_ratio)

        bbox_area_px.append(box_area)
        bbox_area_ratio.append(area_ratio)
        bbox_width_ratio.append(width_ratio)
        bbox_height_ratio.append(height_ratio)
        bbox_center_x_norm.append(cx_norm)
        bbox_center_y_norm.append(cy_norm)
        visible_ratio_proxy.append(vis_ratio)
        clip_ratios.append(clip_ratio)
        occlusion_counter[occ] += 1

        per_category_rows[cid].append({
            "bbox_area_ratio": area_ratio,
            "bbox_width_ratio": width_ratio,
            "bbox_height_ratio": height_ratio,
            "visible_ratio_proxy": vis_ratio,
            "clip_ratio": clip_ratio,
            "occlusion_level_proxy": occ
        })

    # Fancy plots
    save_bar(
        category_counts,
        categories,
        save_path / "category_counts.png",
        # "Category Counts",
        "Instance Count per Category",
        # subtitle="Instance count per COCO category"
    )

    save_hist(
        bbox_area_ratio,
        save_path / "bbox_area_ratio.png",
        "BBox Area Ratio Distribution",
        "BBox Area / Image Area",
        subtitle="Object apparent scale in the image"
    )

    save_hist(
        bbox_width_ratio,
        save_path / "bbox_width_ratio.png",
        "BBox Width Ratio Distribution",
        "BBox Width / Image Width",
        subtitle="Horizontal object scale"
    )

    save_hist(
        bbox_height_ratio,
        save_path / "bbox_height_ratio.png",
        "BBox Height Ratio Distribution",
        "BBox Height / Image Height",
        subtitle="Vertical object scale"
    )

    save_hist(
        visible_ratio_proxy,
        save_path / "visible_ratio_proxy.png",
        "Approx. Visible Ratio Proxy Distribution",
        "Mask Area / BBox Area, or BBox Clip Ratio",
        subtitle="Proxy only; not true 3D visible ratio"
    )

    save_hist(
        clip_ratios,
        save_path / "bbox_clip_ratio.png",
        "BBox Clip Ratio Distribution",
        "Clipped BBox Area / Raw BBox Area",
        subtitle="Lower values indicate image-border truncation"
    )

    save_bar(
        occlusion_counter,
        {k: k for k in occlusion_counter.keys()},
        save_path / "occlusion_level_proxy.png",
        "Approx. Occlusion / Truncation Level",
        subtitle="Heuristic label derived from COCO bbox and mask"
    )

    if objects_per_image:
        save_hist(
            list(objects_per_image.values()),
            save_path / "objects_per_image.png",
            "Objects Per Image Distribution",
            "Objects per Image",
            bins=max(5, min(80, max(objects_per_image.values()) + 1)),
            subtitle="Scene density distribution"
        )

    save_heatmap(
        bbox_center_x_norm,
        bbox_center_y_norm,
        save_path / "bbox_center_heatmap.png",
        "Normalized BBox Center Heatmap",
        "Center X / Image Width",
        "Center Y / Image Height",
        bins=(55, 55),
        subtitle="Where object centers appear in the frame"
    )

    # CSV Summary
    summary_csv = save_path / "per_category_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Id", "Category", "Count",
            "BBox Area Ratio Mean",
            "BBox Area Ratio P10",
            "BBox Area Ratio P50",
            "BBox Area Ratio P90",
            "Visible Ratio Proxy Mean",
            "Visible Ratio Proxy P10",
            "Visible Ratio Proxy P50",
            "Visible Ratio Proxy P90",
            "Clip Ratio Mean"
        ])

        for cid, rows in sorted(per_category_rows.items(), key=lambda kv: categories.get(kv[0], str(kv[0]))):
            areas = np.array([r["bbox_area_ratio"] for r in rows], dtype=np.float64)
            vis = np.array([r["visible_ratio_proxy"] for r in rows], dtype=np.float64)
            clips = np.array([r["clip_ratio"] for r in rows], dtype=np.float64)

            writer.writerow([
                    cid, categories.get(cid, str(cid)), len(rows),
                    f"{np.mean(areas):.3g}",
                    f"{np.percentile(areas, 10):.3g}",
                    f"{np.percentile(areas, 50):.3g}",
                    f"{np.percentile(areas, 90):.3g}",
                    f"{np.mean(vis):.3g}",
                    f"{np.percentile(vis, 10):.3g}",
                    f"{np.percentile(vis, 50):.3g}",
                    f"{np.percentile(vis, 90):.3g}",
                    f"{np.mean(clips):.3g}"
                ])

    dataset_scores = compute_dataset_score(
        total_images=total_images,
        total_annotations=total_annotations,
        category_counts=category_counts,
        bbox_area_ratio=bbox_area_ratio,
        visible_ratio_proxy=visible_ratio_proxy,
        clip_ratios=clip_ratios
    )
    dataset_rating = _rating_from_score(dataset_scores["overall"])

    score_csv = save_path / "dataset_score.csv"
    with open(score_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Score"])
        writer.writerow(["Overall", f"{dataset_scores['overall']:.1f}"])
        writer.writerow(["Rating", dataset_rating])
        writer.writerow(["Coverage", f"{dataset_scores['coverage']:.1f}"])
        writer.writerow(["Category Balance", f"{dataset_scores['category_balance']:.1f}"])
        writer.writerow(["Object Scale", f"{dataset_scores['object_scale']:.1f}"])
        writer.writerow(["Visibility", f"{dataset_scores['visibility']:.1f}"])
        writer.writerow(["Clipping", f"{dataset_scores['clipping']:.1f}"])

    logger.info("Category counts:")
    for cid, count in category_counts.most_common():
        logger.info(f"      {categories.get(cid, str(cid))}: {count:,}")

    logger.info("Approx occlusion / truncation counts:")
    for k, v in occlusion_counter.most_common():
        logger.info(f"      {k}: {v:,}")

    if dataset_scores["overall"] < DATASET_SCORE_WARNING_THRESHOLD:
        logger.warning("This is not a good dataset!")
    logger.info(
        f"Dataset score:\n"
        f"\t\t Overall:          {dataset_scores['overall']:>5.1f}/100 ({dataset_rating})\n"
        f"\t\t Coverage:         {dataset_scores['coverage']:>5.1f}/100\n"
        f"\t\t Category balance: {dataset_scores['category_balance']:>5.1f}/100\n"
        f"\t\t Object scale:     {dataset_scores['object_scale']:>5.1f}/100\n"
        f"\t\t Visibility:       {dataset_scores['visibility']:>5.1f}/100\n"
        f"\t\t Clipping:         {dataset_scores['clipping']:>5.1f}/100"
    )

# def main() -> None:
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--coco_json", required=True, help="Path to COCO annotation JSON")
    # args = parser.parse_args()
    # audit_coco(args.coco_json)


if __name__ == "__main__":
    audit_coco("/media/avent/DATA/generated_data/train/2026.06.15-13:51/coco_annotations_jjivftpu.json")
