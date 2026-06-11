"""Open-vocabulary inventory for 3DGS maps.

Positions and extents are in camera-height gauge units from the reconstruction, not meters. Counts depend on the
CLIPSeg threshold and draft map quality, so this is a census, not ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from gs_sim2real.robotics.language_query import QueryParams

DEFAULT_VOCAB = ("car", "tree", "building", "traffic sign", "pole", "fence", "bush", "person")

_PALETTE = (
    (214, 69, 65),
    (243, 156, 18),
    (39, 174, 96),
    (41, 128, 185),
    (142, 68, 173),
    (22, 160, 133),
    (203, 67, 144),
    (105, 75, 56),
)


def build_inventory(
    session_dir: Path,
    *,
    vocab: Sequence[str],
    round_index: int | None = None,
    params: "QueryParams | None" = None,
    heatmap_fn=None,
    device: str = "cuda",
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run open-vocabulary queries and aggregate 3D hit clusters by category."""
    from gs_sim2real.robotics.language_query import clipseg_heatmap_fn, query_map

    prompts = [prompt.strip() for prompt in vocab if prompt.strip()]
    if not prompts:
        raise ValueError("vocabulary is empty - pass prompts")

    heatmap_fn = heatmap_fn or clipseg_heatmap_fn(device)
    categories: list[dict] = []
    points = np.zeros((0, 3), dtype=np.float64)
    basis = np.eye(3, dtype=np.float64)
    camera_height = 0.0

    for prompt in prompts:
        result, points = query_map(
            Path(session_dir),
            prompt,
            round_index=round_index,
            params=params,
            heatmap_fn=heatmap_fn,
            device=device,
        )
        basis = np.asarray(result.basis, dtype=np.float64)
        camera_height = float(result.camera_height)
        hits = [hit.to_json() for hit in result.hits]
        categories.append(
            {
                "prompt": prompt,
                "clusters": len(result.hits),
                "gaussians": int(sum(hit.gaussians for hit in result.hits)),
                "hits": hits,
            }
        )

    categories.sort(key=lambda category: category["clusters"], reverse=True)
    report = {
        "session": str(session_dir),
        "camera_height": camera_height,
        "vocab": prompts,
        "categories": categories,
        "total_clusters": int(sum(category["clusters"] for category in categories)),
        "note": "positions in the map's reconstruction gauge (camera-height units), counts depend on the CLIPSeg threshold",
    }
    return report, np.asarray(points, dtype=np.float64), basis


def write_inventory_preview(
    report: dict,
    points: np.ndarray,
    basis: np.ndarray,
    output_path: Path,
    *,
    image_width: int = 1600,
) -> Path:
    """Write a top-down inventory preview PNG."""
    from PIL import Image, ImageDraw

    points = np.asarray(points, dtype=np.float64)
    basis2 = np.asarray(basis, dtype=np.float64)[:2]
    if points.size:
        xy = points @ basis2.T
    else:
        xy = np.zeros((1, 2), dtype=np.float64)

    min_xy = xy.min(axis=0)
    span = np.maximum(xy.max(axis=0) - min_xy, 1e-9)
    scale = (image_width - 1) / span[0]
    height = max(int(np.ceil(span[1] * scale)) + 1, 1)

    cols = np.clip(((xy[:, 0] - min_xy[0]) * scale).astype(np.int64), 0, image_width - 1)
    rows = np.clip(((xy[:, 1] - min_xy[1]) * scale).astype(np.int64), 0, height - 1)

    image = np.full((height, image_width, 3), 255, dtype=np.uint8)
    if points.size:
        image[rows, cols] = (200, 200, 200)

    pil = Image.fromarray(image[::-1])
    draw = ImageDraw.Draw(pil)
    camera_height = float(report.get("camera_height") or 0.0)

    def to_pixel(map_xy: np.ndarray) -> tuple[float, float]:
        col = float(np.clip((map_xy[0] - min_xy[0]) * scale, 0, image_width - 1))
        row = float(np.clip(height - 1 - (map_xy[1] - min_xy[1]) * scale, 0, height - 1))
        return col, row

    legend_rows: list[tuple[tuple[int, int, int], str]] = []
    for color_index, category in enumerate(report.get("categories") or []):
        hits = list(category.get("hits") or [])
        if not hits:
            continue
        color = _PALETTE[color_index % len(_PALETTE)]
        legend_rows.append((color, f"{category.get('prompt', '')} ({len(hits)})"))
        for hit_index, hit in enumerate(hits, start=1):
            centroid = np.asarray(hit.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float64)
            extent = np.asarray(hit.get("extent", [0.0, 0.0, 0.0]), dtype=np.float64)
            center_xy = centroid @ basis2.T
            center_col, center_row = to_pixel(center_xy)
            radius_units = max(float(extent.mean()) * 0.5, camera_height * 0.25)
            radius = int(np.clip(radius_units * scale, 5, 32))
            draw.ellipse(
                [center_col - radius, center_row - radius, center_col + radius, center_row + radius],
                fill=color,
                outline=(35, 35, 35),
                width=1,
            )
            draw.text((center_col + radius + 3, center_row - radius), str(hit_index), fill=(20, 20, 20))

    if legend_rows:
        row_height = 18
        width = max(120, max(8 + len(label) * 7 for _, label in legend_rows))
        draw.rectangle([8, 8, width + 24, 16 + row_height * len(legend_rows)], fill=(255, 255, 255))
        for row_index, (color, label) in enumerate(legend_rows):
            y = 14 + row_index * row_height
            draw.rectangle([14, y, 25, y + 11], fill=color, outline=(30, 30, 30))
            draw.text((31, y - 2), label, fill=(20, 20, 20))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path)
    return output_path


def write_inventory_markdown(report: dict, output_path: Path) -> Path:
    """Write a Markdown inventory report."""
    lines = [
        "# Map inventory",
        "",
        f"Session: {report.get('session', '')}",
    ]
    if report.get("round") is not None:
        lines.append(f"Round: {report['round']}")
    lines.extend(
        [
            f"Note: {report.get('note', '')}",
            "",
            "| object | clusters | gaussians | best hit (x, y) | best score |",
            "| --- | ---: | ---: | --- | ---: |",
        ]
    )

    not_found: list[str] = []
    for category in report.get("categories") or []:
        prompt = str(category.get("prompt", ""))
        hits = list(category.get("hits") or [])
        clusters = int(category.get("clusters") or 0)
        gaussians = int(category.get("gaussians") or 0)
        if not hits:
            not_found.append(prompt)
            continue
        best = hits[0]
        goal_xy = best.get("goal_xy") or [0.0, 0.0]
        score = float(best.get("mean_score") or 0.0)
        lines.append(f"| {prompt} | {clusters} | {gaussians} | ({goal_xy[0]:.3f}, {goal_xy[1]:.3f}) | {score:.3f} |")

    if not_found:
        lines.extend(["", f"not found: {', '.join(not_found)}"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
