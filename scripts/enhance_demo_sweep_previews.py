#!/usr/bin/env python3
"""Crop README / Pages splat previews so outdoor structure fills the frame.

The WebGL capture script records the full canvas. That is useful for smoke
tests, but several outdoor splats sit in a small island of pixels inside a
large black frame. This post-process keeps the 1280x720 contract while
cropping around the visible splat, making README and Pages thumbnails read
more like outdoor GS demos.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
TARGET_SIZE = (1280, 720)
HERO_SIZE = (720, 405)
HERO_FRAMES_PER_SCENE = 4
HERO_PALETTE_COLORS = 192
HERO_PREVIEW_PRIORITY = (
    "04_bag6-mast3r.png",
    "07_bag6-vggt-slam.png",
    "08_bag6-mast3r-slam.png",
    "09_bag6-pi3x.png",
    "01_outdoor-demo.png",
    "02_outdoor-demo-dust3r.png",
    "06_mcd-ntu-day02-supervised.png",
    "05_mcd-tuhh-day04-mast3r.png",
    "03_mcd-tuhh-day04.png",
)


@dataclass(frozen=True, slots=True)
class PreviewEnhancement:
    source: Path
    bbox: tuple[int, int, int, int] | None
    crop: tuple[int, int, int, int]
    output: Path


def enhance_preview(
    source: Path,
    *,
    output: Path | None = None,
    padding: float = 1.18,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> PreviewEnhancement:
    """Crop ``source`` around visible splat pixels and write ``output``."""

    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    bbox = _foreground_bbox(image)
    crop = _crop_for_bbox(image.size, bbox, target_size=target_size, padding=padding)
    cropped = image.crop(crop).resize(target_size, Image.Resampling.LANCZOS)
    destination = output or source
    destination.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(destination, optimize=True)
    return PreviewEnhancement(source=source, bbox=bbox, crop=crop, output=destination)


def enhance_scene_manifest_previews(
    docs_dir: Path = DOCS,
    *,
    output_dir: Path | None = None,
    padding: float = 1.18,
) -> list[PreviewEnhancement]:
    """Enhance every preview referenced by ``docs/scenes-list.json``."""

    manifest = json.loads((docs_dir / "scenes-list.json").read_text(encoding="utf-8"))
    results: list[PreviewEnhancement] = []
    for scene in manifest.get("scenes", []):
        preview = docs_dir / scene["preview"]
        output = None if output_dir is None else output_dir / Path(scene["preview"]).name
        results.append(enhance_preview(preview, output=output, padding=padding))
    return results


def build_static_preview_hero_gif(
    docs_dir: Path = DOCS,
    *,
    output: Path | None = None,
    frame_duration_ms: int = 95,
) -> Path:
    """Build a compact animated hero GIF from the enhanced production previews."""

    manifest = json.loads((docs_dir / "scenes-list.json").read_text(encoding="utf-8"))
    scenes = list(manifest.get("scenes", []))
    if not scenes:
        raise ValueError("scenes-list.json has no scenes")
    frames: list[Image.Image] = []
    hero_scenes = _hero_scene_order(scenes)
    for index, scene in enumerate(hero_scenes, start=1):
        with Image.open(docs_dir / scene["preview"]) as source:
            preview = ImageOps.exif_transpose(source).convert("RGB")
        for step in range(HERO_FRAMES_PER_SCENE):
            progress = step / max(1, HERO_FRAMES_PER_SCENE - 1)
            frame = _render_hero_frame(
                preview, scene=scene, scene_index=index, scene_count=len(scenes), progress=progress
            )
            frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=HERO_PALETTE_COLORS))
    destination = output or (docs_dir / "images" / "demo-sweep" / "hero.gif")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return destination


def _hero_scene_order(scenes: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep coverage from the manifest but open the GIF with the strongest outdoor visuals."""

    priority = {name: index for index, name in enumerate(HERO_PREVIEW_PRIORITY)}

    def sort_key(item: tuple[int, dict[str, object]]) -> tuple[int, int]:
        original_index, scene = item
        preview_name = Path(str(scene.get("preview", ""))).name
        return (priority.get(preview_name, len(priority)), original_index)

    return [scene for _, scene in sorted(enumerate(scenes), key=sort_key)]


def _render_hero_frame(
    preview: Image.Image,
    *,
    scene: dict[str, object],
    scene_index: int,
    scene_count: int,
    progress: float,
) -> Image.Image:
    """Render one kinetic README hero frame from a production preview image."""

    eased = _ease_in_out(progress)
    frame = _ken_burns_crop(preview, HERO_SIZE, progress=eased, reverse=scene_index % 2 == 0)
    frame = ImageEnhance.Color(frame).enhance(1.18)
    frame = ImageEnhance.Contrast(frame).enhance(1.10)
    frame = ImageEnhance.Sharpness(frame).enhance(1.08)
    overlay = Image.new("RGBA", HERO_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    _draw_vignette(draw, HERO_SIZE)
    _draw_bottom_gradient(draw, HERO_SIZE)
    _draw_hero_copy(draw, scene=scene, scene_index=scene_index, scene_count=scene_count, progress=progress)
    return Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")


def _ken_burns_crop(
    image: Image.Image,
    target_size: tuple[int, int],
    *,
    progress: float,
    reverse: bool,
) -> Image.Image:
    width, height = image.size
    target_width, target_height = target_size
    aspect = target_width / target_height
    zoom = 1.035 + 0.115 * progress
    crop_width = width / zoom
    crop_height = crop_width / aspect
    if crop_height > height:
        crop_height = height / zoom
        crop_width = crop_height * aspect
    max_x = max(0.0, width - crop_width)
    max_y = max(0.0, height - crop_height)
    pan = 1.0 - progress if reverse else progress
    crop_left = max_x * (0.18 + 0.64 * pan)
    crop_top = max_y * (0.40 + 0.20 * (1.0 - progress))
    crop = (
        int(round(crop_left)),
        int(round(crop_top)),
        int(round(crop_left + crop_width)),
        int(round(crop_top + crop_height)),
    )
    return image.crop(crop).resize(target_size, Image.Resampling.LANCZOS)


def _draw_vignette(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    for x in range(width):
        edge = min(x, width - x - 1) / max(1, width / 2)
        alpha = int(90 * (1.0 - min(1.0, edge)) ** 1.8)
        if alpha:
            draw.line((x, 0, x, height), fill=(0, 0, 0, alpha))
    for y in range(height):
        edge = min(y, height - y - 1) / max(1, height / 2)
        alpha = int(70 * (1.0 - min(1.0, edge)) ** 1.8)
        if alpha:
            draw.line((0, y, width, y), fill=(0, 0, 0, alpha))


def _draw_hero_copy(
    draw: ImageDraw.ImageDraw,
    *,
    scene: dict[str, object],
    scene_index: int,
    scene_count: int,
    progress: float,
) -> None:
    width, height = HERO_SIZE
    font_kicker = _load_font(13)
    font_title = _load_font(27)
    font_meta = _load_font(15)
    font_mono = _load_font(12)
    accent = (89, 230, 120, 255)
    panel = (5, 11, 18, 188)
    draw.rounded_rectangle((18, 18, 272, 44), radius=4, fill=panel, outline=(255, 255, 255, 34))
    draw.text((30, 25), "REAL ROBOT LOGS -> 3DGS", font=font_kicker, fill=(235, 244, 255, 255))
    draw.rectangle((18, height - 8, width - 18, height - 4), fill=(255, 255, 255, 44))
    progress_total = ((scene_index - 1) + progress) / max(1, scene_count)
    draw.rectangle((18, height - 8, 18 + int((width - 36) * progress_total), height - 4), fill=accent)
    label = _short_scene_label(str(scene.get("label", "Outdoor GS scene")))
    draw.text((22, height - 88), f"OUTDOOR GS {scene_index:02d}/{scene_count:02d}", font=font_mono, fill=accent)
    draw.text((22, height - 62), label, font=font_title, fill=(244, 249, 255, 255))
    draw.text(
        (22, height - 27),
        "browser .splat / supervised + pose-free + external SLAM",
        font=font_meta,
        fill=(211, 224, 236, 255),
    )


def _short_scene_label(label: str) -> str:
    cleaned = (
        label.replace("Autoware ", "")
        .replace("MCD ", "")
        .replace(" — ", " / ")
        .replace(" (metric)", "")
        .replace("supervised default", "supervised")
    )
    return _truncate_text(cleaned, 48)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _ease_in_out(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _foreground_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    pixels = image.load()
    width, height = image.size
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    stride = 2
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            r, g, b = pixels[x, y]
            value = max(r, g, b)
            chroma = value - min(r, g, b)
            if value > 42 and (chroma > 8 or value > 92):
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    if max_x < min_x or max_y < min_y:
        return None
    return (
        max(0, min_x - stride),
        max(0, min_y - stride),
        min(width, max_x + stride + 1),
        min(height, max_y + stride + 1),
    )


def _crop_for_bbox(
    image_size: tuple[int, int],
    bbox: tuple[int, int, int, int] | None,
    *,
    target_size: tuple[int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    if bbox is None:
        return (0, 0, width, height)
    left, top, right, bottom = bbox
    box_width = max(1.0, (right - left) * padding)
    box_height = max(1.0, (bottom - top) * padding)
    aspect = target_size[0] / target_size[1]
    if box_width / box_height < aspect:
        box_width = box_height * aspect
    else:
        box_height = box_width / aspect
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_left = center_x - box_width / 2.0
    crop_top = center_y - box_height / 2.0
    crop_left = max(0.0, min(crop_left, width - box_width))
    crop_top = max(0.0, min(crop_top, height - box_height))
    crop_right = min(width, crop_left + box_width)
    crop_bottom = min(height, crop_top + box_height)
    crop = (
        int(round(crop_left)),
        int(round(crop_top)),
        int(round(crop_right)),
        int(round(crop_bottom)),
    )
    if crop[0] <= 2 and crop[1] <= 2 and width - crop[2] <= 2 and height - crop[3] <= 2:
        return (0, 0, width, height)
    return crop


def _draw_bottom_gradient(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    start_y = int(height * 0.58)
    for y in range(start_y, height):
        alpha = int(210 * ((y - start_y) / max(1, height - start_y)) ** 1.4)
        draw.line((0, y, width, y), fill=(0, 0, 0, alpha))


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        candidate = Path(path)
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=DOCS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for enhanced copies; default overwrites scene-list previews.",
    )
    parser.add_argument("--padding", type=float, default=1.18)
    parser.add_argument(
        "--hero-gif",
        action="store_true",
        help="Also rebuild docs/images/demo-sweep/hero.gif from the enhanced previews.",
    )
    parser.add_argument("--hero-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = enhance_scene_manifest_previews(
        args.docs_dir,
        output_dir=args.output_dir,
        padding=args.padding,
    )
    for result in results:
        bbox = "none" if result.bbox is None else ",".join(str(value) for value in result.bbox)
        crop = ",".join(str(value) for value in result.crop)
        print(f"wrote {result.output} bbox={bbox} crop={crop}")
    if args.hero_gif:
        output = build_static_preview_hero_gif(args.docs_dir, output=args.hero_output)
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
