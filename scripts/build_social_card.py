#!/usr/bin/env python3
"""Build the GS Mapper social preview card from production 3DGS imagery."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO / "docs" / "images" / "demo-sweep" / "04_bag6-mast3r.png"
DEFAULT_OUTPUT = REPO / "docs" / "images" / "social-card.png"
CARD_SIZE = (1200, 630)


def build_social_card(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    *,
    size: tuple[int, int] = CARD_SIZE,
) -> Path:
    """Render a GitHub/social card from a production outdoor splat preview."""

    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    background = _cover(image, size)
    background = ImageEnhance.Color(background).enhance(1.18)
    background = ImageEnhance.Contrast(background).enhance(1.12)
    background = ImageEnhance.Sharpness(background).enhance(1.08)
    canvas = background.convert("RGBA")
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    _draw_gradient(draw, size)
    _draw_copy(draw, size)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(canvas, overlay).convert("RGB").save(output, optimize=True, quality=92)
    return output


def _cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = image.size
    target_width, target_height = size
    source_aspect = width / height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:
        crop_height = height
        crop_width = int(round(height * target_aspect))
    else:
        crop_width = width
        crop_height = int(round(width / target_aspect))
    left = max(0, (width - crop_width) // 2)
    top = max(0, int((height - crop_height) * 0.38))
    crop = (left, top, left + crop_width, top + crop_height)
    return image.crop(crop).resize(size, Image.Resampling.LANCZOS)


def _draw_gradient(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    for x in range(width):
        ratio = x / max(1, width - 1)
        alpha = int(238 * (1.0 - ratio) ** 1.75 + 54)
        draw.line((x, 0, x, height), fill=(5, 9, 14, min(255, alpha)))
    for y in range(height):
        ratio = y / max(1, height - 1)
        alpha = int(118 * ratio**1.55)
        draw.line((0, y, width, y), fill=(0, 0, 0, alpha))


def _draw_copy(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    title_font = _load_font(84)
    subtitle_font = _load_font(38)
    meta_font = _load_font(30)
    chip_font = _load_font(24)
    accent = (91, 232, 120, 255)
    text = (243, 248, 255, 255)
    muted = (203, 216, 229, 255)
    draw.rounded_rectangle((64, 56, 370, 105), radius=8, fill=(8, 15, 24, 218), outline=(255, 255, 255, 42))
    draw.text((86, 68), "REAL ROBOT LOGS -> 3DGS", font=chip_font, fill=text)
    draw.text((64, 176), "GS Mapper", font=title_font, fill=text)
    draw.text((69, 282), "Outdoor 3D Gaussian Splatting", font=subtitle_font, fill=text)
    draw.text((69, 333), "for Physical AI scenario CI", font=subtitle_font, fill=accent)
    bullets = (
        "9 browser-ready production splats",
        "DUSt3R / MASt3R / VGGT-SLAM / Pi3X",
        "Route-policy benchmarks + review bundles",
    )
    y = 424
    for bullet in bullets:
        draw.ellipse((70, y + 11, 86, y + 27), fill=accent)
        draw.text((104, y), bullet, font=meta_font, fill=muted)
        y += 48
    draw.line((64, height - 54, width - 74, height - 54), fill=(255, 255, 255, 54), width=2)
    draw.text((64, height - 38), "github.com/rsasaki0109/3dgs-robotics", font=chip_font, fill=(235, 245, 255, 230))


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
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = build_social_card(args.source, args.output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
