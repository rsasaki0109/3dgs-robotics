"""Tests for gsplat trainer data loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.train.gsplat_trainer import GsplatTrainer


def test_load_images_txt_preserves_entries_with_blank_points_lines(tmp_path: Path) -> None:
    images_txt = tmp_path / "images.txt"
    images_txt.write_text(
        "\n".join(
            [
                "# Image list",
                "1 1 0 0 0 0 0 0 1 frame_000000.jpg",
                "",
                "2 1 0 0 0 1 0 0 1 frame_000001.jpg",
                "",
                "3 1 0 0 0 2 0 0 1 nested/frame_000002.jpg",
                "",
            ]
        ),
        encoding="utf-8",
    )

    images = GsplatTrainer._load_images_txt(object(), images_txt)

    assert list(images) == [1, 2, 3]
    assert images[2]["tvec"] == [1.0, 0.0, 0.0]
    assert images[3]["name"] == "nested/frame_000002.jpg"


def test_initialize_gaussians_uses_finite_scales_for_two_point_tiles() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    trainer = GsplatTrainer(config={"sh_degree": 0})
    points = np.array(
        [
            [0.0, 0.0, 3.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 3.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    model = trainer._initialize_gaussians(points, torch.device("cpu"))
    scales = model.scales.detach().cpu().numpy()

    assert np.isfinite(scales).all()
    assert (np.exp(scales) > 0).all()
