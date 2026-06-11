from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gs_sim2real.robotics.inventory import build_inventory, write_inventory_markdown, write_inventory_preview


class _FakeHit:
    def __init__(self, index: int, gaussians: int = 10, score: float = 0.8) -> None:
        self.gaussians = gaussians
        self._payload = {
            "centroid": [float(index), float(index + 1), 0.0],
            "extent": [0.5, 0.5, 0.25],
            "gaussians": gaussians,
            "mean_score": score,
            "goal_xy": [float(index), float(index + 1)],
        }

    def to_json(self) -> dict:
        return dict(self._payload)


def test_build_inventory_aggregates_sorts_and_reuses_heatmap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    heatmap_builds: list[str] = []
    heatmap_seen: list[object] = []

    def fake_clipseg_heatmap_fn(device: str):
        heatmap_builds.append(device)
        return sentinel

    def fake_query_map(session_dir, prompt, *, round_index=None, params=None, heatmap_fn=None, device="cuda"):
        heatmap_seen.append(heatmap_fn)
        counts = {"tree": 2, "car": 1, "pole": 0}
        hits = [_FakeHit(index, gaussians=100 + index) for index in range(counts[prompt])]
        result = SimpleNamespace(hits=hits, basis=np.eye(3), camera_height=1.0)
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        return result, points

    monkeypatch.setattr("gs_sim2real.robotics.language_query.clipseg_heatmap_fn", fake_clipseg_heatmap_fn)
    monkeypatch.setattr("gs_sim2real.robotics.language_query.query_map", fake_query_map)

    report, points, basis = build_inventory(tmp_path, vocab=["car", "tree", "pole"], device="cpu")

    assert heatmap_builds == ["cpu"]
    assert heatmap_seen == [sentinel, sentinel, sentinel]
    assert [category["prompt"] for category in report["categories"]] == ["tree", "car", "pole"]
    assert [category["clusters"] for category in report["categories"]] == [2, 1, 0]
    assert report["total_clusters"] == 3
    assert report["categories"][-1]["gaussians"] == 0
    assert points.shape == (2, 3)
    assert np.array_equal(basis, np.eye(3))


def test_build_inventory_rejects_empty_vocab(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vocabulary is empty - pass prompts"):
        build_inventory(tmp_path, vocab=[])


def test_markdown_contains_top_row_and_not_found(tmp_path: Path) -> None:
    report = {
        "session": str(tmp_path),
        "camera_height": 1.0,
        "note": "positions in gauge units",
        "categories": [
            {
                "prompt": "tree",
                "clusters": 2,
                "gaussians": 201,
                "hits": [{"goal_xy": [1.25, 2.5], "mean_score": 0.91}],
            },
            {"prompt": "pole", "clusters": 0, "gaussians": 0, "hits": []},
        ],
    }

    path = write_inventory_markdown(report, tmp_path / "inventory.md")
    text = path.read_text(encoding="utf-8")

    assert "| tree | 2 | 201 | (1.250, 2.500) | 0.910 |" in text
    assert "not found: pole" in text


def test_preview_writes_png_with_expected_width(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    report = {
        "camera_height": 1.0,
        "categories": [
            {
                "prompt": "tree",
                "clusters": 1,
                "gaussians": 10,
                "hits": [
                    {
                        "centroid": [0.5, 0.5, 0.0],
                        "extent": [0.5, 0.5, 0.25],
                        "goal_xy": [0.5, 0.5],
                        "mean_score": 0.8,
                    }
                ],
            }
        ],
    }
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)

    path = write_inventory_preview(report, points, np.eye(3), tmp_path / "inventory.png", image_width=320)

    assert path.exists()
    with Image.open(path) as image:
        assert image.size[0] == 320
