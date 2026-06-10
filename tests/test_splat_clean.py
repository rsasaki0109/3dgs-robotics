"""Tests for language-prompted object removal (CLIPSeg stubbed out)."""

from __future__ import annotations

import numpy as np

import json
from pathlib import Path

from gs_sim2real.robotics.splat_clean import (
    CleanParams,
    clean_map,
    removal_mask,
    write_clean_preview,
)


def _write_clean_session(session: Path) -> np.ndarray:
    """Optical-convention scene: ground y=0, cameras 1.5 above it along +z,
    an elevated 'object' blob at x~0.5-1.1, y=-1.3, z~2 (projects to the
    upper-right of camera 0 while the ground stays in the lower image rows)."""
    import cv2

    keyframes = session / "keyframes"
    keyframes.mkdir(parents=True)
    live = session / "live"
    live.mkdir()
    (live / "state.json").write_text(json.dumps({"lastSuccessfulRound": {"round": 1}}), encoding="utf-8")
    sparse = session / "rounds" / "round_001" / "sparse_input" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 64 48 50 50 32 24\n", encoding="utf-8")
    lines = []
    for i in range(3):
        lines.append(f"{i + 1} 1 0 0 0 0 1.5 {-float(i)} 1 kf_{i:06d}.jpg\n\n")
        cv2.imwrite(str(keyframes / f"kf_{i:06d}.jpg"), np.full((48, 64, 3), 127, dtype=np.uint8))
    (sparse / "images.txt").write_text("".join(lines), encoding="utf-8")

    xs = np.linspace(-2.0, 2.0, 11)
    zs = np.linspace(0.0, 4.0, 11)
    gx, gz = np.meshgrid(xs, zs)
    ground = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    blob = np.stack(
        [0.5 + 0.02 * np.arange(30), np.full(30, -1.3), 2.0 + 0.01 * np.arange(30)],
        axis=1,
    )
    points = np.vstack([ground, blob])
    train = session / "rounds" / "round_001" / "train"
    train.mkdir(parents=True)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    body = [f"{x} {y} {z}" for x, y, z in points]
    (train / "point_cloud.ply").write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return blob


def _blob_heatmap_fn(image_bgr: np.ndarray, prompt: str) -> np.ndarray:
    """Stub CLIPSeg: relevance only in the upper-right rows where the elevated
    blob projects (v~28-34); the ground only ever lands at v>=42."""
    height, width = image_bgr.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    heatmap[: 3 * height // 4, width // 2 :] = 1.0
    return heatmap


class TestRemovalMask:
    def test_removes_cluster_keeps_stray(self):
        blob = np.stack([0.5 + 0.01 * np.arange(30), np.full(30, -0.5), np.full(30, 2.0)], axis=1)
        stray = np.array([[5.0, 5.0, 5.0]])
        points = np.vstack([blob, stray])
        scores = np.concatenate([np.full(30, 0.9), [0.95]])
        views = np.ones(len(points), dtype=np.int64)
        mask = removal_mask(
            points,
            scores,
            views,
            voxel_size=0.25,
            dilate_radius=0.0,
            params=CleanParams(min_cluster_gaussians=10),
        )
        assert mask[:30].all()
        assert not mask[30]

    def test_dilation_pulls_in_unscored_smear(self):
        blob = np.stack([0.5 + 0.01 * np.arange(30), np.full(30, -0.5), np.full(30, 2.0)], axis=1)
        smear = blob[:1] + np.array([[0.0, 0.05, 0.0]])  # transparent neighbor, never scored
        far = np.array([[3.0, 3.0, 3.0]])
        points = np.vstack([blob, smear, far])
        scores = np.concatenate([np.full(30, 0.9), [0.0], [0.0]])
        views = np.ones(len(points), dtype=np.int64)
        mask = removal_mask(
            points,
            scores,
            views,
            voxel_size=0.25,
            dilate_radius=0.1,
            params=CleanParams(min_cluster_gaussians=10),
        )
        assert mask[:31].all()
        assert not mask[31]

    def test_nothing_selected(self):
        points = np.zeros((5, 3))
        mask = removal_mask(
            points,
            np.full(5, 0.1),
            np.ones(5, dtype=np.int64),
            voxel_size=0.25,
            dilate_radius=0.5,
            params=CleanParams(),
        )
        assert not mask.any()


class TestCleanMap:
    def test_erases_the_blob_keeps_the_ground(self, tmp_path):
        from gs_sim2real.robotics.map_merge import read_raw_gaussian_ply

        session = tmp_path / "session"
        _write_clean_session(session)
        output = tmp_path / "cleaned.ply"
        stats, points, mask = clean_map(
            session,
            "object",
            output,
            params=CleanParams(score_threshold=0.6, min_cluster_gaussians=10),
            heatmap_fn=_blob_heatmap_fn,
        )
        assert stats["removed"] == 30  # exactly the blob rows
        assert stats["kept"] == len(points) - 30
        assert not mask[:121].any()  # 11x11 ground grid survives
        assert mask[121:].all()

        cleaned = read_raw_gaussian_ply(output)
        assert len(cleaned) == stats["kept"]
        assert (np.asarray(cleaned.columns(["x", "y", "z"]))[:, 1] == 0.0).all()  # only ground (y=0) remains

        preview = write_clean_preview(points, mask, np.asarray(stats["basis"]), tmp_path / "cleaned.png")
        assert preview.is_file()

    def test_no_match_keeps_everything(self, tmp_path):
        session = tmp_path / "session"
        _write_clean_session(session)
        stats, _points, mask = clean_map(
            session,
            "object",
            tmp_path / "cleaned.ply",
            params=CleanParams(score_threshold=0.6),
            heatmap_fn=lambda image, prompt: np.zeros(image.shape[:2], dtype=np.float32),
        )
        assert stats["removed"] == 0
        assert not mask.any()


class TestCliSplatClean:
    def test_cli_splat_clean(self, tmp_path, capsys, monkeypatch):
        from gs_sim2real import cli

        session = tmp_path / "session"
        _write_clean_session(session)
        monkeypatch.setattr(
            "gs_sim2real.robotics.language_query.clipseg_heatmap_fn",
            lambda device="cuda": _blob_heatmap_fn,
        )
        args = cli.build_parser().parse_args(
            [
                "splat-clean",
                "object",
                "--map",
                str(session),
                "--threshold",
                "0.6",
                "--min-cluster-gaussians",
                "10",
                "--output",
                str(tmp_path / "cleaned.ply"),
            ]
        )
        cli.cmd_splat_clean(args)
        out = capsys.readouterr().out
        assert "Removed 30" in out
        assert (tmp_path / "cleaned.ply").is_file()
        assert (tmp_path / "cleaned.png").is_file()
