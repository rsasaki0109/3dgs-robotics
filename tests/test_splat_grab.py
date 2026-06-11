"""Tests for language-prompted object grab and paste helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gs_sim2real.robotics import splat_grab
from gs_sim2real.robotics.map_merge import RawGaussianPly, read_raw_gaussian_ply, write_raw_gaussian_ply
from gs_sim2real.robotics.splat_clean import CleanParams
from gs_sim2real.robotics.splat_grab import (
    _rotation_between,
    load_grab_sidecar,
    placement_sim3,
    write_paste_preview,
)


def _write_raw(path: Path, points: np.ndarray) -> None:
    data = np.column_stack([points, np.linspace(0.1, 0.4, len(points), dtype=np.float32)]).astype(np.float32)
    write_raw_gaussian_ply(path, RawGaussianPly(properties=["x", "y", "z", "opacity"], data=data))


def test_grab_keeps_exactly_masked_rows_and_writes_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_dir = tmp_path / "session"
    train = session_dir / "rounds" / "round_007" / "train"
    train.mkdir(parents=True)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 4.0],
            [9.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    ply_path = train / "point_cloud.ply"
    _write_raw(ply_path, points)

    def fake_query_map(*args, **kwargs):
        result = SimpleNamespace(
            scores=np.array([0.0, 0.9, 0.95, 0.1], dtype=np.float64),
            views=np.ones(4, dtype=np.int64),
            camera_height=2.0,
            basis=np.eye(3),
            hits=[SimpleNamespace(centroid=[0.0, 0.0, 0.0]), SimpleNamespace(centroid=[9.0, 9.0, 9.0])],
        )
        return result, points

    def fake_removal_mask(mask_points, scores, views, *, voxel_size, dilate_radius, params):
        assert np.array_equal(mask_points, points)
        assert voxel_size == 0.5
        assert dilate_radius == 0.25
        assert params.score_threshold == 0.6
        return np.array([False, True, True, False])

    round_obj = SimpleNamespace(ply_path=ply_path, round_index=7)
    monkeypatch.setattr("gs_sim2real.robotics.language_query.query_map", fake_query_map)
    monkeypatch.setattr(
        splat_grab, "resolve_live_map_session", lambda *args, **kwargs: SimpleNamespace(round=round_obj)
    )
    monkeypatch.setattr(splat_grab, "removal_mask", fake_removal_mask)

    output = tmp_path / "grabbed.ply"
    stats, scored_points, mask = splat_grab.grab_map(
        session_dir,
        "object",
        output,
        params=CleanParams(score_threshold=0.6, dilate_camera_heights=0.125),
        device="cpu",
        best_cluster=False,
    )

    assert stats["grabbed"] == 2
    assert stats["total"] == 4
    assert stats["clusters"] == 2
    assert np.array_equal(scored_points, points)
    assert np.array_equal(mask, np.array([False, True, True, False]))

    grabbed = read_raw_gaussian_ply(output)
    assert len(grabbed) == 2
    assert np.allclose(grabbed.columns(["x", "y", "z"]), points[[1, 2]])

    sidecar = load_grab_sidecar(output)
    assert sidecar["prompt"] == "object"
    assert sidecar["source_round"] == 7
    assert sidecar["gaussians"] == 2
    assert sidecar["camera_height"] == 2.0
    assert np.allclose(sidecar["centroid"], points[[1, 2]].mean(axis=0))
    assert sidecar["bottom"] == 3.0


def test_grab_nothing_matched_raises_threshold_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_dir = tmp_path / "session"
    train = session_dir / "rounds" / "round_001" / "train"
    train.mkdir(parents=True)
    points = np.zeros((3, 3), dtype=np.float64)
    ply_path = train / "point_cloud.ply"
    _write_raw(ply_path, points)

    def fake_query_map(*args, **kwargs):
        result = SimpleNamespace(
            scores=np.zeros(3, dtype=np.float64),
            views=np.ones(3, dtype=np.int64),
            camera_height=1.5,
            basis=np.eye(3),
            hits=[],
        )
        return result, points

    monkeypatch.setattr("gs_sim2real.robotics.language_query.query_map", fake_query_map)
    monkeypatch.setattr(
        splat_grab,
        "resolve_live_map_session",
        lambda *args, **kwargs: SimpleNamespace(round=SimpleNamespace(ply_path=ply_path, round_index=1)),
    )
    monkeypatch.setattr(splat_grab, "removal_mask", lambda *args, **kwargs: np.zeros(3, dtype=bool))

    with pytest.raises(ValueError, match="threshold"):
        splat_grab.grab_map(session_dir, "object", tmp_path / "grabbed.ply", device="cpu")


def test_rotation_between_general_parallel_and_antiparallel() -> None:
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])

    general = _rotation_between(x_axis, y_axis)
    assert np.allclose(general @ x_axis, y_axis)

    parallel = _rotation_between(x_axis, x_axis)
    assert np.allclose(parallel, np.eye(3))

    antiparallel = _rotation_between(x_axis, -x_axis)
    assert np.allclose(antiparallel @ x_axis, -x_axis)
    assert np.isclose(np.linalg.det(antiparallel), 1.0)


def test_placement_sim3_auto_scale_bottom_anchor_and_yaw() -> None:
    sidecar = {
        "up": [0.0, 0.0, 1.0],
        "camera_height": 1.5,
        "centroid": [2.0, 3.0, 5.0],
        "bottom": 4.0,
    }
    sim3 = placement_sim3(
        sidecar,
        up_t=np.array([0.0, 0.0, 1.0]),
        e1=np.array([1.0, 0.0, 0.0]),
        e2=np.array([0.0, 1.0, 0.0]),
        ground_t=0.5,
        camera_height_t=3.0,
        at_xy=(10.0, 20.0),
        yaw_deg=90.0,
        scale=None,
    )
    scale, rotation, translation = sim3
    assert scale == 2.0
    assert np.allclose(rotation @ np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), atol=1e-12)

    anchor = np.array([2.0, 3.0, 4.0])
    target_world = np.array([10.0, 20.0, 0.5])
    assert np.allclose(scale * (rotation @ anchor) + translation, target_world)


def test_load_grab_sidecar_missing_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Re-run splat-grab"):
        load_grab_sidecar(tmp_path / "missing.ply")


def test_write_paste_preview_smoke(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    target = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    pasted = np.array([[0.5, 0.25, 0.0]])
    basis2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    preview = write_paste_preview(target, pasted, basis2, tmp_path / "paste.png", image_width=64)
    assert preview.is_file()


def test_keep_cluster_nearest_picks_target_blob() -> None:
    points = np.vstack(
        [
            np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
            np.array([[5.0, 0.0, 0.0], [5.1, 0.0, 0.0]]),
        ]
    )
    mask = np.ones(len(points), dtype=bool)

    near_first = splat_grab._keep_cluster_nearest(points, mask, 0.5, np.array([0.0, 0.0, 0.0]))
    near_second = splat_grab._keep_cluster_nearest(points, mask, 0.5, np.array([5.0, 0.0, 0.0]))

    assert near_first.tolist() == [True, True, True, False, False]
    assert near_second.tolist() == [False, False, False, True, True]
