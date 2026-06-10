"""CPU tests for gs_sim2real.robotics.localize."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gs_sim2real.robotics.localize import (
    compute_thumbnail,
    interpolate_gt_pose,
    keyframe_index,
    list_non_round_keyframes,
    mapped_records_by_index,
    median_neighbor_spacing,
    resolve_live_map_session,
    retrieve_seed_keyframe,
    viewmat_to_colmap,
)
from gs_sim2real.train.large_scale_3dgs import ColmapImageRecord


def _record(name: str, center: tuple[float, float, float], qvec=(1.0, 0.0, 0.0, 0.0)) -> ColmapImageRecord:
    tvec = (-center[0], -center[1], -center[2])
    return ColmapImageRecord(
        image_id=keyframe_index(name),
        camera_id=1,
        name=name,
        qvec=qvec,
        tvec=tvec,
        center=center,
        metadata_line="",
        points2d_line="",
    )


def test_keyframe_index_parses_live_mapping_names() -> None:
    assert keyframe_index("kf_000042.jpg") == 42


def test_resolve_live_map_session_uses_state_json(tmp_path: Path) -> None:
    session = tmp_path / "session"
    keyframes = session / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "kf_000000.jpg").write_bytes(b"fake")

    round_dir = session / "rounds" / "round_002"
    sparse = round_dir / "sparse_input" / "sparse" / "0"
    train = round_dir / "train"
    sparse.mkdir(parents=True)
    train.mkdir(parents=True)
    (round_dir / "scene.splat").write_bytes(b"x" * 32)
    (train / "point_cloud.ply").write_text("ply\nformat ascii 1.0\n", encoding="ascii")
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 kf_000000.jpg\n\n",
        encoding="utf-8",
    )
    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 640 480 500 500 320 240\n",
        encoding="utf-8",
    )
    live = session / "live"
    live.mkdir()
    (live / "state.json").write_text(
        '{"lastSuccessfulRound": {"round": 2}, "completedRounds": 2}',
        encoding="utf-8",
    )

    resolved = resolve_live_map_session(session)
    assert resolved.round.round_index == 2
    assert resolved.round.ply_path.is_file()


def test_retrieve_seed_keyframe_picks_closest_mapped_thumbnail(tmp_path: Path) -> None:
    keyframes = tmp_path / "keyframes"
    keyframes.mkdir()
    near = np.full((32, 32, 3), 200, dtype=np.uint8)
    far = np.full((32, 32, 3), 20, dtype=np.uint8)
    query = np.full((32, 32, 3), 190, dtype=np.uint8)
    cv2.imwrite(str(keyframes / "kf_000000.jpg"), near)
    cv2.imwrite(str(keyframes / "kf_000010.jpg"), far)
    records = [
        _record("kf_000000.jpg", (0.0, 0.0, 0.0)),
        _record("kf_000010.jpg", (1.0, 0.0, 0.0)),
    ]
    name, distance = retrieve_seed_keyframe(
        query,
        mapped_records=records,
        keyframes_dir=keyframes,
    )
    assert name == "kf_000000.jpg"
    assert distance < 0.1


def test_interpolate_gt_pose_between_mapped_keyframes() -> None:
    records = [
        _record("kf_000000.jpg", (0.0, 0.0, 0.0)),
        _record("kf_000010.jpg", (10.0, 0.0, 0.0)),
    ]
    mapped = mapped_records_by_index(records)
    qvec, tvec, center = interpolate_gt_pose("kf_000005.jpg", mapped)
    assert center is not None
    np.testing.assert_allclose(center, [5.0, 0.0, 0.0], atol=1e-6)
    assert len(qvec) == 4
    assert len(tvec) == 3


def test_list_non_round_keyframes(tmp_path: Path) -> None:
    session = tmp_path / "session"
    keyframes = session / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "kf_000000.jpg").write_bytes(b"a")
    (keyframes / "kf_000001.jpg").write_bytes(b"b")
    (keyframes / "kf_000002.jpg").write_bytes(b"c")
    records = [_record("kf_000000.jpg", (0.0, 0.0, 0.0)), _record("kf_000002.jpg", (2.0, 0.0, 0.0))]
    resolved = resolve_live_map_session
    # Build minimal round artifacts for resolve helper usage in list_non_round only:
    round_dir = session / "rounds" / "round_001"
    sparse = round_dir / "sparse_input" / "sparse" / "0"
    train = round_dir / "train"
    sparse.mkdir(parents=True)
    train.mkdir(parents=True)
    (round_dir / "scene.splat").write_bytes(b"x" * 32)
    (train / "point_cloud.ply").write_text("ply\n", encoding="ascii")
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 kf_000000.jpg\n\n3 1 0 0 0 -2 0 0 1 kf_000002.jpg\n\n",
        encoding="utf-8",
    )
    (sparse / "cameras.txt").write_text("1 PINHOLE 640 480 500 500 320 240\n", encoding="utf-8")
    (session / "live").mkdir()
    (session / "live" / "state.json").write_text('{"lastSuccessfulRound": {"round": 1}}', encoding="utf-8")

    live_session = resolved(session)
    missing = list_non_round_keyframes(live_session, records)
    assert [path.name for path in missing] == ["kf_000001.jpg"]


def test_median_neighbor_spacing() -> None:
    centers = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    assert median_neighbor_spacing(centers) == pytest.approx(2.5)


def test_viewmat_to_colmap_roundtrip_identity() -> None:
    viewmat = np.eye(4, dtype=np.float64)
    viewmat[:3, 3] = [0.5, -1.0, 2.0]
    qvec, tvec, center = viewmat_to_colmap(viewmat)
    np.testing.assert_allclose(center, [-0.5, 1.0, -2.0], atol=1e-6)
    assert qvec[0] == pytest.approx(1.0, abs=1e-5)


def test_compute_thumbnail_normalizes_to_unit_range() -> None:
    image = np.full((80, 120, 3), 128, dtype=np.uint8)
    thumb = compute_thumbnail(image)
    assert thumb.shape == (64, 64)
    assert 0.0 <= float(thumb.mean()) <= 1.0
