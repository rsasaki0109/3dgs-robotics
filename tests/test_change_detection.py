"""Tests for 3DGS change detection (inspection)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gs_sim2real.robotics.change_detection import (
    align_by_localization,
    align_shared_keyframes,
    detect_changes,
    detect_session_changes,
    write_change_preview,
)
from gs_sim2real.robotics.gauge_alignment import (
    apply_to_points,
    invert,
    quat_to_rotation,
    rotation_to_quat,
)

# Same optical-convention synthetic world as the occupancy tests: identity
# camera orientation means world up is (0, -1, 0); ground y=0, cameras at
# y=-1.5 driving along +z.
_IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)


def _camera_track(count: int = 5) -> tuple[np.ndarray, list]:
    centers = np.stack([np.zeros(count), np.full(count, -1.5), np.arange(float(count))], axis=1)
    return centers, [_IDENTITY_Q] * count


def _ground_and_wall() -> np.ndarray:
    xs = np.linspace(-2.0, 2.0, 21)
    zs = np.linspace(0.0, 4.0, 21)
    gx, gz = np.meshgrid(xs, zs)
    ground = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    wy = np.linspace(-0.4, -2.0, 9)
    wz = np.linspace(1.0, 3.0, 9)
    gy, gz2 = np.meshgrid(wy, wz)
    wall = np.stack([np.full(gy.size, 2.0), gy.ravel(), gz2.ravel()], axis=1)
    return np.vstack([ground, wall])


def _box(center=(-1.0, -0.85, 2.25), size=(0.4, 0.7, 0.5), n: int = 5) -> np.ndarray:
    axes = [np.linspace(c - s / 2, c + s / 2, n) for c, s in zip(center, size)]
    gx, gy, gz = np.meshgrid(*axes)
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def _detect(points_a, points_b, **kwargs):
    centers, qvecs = _camera_track()
    return detect_changes(
        points_a,
        np.ones(len(points_a)),
        points_b,
        np.ones(len(points_b)),
        centers,
        qvecs,
        **kwargs,
    )


class TestDetectChanges:
    def test_identical_maps_report_nothing(self):
        scene = _ground_and_wall()
        report = _detect(scene, scene.copy())
        assert report.clusters == []
        assert report.camera_height == pytest.approx(1.5, abs=0.05)

    def test_appeared_box_is_found(self):
        scene = _ground_and_wall()
        report = _detect(scene, np.vstack([scene, _box()]))
        assert len(report.appeared) == 1
        assert report.disappeared == []
        centroid = np.asarray(report.appeared[0].centroid)
        assert centroid == pytest.approx((-1.0, -0.85, 2.25), abs=report.voxel_size * 1.5)

    def test_disappeared_wall_is_found(self):
        scene = _ground_and_wall()
        without_wall = scene[scene[:, 0] < 1.9]
        report = _detect(scene, without_wall)
        assert report.appeared == []
        assert len(report.disappeared) >= 1
        assert report.disappeared[0].centroid[0] == pytest.approx(2.0, abs=report.voxel_size * 1.5)

    def test_small_blobs_are_filtered(self):
        scene = _ground_and_wall()
        speck = np.tile(np.array([[-1.0, -1.0, 0.5]]), (4, 1))  # one solid voxel only
        report = _detect(scene, np.vstack([scene, speck]))
        assert report.clusters == []

    def test_transparent_gaussians_are_ignored(self):
        scene = _ground_and_wall()
        box = _box()
        centers, qvecs = _camera_track()
        report = detect_changes(
            scene,
            np.ones(len(scene)),
            np.vstack([scene, box]),
            np.concatenate([np.ones(len(scene)), np.full(len(box), 0.05)]),
            centers,
            qvecs,
        )
        assert report.appeared == []

    def test_json_report_shape(self):
        scene = _ground_and_wall()
        report = _detect(scene, np.vstack([scene, _box()]))
        payload = report.to_json()
        assert payload["appeared"][0]["kind"] == "appeared"
        assert len(payload["basis"]) == 3
        assert payload["camera_height"] == pytest.approx(report.camera_height)


def _sim3_example():
    angle = np.radians(30.0)
    up = np.array([0.0, -1.0, 0.0])
    k = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    return (2.0, rotation, np.array([0.3, -0.1, 0.5]))


def _write_images_txt(path: Path, names, centers, rotations) -> None:
    lines = []
    for i, (name, center, r_wc) in enumerate(zip(names, centers, rotations)):
        r_cw = r_wc.T
        q = rotation_to_quat(r_cw)
        t = -r_cw @ np.asarray(center, dtype=np.float64)
        lines.append(f"{i + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")
    path.write_text("".join(lines), encoding="utf-8")


class TestAlignSharedKeyframes:
    def test_recovers_known_sim3(self, tmp_path):
        transform = _sim3_example()
        names = [f"kf_{i:06d}.jpg" for i in range(5)]
        centers_a, _ = _camera_track()
        rotations_a = [np.eye(3)] * 5
        inverse = invert(transform)
        centers_b = apply_to_points(inverse, centers_a)
        rotations_b = [inverse[1] @ r for r in rotations_a]
        _write_images_txt(tmp_path / "a.txt", names, centers_a, rotations_a)
        _write_images_txt(tmp_path / "b.txt", names, centers_b, rotations_b)

        recovered, matched = align_shared_keyframes(tmp_path / "a.txt", tmp_path / "b.txt")
        assert matched == 5
        sample = np.array([[0.7, -1.1, 2.3], [2.0, 0.0, 0.0]])
        assert apply_to_points(recovered, apply_to_points(inverse, sample)) == pytest.approx(sample, abs=1e-6)

    def test_requires_two_shared_keyframes(self, tmp_path):
        centers_a, _ = _camera_track()
        _write_images_txt(tmp_path / "a.txt", ["kf_000000.jpg"], centers_a[:1], [np.eye(3)])
        _write_images_txt(tmp_path / "b.txt", ["other.jpg"], centers_a[:1], [np.eye(3)])
        with pytest.raises(ValueError):
            align_shared_keyframes(tmp_path / "a.txt", tmp_path / "b.txt")


class _StubLocalizer:
    """Returns map-A poses for map-B keyframes via a known Sim3."""

    def __init__(self, transform, records_by_name, off_map: set[str] | None = None) -> None:
        self.transform = transform
        self.records = records_by_name
        self.off_map = off_map or set()

    def localize(self, image_bgr, *, query_name="query"):
        record = self.records[query_name]
        r_wc_b = quat_to_rotation(np.asarray(record.qvec, dtype=np.float64)).T
        r_wc_a = self.transform[1] @ r_wc_b
        center_a = apply_to_points(self.transform, np.asarray(record.center, dtype=np.float64)[None, :])[0]
        return SimpleNamespace(
            qvec=tuple(rotation_to_quat(r_wc_a.T)),
            tvec=(0.0, 0.0, 0.0),
            center=center_a,
            seed_distance=0.9 if query_name in self.off_map else 0.05,
        )


def _stub_records(centers) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(name=f"kf_{i:06d}.jpg", qvec=_IDENTITY_Q, center=tuple(center), camera_id=1)
        for i, center in enumerate(centers)
    ]


def _write_keyframe_images(directory: Path, names) -> None:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    for name in names:
        cv2.imwrite(str(directory / name), image)


class TestAlignByLocalization:
    def test_recovers_known_sim3(self, tmp_path):
        transform = _sim3_example()
        inverse = invert(transform)
        centers_a, _ = _camera_track()
        centers_b = apply_to_points(inverse, centers_a)
        records = _stub_records(centers_b)
        _write_keyframe_images(tmp_path / "keyframes", [r.name for r in records])
        session_b = SimpleNamespace(keyframes_dir=tmp_path / "keyframes")

        # records carry identity orientations in B's gauge, so the stub's A
        # poses follow transform directly
        localizer = _StubLocalizer(transform, {r.name: r for r in records})
        recovered, matched = align_by_localization(tmp_path, session_b, records, localizer=localizer)
        assert matched == 5
        sample = np.array([[0.4, -0.9, 1.7]])
        assert apply_to_points(recovered, apply_to_points(inverse, sample)) == pytest.approx(sample, abs=1e-6)

    def test_off_map_keyframes_are_skipped(self, tmp_path):
        transform = _sim3_example()
        centers_b = apply_to_points(invert(transform), _camera_track()[0])
        records = _stub_records(centers_b)
        _write_keyframe_images(tmp_path / "keyframes", [r.name for r in records])
        session_b = SimpleNamespace(keyframes_dir=tmp_path / "keyframes")
        off_map = {r.name for r in records[2:]}
        localizer = _StubLocalizer(transform, {r.name: r for r in records}, off_map=off_map)
        _, matched = align_by_localization(tmp_path, session_b, records, localizer=localizer)
        assert matched == 2

    def test_too_few_matches_raise(self, tmp_path):
        transform = _sim3_example()
        centers_b = apply_to_points(invert(transform), _camera_track()[0])
        records = _stub_records(centers_b)
        _write_keyframe_images(tmp_path / "keyframes", [r.name for r in records])
        session_b = SimpleNamespace(keyframes_dir=tmp_path / "keyframes")
        localizer = _StubLocalizer(transform, {r.name: r for r in records}, off_map={r.name for r in records[1:]})
        with pytest.raises(ValueError):
            align_by_localization(tmp_path, session_b, records, localizer=localizer)


def _write_session(session: Path, points: np.ndarray, centers, rotations, names) -> None:
    (session / "keyframes").mkdir(parents=True)
    live = session / "live"
    live.mkdir()
    (live / "state.json").write_text(json.dumps({"lastSuccessfulRound": {"round": 1}}), encoding="utf-8")
    sparse = session / "rounds" / "round_001" / "sparse_input" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 64 48 50 50 32 24\n", encoding="utf-8")
    _write_images_txt(sparse / "images.txt", names, centers, rotations)
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


def _write_session_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Session B = session A in a different gauge, plus a new box."""
    transform = _sim3_example()
    inverse = invert(transform)
    names = [f"kf_{i:06d}.jpg" for i in range(5)]
    centers_a, _ = _camera_track()
    rotations_a = [np.eye(3)] * 5
    points_a = _ground_and_wall()

    session_a = tmp_path / "session_a"
    _write_session(session_a, points_a, centers_a, rotations_a, names)

    points_b_world = np.vstack([points_a, _box()])
    session_b = tmp_path / "session_b"
    _write_session(
        session_b,
        apply_to_points(inverse, points_b_world),
        apply_to_points(inverse, centers_a),
        [inverse[1] @ r for r in rotations_a],
        names,
    )
    return session_a, session_b


class TestDetectSessionChanges:
    def test_cross_gauge_appeared_box(self, tmp_path):
        session_a, session_b = _write_session_pair(tmp_path)
        report, points_a, aligned_b = detect_session_changes(session_a, session_b, align="shared")
        assert report.alignment["mode"] == "shared"
        assert report.alignment["scale"] == pytest.approx(2.0, rel=1e-3)
        assert len(report.appeared) == 1
        assert report.disappeared == []
        assert np.asarray(report.appeared[0].centroid) == pytest.approx(
            (-1.0, -0.85, 2.25), abs=report.voxel_size * 1.5
        )
        preview = write_change_preview(report, points_a, aligned_b, tmp_path / "changes.png")
        assert preview.is_file()

    def test_same_round_is_rejected(self, tmp_path):
        session_a, _ = _write_session_pair(tmp_path)
        with pytest.raises(ValueError):
            detect_session_changes(session_a, session_a)


class TestCli:
    def test_cli_detect_changes(self, tmp_path, capsys):
        from gs_sim2real import cli

        session_a, session_b = _write_session_pair(tmp_path)
        args = cli.build_parser().parse_args(
            [
                "detect-changes",
                "--map-a",
                str(session_a),
                "--map-b",
                str(session_b),
                "--align",
                "shared",
                "--output",
                str(tmp_path / "changes.json"),
            ]
        )
        cli.cmd_detect_changes(args)
        out = capsys.readouterr().out
        assert "Appeared: 1 cluster(s)" in out
        assert (tmp_path / "changes.json").is_file()
        assert (tmp_path / "changes.png").is_file()
        payload = json.loads((tmp_path / "changes.json").read_text(encoding="utf-8"))
        assert payload["alignment"]["mode"] == "shared"

    def test_cli_rejects_self_compare_without_rounds(self):
        from gs_sim2real import cli

        args = cli.build_parser().parse_args(
            ["detect-changes", "--map-a", "x", "--output", "changes.json", "--align", "none"]
        )
        with pytest.raises(SystemExit):
            cli.cmd_detect_changes(args)

    def test_cli_rejects_non_json_output(self):
        from gs_sim2real import cli

        args = cli.build_parser().parse_args(
            ["detect-changes", "--map-a", "x", "--map-b", "y", "--output", "changes.txt"]
        )
        with pytest.raises(SystemExit):
            cli.cmd_detect_changes(args)
