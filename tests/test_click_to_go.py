"""Tests for click-to-go navigation server."""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pytest

from gs_sim2real.robotics import click_to_go
from gs_sim2real.robotics.viewer_overlay import SplatFrameMapper


def test_splat_ray_to_goal_round_trip() -> None:
    frame = _synthetic_frame()
    goal_xy = (1.25, -0.75)
    point_splat, origin_splat, direction_splat = _ray_for_goal(frame, goal_xy)

    assert point_splat.shape == (3,)
    actual = click_to_go.splat_ray_to_goal(origin_splat, direction_splat, frame)

    assert actual == pytest.approx(goal_xy, abs=1e-6)


def test_splat_ray_to_goal_rejects_parallel_ray() -> None:
    frame = _synthetic_frame()

    with pytest.raises(ValueError, match="does not hit the ground plane"):
        click_to_go.splat_ray_to_goal([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], frame)


def test_splat_ray_to_goal_rejects_ray_pointing_away() -> None:
    frame = _synthetic_frame()
    _, origin_splat, direction_splat = _ray_for_goal(frame, (1.0, 2.0))

    with pytest.raises(ValueError, match="does not hit the ground plane"):
        click_to_go.splat_ray_to_goal(origin_splat, -direction_splat, frame)


def test_run_navigate_and_overlay_invokes_cli_and_summarizes(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    calls: list[list[str]] = []

    def fake_run_cli(args: Sequence[str]) -> SimpleNamespace:
        calls.append(list(args))
        if args[0] == "navigate":
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps({"reached": True, "steps": 42, "extra": "ignored"}),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    config = click_to_go.ClickToGoConfig(round_index=3, localize_every=5, odom_noise=0.25, device="cpu")
    result = click_to_go.run_navigate_and_overlay(session_dir, (1.5, -2.0), config, run_cli=fake_run_cli)

    nav_json = session_dir / "clickgo" / "nav_result.json"
    overlay_json = session_dir / "clickgo" / "overlay.json"
    assert calls == [
        [
            "navigate",
            "--map",
            str(session_dir),
            "--goal",
            "1.5,-2.0",
            "--output",
            str(nav_json),
            "--localize-every",
            "5",
            "--odom-noise",
            "0.25",
            "--device",
            "cpu",
            "--round",
            "3",
        ],
        [
            "export-overlay",
            "--map",
            str(session_dir),
            "--output",
            str(overlay_json),
            "--nav",
            str(nav_json),
            "--round",
            "3",
        ],
    ]
    assert result == {
        "reached": True,
        "steps": 42,
        "goal_xy": [1.5, -2.0],
        "overlay": "/clickgo/overlay.json",
        "nav_json": "/clickgo/nav_result.json",
    }


def test_run_query_and_overlay_invokes_cli_and_summarizes(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    calls: list[list[str]] = []

    def fake_run_cli(args: Sequence[str]) -> SimpleNamespace:
        calls.append(list(args))
        if args[0] == "query-map":
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps({"hits": [{"centroid": [0, 0, 0]}, {"centroid": [1, 1, 1]}]}),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    config = click_to_go.ClickToGoConfig(round_index=3, device="cpu")
    result = click_to_go.run_query_and_overlay(session_dir, "car", config, run_cli=fake_run_cli)

    query_json = session_dir / "clickgo" / "query.json"
    overlay_json = session_dir / "clickgo" / "overlay.json"
    assert calls == [
        [
            "query-map",
            "car",
            "--map",
            str(session_dir),
            "--output",
            str(query_json),
            "--device",
            "cpu",
            "--round",
            "3",
        ],
        [
            "export-overlay",
            "--map",
            str(session_dir),
            "--output",
            str(overlay_json),
            "--query",
            str(query_json),
            "--round",
            "3",
        ],
    ]
    assert result == {
        "prompt": "car",
        "hits": 2,
        "overlay": "/clickgo/overlay.json",
        "query_json": "/clickgo/query.json",
    }


def test_run_clean_and_swap_invokes_cli_and_exports(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    calls: list[list[str]] = []
    exports: list[tuple[Path, Path, Path, int | None]] = []

    def fake_run_cli(args: Sequence[str]) -> SimpleNamespace:
        calls.append(list(args))
        return SimpleNamespace(returncode=0)

    def fake_export(session: Path, cleaned_ply: Path, output_splat: Path, *, round_index: int | None = None) -> int:
        exports.append((Path(session), Path(cleaned_ply), Path(output_splat), round_index))
        Path(output_splat).write_bytes(b"\x00" * 64)  # two 32-byte gaussians
        return 2

    config = click_to_go.ClickToGoConfig(round_index=3, device="cpu")
    result = click_to_go.run_clean_and_swap(session_dir, "car", config, run_cli=fake_run_cli, export_splat=fake_export)

    cleaned_ply = session_dir / "clickgo" / "cleaned.ply"
    cleaned_splat = session_dir / "clickgo" / "cleaned.splat"
    assert calls == [
        [
            "splat-clean",
            "car",
            "--map",
            str(session_dir),
            "--output",
            str(cleaned_ply),
            "--device",
            "cpu",
            "--round",
            "3",
        ],
    ]
    assert exports == [(session_dir, cleaned_ply, cleaned_splat, 3)]
    assert result == {"prompt": "car", "splat": "/clickgo/cleaned.splat", "gaussians": 2}


def test_run_grab_and_swap_invokes_cli_and_exports(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    calls: list[list[str]] = []
    exports: list[tuple[Path, Path, Path, int | None]] = []

    def fake_run_cli(args: Sequence[str]) -> SimpleNamespace:
        calls.append(list(args))
        return SimpleNamespace(returncode=0)

    def fake_export(session: Path, grabbed_ply: Path, output_splat: Path, *, round_index: int | None = None) -> int:
        exports.append((Path(session), Path(grabbed_ply), Path(output_splat), round_index))
        Path(output_splat).write_bytes(b"\x00" * 96)  # three 32-byte gaussians
        return 3

    config = click_to_go.ClickToGoConfig(round_index=2, device="cpu")
    result = click_to_go.run_grab_and_swap(session_dir, "car", config, run_cli=fake_run_cli, export_splat=fake_export)

    grabbed_ply = session_dir / "clickgo" / "grabbed.ply"
    grabbed_splat = session_dir / "clickgo" / "grabbed.splat"
    assert calls == [
        [
            "splat-grab",
            "car",
            "--map",
            str(session_dir),
            "--output",
            str(grabbed_ply),
            "--device",
            "cpu",
            "--round",
            "2",
        ],
    ]
    assert exports == [(session_dir, grabbed_ply, grabbed_splat, 2)]
    assert result == {"prompt": "car", "splat": "/clickgo/grabbed.splat", "gaussians": 3}


def test_highlight_splat_file_glows_inside_box_and_dims_rest(tmp_path: Path) -> None:
    # three gaussians: one inside the box, two outside, all opaque (alpha 200)
    records = [
        (0.0, 0.0, 0.0, (10, 20, 30, 200)),  # inside the unit box at the origin
        (5.0, 5.0, 5.0, (40, 50, 60, 200)),  # far outside
        (-5.0, 0.0, 0.0, (70, 80, 90, 200)),  # far outside
    ]
    raw = bytearray()
    for x, y, z, rgba in records:
        raw += np.asarray([x, y, z, 0.1, 0.1, 0.1], dtype=np.float32).tobytes()
        raw += bytes(rgba)  # RGBA at offset 24
        raw += bytes((0, 0, 0, 0))  # rotation
    scene_splat = tmp_path / "scene.splat"
    scene_splat.write_bytes(bytes(raw))
    output_splat = tmp_path / "highlighted.splat"

    aabb = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    lit = click_to_go._highlight_splat_file(scene_splat, output_splat, [aabb])

    assert lit == 1
    out = np.frombuffer(output_splat.read_bytes(), dtype=np.uint8).reshape(3, 32)
    # the inside gaussian is repainted to the glow colour at full alpha
    assert tuple(out[0, 24:28]) == click_to_go.GLOW_RGBA
    # the outside gaussians keep their colour but their alpha is faded
    assert tuple(out[1, 24:27]) == (40, 50, 60)
    assert out[1, 27] == int(200 * click_to_go.DIM_ALPHA_SCALE)
    assert out[2, 27] == int(200 * click_to_go.DIM_ALPHA_SCALE)


def test_run_highlight_and_swap_queries_then_recolors(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    calls: list[list[str]] = []
    highlights: list[tuple[Path, Path, Path, int | None]] = []

    def fake_run_cli(args: Sequence[str]) -> SimpleNamespace:
        calls.append(list(args))
        if args[0] == "query-map":
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps({"hits": [{"centroid": [0, 0, 0]}, {"centroid": [1, 1, 1]}]}),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    def fake_highlight(session: Path, overlay_json: Path, output_splat: Path, *, round_index: int | None = None) -> int:
        highlights.append((Path(session), Path(overlay_json), Path(output_splat), round_index))
        Path(output_splat).write_bytes(b"\x00" * 96)
        return 2

    config = click_to_go.ClickToGoConfig(round_index=4, device="cpu")
    result = click_to_go.run_highlight_and_swap(
        session_dir, "car", config, run_cli=fake_run_cli, highlight_splat=fake_highlight
    )

    overlay_json = session_dir / "clickgo" / "overlay.json"
    highlighted_splat = session_dir / "clickgo" / "highlighted.splat"
    # the query + overlay pass runs first, exactly as run_query_and_overlay does
    assert [call[0] for call in calls] == ["query-map", "export-overlay"]
    assert highlights == [(session_dir, overlay_json, highlighted_splat, 4)]
    assert result == {
        "prompt": "car",
        "hits": 2,
        "highlighted": 2,
        "splat": "/clickgo/highlighted.splat",
        "overlay": "/clickgo/overlay.json",
    }


def test_quality_splat_file_heatmaps_by_opacity(tmp_path: Path) -> None:
    # three gaussians at opacity 0.0, ~0.5, 1.0 — low / mid / high confidence
    alphas = [0, 128, 255]
    raw = bytearray()
    for alpha in alphas:
        raw += np.asarray([0.0, 0.0, 0.0, 0.1, 0.1, 0.1], dtype=np.float32).tobytes()
        raw += bytes((10, 20, 30, alpha))  # RGBA at offset 24
        raw += bytes((0, 0, 0, 0))  # rotation
    scene_splat = tmp_path / "scene.splat"
    scene_splat.write_bytes(bytes(raw))
    output_splat = tmp_path / "quality.splat"

    stats = click_to_go._quality_splat_file(scene_splat, output_splat)

    assert stats["gaussians"] == 3
    assert stats["low_confidence"] == 1  # only opacity 0.0 is below the 0.3 cut
    assert stats["median_opacity"] == pytest.approx(128 / 255.0)
    out = np.frombuffer(output_splat.read_bytes(), dtype=np.uint8).reshape(3, 32)
    # alpha is pinned to the readable diagnostic constant for every gaussian
    assert list(out[:, 27]) == [click_to_go.QUALITY_ALPHA] * 3
    # the heatmap runs warm (low opacity) -> cool (high opacity): the low one is
    # redder, the high one is greener (RGB lives at byte offsets 24, 25, 26)
    assert out[0, 24] > out[2, 24]  # more red at low confidence
    assert out[2, 25] > out[0, 25]  # more green at high confidence
    # the endpoints land on the configured heatmap stops
    assert tuple(out[0, 24:27]) == click_to_go._CONFIDENCE_STOPS[0][1]
    assert tuple(out[2, 24:27]) == click_to_go._CONFIDENCE_STOPS[-1][1]


def test_quality_splat_file_penalizes_bloated_gaussians(tmp_path: Path) -> None:
    # every gaussian is fully opaque, so only the footprint can lower confidence:
    # scales sweep small -> large, and the bloated tail should read as low quality
    sizes = [round(0.1 * (i + 1), 3) for i in range(10)]  # 0.1 .. 1.0
    raw = bytearray()
    for size in sizes:
        raw += np.asarray([0.0, 0.0, 0.0, size, size, size], dtype=np.float32).tobytes()
        raw += bytes((10, 20, 30, 255))  # RGBA at offset 24 — uniform full opacity
        raw += bytes((0, 0, 0, 0))  # rotation
    scene_splat = tmp_path / "scene.splat"
    scene_splat.write_bytes(bytes(raw))
    output_splat = tmp_path / "quality.splat"

    stats = click_to_go._quality_splat_file(scene_splat, output_splat)

    assert stats["gaussians"] == 10
    # opacity is uniform, so the only confidence drop comes from oversized blobs
    assert stats["low_confidence"] >= 1
    out = np.frombuffer(output_splat.read_bytes(), dtype=np.uint8).reshape(10, 32)
    # the smallest gaussian stays cool/high; the largest is pushed warm/low
    assert out[-1, 24] > out[0, 24]  # biggest blob is redder
    assert out[0, 25] > out[-1, 25]  # smallest blob is greener


def test_surfel_splat_file_heatmaps_by_flatness(tmp_path: Path) -> None:
    # three gaussians: a flat disc, an isotropic blob, a needle. Only the disc is
    # a clean surface element; the blob and the needle should both read warm.
    shapes = [
        (0.3, 0.3, 0.01),  # disc: thin third axis -> high flatness
        (0.2, 0.2, 0.2),   # sphere: isotropic -> low flatness
        (0.9, 0.02, 0.02),  # needle: one long spike -> low flatness
    ]
    raw = bytearray()
    for sx, sy, sz in shapes:
        raw += np.asarray([0.0, 0.0, 0.0, sx, sy, sz], dtype=np.float32).tobytes()
        raw += bytes((10, 20, 30, 255))  # RGBA at offset 24
        raw += bytes((0, 0, 0, 0))  # rotation
    scene_splat = tmp_path / "scene.splat"
    scene_splat.write_bytes(bytes(raw))
    output_splat = tmp_path / "surfel.splat"

    stats = click_to_go._surfel_splat_file(scene_splat, output_splat)

    assert stats["gaussians"] == 3
    assert stats["surfels"] == 1  # only the disc clears the flat-surfel cut
    out = np.frombuffer(output_splat.read_bytes(), dtype=np.uint8).reshape(3, 32)
    # alpha is pinned to the readable diagnostic constant for every gaussian
    assert list(out[:, 27]) == [click_to_go.QUALITY_ALPHA] * 3
    # the disc runs cool/green; the blob and needle run warm/red
    assert out[0, 25] > out[1, 25]  # disc greener than the sphere
    assert out[0, 25] > out[2, 25]  # disc greener than the needle
    assert out[1, 24] > out[0, 24]  # sphere redder than the disc
    assert out[2, 24] > out[0, 24]  # needle redder than the disc


def test_run_surfel_and_swap_recolors_served_splat(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    builds: list[tuple[Path, Path, int | None]] = []

    def fake_surfel(session: Path, output_splat: Path, *, round_index: int | None = None) -> dict:
        builds.append((Path(session), Path(output_splat), round_index))
        Path(output_splat).write_bytes(b"\x00" * 64)
        return {"gaussians": 2, "surfels": 1, "median_flatness": 0.5}

    config = click_to_go.ClickToGoConfig(round_index=6, device="cpu")
    result = click_to_go.run_surfel_and_swap(session_dir, config, surfel_splat=fake_surfel)

    surfel_splat = session_dir / "clickgo" / "surfel.splat"
    assert builds == [(session_dir, surfel_splat, 6)]
    assert result == {
        "splat": "/clickgo/surfel.splat",
        "gaussians": 2,
        "surfels": 1,
        "median_flatness": 0.5,
    }


def test_run_quality_and_swap_recolors_served_splat(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    builds: list[tuple[Path, Path, int | None]] = []

    def fake_quality(session: Path, output_splat: Path, *, round_index: int | None = None) -> dict:
        builds.append((Path(session), Path(output_splat), round_index))
        Path(output_splat).write_bytes(b"\x00" * 64)
        return {"gaussians": 2, "low_confidence": 1, "median_opacity": 0.5}

    config = click_to_go.ClickToGoConfig(round_index=6, device="cpu")
    result = click_to_go.run_quality_and_swap(session_dir, config, quality_splat=fake_quality)

    quality_splat = session_dir / "clickgo" / "quality.splat"
    assert builds == [(session_dir, quality_splat, 6)]
    assert result == {
        "splat": "/clickgo/quality.splat",
        "gaussians": 2,
        "low_confidence": 1,
        "median_opacity": 0.5,
    }


def test_run_changes_and_overlay_invokes_cli_and_summarizes(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    calls: list[list[str]] = []

    def fake_run_cli(args: Sequence[str]) -> SimpleNamespace:
        calls.append(list(args))
        if args[0] == "detect-changes":
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps({"appeared": [{"points": 9}], "disappeared": [{"points": 4}, {"points": 7}]}),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    config = click_to_go.ClickToGoConfig(round_index=5, baseline_round=1, device="cpu")
    result = click_to_go.run_changes_and_overlay(session_dir, config, run_cli=fake_run_cli)

    changes_json = session_dir / "clickgo" / "changes.json"
    overlay_json = session_dir / "clickgo" / "overlay.json"
    assert calls == [
        [
            "detect-changes",
            "--map-a",
            str(session_dir),
            "--round-b",
            "1",
            "--align",
            "auto",
            "--output",
            str(changes_json),
            "--device",
            "cpu",
            "--round-a",
            "5",
        ],
        [
            "export-overlay",
            "--map",
            str(session_dir),
            "--output",
            str(overlay_json),
            "--changes",
            str(changes_json),
            "--round",
            "5",
        ],
    ]
    assert result == {
        "appeared": 1,
        "disappeared": 2,
        "overlay": "/clickgo/overlay.json",
        "changes_json": "/clickgo/changes.json",
    }


def test_run_changes_and_overlay_requires_baseline(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    config = click_to_go.ClickToGoConfig(device="cpu")  # no baseline_round

    with pytest.raises(ValueError, match="no baseline round configured"):
        click_to_go.run_changes_and_overlay(session_dir, config, run_cli=lambda args: SimpleNamespace(returncode=0))


def test_http_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "hello.txt").write_text("hello", encoding="utf-8")
    frame = _synthetic_frame()
    monkeypatch.setattr(click_to_go, "load_scene_frame", lambda *args, **kwargs: frame)

    def fake_run_cli(args: Sequence[str]) -> SimpleNamespace:
        if args[0] == "navigate":
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps({"reached": True, "steps": 7}),
                encoding="utf-8",
            )
        elif args[0] == "query-map":
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps({"hits": [{"centroid": [0, 0, 0]}]}),
                encoding="utf-8",
            )
        elif args[0] == "export-overlay":
            Path(args[args.index("--output") + 1]).write_text("{}", encoding="utf-8")
        elif args[0] in ("splat-clean", "splat-grab"):
            Path(args[args.index("--output") + 1]).write_text("ply", encoding="utf-8")
        elif args[0] == "detect-changes":
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps({"appeared": [{"points": 5}, {"points": 6}], "disappeared": [{"points": 3}]}),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    def fake_export(session: Path, cleaned_ply: Path, output_splat: Path, *, round_index: int | None = None) -> int:
        Path(output_splat).write_bytes(b"\x00" * 96)  # three 32-byte gaussians
        return 3

    def fake_highlight(session: Path, overlay_json: Path, output_splat: Path, *, round_index: int | None = None) -> int:
        Path(output_splat).write_bytes(b"\x00" * 128)  # four 32-byte gaussians
        return 4

    def fake_quality(session: Path, output_splat: Path, *, round_index: int | None = None) -> dict:
        Path(output_splat).write_bytes(b"\x00" * 160)  # five 32-byte gaussians
        return {"gaussians": 5, "low_confidence": 2, "median_opacity": 0.4}

    def fake_surfel(session: Path, output_splat: Path, *, round_index: int | None = None) -> dict:
        Path(output_splat).write_bytes(b"\x00" * 192)  # six 32-byte gaussians
        return {"gaussians": 6, "surfels": 4, "median_flatness": 0.7}

    server = click_to_go.make_server(
        session_dir,
        click_to_go.ClickToGoConfig(port=0, device="cpu", baseline_round=1),
        run_cli=fake_run_cli,
        export_splat=fake_export,
        highlight_splat=fake_highlight,
        quality_splat=fake_quality,
        surfel_splat=fake_surfel,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    try:
        _, origin_splat, direction_splat = _ray_for_goal(frame, (0.5, -0.25))

        status, headers, body = _request_json(
            host,
            port,
            "POST",
            "/goal",
            {"origin": origin_splat.tolist(), "direction": direction_splat.tolist()},
        )
        assert status == 200
        assert headers["Access-Control-Allow-Origin"] == "*"
        assert body["reached"] is True
        assert body["steps"] == 7
        assert body["overlay"] == "/clickgo/overlay.json"

        status, _, body = _request_json(host, port, "POST", "/query", {"prompt": "car"})
        assert status == 200
        assert body["prompt"] == "car"
        assert body["hits"] == 1
        assert body["overlay"] == "/clickgo/overlay.json"

        status, _, body = _request_json(host, port, "POST", "/query", {"prompt": "   "})
        assert status == 400

        status, _, body = _request_json(host, port, "POST", "/clean", {"prompt": "car"})
        assert status == 200
        assert body["prompt"] == "car"
        assert body["splat"] == "/clickgo/cleaned.splat"
        assert body["gaussians"] == 3
        assert (session_dir / "clickgo" / "cleaned.splat").stat().st_size == 96

        status, _, body = _request_json(host, port, "POST", "/clean", {"prompt": "   "})
        assert status == 400

        status, _, body = _request_json(host, port, "POST", "/grab", {"prompt": "car"})
        assert status == 200
        assert body["prompt"] == "car"
        assert body["splat"] == "/clickgo/grabbed.splat"
        assert body["gaussians"] == 3
        assert (session_dir / "clickgo" / "grabbed.splat").stat().st_size == 96

        status, _, body = _request_json(host, port, "POST", "/grab", {"prompt": "   "})
        assert status == 400

        status, _, body = _request_json(host, port, "POST", "/highlight", {"prompt": "car"})
        assert status == 200
        assert body["prompt"] == "car"
        assert body["splat"] == "/clickgo/highlighted.splat"
        assert body["highlighted"] == 4
        assert body["overlay"] == "/clickgo/overlay.json"
        assert (session_dir / "clickgo" / "highlighted.splat").stat().st_size == 128

        status, _, body = _request_json(host, port, "POST", "/highlight", {"prompt": "   "})
        assert status == 400

        status, _, body = _request_json(host, port, "POST", "/quality", {})
        assert status == 200
        assert body["splat"] == "/clickgo/quality.splat"
        assert body["gaussians"] == 5
        assert body["low_confidence"] == 2
        assert body["median_opacity"] == 0.4
        assert (session_dir / "clickgo" / "quality.splat").stat().st_size == 160

        status, _, body = _request_json(host, port, "POST", "/surfel", {})
        assert status == 200
        assert body["splat"] == "/clickgo/surfel.splat"
        assert body["gaussians"] == 6
        assert body["surfels"] == 4
        assert body["median_flatness"] == 0.7
        assert (session_dir / "clickgo" / "surfel.splat").stat().st_size == 192

        status, _, body = _request_json(host, port, "POST", "/changes", {})
        assert status == 200
        assert body["appeared"] == 2
        assert body["disappeared"] == 1
        assert body["overlay"] == "/clickgo/overlay.json"

        status, _, body = _request_raw(host, port, "POST", "/goal", b"{", "application/json")
        assert status == 400
        assert "error" in body

        status, _, body = _request_json(
            host,
            port,
            "POST",
            "/goal",
            {"origin": [0.0, 0.0, 1.0], "direction": [1.0, 0.0, 0.0]},
        )
        assert status == 422
        assert "does not hit the ground plane" in body["error"]

        conn = http.client.HTTPConnection(host, port)
        conn.request("OPTIONS", "/goal")
        response = conn.getresponse()
        response.read()
        assert response.status == 204
        assert response.getheader("Access-Control-Allow-Origin") == "*"
        conn.close()

        conn = http.client.HTTPConnection(host, port)
        conn.request("GET", "/hello.txt")
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("Access-Control-Allow-Origin") == "*"
        assert response.read() == b"hello"
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def _synthetic_frame() -> click_to_go.SceneFrame:
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mapper = SplatFrameMapper(
        scale=2.0,
        rotation=rotation,
        translation=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        centroid=np.array([0.5, 0.0, 0.0], dtype=np.float64),
        factor=3.0,
    )
    basis = np.eye(3, dtype=np.float64)
    return click_to_go.SceneFrame(
        mapper=mapper,
        basis=basis,
        ground_height=0.0,
        camera_height=1.0,
        splat_rel="rounds/round_001/scene.splat",
    )


def _ray_for_goal(
    frame: click_to_go.SceneFrame,
    goal_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world = goal_xy[0] * frame.basis[0] + goal_xy[1] * frame.basis[1] + frame.ground_height * frame.basis[2]
    point_splat = frame.mapper.points(np.asarray([world], dtype=np.float64))[0]
    up_splat = frame.basis[2] @ frame.mapper.rotation.T
    up_splat = up_splat / np.linalg.norm(up_splat)
    origin_splat = point_splat + up_splat
    direction_splat = -up_splat
    return point_splat, origin_splat, direction_splat


def _request_json(
    host: str,
    port: int,
    method: str,
    path: str,
    payload: dict[str, object],
) -> tuple[int, dict[str, str], dict[str, object]]:
    return _request_raw(host, port, method, path, json.dumps(payload).encode("utf-8"), "application/json")


def _request_raw(
    host: str,
    port: int,
    method: str,
    path: str,
    body: bytes,
    content_type: str,
) -> tuple[int, dict[str, str], dict[str, object]]:
    conn = http.client.HTTPConnection(host, port)
    conn.request(method, path, body=body, headers={"Content-Type": content_type})
    response = conn.getresponse()
    raw = response.read()
    headers = {key: value for key, value in response.getheaders()}
    conn.close()
    return response.status, headers, json.loads(raw.decode("utf-8"))
