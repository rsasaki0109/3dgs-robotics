"""Rerun replay export for live-mapping sessions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

Sim3 = tuple[float, np.ndarray, np.ndarray]


@dataclass
class RoundEntry:
    round_index: int
    positions: np.ndarray
    colors: np.ndarray
    centers: np.ndarray
    image_path: Path | None


def resolve_live_map_session(*args: Any, **kwargs: Any) -> Any:
    """Lazy seam for tests; resolves a live-map session round."""
    from gs_sim2real.robotics.localize import resolve_live_map_session as _resolve_live_map_session

    return _resolve_live_map_session(*args, **kwargs)


def load_mapped_records(*args: Any, **kwargs: Any) -> Any:
    """Lazy seam for tests; loads localized keyframe records."""
    from gs_sim2real.robotics.localize import load_mapped_records as _load_mapped_records

    return _load_mapped_records(*args, **kwargs)


def load_ply(*args: Any, **kwargs: Any) -> Any:
    """Lazy seam for tests; loads a raw gaussian PLY."""
    from gs_sim2real.viewer.web_viewer import load_ply as _load_ply

    return _load_ply(*args, **kwargs)


def read_gauge_transform(*args: Any, **kwargs: Any) -> Any:
    """Lazy seam for tests; loads a round-gauge to session-gauge transform."""
    from gs_sim2real.robotics.gauge_alignment import read_gauge_transform as _read_gauge_transform

    return _read_gauge_transform(*args, **kwargs)


def apply_to_points(*args: Any, **kwargs: Any) -> Any:
    """Lazy seam for tests; applies a similarity transform to points."""
    from gs_sim2real.robotics.gauge_alignment import apply_to_points as _apply_to_points

    return _apply_to_points(*args, **kwargs)


def estimate_ground_frame(*args: Any, **kwargs: Any) -> Any:
    """Lazy seam for tests; estimates the navigation plane frame."""
    from gs_sim2real.robotics.occupancy_grid import estimate_ground_frame as _estimate_ground_frame

    return _estimate_ground_frame(*args, **kwargs)


def _identity_transform() -> Sim3:
    return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)


def _round_index(round_dir: Path) -> int:
    match = re.search(r"round_(\d+)", round_dir.name)
    if match is None:
        raise ValueError(f"invalid round directory name: {round_dir.name}")
    return int(match.group(1))


def _successful_round_dirs(session_dir: Path) -> list[Path]:
    round_dirs = []
    for round_dir in sorted((session_dir / "rounds").glob("round_*")):
        if (round_dir / "train" / "point_cloud.ply").is_file() and (round_dir / "scene.splat").is_file():
            round_dirs.append(round_dir)
    return round_dirs


def _round_transform(round_dir: Path) -> Sim3:
    loaded = read_gauge_transform(round_dir)
    if loaded is None:
        return _identity_transform()
    transform, _rebased = loaded
    return transform


def _round_records(session_dir: Path, round_index: int) -> list[Any]:
    session = resolve_live_map_session(session_dir, round_index=round_index)
    return list(load_mapped_records(session))


def _centers(records: list[Any]) -> np.ndarray:
    return np.asarray([record.center for record in records], dtype=np.float64)


def _qvecs(records: list[Any]) -> list[Any]:
    return [record.qvec for record in records]


def _colors_from_ply(ply: Any, count: int) -> np.ndarray:
    colors = getattr(ply, "colors", None)
    if colors is None:
        return np.full((count, 3), 128, dtype=np.uint8)
    return np.clip(np.asarray(colors, dtype=np.float64) * 255.0, 0.0, 255.0).astype(np.uint8)


def _subsample(positions: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0:
        raise ValueError("max_points_per_round must be positive")
    if len(positions) <= max_points:
        return positions, colors
    indices = np.linspace(0, len(positions) - 1, max_points, dtype=np.int64)
    return positions[indices], colors[indices]


def _newest_image(round_dir: Path) -> Path | None:
    image_dir = round_dir / "images"
    if not image_dir.is_dir():
        return None
    images = sorted(path for path in image_dir.iterdir() if path.is_file())
    return images[-1] if images else None


def _loop_edges(session_dir: Path, final_centers: np.ndarray) -> list[list[list[float]]]:
    path = session_dir / "live" / "loop_candidates.json"
    if not path.is_file():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    edges: list[list[list[float]]] = []
    for candidate in data.get("loopCandidates") or []:
        query_index = candidate.get("queryIndex")
        match_index = candidate.get("matchIndex")
        if not isinstance(query_index, int) or not isinstance(match_index, int):
            continue
        if 0 <= query_index < len(final_centers) and 0 <= match_index < len(final_centers):
            edges.append([final_centers[query_index].tolist(), final_centers[match_index].tolist()])
    return edges


def _nav_points(
    nav_json: Path,
    final_ply_path: Path,
    final_records: list[Any],
    final_transform: Sim3,
) -> list[list[float]] | None:
    from gs_sim2real.robotics.viewer_overlay import _lift_plane_points

    nav = json.loads(nav_json.read_text(encoding="utf-8"))
    vertices = nav.get("path_vertices")
    if not vertices:
        return None

    centers = _centers(final_records)
    ply_points = np.asarray(load_ply(final_ply_path).positions, dtype=np.float64)
    up, forward, _ground, camera_height = estimate_ground_frame(
        centers,
        _qvecs(final_records),
        lambda candidate_up: ply_points @ candidate_up,
        ground_percentile=30.0,
    )
    basis = np.stack([forward, np.cross(up, forward), up])
    lifted = _lift_plane_points(np.asarray(vertices, dtype=np.float64), basis, centers, float(camera_height))
    return apply_to_points(final_transform, lifted).tolist()


def session_timeline(
    session_dir: Path,
    *,
    max_points_per_round: int = 200_000,
    nav_json: Path | None = None,
) -> tuple[list[RoundEntry], dict[str, Any]]:
    """Assemble a rerun timeline without importing rerun."""
    session_dir = Path(session_dir)
    entries: list[RoundEntry] = []
    round_dirs = _successful_round_dirs(session_dir)
    if not round_dirs:
        raise FileNotFoundError(
            f"no successful rounds under {session_dir / 'rounds'}; expected round_*/train/point_cloud.ply and scene.splat"
        )

    final_records: list[Any] | None = None
    final_transform: Sim3 | None = None
    final_ply_path: Path | None = None

    for round_dir in round_dirs:
        round_index = _round_index(round_dir)
        ply_path = round_dir / "train" / "point_cloud.ply"
        transform = _round_transform(round_dir)

        ply = load_ply(ply_path)
        positions = apply_to_points(transform, np.asarray(ply.positions, dtype=np.float64))
        colors = _colors_from_ply(ply, len(positions))
        positions, colors = _subsample(positions, colors, max_points_per_round)

        records = _round_records(session_dir, round_index)
        centers = apply_to_points(transform, _centers(records))

        entries.append(
            RoundEntry(
                round_index=round_index,
                positions=positions,
                colors=colors,
                centers=centers,
                image_path=_newest_image(round_dir),
            )
        )
        final_records = records
        final_transform = transform
        final_ply_path = ply_path

    final_centers = entries[-1].centers
    extras = {
        "loop_edges": _loop_edges(session_dir, final_centers),
        "nav_points": None,
        "session": str(session_dir),
    }
    if (
        nav_json is not None
        and final_records is not None
        and final_transform is not None
        and final_ply_path is not None
    ):
        extras["nav_points"] = _nav_points(Path(nav_json), final_ply_path, final_records, final_transform)

    return entries, extras


def _import_rerun() -> Any:
    try:
        import rerun as rr
    except ImportError as error:
        raise ImportError('rerun-sdk is required: pip install "3dgs-robotics[rerun]"') from error
    return rr


def _set_round(rr: Any, index: int) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("round", index)
    else:
        rr.set_time("round", sequence=index)


def log_session(
    entries: list[RoundEntry],
    extras: dict[str, Any],
    *,
    rr: Any | None = None,
    save: Path | None = None,
    spawn: bool = False,
    application_id: str = "3dgs-robotics",
) -> dict[str, Any]:
    """Log assembled replay entries to rerun."""
    rr = rr or _import_rerun()
    rr.init(application_id, spawn=spawn)
    if save is not None:
        rr.save(str(save))

    total_points = 0
    for entry in entries:
        _set_round(rr, entry.round_index)
        rr.log("world/map", rr.Points3D(entry.positions, colors=entry.colors))
        total_points += int(len(entry.positions))

        if len(entry.centers) >= 2:
            rr.log("world/trajectory", rr.LineStrips3D([entry.centers], colors=[(80, 200, 120)]))

        if entry.image_path is not None:
            import cv2

            bgr = cv2.imread(str(entry.image_path))
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rr.log("camera/image", rr.Image(rgb))

    if entries:
        _set_round(rr, entries[-1].round_index)

    loop_edges = extras.get("loop_edges") or []
    if loop_edges:
        rr.log("world/loops", rr.LineStrips3D(loop_edges, colors=[(240, 200, 60)]))

    nav_points = extras.get("nav_points")
    if nav_points is not None and len(nav_points) >= 2:
        rr.log("world/nav/planned_path", rr.LineStrips3D([nav_points], colors=[(64, 158, 244)]))

    return {
        "rounds": len(entries),
        "points_logged": total_points,
        "loop_edges": len(loop_edges),
        "nav": nav_points is not None,
        "rrd": str(save) if save else None,
    }
