"""Tests for replay-driven active mapping over growing session-gauge maps."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gs_sim2real.robotics.active_mapping as active_mapping
from gs_sim2real.robotics.active_mapping import (
    ActiveMappingConfig,
    ActiveMappingDriver,
    SessionGrid,
    map_frontier_clusters,
)
from gs_sim2real.robotics.gauge_alignment import quat_multiply, rotation_to_quat
from gs_sim2real.robotics.occupancy_grid import OccupancyGridMap
from gs_sim2real.robotics.splat_explore import reachable_free_mask
from gs_sim2real.robotics.splat_nav import Pose2D


class FakeSession:
    def __init__(self, build_results: list[bool] | None = None) -> None:
        self.workdir = Path("/tmp/fake-active-mapping")
        self.frames: list[tuple[np.ndarray, float]] = []
        self._build_results = list(build_results) if build_results is not None else []

    def add_frame(self, image: np.ndarray, timestamp: float) -> bool:
        self.frames.append((image, timestamp))
        return True

    def build_pending_round(self) -> bool:
        if self._build_results:
            return self._build_results.pop(0)
        return True


def _corridor_grid(
    mapped_width: int = 40,
    *,
    round_index: int = 1,
    camera_height: float = 1.0,
    centers_max_col: int | None = None,
    centers_step: int = 5,
) -> SessionGrid:
    """Free corridor with side walls and unknown beyond the right mapping edge.

    ``centers_max_col`` / ``centers_step`` shape the mapped keyframe centers so
    tests can place them short of (or dense across) the corridor's open end.
    """
    height = 21
    start_col = 2
    end_col = start_col + mapped_width
    total_width = end_col + 8

    data = np.full((height, total_width), -1, dtype=np.int8)
    data[4, start_col - 1 : end_col] = 100
    data[14, start_col - 1 : end_col] = 100
    data[5:14, start_col - 1] = 100
    data[5:14, start_col:end_col] = 0

    grid = OccupancyGridMap(
        data=data,
        resolution=0.25,
        origin=(0.0, 0.0),
        up=np.array([0.0, 0.0, 1.0]),
        basis=np.eye(3),
        ground_height=0.0,
        camera_height=camera_height,
    )

    center_y = (5.0 + 14.0) * 0.5 * grid.resolution
    limit = centers_max_col if centers_max_col is not None else max(mapped_width - 2, 1)
    xs = [(start_col + 2 + index + 0.5) * grid.resolution for index in range(0, limit, centers_step)]
    centers = np.asarray([[x, center_y, 0.0] for x in xs], dtype=np.float64)
    return SessionGrid(grid=grid, centers=centers, round_index=round_index, gaussian_count=mapped_width * 9)


def _loader(*states: SessionGrid):
    remaining = list(states)
    last = remaining[-1]

    def load(_session_dir: Path) -> SessionGrid:
        nonlocal last
        if remaining:
            last = remaining.pop(0)
        return last

    return load


def _frames(count: int) -> list[tuple[np.ndarray, float]]:
    return [(np.zeros((2, 2, 3), dtype=np.uint8), float(index)) for index in range(count)]


def _patch_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_plan_path(_grid, start_xy, goal_xy, **_kwargs):
        return np.vstack([np.asarray(start_xy, dtype=np.float64), np.asarray(goal_xy, dtype=np.float64)])

    def fake_run_navigation(_grid, start, goal_xy, **_kwargs):
        goal_xy = np.asarray(goal_xy, dtype=np.float64)
        frame = SimpleNamespace(true_pose=Pose2D(float(goal_xy[0]), float(goal_xy[1]), float(start.yaw)))
        return SimpleNamespace(frames=[frame], reached=True)

    monkeypatch.setattr(active_mapping, "plan_path", fake_plan_path)
    monkeypatch.setattr(active_mapping, "run_navigation", fake_run_navigation)


def test_map_frontier_clusters_finds_open_corridor_end() -> None:
    state = _corridor_grid(mapped_width=20)
    start_cell = (9, 4)
    reachable = reachable_free_mask(state.grid, start_cell, robot_radius=0.0)

    frontiers = map_frontier_clusters(state.grid, reachable, min_cells=8)

    assert len(frontiers) == 1
    assert frontiers[0].size == 9
    assert frontiers[0].centroid_xy[0] > 5.0
    assert map_frontier_clusters(state.grid, reachable, min_cells=10) == []


def test_driver_step_grows_toward_frontier(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_navigation(monkeypatch)
    first = _corridor_grid(mapped_width=40, round_index=1, centers_max_col=30)
    grown = _corridor_grid(mapped_width=55, round_index=2, centers_step=2)
    driver = ActiveMappingDriver(
        FakeSession(),
        iter(_frames(8)),
        ActiveMappingConfig(batch_frames=3, max_rounds=3, min_frontier_cells=4, growth_tolerance=0.3),
        grid_loader=_loader(first, first, grown),
    )

    driver.bootstrap(initial_frames=2)
    entry = driver.step()

    assert entry is not None
    assert entry["frames_fed"] == 3
    assert entry["frames_accepted"] == 3
    assert entry["built"] is True
    assert entry["new_keyframes"] > 0
    assert entry["grew"] is True
    assert driver.exhausted_frontiers == []


def test_driver_exhausts_frontier_when_round_does_not_grow(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_navigation(monkeypatch)
    state = _corridor_grid(mapped_width=40, round_index=1, centers_max_col=30)
    driver = ActiveMappingDriver(
        FakeSession(),
        iter(_frames(8)),
        ActiveMappingConfig(batch_frames=3, max_rounds=3, min_frontier_cells=4, growth_tolerance=0.3),
        grid_loader=_loader(state, state, state, state),
    )

    driver.bootstrap(initial_frames=2)
    entry = driver.step()
    next_entry = driver.step()

    assert entry is not None
    assert entry["grew"] is False
    assert len(driver.exhausted_frontiers) == 1
    assert next_entry is None
    assert driver.stop_reason == "no-frontiers"


def test_driver_stops_when_frames_are_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_navigation(monkeypatch)
    state = _corridor_grid(mapped_width=40, round_index=1)
    driver = ActiveMappingDriver(
        FakeSession(),
        iter(_frames(1)),
        ActiveMappingConfig(batch_frames=3, max_rounds=3, min_frontier_cells=4, growth_tolerance=0.3),
        grid_loader=_loader(state, state),
    )

    driver.bootstrap(initial_frames=1)
    entry = driver.step()

    assert entry is None
    assert driver.stop_reason == "frames-exhausted"


def test_run_honors_max_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_navigation(monkeypatch)
    first = _corridor_grid(mapped_width=40, round_index=1)
    grown = _corridor_grid(mapped_width=55, round_index=2)
    driver = ActiveMappingDriver(
        FakeSession(),
        iter(_frames(8)),
        ActiveMappingConfig(batch_frames=3, max_rounds=1, min_frontier_cells=4, growth_tolerance=0.3),
        grid_loader=_loader(first, first, grown),
    )

    driver.bootstrap(initial_frames=2)
    result = driver.run()

    assert len(result["entries"]) == 1
    assert result["stop_reason"] == "max-rounds"


def test_bootstrap_failure_mentions_backend_and_keyframes() -> None:
    state = _corridor_grid(mapped_width=40, round_index=1)
    driver = ActiveMappingDriver(
        FakeSession(build_results=[False]),
        iter(_frames(2)),
        ActiveMappingConfig(),
        grid_loader=_loader(state),
    )

    with pytest.raises(RuntimeError, match="backend/keyframes"):
        driver.bootstrap(initial_frames=2)


def test_qvec_identity_gauge_rotation_round_trips() -> None:
    qvec = np.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float64)
    rotated = quat_multiply(qvec, rotation_to_quat(np.eye(3).T))

    assert np.allclose(rotated, qvec)
