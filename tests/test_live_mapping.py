"""Tests for the rclpy-free live mapping core (fake builder, no GPU)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.robotics.live_mapping import (
    Keyframe,
    KeyframeSelector,
    LiveMapperConfig,
    LiveMappingSession,
    select_round_frames,
)


def _frame(seed: int, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


class TestKeyframeSelector:
    def test_first_frame_is_always_a_keyframe(self):
        selector = KeyframeSelector()
        assert selector.consider(_frame(0), timestamp=0.0)

    def test_time_gate_rejects_rapid_frames(self):
        selector = KeyframeSelector(min_gap_s=1.0, min_motion=0.0)
        assert selector.consider(_frame(0), timestamp=0.0)
        assert not selector.consider(_frame(1), timestamp=0.5)
        assert selector.consider(_frame(2), timestamp=1.5)

    def test_motion_gate_rejects_static_scene(self):
        selector = KeyframeSelector(min_gap_s=0.0, min_motion=0.04)
        frame = _frame(0)
        assert selector.consider(frame, timestamp=0.0)
        assert not selector.consider(frame.copy(), timestamp=1.0)
        assert selector.consider(_frame(99), timestamp=2.0)

    def test_translation_gate_used_when_positions_provided(self):
        selector = KeyframeSelector(min_gap_s=0.0, min_motion=0.0, min_translation_m=1.0)
        frame = _frame(0)
        assert selector.consider(frame, timestamp=0.0, position=np.zeros(3))
        assert not selector.consider(frame, timestamp=1.0, position=np.array([0.5, 0.0, 0.0]))
        assert selector.consider(frame, timestamp=2.0, position=np.array([1.5, 0.0, 0.0]))


class TestSelectRoundFrames:
    def _keyframes(self, count: int) -> list[Keyframe]:
        return [Keyframe(index=i, timestamp=float(i), path=Path(f"kf_{i}.jpg")) for i in range(count)]

    def test_returns_all_when_under_cap(self):
        keyframes = self._keyframes(5)
        assert select_round_frames(keyframes, 10) == keyframes
        assert select_round_frames(keyframes, 0) == keyframes

    def test_strides_evenly_keeping_endpoints(self):
        keyframes = self._keyframes(100)
        selected = select_round_frames(keyframes, 10)
        assert len(selected) == 10
        assert selected[0].index == 0
        assert selected[-1].index == 99
        assert [k.index for k in selected] == sorted(k.index for k in selected)


class _FakeBuilder:
    def __init__(self, fail_rounds: set[int] | None = None):
        self.calls: list[tuple[Path, Path]] = []
        self.fail_rounds = fail_rounds or set()

    def __call__(self, images_dir: Path, round_dir: Path) -> Path:
        self.calls.append((images_dir, round_dir))
        if len(self.calls) in self.fail_rounds:
            raise RuntimeError("synthetic backend failure")
        splat = round_dir / "scene.splat"
        image_count = len(list(images_dir.glob("*.jpg")))
        splat.write_bytes(b"\x00" * 32 * image_count)
        return splat


def _config(tmp_path: Path, **overrides) -> LiveMapperConfig:
    defaults = dict(
        workdir=tmp_path / "session",
        min_keyframe_gap_s=0.0,
        min_keyframe_motion=0.0,
        rebuild_min_new_keyframes=3,
        num_frames=8,
    )
    defaults.update(overrides)
    return LiveMapperConfig(**defaults)


def _wait_for(predicate, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for condition")


class TestLiveMappingSession:
    def test_keyframes_are_persisted_as_jpegs(self, tmp_path):
        session = LiveMappingSession(_config(tmp_path), builder=_FakeBuilder())
        assert session.add_frame(_frame(0), timestamp=0.0)
        assert session.add_frame(_frame(1), timestamp=1.0)
        files = sorted((tmp_path / "session" / "keyframes").glob("kf_*.jpg"))
        assert len(files) == 2

    def test_background_round_publishes_latest_splat_and_state(self, tmp_path):
        builder = _FakeBuilder()
        session = LiveMappingSession(_config(tmp_path), builder=builder)
        session.start()
        for i in range(4):
            session.add_frame(_frame(i), timestamp=float(i))
        _wait_for(lambda: len(session.rounds) >= 1)
        session.stop()

        live = tmp_path / "session" / "live"
        assert (live / "latest.splat").exists()
        state = json.loads((live / "state.json").read_text())
        assert state["completedRounds"] >= 1
        assert state["keyframesTotal"] == 4
        assert state["lastSuccessfulRound"]["error"] is None
        assert state["splatUrl"] == "latest.splat"

    def test_stop_flushes_pending_keyframes_into_final_round(self, tmp_path):
        builder = _FakeBuilder()
        session = LiveMappingSession(_config(tmp_path, rebuild_min_new_keyframes=100), builder=builder)
        session.start()
        session.add_frame(_frame(0), timestamp=0.0)
        session.add_frame(_frame(1), timestamp=1.0)
        session.stop(wait=True, timeout=10.0)
        assert len(builder.calls) == 1
        assert (tmp_path / "session" / "live" / "latest.splat").exists()

    def test_builder_failure_is_recorded_and_mapping_continues(self, tmp_path):
        builder = _FakeBuilder(fail_rounds={1})
        session = LiveMappingSession(_config(tmp_path), builder=builder)
        session.start()
        for i in range(4):
            session.add_frame(_frame(i), timestamp=float(i))
        _wait_for(lambda: len(session.rounds) >= 1)
        for i in range(4, 8):
            session.add_frame(_frame(i), timestamp=float(i))
        _wait_for(lambda: any(r.error is None for r in session.rounds))
        session.stop()

        assert session.rounds[0].error is not None
        assert (tmp_path / "session" / "live" / "latest.splat").exists()
        state = json.loads((tmp_path / "session" / "live" / "state.json").read_text())
        assert state["rounds"][0]["error"] is not None

    def test_max_keyframes_cap(self, tmp_path):
        session = LiveMappingSession(_config(tmp_path, max_keyframes=3), builder=_FakeBuilder())
        accepted = sum(session.add_frame(_frame(i), timestamp=float(i)) for i in range(10))
        assert accepted == 3

    def test_viewer_html_copied_into_live_dir(self, tmp_path):
        viewer = tmp_path / "viewer.html"
        viewer.write_text("<html>live</html>")
        config = _config(tmp_path, viewer_html=viewer)
        LiveMappingSession(config, builder=_FakeBuilder())
        assert (tmp_path / "session" / "live" / "index.html").read_text() == "<html>live</html>"

    def test_round_uses_strided_subset_when_over_cap(self, tmp_path):
        builder = _FakeBuilder()
        session = LiveMappingSession(_config(tmp_path, num_frames=4, rebuild_min_new_keyframes=6), builder=builder)
        session.start()
        for i in range(6):
            session.add_frame(_frame(i), timestamp=float(i))
        _wait_for(lambda: len(session.rounds) >= 1)
        session.stop()
        staged_dir = builder.calls[0][0]
        assert len(list(staged_dir.glob("kf_*.jpg"))) == 4


@pytest.mark.parametrize("status_field", ["status", "keyframesTotal", "rounds", "updatedUnix"])
def test_initial_state_json_schema(tmp_path, status_field):
    LiveMappingSession(_config(tmp_path), builder=_FakeBuilder())
    state = json.loads((tmp_path / "session" / "live" / "state.json").read_text())
    assert status_field in state
