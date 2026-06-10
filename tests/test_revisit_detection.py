"""Revisit (loop candidate) detection v1: record + persist, no correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gs_sim2real.robotics.live_mapping import (
    LiveMapperConfig,
    LiveMappingSession,
    RevisitDetector,
)


def _frame(seed: int, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def _detector(**overrides) -> RevisitDetector:
    defaults = dict(min_time_separation_s=10.0, min_index_separation=3, max_distance=0.04)
    defaults.update(overrides)
    return RevisitDetector(**defaults)


class TestRevisitDetector:
    def test_flags_revisit_of_an_old_keyframe(self) -> None:
        detector = _detector()
        detector.add_keyframe(_frame(0), timestamp=0.0)
        for i in range(1, 5):
            assert detector.add_keyframe(_frame(i), timestamp=float(i)) is None
        candidate = detector.add_keyframe(_frame(0), timestamp=60.0)
        assert candidate is not None
        assert candidate.match_index == 0
        assert candidate.query_index == 5
        assert candidate.distance == 0.0
        assert detector.candidates == [candidate]

    def test_recent_keyframes_never_match(self) -> None:
        detector = _detector(min_index_separation=10)
        detector.add_keyframe(_frame(0), timestamp=0.0)
        # identical image, huge time gap, but only 1 keyframe apart
        assert detector.add_keyframe(_frame(0), timestamp=1000.0) is None

    def test_time_separation_gate(self) -> None:
        detector = _detector(min_time_separation_s=100.0)
        detector.add_keyframe(_frame(0), timestamp=0.0)
        for i in range(1, 5):
            detector.add_keyframe(_frame(i), timestamp=float(i))
        assert detector.add_keyframe(_frame(0), timestamp=50.0) is None
        assert detector.add_keyframe(_frame(0), timestamp=150.0) is not None

    def test_dissimilar_images_do_not_match(self) -> None:
        detector = _detector()
        detector.add_keyframe(_frame(0), timestamp=0.0)
        for i in range(1, 5):
            detector.add_keyframe(_frame(i), timestamp=float(i))
        assert detector.add_keyframe(_frame(99), timestamp=60.0) is None
        assert detector.candidates == []

    def test_best_match_wins_over_first_match(self) -> None:
        base = _frame(0).astype(np.int16)
        near = np.clip(base + 4, 0, 255).astype(np.uint8)  # slightly off
        detector = _detector(max_distance=0.1)
        detector.add_keyframe(near, timestamp=0.0)
        detector.add_keyframe(_frame(0), timestamp=1.0)
        for i in range(2, 6):
            detector.add_keyframe(_frame(i + 10), timestamp=float(i))
        candidate = detector.add_keyframe(_frame(0), timestamp=60.0)
        assert candidate is not None
        assert candidate.match_index == 1  # exact copy beats the offset one


class TestSessionLoopCandidatePersistence:
    def _config(self, tmp_path: Path) -> LiveMapperConfig:
        return LiveMapperConfig(
            workdir=tmp_path / "session",
            min_keyframe_gap_s=0.0,
            min_keyframe_motion=0.0,
            rebuild_min_new_keyframes=100,  # no background rounds needed here
            revisit_min_time_separation_s=10.0,
            revisit_min_index_separation=3,
            revisit_max_distance=0.04,
        )

    def test_candidates_are_persisted_and_counted(self, tmp_path: Path) -> None:
        session = LiveMappingSession(self._config(tmp_path), builder=lambda images, round_dir: round_dir)
        session.add_frame(_frame(0), timestamp=0.0)
        for i in range(1, 5):
            session.add_frame(_frame(i), timestamp=float(i))
        session.add_frame(_frame(0), timestamp=60.0)

        live = tmp_path / "session" / "live"
        payload = json.loads((live / "loop_candidates.json").read_text())
        assert len(payload["loopCandidates"]) == 1
        candidate = payload["loopCandidates"][0]
        assert candidate["matchIndex"] == 0
        assert candidate["queryIndex"] == 5

        session._write_state(status="idle")
        state = json.loads((live / "state.json").read_text())
        assert state["loopCandidates"] == 1

    def test_detection_can_be_disabled(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        config.revisit_detection = False
        session = LiveMappingSession(config, builder=lambda images, round_dir: round_dir)
        session.add_frame(_frame(0), timestamp=0.0)
        for i in range(1, 5):
            session.add_frame(_frame(i), timestamp=float(i))
        session.add_frame(_frame(0), timestamp=60.0)
        assert session.revisit_detector is None
        assert not (tmp_path / "session" / "live" / "loop_candidates.json").exists()
        state = json.loads((tmp_path / "session" / "live" / "state.json").read_text())
        assert state["loopCandidates"] == 0
