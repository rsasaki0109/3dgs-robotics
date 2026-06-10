"""Incremental live 3DGS mapping: camera frames in, growing browser .splat out.

The session accumulates keyframes on disk and rebuilds a draft-quality splat in
a background thread whenever enough new keyframes arrive. Each rebuild covers
the whole trajectory so far (evenly strided), so the published map grows as the
robot drives. ``live/latest.splat`` and ``live/state.json`` are replaced
atomically so a polling web viewer never sees a partial file.

This module is intentionally rclpy-free: the same session is driven by the
``live_mapper`` ROS 2 node and by ``scripts/run_live_mapping_demo.py`` (folder
replay), and unit tests inject a fake builder instead of the real
DUSt3R + gsplat backend.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# images_dir, round_dir -> path of the built .splat
SplatBuilder = Callable[[Path, Path], Path]

_MOTION_THUMB_SIZE = (64, 64)


@dataclass
class LiveMapperConfig:
    """Tuning knobs for keyframe selection and incremental rebuilds."""

    workdir: Path
    method: str = "dust3r"  # dust3r | mast3r | vggt | simple
    # keyframe gating
    min_keyframe_gap_s: float = 1.0
    min_keyframe_motion: float = 0.04  # mean abs diff (0..1) on a gray thumbnail
    min_translation_m: float = 0.5  # used instead of image motion when poses arrive
    # rebuild scheduling
    rebuild_min_new_keyframes: int = 4
    max_keyframes: int = 512  # hard cap on stored keyframes
    num_frames: int = 24  # frame cap per rebuild (evenly strided over the run)
    # revisit (loop candidate) detection — v1: record + visualize only
    revisit_detection: bool = True
    revisit_min_time_separation_s: float = 30.0
    revisit_min_index_separation: int = 20
    revisit_max_distance: float = 0.04
    # reconstruction quality (draft-leaning: latency beats fidelity here)
    iterations: int = 1500
    align_iters: int = 150
    scene_graph: str = "swin-3"
    splat_max_points: int = 400000
    splat_normalize_extent: float | None = 17.0
    jpeg_quality: int = 92
    device: str = "cuda"
    # backend overrides (None -> pose_free defaults / env vars)
    checkpoint: Path | str | None = None
    dust3r_root: Path | None = None
    mast3r_root: Path | None = None
    vggt_root: Path | None = None
    # optional page copied to live/index.html so one HTTP server serves the demo
    viewer_html: Path | None = None


@dataclass
class Keyframe:
    index: int
    timestamp: float
    path: Path
    position: np.ndarray | None = None


class KeyframeSelector:
    """Time + parallax (or translation) gate deciding which frames to keep."""

    def __init__(
        self,
        *,
        min_gap_s: float = 1.0,
        min_motion: float = 0.04,
        min_translation_m: float = 0.5,
    ) -> None:
        self.min_gap_s = min_gap_s
        self.min_motion = min_motion
        self.min_translation_m = min_translation_m
        self._last_timestamp: float | None = None
        self._last_thumb: np.ndarray | None = None
        self._last_position: np.ndarray | None = None

    @staticmethod
    def _thumbnail(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        thumb = cv2.resize(gray, _MOTION_THUMB_SIZE, interpolation=cv2.INTER_AREA)
        return thumb.astype(np.float32) / 255.0

    def consider(
        self,
        image_bgr: np.ndarray,
        timestamp: float,
        position: np.ndarray | None = None,
    ) -> bool:
        """Return True (and latch state) when the frame should become a keyframe."""
        if self._last_timestamp is None:
            self._accept(image_bgr, timestamp, position)
            return True
        if timestamp - self._last_timestamp < self.min_gap_s:
            return False

        if position is not None and self._last_position is not None:
            moved = float(np.linalg.norm(np.asarray(position, dtype=np.float64) - self._last_position))
            if moved < self.min_translation_m:
                return False
        elif self._last_thumb is not None and self.min_motion > 0:
            motion = float(np.abs(self._thumbnail(image_bgr) - self._last_thumb).mean())
            if motion < self.min_motion:
                return False

        self._accept(image_bgr, timestamp, position)
        return True

    def _accept(self, image_bgr: np.ndarray, timestamp: float, position: np.ndarray | None) -> None:
        self._last_timestamp = timestamp
        self._last_thumb = self._thumbnail(image_bgr)
        self._last_position = None if position is None else np.asarray(position, dtype=np.float64).copy()


@dataclass(frozen=True)
class LoopCandidate:
    """A "temporally distant but visually near" keyframe pair (possible revisit).

    v1 records and visualizes candidates only — no map correction yet. The
    distance is the same gray-thumbnail mean-abs-diff used by keyframe motion
    gating, so it is map-quality independent (original images, not splats).
    """

    query_index: int
    query_timestamp: float
    match_index: int
    match_timestamp: float
    distance: float

    def to_json(self) -> dict:
        return {
            "queryIndex": self.query_index,
            "queryTimestamp": round(self.query_timestamp, 3),
            "matchIndex": self.match_index,
            "matchTimestamp": round(self.match_timestamp, 3),
            "distance": round(self.distance, 5),
        }


class RevisitDetector:
    """Match each new keyframe against past keyframes outside a temporal window.

    Candidates must be separated by both ``min_time_separation_s`` and
    ``min_index_separation`` so adjacent keyframes (which are always similar)
    never count as loops. The best match below ``max_distance`` is kept.
    """

    def __init__(
        self,
        *,
        min_time_separation_s: float = 30.0,
        min_index_separation: int = 20,
        max_distance: float = 0.04,
    ) -> None:
        self.min_time_separation_s = min_time_separation_s
        self.min_index_separation = min_index_separation
        self.max_distance = max_distance
        self._thumbs: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self.candidates: list[LoopCandidate] = []

    def add_keyframe(self, image_bgr: np.ndarray, timestamp: float) -> LoopCandidate | None:
        """Register a keyframe; returns a loop candidate when a revisit is found."""
        thumb = KeyframeSelector._thumbnail(image_bgr)
        index = len(self._thumbs)
        best: LoopCandidate | None = None
        last_eligible = index - self.min_index_separation
        for past_index in range(min(len(self._thumbs), max(last_eligible, 0))):
            if timestamp - self._timestamps[past_index] < self.min_time_separation_s:
                continue
            distance = float(np.abs(thumb - self._thumbs[past_index]).mean())
            if distance > self.max_distance:
                continue
            if best is None or distance < best.distance:
                best = LoopCandidate(
                    query_index=index,
                    query_timestamp=timestamp,
                    match_index=past_index,
                    match_timestamp=self._timestamps[past_index],
                    distance=distance,
                )
        self._thumbs.append(thumb)
        self._timestamps.append(timestamp)
        if best is not None:
            self.candidates.append(best)
        return best


def select_round_frames(keyframes: list[Keyframe], num_frames: int) -> list[Keyframe]:
    """Evenly strided subset covering the whole run (keeps first and latest)."""
    if num_frames <= 0 or len(keyframes) <= num_frames:
        return list(keyframes)
    idx = np.linspace(0, len(keyframes) - 1, num_frames).round().astype(int)
    return [keyframes[i] for i in sorted(set(idx.tolist()))]


class SplatRebuilder:
    """Real backend: pose-free preprocess -> gsplat training -> .splat export.

    Every round is an independent pose-free reconstruction with its own gauge.
    The rebuilder chains each round onto the session gauge (= the first
    round's) via the shared keyframes' COLMAP poses before exporting
    ``scene.splat``, and freezes the viewer normalization on the first round —
    so ``live/latest.splat`` grows cumulatively instead of jumping every
    round. The cumulative transform is persisted as
    ``rounds/round_NNN/gauge_transform.json`` for offline consumers (GIF
    timeline, loop closure).
    """

    def __init__(self, config: LiveMapperConfig) -> None:
        self.config = config
        from gs_sim2real.robotics.gauge_alignment import SessionGaugeChain

        self._gauge_chain = SessionGaugeChain()
        self._normalize_params: tuple[np.ndarray, float] | None = None

    def __call__(self, images_dir: Path, round_dir: Path) -> Path:
        from gs_sim2real.preprocess.pose_free import PoseFreeProcessor
        from gs_sim2real.train.gsplat_trainer import train_gsplat
        from gs_sim2real.viewer.web_export import ply_to_splat

        cfg = self.config
        processor_kwargs: dict = {
            "method": cfg.method,
            "num_frames": 0,  # the session already strided the frames
            "align_iters": cfg.align_iters,
            "scene_graph": cfg.scene_graph,
            "device": cfg.device,
        }
        if cfg.checkpoint:
            processor_kwargs["checkpoint"] = cfg.checkpoint
        if cfg.dust3r_root:
            processor_kwargs["dust3r_root"] = cfg.dust3r_root
        if cfg.mast3r_root:
            processor_kwargs["mast3r_root"] = cfg.mast3r_root
        if cfg.vggt_root:
            processor_kwargs["vggt_root"] = cfg.vggt_root

        sparse_dir = round_dir / "sparse_input"
        PoseFreeProcessor(**processor_kwargs).estimate_poses(images_dir, sparse_dir)
        ply_path = train_gsplat(
            data_dir=sparse_dir,
            output_dir=round_dir / "train",
            num_iterations=cfg.iterations,
        )
        transform, rebased = self._align_round_gauge(round_dir)
        normalize_params = self._session_normalize_params(Path(ply_path), transform, rebased)
        splat_path = round_dir / "scene.splat"
        ply_to_splat(
            ply_path,
            splat_path,
            max_points=cfg.splat_max_points,
            normalize_params=normalize_params,
            min_opacity=0.02,
            max_scale=2.0,
            similarity_transform=transform,
        )
        return splat_path

    def _align_round_gauge(self, round_dir: Path):
        """Chain this round onto the session gauge from its COLMAP poses.

        Alignment uses ``train/point_cloud.ply``'s gauge via ``images.txt`` —
        never the normalized ``scene.splat``. A missing/short pose file
        rebases the chain (one visible jump) rather than exporting a garbage
        alignment.
        """
        from gs_sim2real.robotics.gauge_alignment import (
            RoundPoses,
            SessionGaugeChain,
            identity_sim3,
            write_gauge_transform,
        )

        images_txt = round_dir / "sparse_input" / "sparse" / "0" / "images.txt"
        if not images_txt.is_file():
            logger.warning("gauge chain: %s missing; rebasing the session gauge", images_txt)
            self._gauge_chain = SessionGaugeChain()  # next round becomes the new anchor
            transform = identity_sim3()
            write_gauge_transform(round_dir, transform, rebased=True, shared_cameras=0)
            return transform, True

        poses = RoundPoses.from_images_txt(images_txt)
        transform, rebased, shared = self._gauge_chain.update(poses)
        write_gauge_transform(round_dir, transform, rebased=rebased, shared_cameras=shared)
        if not rebased and shared:
            logger.info(
                "gauge chain: aligned onto session gauge via %d shared cameras (scale %.4f)",
                shared,
                transform[0],
            )
        return transform, rebased

    def _session_normalize_params(self, ply_path: Path, transform, rebased: bool):
        """Freeze the viewer normalization on the session's first (or rebased) round."""
        if self.config.splat_normalize_extent is None or self.config.splat_normalize_extent <= 0:
            return None
        if self._normalize_params is not None and not rebased:
            return self._normalize_params
        from gs_sim2real.robotics.gauge_alignment import apply_to_points
        from gs_sim2real.viewer.web_export import compute_splat_normalization
        from gs_sim2real.viewer.web_viewer import load_ply

        positions = np.asarray(load_ply(str(ply_path)).positions, dtype=np.float64)
        self._normalize_params = compute_splat_normalization(
            apply_to_points(transform, positions),
            float(self.config.splat_normalize_extent),
        )
        return self._normalize_params


@dataclass
class _RoundRecord:
    round_index: int
    keyframes_used: int
    keyframes_total: int
    build_seconds: float
    splat_bytes: int
    finished_unix: float
    error: str | None = None

    def to_json(self) -> dict:
        return {
            "round": self.round_index,
            "keyframesUsed": self.keyframes_used,
            "keyframesTotal": self.keyframes_total,
            "buildSeconds": round(self.build_seconds, 2),
            "splatBytes": self.splat_bytes,
            "finishedUnix": round(self.finished_unix, 3),
            "error": self.error,
        }


class LiveMappingSession:
    """Accumulate keyframes and rebuild ``live/latest.splat`` in the background.

    Layout under ``config.workdir``::

        keyframes/kf_000042.jpg   incoming keyframes
        rounds/round_003/...      per-round artifacts (kept for GIF timelines)
        live/latest.splat         atomically replaced after each round
        live/state.json           round/keyframe counters for viewers
        live/index.html           optional viewer page (config.viewer_html)
    """

    def __init__(self, config: LiveMapperConfig, builder: SplatBuilder | None = None) -> None:
        self.config = config
        self.builder: SplatBuilder = builder if builder is not None else SplatRebuilder(config)
        self.workdir = Path(config.workdir)
        self.keyframes_dir = self.workdir / "keyframes"
        self.rounds_dir = self.workdir / "rounds"
        self.live_dir = self.workdir / "live"
        for directory in (self.keyframes_dir, self.rounds_dir, self.live_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.selector = KeyframeSelector(
            min_gap_s=config.min_keyframe_gap_s,
            min_motion=config.min_keyframe_motion,
            min_translation_m=config.min_translation_m,
        )
        self.revisit_detector: RevisitDetector | None = (
            RevisitDetector(
                min_time_separation_s=config.revisit_min_time_separation_s,
                min_index_separation=config.revisit_min_index_separation,
                max_distance=config.revisit_max_distance,
            )
            if config.revisit_detection
            else None
        )
        self.keyframes: list[Keyframe] = []
        self.rounds: list[_RoundRecord] = []
        self._built_keyframe_count = 0
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

        if config.viewer_html and Path(config.viewer_html).is_file():
            shutil.copy2(config.viewer_html, self.live_dir / "index.html")
        self._write_state(status="idle")

    # ------------------------------------------------------------------ input

    def add_frame(
        self,
        image_bgr: np.ndarray,
        timestamp: float,
        position: np.ndarray | None = None,
    ) -> bool:
        """Feed a camera frame; returns True when it was kept as a keyframe."""
        loop_candidate: LoopCandidate | None = None
        with self._lock:
            if len(self.keyframes) >= self.config.max_keyframes:
                return False
            if not self.selector.consider(image_bgr, timestamp, position):
                return False
            index = len(self.keyframes)
            path = self.keyframes_dir / f"kf_{index:06d}.jpg"
            cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
            self.keyframes.append(Keyframe(index=index, timestamp=timestamp, path=path, position=position))
            if self.revisit_detector is not None:
                loop_candidate = self.revisit_detector.add_keyframe(image_bgr, timestamp)
        if loop_candidate is not None:
            logger.info(
                "loop candidate: kf_%06d revisits kf_%06d (dt=%.1fs, thumbnail dist %.4f)",
                loop_candidate.query_index,
                loop_candidate.match_index,
                loop_candidate.query_timestamp - loop_candidate.match_timestamp,
                loop_candidate.distance,
            )
            self._write_loop_candidates()
        self._wakeup.set()
        return True

    # ----------------------------------------------------------------- worker

    def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._run, name="live-mapping-worker", daemon=True)
        self._worker.start()

    def stop(self, wait: bool = True, timeout: float | None = None) -> None:
        self._stop.set()
        self._wakeup.set()
        if wait and self._worker is not None:
            self._worker.join(timeout=timeout)

    def _pending_new_keyframes(self) -> int:
        with self._lock:
            return len(self.keyframes) - self._built_keyframe_count

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._pending_new_keyframes() < self.config.rebuild_min_new_keyframes:
                self._wakeup.wait(timeout=0.2)
                self._wakeup.clear()
                continue
            self._build_round()
        # Final pass so the last keyframes of a run still land in the map.
        if self._pending_new_keyframes() > 0:
            self._build_round()

    def _build_round(self) -> None:
        with self._lock:
            snapshot = list(self.keyframes)
        if len(snapshot) < 2:
            return
        round_index = len(self.rounds) + 1
        selected = select_round_frames(snapshot, self.config.num_frames)
        round_dir = self.rounds_dir / f"round_{round_index:03d}"
        staged = round_dir / "images"
        staged.mkdir(parents=True, exist_ok=True)
        for keyframe in selected:
            target = staged / keyframe.path.name
            if not target.exists():
                try:
                    os.link(keyframe.path, target)
                except OSError:
                    shutil.copy2(keyframe.path, target)

        self._write_state(status="building", building_round=round_index)
        logger.info("round %d: rebuilding from %d/%d keyframes", round_index, len(selected), len(snapshot))
        start = time.time()
        error: str | None = None
        splat_bytes = 0
        try:
            splat_path = self.builder(staged, round_dir)
            splat_bytes = Path(splat_path).stat().st_size
            self._publish(Path(splat_path))
        except Exception as exc:  # keep mapping on transient backend failures
            logger.exception("round %d failed", round_index)
            error = str(exc)
        elapsed = time.time() - start

        record = _RoundRecord(
            round_index=round_index,
            keyframes_used=len(selected),
            keyframes_total=len(snapshot),
            build_seconds=elapsed,
            splat_bytes=splat_bytes,
            finished_unix=time.time(),
            error=error,
        )
        self.rounds.append(record)
        self._built_keyframe_count = len(snapshot)
        self._write_state(status="idle")
        if error is None:
            logger.info(
                "round %d published: %d keyframes -> %.1f KB in %.1fs",
                round_index,
                len(selected),
                splat_bytes / 1024,
                elapsed,
            )

    # ---------------------------------------------------------------- outputs

    def _publish(self, splat_path: Path) -> None:
        target = self.live_dir / "latest.splat"
        tmp = target.with_suffix(".splat.tmp")
        shutil.copy2(splat_path, tmp)
        os.replace(tmp, target)

    def _write_loop_candidates(self) -> None:
        """Atomically persist loop candidates to ``live/loop_candidates.json``."""
        if self.revisit_detector is None:
            return
        with self._lock:
            candidates = [c.to_json() for c in self.revisit_detector.candidates]
        payload = {"loopCandidates": candidates, "updatedUnix": round(time.time(), 3)}
        target = self.live_dir / "loop_candidates.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, target)

    def _write_state(self, *, status: str, building_round: int | None = None) -> None:
        with self._lock:
            keyframes_total = len(self.keyframes)
            loop_candidates = len(self.revisit_detector.candidates) if self.revisit_detector is not None else 0
        successful = [r for r in self.rounds if r.error is None]
        state = {
            "status": status,
            "buildingRound": building_round,
            "keyframesTotal": keyframes_total,
            "completedRounds": len(self.rounds),
            "lastSuccessfulRound": successful[-1].to_json() if successful else None,
            "rounds": [r.to_json() for r in self.rounds],
            "loopCandidates": loop_candidates,
            "updatedUnix": round(time.time(), 3),
            "splatUrl": "latest.splat",
        }
        target = self.live_dir / "state.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, target)


class _NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:  # noqa: N802 - http.server API
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - http.server API
        logger.debug("http: " + format, *args)


def serve_live_dir(live_dir: Path, port: int) -> ThreadingHTTPServer:
    """Serve the live output directory (viewer + latest.splat) without caching."""
    handler = partial(_NoCacheHandler, directory=str(live_dir))
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=server.serve_forever, name="live-mapping-http", daemon=True)
    thread.start()
    logger.info("live viewer at http://localhost:%d/ (splat: /latest.splat)", port)
    return server
