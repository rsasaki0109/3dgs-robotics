"""Sim3 gauge alignment between live-mapping rebuild rounds.

Each live-mapping round is an independent pose-free reconstruction, so every
round lives in its own gauge (scale / rotation / translation). Consecutive
rounds share keyframes; Kabsch over the shared cameras' orientations plus the
center variance ratio gives a similarity transform, and chaining those steps
expresses every round in one session gauge. Two shared cameras suffice because
the rotation comes from camera orientations, not centers.

Used at runtime by :class:`gs_sim2real.robotics.live_mapping.SplatRebuilder`
(each round is aligned onto the session gauge before ``live/latest.splat`` is
exported) and offline by ``scripts/build_live_mapping_gif.py``.

Alignment always uses the round's COLMAP poses (``images.txt``) together with
the full-precision ``train/point_cloud.ply`` gauge — never ``scene.splat``,
whose coordinates are normalized for the browser viewer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# (scale, rotation, translation): x_dst = scale * rotation @ x_src + translation
Sim3 = tuple[float, np.ndarray, np.ndarray]

MIN_SHARED_CAMERAS = 2

GAUGE_TRANSFORM_FILENAME = "gauge_transform.json"


def identity_sim3() -> Sim3:
    return 1.0, np.eye(3), np.zeros(3)


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_to_quat(rotation: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (w, x, y, z) quaternion."""
    w = np.sqrt(max(0.0, 1.0 + rotation[0, 0] + rotation[1, 1] + rotation[2, 2])) / 2.0
    if w < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    x = (rotation[2, 1] - rotation[1, 2]) / (4 * w)
    y = (rotation[0, 2] - rotation[2, 0]) / (4 * w)
    z = (rotation[1, 0] - rotation[0, 1]) / (4 * w)
    return np.array([w, x, y, z])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (..., 4) (w, x, y, z) quaternion arrays."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def parse_images_txt(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """COLMAP image lines -> (names, camera centers, world-from-camera rotations)."""
    names: list[str] = []
    centers: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # image lines have 10 fields ending in a filename; POINTS2D lines are all-numeric
        if len(parts) != 10 or parts[9].replace(".", "").isdigit():
            continue
        qw, qx, qy, qz, tx, ty, tz = (float(v) for v in parts[1:8])
        r_cw = quat_to_rotation(np.array([qw, qx, qy, qz]))
        t = np.array([tx, ty, tz])
        rotations.append(r_cw.T)
        centers.append(-r_cw.T @ t)
        names.append(parts[9])
    return names, np.asarray(centers), np.asarray(rotations)


def similarity_from_poses(
    src_centers: np.ndarray,
    src_rotations: np.ndarray,
    dst_centers: np.ndarray,
    dst_rotations: np.ndarray,
) -> Sim3:
    """Similarity (s, R, t) with dst ~= s * R @ src + t.

    Rotation comes from the shared cameras' orientations (Kabsch over
    R_dst @ R_src^T), so two shared cameras are enough — centers alone would
    leave the rotation about their connecting axis unconstrained.
    """
    acc = np.zeros((3, 3))
    for r_src, r_dst in zip(src_rotations, dst_rotations):
        acc += r_dst @ r_src.T
    u, _d, vt = np.linalg.svd(acc)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt

    mu_src, mu_dst = src_centers.mean(axis=0), dst_centers.mean(axis=0)
    var_src = ((src_centers - mu_src) ** 2).sum()
    var_dst = ((dst_centers - mu_dst) ** 2).sum()
    scale = float(np.sqrt(var_dst / max(var_src, 1e-12)))
    translation = mu_dst - scale * rotation @ mu_src
    return scale, rotation, translation


def compose(second: Sim3, first: Sim3) -> Sim3:
    """Composition second∘first as a similarity transform."""
    s1, r1, t1 = first
    s2, r2, t2 = second
    return s2 * s1, r2 @ r1, s2 * r2 @ t1 + t2


def invert(transform: Sim3) -> Sim3:
    """Inverse similarity: maps dst back to src."""
    s, r, t = transform
    s_inv = 1.0 / s
    r_inv = r.T
    return s_inv, r_inv, -s_inv * r_inv @ t


def apply_to_points(transform: Sim3, points: np.ndarray) -> np.ndarray:
    s, r, t = transform
    return np.asarray(points, dtype=np.float64) @ r.T * s + t


def shared_camera_indices(src_names: list[str], dst_names: list[str]) -> tuple[list[int], list[int]]:
    """Index pairs of cameras present in both rounds (matched by keyframe filename)."""
    dst_index = {name: i for i, name in enumerate(dst_names)}
    pairs = [(i, dst_index[name]) for i, name in enumerate(src_names) if name in dst_index]
    return [i for i, _ in pairs], [j for _, j in pairs]


@dataclass
class RoundPoses:
    """One round's COLMAP poses (this round's own gauge)."""

    names: list[str]
    centers: np.ndarray  # (N, 3) camera centers
    rotations: np.ndarray  # (N, 3, 3) world-from-camera

    @classmethod
    def from_images_txt(cls, path: Path) -> RoundPoses:
        names, centers, rotations = parse_images_txt(path)
        return cls(names=names, centers=centers, rotations=rotations)


def transform_to_json(transform: Sim3, *, rebased: bool, shared_cameras: int, optimized: bool = False) -> dict:
    scale, rotation, translation = transform
    return {
        "scale": float(scale),
        "rotation": np.asarray(rotation, dtype=np.float64).tolist(),
        "translation": np.asarray(translation, dtype=np.float64).tolist(),
        "rebased": bool(rebased),
        "sharedCameras": int(shared_cameras),
        "optimized": bool(optimized),
    }


def transform_from_json(data: dict) -> Sim3:
    return (
        float(data["scale"]),
        np.asarray(data["rotation"], dtype=np.float64),
        np.asarray(data["translation"], dtype=np.float64),
    )


def write_gauge_transform(
    round_dir: Path, transform: Sim3, *, rebased: bool, shared_cameras: int, optimized: bool = False
) -> Path:
    """Persist a round's cumulative round-gauge -> session-gauge transform."""
    path = Path(round_dir) / GAUGE_TRANSFORM_FILENAME
    payload = transform_to_json(transform, rebased=rebased, shared_cameras=shared_cameras, optimized=optimized)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_gauge_transform(round_dir: Path) -> tuple[Sim3, bool] | None:
    """Load (transform, rebased) written by :func:`write_gauge_transform`, or None."""
    path = Path(round_dir) / GAUGE_TRANSFORM_FILENAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return transform_from_json(data), bool(data.get("rebased", False))


class SessionGaugeChain:
    """Chains each round's gauge onto the session gauge (= first round's gauge).

    Feed every successful round's poses in order; :meth:`update` returns the
    cumulative Sim3 mapping that round into the session gauge. When a round
    shares fewer than ``MIN_SHARED_CAMERAS`` keyframes with the previous one,
    the chain rebases: that round becomes the new session anchor (the map
    jumps once instead of receiving a garbage alignment).
    """

    def __init__(self) -> None:
        self._prev_poses: RoundPoses | None = None
        self._cumulative: Sim3 = identity_sim3()

    def update(self, poses: RoundPoses) -> tuple[Sim3, bool, int]:
        """Returns (cumulative transform, rebased, shared camera count)."""
        if self._prev_poses is None:
            self._prev_poses = poses
            self._cumulative = identity_sim3()
            return self._cumulative, False, 0

        src_ids, dst_ids = shared_camera_indices(poses.names, self._prev_poses.names)
        shared = len(src_ids)
        if shared < MIN_SHARED_CAMERAS:
            logger.warning(
                "gauge chain: only %d shared cameras with the previous round; rebasing the session gauge",
                shared,
            )
            self._prev_poses = poses
            self._cumulative = identity_sim3()
            return self._cumulative, True, shared

        step = similarity_from_poses(
            poses.centers[src_ids],
            poses.rotations[src_ids],
            self._prev_poses.centers[dst_ids],
            self._prev_poses.rotations[dst_ids],
        )
        self._cumulative = compose(self._cumulative, step)
        self._prev_poses = poses
        return self._cumulative, False, shared

    def set_cumulative(self, transform: Sim3) -> None:
        """Replace the latest cumulative transform (e.g. after pose-graph refinement)."""
        self._cumulative = transform
