"""Round-level Sim3 pose graph: globally consistent session-gauge transforms.

The session gauge chain (gauge_alignment.py) aligns each round only to its
predecessor, so alignment error compounds over a long session. Because every
rebuild round re-strides the *whole* keyframe history, temporally distant
rounds usually share keyframes as well — including across revisited places —
and every such pair yields a direct relative Sim3 measurement. This module
builds that graph (nodes = rounds, edges = shared-keyframe similarities
weighted by the shared count) and refines the per-round session-gauge
transforms with a damped Gauss-Newton on a 7-parameter Sim3 chart
(log-scale, rotation vector, translation). Pure numpy: sessions have tens of
rounds, so the solve takes milliseconds and no new dependency is needed.

Limitation (documented, not hidden): this corrects *between-round* gauge
drift. Drift inside a single round's pose-free reconstruction (e.g. a swin
pair graph that never connects the two passes of a loop) is out of scope
here and belongs to the per-round backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from gs_sim2real.robotics.gauge_alignment import (
    RoundPoses,
    Sim3,
    compose,
    invert,
    quat_to_rotation,
    rotation_to_quat,
    shared_camera_indices,
    similarity_from_poses,
)

logger = logging.getLogger(__name__)

MIN_EDGE_SHARED_CAMERAS = 2


@dataclass(frozen=True)
class Sim3Edge:
    """Relative Sim3 measurement between two rounds: x_dst_gauge = transform(x_src_gauge)."""

    src: int
    dst: int
    transform: Sim3
    weight: float  # shared camera count


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    """3x3 rotation -> rotation vector (axis * angle)."""
    w, x, y, z = rotation_to_quat(rotation)
    vec = np.array([x, y, z])
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * float(np.arctan2(norm, w))
    return vec / norm * angle


def rotation_exp(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector -> 3x3 rotation."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    axis = np.asarray(rotvec, dtype=np.float64) / angle
    half = 0.5 * angle
    quat = np.array([np.cos(half), *(np.sin(half) * axis)])
    return quat_to_rotation(quat)


def sim3_to_params(transform: Sim3) -> np.ndarray:
    """Sim3 -> 7-vector chart [log scale, rotvec (3), translation (3)]."""
    scale, rotation, translation = transform
    return np.concatenate([[np.log(scale)], rotation_log(rotation), np.asarray(translation, dtype=np.float64)])


def sim3_from_params(params: np.ndarray) -> Sim3:
    """Inverse of :func:`sim3_to_params`."""
    return float(np.exp(params[0])), rotation_exp(params[1:4]), np.asarray(params[4:7], dtype=np.float64)


def build_round_edges(rounds: list[RoundPoses], *, min_shared: int = MIN_EDGE_SHARED_CAMERAS) -> list[Sim3Edge]:
    """Relative Sim3 edges between every round pair sharing enough keyframes."""
    edges: list[Sim3Edge] = []
    for i in range(len(rounds)):
        for j in range(i + 1, len(rounds)):
            src_ids, dst_ids = shared_camera_indices(rounds[i].names, rounds[j].names)
            if len(src_ids) < min_shared:
                continue
            transform = similarity_from_poses(
                rounds[i].centers[src_ids],
                rounds[i].rotations[src_ids],
                rounds[j].centers[dst_ids],
                rounds[j].rotations[dst_ids],
            )
            edges.append(Sim3Edge(src=i, dst=j, transform=transform, weight=float(len(src_ids))))
    return edges


def _edge_residual(transforms: list[Sim3], edge: Sim3Edge) -> np.ndarray:
    # constraint: X_src = X_dst ∘ T_dst<-src, so this error transform is identity at optimum
    error = compose(invert(transforms[edge.src]), compose(transforms[edge.dst], edge.transform))
    return np.sqrt(edge.weight) * sim3_to_params(error)


def optimize_session_transforms(
    initial: list[Sim3],
    edges: list[Sim3Edge],
    *,
    max_iterations: int = 25,
    damping: float = 1e-4,
    epsilon: float = 1e-6,
) -> list[Sim3]:
    """Refine round->session transforms; node 0 stays fixed as the session anchor.

    Damped Gauss-Newton with a forward-difference Jacobian over the Sim3
    chart. Problem sizes here are tiny (7 x (rounds - 1) parameters), so the
    numerical Jacobian is the simplest correct tool.
    """
    n = len(initial)
    if n < 2 or not edges:
        return list(initial)

    def unpack(x: np.ndarray) -> list[Sim3]:
        return [initial[0]] + [sim3_from_params(x[7 * k : 7 * k + 7]) for k in range(n - 1)]

    def residuals(x: np.ndarray) -> np.ndarray:
        transforms = unpack(x)
        return np.concatenate([_edge_residual(transforms, edge) for edge in edges])

    x = np.concatenate([sim3_to_params(t) for t in initial[1:]])
    previous_cost = float("inf")
    for iteration in range(max_iterations):
        r0 = residuals(x)
        cost = float(r0 @ r0)
        if not np.isfinite(cost):
            logger.warning("gauge pose graph: non-finite cost at iteration %d; keeping initial transforms", iteration)
            return list(initial)
        if previous_cost - cost < 1e-12 * max(previous_cost, 1.0) and iteration > 0:
            break
        previous_cost = cost

        jacobian = np.empty((r0.size, x.size))
        for column in range(x.size):
            probe = x.copy()
            probe[column] += epsilon
            jacobian[:, column] = (residuals(probe) - r0) / epsilon
        hessian = jacobian.T @ jacobian
        hessian[np.diag_indices_from(hessian)] += damping
        try:
            step = np.linalg.solve(hessian, -jacobian.T @ r0)
        except np.linalg.LinAlgError:
            logger.warning("gauge pose graph: singular normal equations; keeping initial transforms")
            return list(initial)
        x = x + step

    final = unpack(x)
    final_cost = float(np.sum(residuals(x) ** 2))
    if final_cost > previous_cost and final_cost > 0:
        # never return something worse than what the chain produced
        initial_cost = float(np.sum(np.concatenate([_edge_residual(list(initial), edge) for edge in edges]) ** 2))
        if final_cost > initial_cost:
            logger.warning("gauge pose graph: optimization diverged; keeping initial transforms")
            return list(initial)
    logger.info(
        "gauge pose graph: %d rounds, %d edges, residual %.3e -> %.3e",
        n,
        len(edges),
        float(np.sum(np.concatenate([_edge_residual(list(initial), edge) for edge in edges]) ** 2)),
        final_cost,
    )
    return final
