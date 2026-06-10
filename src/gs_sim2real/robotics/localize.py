"""Localize a camera image against a trained live-mapping 3DGS round.

Stage 1 retrieves the nearest mapped keyframe thumbnail as a pose seed.
Stage 2 refines the COLMAP view matrix with differentiable gsplat rendering
and a photometric L1 + SSIM loss (MonoGS / iComMa pattern).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from gs_sim2real.train.gsplat_trainer import GaussianModel, GsplatTrainer
from gs_sim2real.train.large_scale_3dgs import ColmapImageRecord, load_colmap_images_text
from gs_sim2real.viewer.web_viewer import load_ply

logger = logging.getLogger(__name__)

_KEYFRAME_RE = re.compile(r"kf_(\d+)")
_THUMB_SIZE = (64, 64)
_C0 = 0.28209479177387814


@dataclass(frozen=True)
class LiveMapRound:
    """One successful live-mapping rebuild round."""

    round_index: int
    round_dir: Path
    ply_path: Path
    sparse_dir: Path
    images_txt: Path
    cameras_txt: Path


@dataclass(frozen=True)
class LiveMapSession:
    """Resolved session pointing at the map gauge used for localization."""

    session_dir: Path
    keyframes_dir: Path
    round: LiveMapRound


@dataclass
class LocalizeConfig:
    """Tuning knobs for retrieval + photometric refinement."""

    device: str = "cuda"
    refine_iters: int = 80
    refine_lr: float = 0.005
    lambda_dssim: float = 0.2
    pyramid_scales: tuple[float, ...] = (0.25, 0.5, 1.0)
    thumb_size: tuple[int, int] = _THUMB_SIZE


@dataclass
class LocalizeResult:
    """Pose estimate for one query image."""

    query_path: Path
    seed_keyframe: str
    seed_distance: float
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    center: np.ndarray
    refine_loss: float
    refine_iterations: int
    gt_center: np.ndarray | None = None
    translation_error: float | None = None
    relative_translation_error: float | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "query": str(self.query_path),
            "seed_keyframe": self.seed_keyframe,
            "seed_distance": self.seed_distance,
            "qvec": list(self.qvec),
            "tvec": list(self.tvec),
            "center": self.center.tolist(),
            "refine_loss": self.refine_loss,
            "refine_iterations": self.refine_iterations,
        }
        if self.gt_center is not None:
            payload["gt_center"] = self.gt_center.tolist()
        if self.translation_error is not None:
            payload["translation_error"] = self.translation_error
        if self.relative_translation_error is not None:
            payload["relative_translation_error"] = self.relative_translation_error
        return payload


@dataclass
class LocalizeSummary:
    """Batch localization output."""

    session_dir: Path
    round_index: int
    results: list[LocalizeResult] = field(default_factory=list)
    median_neighbor_spacing: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "session_dir": str(self.session_dir),
            "round_index": self.round_index,
            "median_neighbor_spacing": self.median_neighbor_spacing,
            "results": [result.to_json() for result in self.results],
        }


def keyframe_index(name: str) -> int:
    """Parse ``kf_000042.jpg`` -> 42."""
    match = _KEYFRAME_RE.search(name)
    if match is None:
        raise ValueError(f"not a live-mapping keyframe name: {name}")
    return int(match.group(1))


def resolve_live_map_session(session_dir: Path, *, round_index: int | None = None) -> LiveMapSession:
    """Resolve the trained round used for localization (defaults to last success)."""
    session_dir = Path(session_dir)
    keyframes_dir = session_dir / "keyframes"
    if not keyframes_dir.is_dir():
        raise FileNotFoundError(f"missing keyframes directory: {keyframes_dir}")

    if round_index is None:
        state_path = session_dir / "live" / "state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            last = state.get("lastSuccessfulRound") or {}
            round_index = int(last.get("round") or state.get("completedRounds") or 0)
        if not round_index:
            round_dirs = sorted(session_dir.glob("rounds/round_*"))
            if not round_dirs:
                raise FileNotFoundError(f"no rounds under {session_dir / 'rounds'}")
            round_index = int(re.search(r"round_(\d+)", round_dirs[-1].name).group(1))  # type: ignore[union-attr]

    round_dir = session_dir / "rounds" / f"round_{round_index:03d}"
    ply_path = round_dir / "train" / "point_cloud.ply"
    sparse_dir = round_dir / "sparse_input"
    images_txt = sparse_dir / "sparse" / "0" / "images.txt"
    cameras_txt = sparse_dir / "sparse" / "0" / "cameras.txt"
    for path in (ply_path, images_txt, cameras_txt):
        if not path.is_file():
            raise FileNotFoundError(f"missing map artifact for round {round_index}: {path}")

    return LiveMapSession(
        session_dir=session_dir,
        keyframes_dir=keyframes_dir,
        round=LiveMapRound(
            round_index=round_index,
            round_dir=round_dir,
            ply_path=ply_path,
            sparse_dir=sparse_dir,
            images_txt=images_txt,
            cameras_txt=cameras_txt,
        ),
    )


def load_mapped_records(session: LiveMapSession) -> list[ColmapImageRecord]:
    """Load COLMAP image records for the localized round."""
    return load_colmap_images_text(session.round.images_txt)


def mapped_records_by_name(records: Sequence[ColmapImageRecord]) -> dict[str, ColmapImageRecord]:
    return {record.name: record for record in records}


def mapped_records_by_index(records: Sequence[ColmapImageRecord]) -> dict[int, ColmapImageRecord]:
    return {keyframe_index(record.name): record for record in records}


def list_non_round_keyframes(session: LiveMapSession, records: Sequence[ColmapImageRecord] | None = None) -> list[Path]:
    """Keyframes stored on disk but absent from the round's ``images.txt``."""
    records = records if records is not None else load_mapped_records(session)
    mapped = {record.name for record in records}
    paths = sorted(session.keyframes_dir.glob("*.jpg"))
    return [path for path in paths if path.name not in mapped]


def compute_thumbnail(image_bgr: np.ndarray, size: tuple[int, int] = _THUMB_SIZE) -> np.ndarray:
    """Gray thumbnail in [0, 1], matching live-mapping keyframe gating."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    thumb = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return thumb.astype(np.float32) / 255.0


def retrieve_seed_keyframe(
    query_bgr: np.ndarray,
    *,
    mapped_records: Sequence[ColmapImageRecord],
    keyframes_dir: Path,
    thumb_size: tuple[int, int] = _THUMB_SIZE,
) -> tuple[str, float]:
    """Return the mapped keyframe name with the closest thumbnail."""
    query_thumb = compute_thumbnail(query_bgr, thumb_size)
    best_name = ""
    best_distance = float("inf")
    for record in mapped_records:
        path = keyframes_dir / record.name
        if not path.is_file():
            continue
        candidate = cv2.imread(str(path))
        if candidate is None:
            continue
        distance = float(np.abs(query_thumb - compute_thumbnail(candidate, thumb_size)).mean())
        if distance < best_distance:
            best_distance = distance
            best_name = record.name
    if not best_name:
        raise RuntimeError("no mapped keyframe thumbnails found for retrieval")
    return best_name, best_distance


def _slerp_quat(
    q0: Sequence[float], q1: Sequence[float], t: float
) -> tuple[float, float, float, float]:
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        out /= np.linalg.norm(out).clip(1e-12)
        return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    w0 = np.sin((1.0 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    out = w0 * a + w1 * b
    return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))


def interpolate_gt_pose(
    query_name: str,
    mapped_by_index: dict[int, ColmapImageRecord],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float], np.ndarray] | None:
    """Interpolate GT qvec/tvec/center between surrounding mapped keyframes."""
    if not mapped_by_index:
        return None
    query_idx = keyframe_index(query_name)
    indices = sorted(mapped_by_index)
    if query_idx <= indices[0]:
        record = mapped_by_index[indices[0]]
        return record.qvec, record.tvec, np.asarray(record.center, dtype=np.float64)
    if query_idx >= indices[-1]:
        record = mapped_by_index[indices[-1]]
        return record.qvec, record.tvec, np.asarray(record.center, dtype=np.float64)

    prev_idx = max(i for i in indices if i <= query_idx)
    next_idx = min(i for i in indices if i >= query_idx)
    if prev_idx == next_idx:
        record = mapped_by_index[prev_idx]
        return record.qvec, record.tvec, np.asarray(record.center, dtype=np.float64)

    prev = mapped_by_index[prev_idx]
    nxt = mapped_by_index[next_idx]
    t = (query_idx - prev_idx) / max(next_idx - prev_idx, 1)
    center = (1.0 - t) * np.asarray(prev.center) + t * np.asarray(nxt.center)
    qvec = _slerp_quat(prev.qvec, nxt.qvec, t)
    tvec = tuple((1.0 - t) * np.asarray(prev.tvec) + t * np.asarray(nxt.tvec))
    return qvec, tvec, center.astype(np.float64)


def median_neighbor_spacing(centers: np.ndarray) -> float:
    """Median distance between consecutive mapped camera centers."""
    if len(centers) < 2:
        return 1.0
    diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return float(np.median(diffs))


def load_gaussian_model_from_ply(ply_path: Path, device: str) -> GaussianModel:
    """Load a trained gsplat PLY into a frozen GaussianModel."""
    import torch

    ply = load_ply(ply_path)
    if not ply.is_gaussian_splat or ply.scales is None or ply.rotations is None or ply.opacities is None:
        raise ValueError(f"PLY is not a gsplat export: {ply_path}")

    num = len(ply.positions)
    model = GaussianModel(num_gaussians=num)
    model.means = torch.tensor(ply.positions, dtype=torch.float32, device=device)
    model.scales = torch.tensor(ply.scales, dtype=torch.float32, device=device)
    model.rotations = torch.tensor(ply.rotations, dtype=torch.float32, device=device)
    model.opacities = torch.tensor(ply.opacities.reshape(-1, 1), dtype=torch.float32, device=device)
    model.sh_coeffs = torch.zeros(num, 1, 3, dtype=torch.float32, device=device)
    dc = (np.asarray(ply.colors, dtype=np.float32) - 0.5) / _C0
    model.sh_coeffs[:, 0, :] = torch.tensor(dc, dtype=torch.float32, device=device)
    for tensor in (model.means, model.scales, model.rotations, model.opacities, model.sh_coeffs):
        tensor.requires_grad_(False)
    return model


def _load_cameras_txt(path: Path) -> dict[int, dict[str, Any]]:
    trainer = GsplatTrainer(config={"num_iterations": 1})
    return trainer._load_cameras_txt(path)


def _load_query_rgb(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"could not read query image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _scale_intrinsics(k: Any, scale: float) -> Any:
    import torch

    scaled = k.clone()
    scaled[0, 0] *= scale
    scaled[1, 1] *= scale
    scaled[0, 2] = (scaled[0, 2] + 0.5) * scale - 0.5
    scaled[1, 2] = (scaled[1, 2] + 0.5) * scale - 0.5
    return scaled


def _apply_se3_delta(viewmat: Any, so3: Any, tvec_delta: Any) -> Any:
    import torch

    theta = torch.linalg.norm(so3) + 1e-8
    axis = so3 / theta
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    ax, ay, az = axis[0], axis[1], axis[2]
    k_skew = torch.stack(
        [
            torch.stack([torch.zeros_like(ax), -az, ay]),
            torch.stack([az, torch.zeros_like(ax), -ax]),
            torch.stack([-ay, ax, torch.zeros_like(ax)]),
        ]
    )
    r_delta = torch.eye(3, device=viewmat.device) + sin_t * k_skew + (1 - cos_t) * (k_skew @ k_skew)
    t_delta = torch.eye(4, device=viewmat.device, dtype=viewmat.dtype)
    t_delta[:3, :3] = r_delta
    t_delta[:3, 3] = tvec_delta
    return t_delta @ viewmat


def refine_pose_photometric(
    gaussians: GaussianModel,
    gt_rgb: np.ndarray,
    viewmat_init: np.ndarray,
    intrinsic: np.ndarray,
    *,
    config: LocalizeConfig,
) -> tuple[np.ndarray, float, int]:
    """Optimize a small SE(3) delta against frozen gaussians."""
    import torch

    trainer = GsplatTrainer(config={"num_iterations": 1, "lambda_dssim": config.lambda_dssim})
    if not trainer._has_gsplat:
        raise RuntimeError("gsplat is required for photometric localization refinement")

    device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
    gaussians = gaussians  # frozen
    viewmat_base = torch.tensor(viewmat_init, dtype=torch.float32, device=device)
    k_base = torch.tensor(intrinsic, dtype=torch.float32, device=device)
    gt_full = torch.tensor(gt_rgb, dtype=torch.float32, device=device)

    so3 = torch.nn.Parameter(torch.zeros(3, device=device, dtype=torch.float32))
    t_delta = torch.nn.Parameter(torch.zeros(3, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([so3, t_delta], lr=config.refine_lr)

    total_iters = 0
    final_loss = float("inf")
    for scale in config.pyramid_scales:
        h = max(8, int(round(gt_full.shape[0] * scale)))
        w = max(8, int(round(gt_full.shape[1] * scale)))
        gt = (
            torch.nn.functional.interpolate(
                gt_full.permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )
        k = _scale_intrinsics(k_base, scale)
        for _ in range(config.refine_iters):
            viewmat = _apply_se3_delta(viewmat_base, so3, t_delta)
            rendered = trainer._render_gsplat(gaussians, viewmat, k, h, w, device)
            l1_loss = torch.abs(rendered - gt).mean()
            ssim_loss = 1.0 - trainer._simple_ssim(rendered, gt)
            loss = (1.0 - config.lambda_dssim) * l1_loss + config.lambda_dssim * ssim_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            total_iters += 1

    viewmat_final = _apply_se3_delta(viewmat_base, so3, t_delta).detach().cpu().numpy()
    return viewmat_final, final_loss, total_iters


def viewmat_to_colmap(viewmat: np.ndarray) -> tuple[tuple[float, float, float, float], tuple[float, float, float], np.ndarray]:
    """Convert a 4x4 world-to-camera matrix to COLMAP qvec/tvec + center."""
    rotation = viewmat[:3, :3]
    tvec = viewmat[:3, 3]
    # rotation matrix -> quaternion (w, x, y, z)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    qvec = (float(qw), float(qx), float(qy), float(qz))
    tvec_tuple = (float(tvec[0]), float(tvec[1]), float(tvec[2]))
    center = -(rotation.T @ tvec)
    return qvec, tvec_tuple, center.astype(np.float64)


def localize_query(
    session: LiveMapSession,
    query_path: Path,
    *,
    gaussians: GaussianModel | None = None,
    mapped_records: Sequence[ColmapImageRecord] | None = None,
    cameras: dict[int, dict[str, Any]] | None = None,
    config: LocalizeConfig | None = None,
    gt_name: str | None = None,
    neighbor_spacing: float | None = None,
) -> LocalizeResult:
    """Localize one query image against the session's final round map."""
    config = config or LocalizeConfig()
    mapped_records = list(mapped_records if mapped_records is not None else load_mapped_records(session))
    mapped_by_name = mapped_records_by_name(mapped_records)
    mapped_by_idx = mapped_records_by_index(mapped_records)
    cameras = cameras if cameras is not None else _load_cameras_txt(session.round.cameras_txt)

    query_bgr = cv2.imread(str(query_path))
    if query_bgr is None:
        raise FileNotFoundError(f"could not read query image: {query_path}")

    seed_name, seed_distance = retrieve_seed_keyframe(
        query_bgr,
        mapped_records=mapped_records,
        keyframes_dir=session.keyframes_dir,
        thumb_size=config.thumb_size,
    )
    seed = mapped_by_name[seed_name]
    trainer = GsplatTrainer(config={"num_iterations": 1})
    device = config.device if trainer._has_gsplat else "cpu"
    viewmat_init = trainer._quat_tvec_to_viewmat(list(seed.qvec), list(seed.tvec), device)
    if hasattr(viewmat_init, "detach"):
        viewmat_init_np = viewmat_init.detach().cpu().numpy()
    else:
        viewmat_init_np = np.asarray(viewmat_init, dtype=np.float64)

    cam = cameras[seed.camera_id]
    k = trainer._make_intrinsic_matrix(cam, device)
    intrinsic = k.detach().cpu().numpy() if hasattr(k, "detach") else np.asarray(k, dtype=np.float64)

    gt_rgb = _load_query_rgb(query_path)
    if gt_rgb.shape[0] != cam["height"] or gt_rgb.shape[1] != cam["width"]:
        gt_rgb = cv2.resize(gt_rgb, (cam["width"], cam["height"]), interpolation=cv2.INTER_AREA)

    if gaussians is None:
        gaussians = load_gaussian_model_from_ply(session.round.ply_path, device)

    viewmat_final, refine_loss, refine_iters = refine_pose_photometric(
        gaussians,
        gt_rgb,
        viewmat_init_np,
        intrinsic,
        config=config,
    )
    qvec, tvec, center = viewmat_to_colmap(viewmat_final)

    eval_name = gt_name or query_path.name
    gt_pose = interpolate_gt_pose(eval_name, mapped_by_idx)
    gt_center = gt_pose[2] if gt_pose is not None else None
    translation_error = None
    relative_translation_error = None
    if gt_center is not None:
        translation_error = float(np.linalg.norm(center - gt_center))
        spacing = neighbor_spacing if neighbor_spacing is not None else median_neighbor_spacing(
            np.asarray([record.center for record in mapped_records], dtype=np.float64)
        )
        relative_translation_error = translation_error / max(spacing, 1e-8)

    return LocalizeResult(
        query_path=Path(query_path),
        seed_keyframe=seed_name,
        seed_distance=seed_distance,
        qvec=qvec,
        tvec=tvec,
        center=center,
        refine_loss=refine_loss,
        refine_iterations=refine_iters,
        gt_center=gt_center,
        translation_error=translation_error,
        relative_translation_error=relative_translation_error,
    )


def localize_queries(
    session_dir: Path,
    query_paths: Sequence[Path],
    *,
    round_index: int | None = None,
    config: LocalizeConfig | None = None,
) -> LocalizeSummary:
    """Localize many query images, loading the map once."""
    config = config or LocalizeConfig()
    session = resolve_live_map_session(session_dir, round_index=round_index)
    mapped_records = load_mapped_records(session)
    centers = np.asarray([record.center for record in mapped_records], dtype=np.float64)
    spacing = median_neighbor_spacing(centers)
    gaussians = load_gaussian_model_from_ply(session.round.ply_path, config.device)
    cameras = _load_cameras_txt(session.round.cameras_txt)

    results: list[LocalizeResult] = []
    for query_path in query_paths:
        logger.info("localizing %s", query_path)
        results.append(
            localize_query(
                session,
                Path(query_path),
                gaussians=gaussians,
                mapped_records=mapped_records,
                cameras=cameras,
                config=config,
                neighbor_spacing=spacing,
            )
        )
    return LocalizeSummary(
        session_dir=session.session_dir,
        round_index=session.round.round_index,
        results=results,
        median_neighbor_spacing=spacing,
    )


def write_localize_summary(summary: LocalizeSummary, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary.to_json(), indent=2) + "\n", encoding="utf-8")
