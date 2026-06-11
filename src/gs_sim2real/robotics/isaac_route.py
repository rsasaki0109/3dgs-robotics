"""Bake robot route results into an Isaac-readable USD layer.

The route geometry is written in the round gauge: the same raw reconstruction
coordinate frame used by the NuRec USDZ generated from ``train/point_cloud.ply``.
Do not apply the browser splat-frame normalization here; the splat and route
share one gauge frame, so Isaac's stage-level corrections affect both equally.

Caveat: this writer uses usd-core and is verified by reading the stage back
with pxr, not by running Isaac Sim itself. Distances are reconstruction-gauge
camera-height units, not meters unless the source reconstruction is metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_USD_IMPORT_HINT = "usd-core is required to write Isaac route layers. Install it with: pip install usd-core"
_ROUTE_SUFFIXES = {".usda", ".usd", ".usdc"}


def resolve_live_map_session(*args: Any, **kwargs: Any) -> Any:
    """Lazy seam for tests; resolves a live-map session round."""
    from gs_sim2real.robotics.localize import resolve_live_map_session as _resolve_live_map_session

    return _resolve_live_map_session(*args, **kwargs)


def load_mapped_records(*args: Any, **kwargs: Any) -> Any:
    """Lazy seam for tests; loads localized keyframe records."""
    from gs_sim2real.robotics.localize import load_mapped_records as _load_mapped_records

    return _load_mapped_records(*args, **kwargs)


def estimate_ground_frame(*args: Any, **kwargs: Any) -> Any:
    """Lazy seam for tests; estimates the navigation plane frame."""
    from gs_sim2real.robotics.occupancy_grid import estimate_ground_frame as _estimate_ground_frame

    return _estimate_ground_frame(*args, **kwargs)


def load_ply(*args: Any, **kwargs: Any) -> Any:
    """Lazy seam for tests; loads a raw gaussian PLY."""
    from gs_sim2real.viewer.web_viewer import load_ply as _load_ply

    return _load_ply(*args, **kwargs)


def route_geometry(
    session_dir: Path,
    *,
    nav_json: Path | None = None,
    query_json: Path | None = None,
    round_index: int | None = None,
    include_trajectory: bool = True,
) -> dict[str, Any]:
    """Prepare route geometry in round-gauge camera-height units.

    The result intentionally mirrors ``viewer_overlay.build_overlay`` up to
    the point where browser splat-frame normalization would be applied.
    """
    from gs_sim2real.robotics.viewer_overlay import _lift_plane_points

    if nav_json is None and query_json is None and not include_trajectory:
        raise ValueError("nothing to export")

    session_dir = Path(session_dir)
    session = resolve_live_map_session(session_dir, round_index=round_index)
    records = load_mapped_records(session)
    centers = np.asarray([record.center for record in records], dtype=np.float64)

    polylines: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []
    ground_frame: tuple[np.ndarray, float] | None = None

    def grid_basis() -> tuple[np.ndarray, float]:
        nonlocal ground_frame
        if ground_frame is None:
            points = np.asarray(load_ply(session.round.ply_path).positions, dtype=np.float64)
            up, forward, _ground, camera_height = estimate_ground_frame(
                centers,
                [record.qvec for record in records],
                lambda candidate_up: points @ candidate_up,
                ground_percentile=30.0,
            )
            basis = np.stack([forward, np.cross(up, forward), up])
            ground_frame = (basis, float(camera_height))
        return ground_frame

    camera_height = 0.0
    if include_trajectory:
        polylines.append(
            {
                "name": "trajectory",
                "points": centers.tolist(),
                "color": [0.30, 0.78, 0.42],
            }
        )

    if nav_json is not None:
        nav = json.loads(Path(nav_json).read_text(encoding="utf-8"))
        basis, camera_height = grid_basis()
        path = _lift_plane_points(np.asarray(nav["path_vertices"], dtype=np.float64), basis, centers, camera_height)
        polylines.append(
            {
                "name": "planned_path",
                "points": path.tolist(),
                "color": [0.25, 0.62, 0.96],
            }
        )
        goal = _lift_plane_points(np.asarray([nav["goal"]], dtype=np.float64), basis, centers, camera_height)
        markers.append(
            {
                "name": "goal",
                "position": goal[0].tolist(),
                "radius": float(camera_height * 0.5),
                "color": [0.96, 0.55, 0.15],
            }
        )

    if query_json is not None:
        query = json.loads(Path(query_json).read_text(encoding="utf-8"))
        camera_height = max(float(camera_height), float(query.get("camera_height", 0.0)))
        for rank, hit in enumerate(query.get("hits", []), start=1):
            radius = max(float(np.mean(hit["extent"])) * 0.5, float(query.get("camera_height", 0.0)) * 0.25)
            markers.append(
                {
                    "name": f"hit_{rank}",
                    "position": np.asarray(hit["centroid"], dtype=np.float64).tolist(),
                    "radius": float(radius),
                    "color": [0.96, 0.35, 0.25],
                }
            )

    return {
        "session": str(session_dir),
        "round": session.round.round_dir.name,
        "camera_height": float(camera_height),
        "polylines": polylines,
        "markers": markers,
    }


def write_route_layer(
    geometry: dict[str, Any], output_path: Path, *, usdz_reference: str | None = None
) -> dict[str, Any]:
    """Write route geometry into a USD layer readable by usdview and Isaac Sim."""
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as error:
        raise ImportError(_USD_IMPORT_HINT) from error

    output_path = Path(output_path)
    if output_path.suffix not in _ROUTE_SUFFIXES:
        raise ValueError(f"output path must end with one of {sorted(_ROUTE_SUFFIXES)}: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    # Do not call UsdGeom.SetStageUpAxis: the splat and route share the same arbitrary reconstruction gauge frame, and
    # any Isaac up correction should apply to both layers equally.

    if usdz_reference is not None:
        splat = UsdGeom.Xform.Define(stage, "/World/Splat")
        splat.GetPrim().GetReferences().AddReference(usdz_reference)

    UsdGeom.Xform.Define(stage, "/World/Route")
    camera_height = float(geometry.get("camera_height", 0.0))
    width = camera_height * 0.08

    for polyline in geometry.get("polylines", []):
        points = [Gf.Vec3f(*point) for point in polyline["points"]]
        curve = UsdGeom.BasisCurves.Define(stage, f"/World/Route/{polyline['name']}")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr(points)
        widths = curve.CreateWidthsAttr([width] * len(points))
        widths.SetMetadata("interpolation", "vertex")
        curve.CreateDisplayColorAttr([Gf.Vec3f(*polyline["color"])])

    for marker in geometry.get("markers", []):
        sphere = UsdGeom.Sphere.Define(stage, f"/World/Route/{marker['name']}")
        sphere.CreateRadiusAttr(float(marker["radius"]))
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*marker["color"])])
        UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(Gf.Vec3d(*marker["position"]))

    root_layer = stage.GetRootLayer()
    root_layer.customLayerData = {
        "generator": "3dgs-robotics export-isaac-route",
        "frame": "round-gauge (camera-height units, not meters)",
        "session": str(geometry.get("session", "")),
        "round": str(geometry.get("round", "")),
    }
    root_layer.Save()

    return {
        "output": str(output_path),
        "polylines": len(geometry.get("polylines", [])),
        "markers": len(geometry.get("markers", [])),
        "usdz_reference": usdz_reference,
    }
