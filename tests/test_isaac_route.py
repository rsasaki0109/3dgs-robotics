"""Tests for Isaac route USD layer export."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gs_sim2real.robotics import isaac_route


def test_route_geometry_uses_round_gauge_and_lifts_nav_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_dir = tmp_path / "session"
    round_dir = session_dir / "rounds" / "round_001"
    ply_path = round_dir / "train" / "point_cloud.ply"
    ply_path.parent.mkdir(parents=True)

    centers = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    records = [SimpleNamespace(center=center, qvec=np.array([1.0, 0.0, 0.0, 0.0])) for center in centers]
    session = SimpleNamespace(round=SimpleNamespace(round_dir=round_dir, ply_path=ply_path))

    nav_json = tmp_path / "nav_result.json"
    nav_json.write_text(
        json.dumps(
            {
                "path_vertices": [[0.0, 0.0], [1.0, 0.0]],
                "goal": [2.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    query_json = tmp_path / "query.json"
    query_json.write_text(
        json.dumps(
            {
                "prompt": "chair",
                "camera_height": 2.0,
                "hits": [
                    {
                        "centroid": [4.0, 5.0, 6.0],
                        "extent": [0.4, 0.6, 0.8],
                        "mean_score": 0.7,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(isaac_route, "resolve_live_map_session", lambda *args, **kwargs: session)
    monkeypatch.setattr(isaac_route, "load_mapped_records", lambda *args, **kwargs: records)
    monkeypatch.setattr(isaac_route, "load_ply", lambda *args, **kwargs: SimpleNamespace(positions=centers))
    monkeypatch.setattr(
        isaac_route,
        "estimate_ground_frame",
        lambda *args, **kwargs: (
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            0.0,
            2.0,
        ),
    )

    geometry = isaac_route.route_geometry(session_dir, nav_json=nav_json, query_json=query_json)

    assert geometry["session"] == str(session_dir)
    assert geometry["round"] == "round_001"
    assert geometry["camera_height"] == 2.0
    assert [polyline["name"] for polyline in geometry["polylines"]] == ["trajectory", "planned_path"]
    assert [marker["name"] for marker in geometry["markers"]] == ["goal", "hit_1"]
    assert geometry["polylines"][0]["points"] == centers.tolist()
    planned_path = np.asarray(geometry["polylines"][1]["points"], dtype=np.float64)
    assert planned_path.shape == (2, 3)
    assert planned_path[0, 2] == pytest.approx(0.3)
    assert planned_path[1, 2] == pytest.approx(1.3)
    assert geometry["markers"][1]["position"] == [4.0, 5.0, 6.0]
    assert geometry["markers"][1]["radius"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="nothing to export"):
        isaac_route.route_geometry(session_dir, include_trajectory=False)


def test_write_route_layer_round_trips_usd(tmp_path: Path) -> None:
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    geometry = {
        "session": "session",
        "round": "round_001",
        "camera_height": 2.0,
        "polylines": [
            {
                "name": "trajectory",
                "points": [[0.0, 0.0, 1.0], [1.0, 0.0, 1.5]],
                "color": [0.30, 0.78, 0.42],
            }
        ],
        "markers": [
            {
                "name": "goal",
                "position": [2.0, 0.0, 1.0],
                "radius": 1.0,
                "color": [0.96, 0.55, 0.15],
            }
        ],
    }

    output = tmp_path / "route.usda"
    summary = isaac_route.write_route_layer(geometry, output, usdz_reference="scene.usdz")
    assert pxr
    assert summary == {
        "output": str(output),
        "polylines": 1,
        "markers": 1,
        "usdz_reference": "scene.usdz",
    }

    stage = Usd.Stage.Open(str(output))
    assert stage.GetDefaultPrim().GetPath().pathString == "/World"
    curve = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Route/trajectory"))
    assert curve
    assert curve.GetCurveVertexCountsAttr().Get() == [2]
    sphere = UsdGeom.Sphere(stage.GetPrimAtPath("/World/Route/goal"))
    assert sphere
    assert sphere.GetRadiusAttr().Get() == pytest.approx(1.0)
    assert stage.GetRootLayer().customLayerData["frame"] == "round-gauge (camera-height units, not meters)"

    splat = stage.GetPrimAtPath("/World/Splat")
    assert splat.HasAuthoredReferences()


def test_write_route_layer_rejects_bad_suffix(tmp_path: Path) -> None:
    geometry = {"camera_height": 1.0, "polylines": [], "markers": []}
    with pytest.raises(ValueError, match="output path must end"):
        isaac_route.write_route_layer(geometry, tmp_path / "route.json")
