from __future__ import annotations

import json
import subprocess
from pathlib import Path

from gs_sim2real.train.large_scale_3dgs import (
    LargeScale3DGSCatalogOptions,
    LargeScale3DGSOptions,
    LargeScale3DGSPilotOptions,
    LargeScale3DGSPreflightOptions,
    LargeScale3DGSRouteOptions,
    LargeScale3DGSRunOptions,
    LargeScale3DGSSmokeDataOptions,
    _default_command_runner,
    build_large_scale_3dgs_catalog,
    build_large_scale_3dgs_plan,
    build_large_scale_3dgs_pilot,
    build_large_scale_3dgs_preflight,
    build_large_scale_3dgs_route,
    build_large_scale_3dgs_web_runbook,
    format_large_scale_3dgs_catalog_text,
    format_large_scale_3dgs_pilot_text,
    format_large_scale_3dgs_preflight_text,
    format_large_scale_3dgs_route_text,
    format_large_scale_3dgs_shell,
    format_large_scale_3dgs_smoke_data_text,
    load_colmap_images_text,
    run_large_scale_3dgs_plan,
    write_large_scale_3dgs_catalog,
    write_large_scale_3dgs_plan,
    write_large_scale_3dgs_pilot,
    write_large_scale_3dgs_preflight,
    write_large_scale_3dgs_route,
    write_large_scale_3dgs_smoke_data,
)


def _write_sparse_fixture(root: Path) -> Path:
    sparse = root / "sparse" / "0"
    images = root / "images"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 640 480 400 400 320 240\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 frame_000.jpg\n"
        "\n"
        "2 1 0 0 0 -12 0 0 1 frame_001.jpg\n"
        "\n"
        "3 1 0 0 0 -28 0 0 1 frame_002.jpg\n"
        "\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text(
        "# Point list\n1 2 0 0 255 0 0 0\n2 18 0 0 0 255 0 0\n3 26 0 0 0 0 255 0\n",
        encoding="utf-8",
    )
    for index in range(3):
        (images / f"frame_{index:03d}.jpg").write_bytes(b"jpg")
    return root


def _write_route_sparse_fixture(root: Path, image_count: int = 8) -> Path:
    sparse = root / "sparse" / "0"
    images = root / "images"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 640 480 400 400 320 240\n",
        encoding="utf-8",
    )

    image_lines = ["# Image list"]
    point_lines = ["# Point list"]
    for index in range(image_count):
        center_x = 2.0 + index * 11.0
        image_name = f"route_{index:03d}.jpg"
        image_lines.append(f"{index + 1} 1 0 0 0 {-center_x:.6f} 0 0 1 {image_name}")
        image_lines.append("")
        point_lines.append(f"{index + 1} {center_x:.6f} 0 0 255 0 0 0")
        (images / image_name).write_bytes(b"jpg")

    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (sparse / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")
    return root


def test_load_colmap_images_text_computes_camera_centers(tmp_path: Path) -> None:
    data_dir = _write_sparse_fixture(tmp_path / "data")

    records = load_colmap_images_text(data_dir / "sparse" / "0" / "images.txt")

    assert [record.image_id for record in records] == [1, 2, 3]
    assert [round(record.center[0], 3) for record in records] == [0.0, 12.0, 28.0]
    assert records[1].name == "frame_001.jpg"


def test_build_large_scale_3dgs_plan_tiles_by_camera_center(tmp_path: Path) -> None:
    data_dir = _write_sparse_fixture(tmp_path / "data")
    output_dir = tmp_path / "large"

    plan = build_large_scale_3dgs_plan(
        LargeScale3DGSOptions(
            data_dir=data_dir,
            output_dir=output_dir,
            tile_size=20,
            overlap=10,
            axes="xy",
            min_images=1,
            iterations=123,
            config=None,
            materialize=False,
        )
    )

    assert plan["type"] == "large-scale-3dgs-plan"
    assert plan["summary"]["registeredImageCount"] == 3
    assert plan["summary"]["chunkCount"] == 2
    assert plan["summary"]["readyChunkCount"] == 2
    assert [chunk["coreImageCount"] for chunk in plan["chunks"]] == [2, 1]
    assert [chunk["imageCount"] for chunk in plan["chunks"]] == [3, 2]
    assert plan["chunks"][0]["pointCount"] == 3
    assert "--config" not in plan["chunks"][0]["trainCommand"]
    assert "--iterations 123" in plan["chunks"][0]["trainCommand"]


def test_write_large_scale_3dgs_smoke_data_feeds_multi_tile_planner(tmp_path: Path) -> None:
    data_dir = tmp_path / "smoke_data"

    manifest = write_large_scale_3dgs_smoke_data(
        LargeScale3DGSSmokeDataOptions(
            output_dir=data_dir,
            axes="xz",
            grid_width=3,
            grid_height=2,
            tile_size=8,
            images_per_tile=2,
            points_per_tile=5,
            image_size=16,
        )
    )
    plan = build_large_scale_3dgs_plan(
        LargeScale3DGSOptions(
            data_dir=data_dir,
            output_dir=tmp_path / "large",
            axes="xz",
            tile_size=8,
            overlap=0,
            min_images=2,
            materialize=True,
            link_mode="copy",
        )
    )
    text = format_large_scale_3dgs_smoke_data_text(manifest)

    assert manifest["summary"] == {
        "tileCount": 6,
        "imageCount": 12,
        "points3DCount": 30,
        "imagesPerTile": 2,
        "pointsPerTile": 5,
        "imageSize": 16,
    }
    assert (data_dir / "large_scale_3dgs_smoke_data.json").exists()
    assert (data_dir / "images" / "tile_x000_z000_view000.ppm").read_bytes().startswith(b"P6\n")
    assert plan["summary"]["chunkCount"] == 6
    assert plan["summary"]["readyChunkCount"] == 6
    assert {chunk["coreImageCount"] for chunk in plan["chunks"]} == {2}
    assert {chunk["pointCount"] for chunk in plan["chunks"]} == {5}
    assert (Path(plan["chunks"][0]["dataDir"]) / "images").exists()
    assert "next plan: gs-mapper large-scale-3dgs-plan" in text
    assert "--axes xz" in text


def test_build_large_scale_3dgs_preflight_recommends_tile_size_and_next_commands(tmp_path: Path) -> None:
    data_dir = _write_sparse_fixture(tmp_path / "data")
    output_dir = tmp_path / "large"

    report = build_large_scale_3dgs_preflight(
        LargeScale3DGSPreflightOptions(
            data_dir=data_dir,
            output_dir=output_dir,
            axes="xy",
            tile_sizes=(10.0, 20.0),
            overlap=0,
            min_images=1,
            target_images_per_chunk=2,
            iterations=77,
            config=None,
        )
    )
    report_path = write_large_scale_3dgs_preflight(report, output_dir)
    text = format_large_scale_3dgs_preflight_text(report, report_path)

    assert report["type"] == "large-scale-3dgs-preflight"
    assert report["summary"]["registeredImageCount"] == 3
    assert report["summary"]["points3DCount"] == 3
    assert report["summary"]["sourceImageBytes"] == 9
    assert report["recommendation"]["tileSize"] == 20.0
    assert [candidate["recommended"] for candidate in report["candidates"]] == [False, True]
    assert "--tile-size 20.0" in report["next"]["planCommand"]
    assert "--iterations 77" in report["next"]["planCommand"]
    assert "--config" not in report["next"]["planCommand"]
    assert str(output_dir / "large_scale_3dgs_plan.json") in report["next"]["runCommand"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["recommendation"]["tileSize"] == 20.0
    assert "Large-scale 3DGS preflight" in text
    assert "recommended: tile_size=20.0" in text
    assert "next plan: gs-mapper large-scale-3dgs-plan" in text


def test_build_large_scale_3dgs_preflight_can_write_recommended_plan(tmp_path: Path) -> None:
    data_dir = _write_sparse_fixture(tmp_path / "data")
    output_dir = tmp_path / "large"

    report = build_large_scale_3dgs_preflight(
        LargeScale3DGSPreflightOptions(
            data_dir=data_dir,
            output_dir=output_dir,
            axes="xy",
            tile_sizes=(10.0, 20.0),
            overlap=0,
            min_images=1,
            target_images_per_chunk=2,
            write_plan=True,
            link_mode="copy",
        )
    )
    text = format_large_scale_3dgs_preflight_text(report)
    plan_path = Path(report["next"]["planPath"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert report["next"]["planWritten"] is True
    assert plan_path == output_dir / "large_scale_3dgs_plan.json"
    assert plan["tiling"]["tileSize"] == 20.0
    assert plan["materialized"] is True
    assert Path(plan["chunks"][0]["dataDir"], "images", "frame_000.jpg").read_bytes() == b"jpg"
    assert f"plan: {plan_path}" in text


def test_build_large_scale_3dgs_pilot_selects_route_contiguous_ready_chunks(tmp_path: Path) -> None:
    data_dir = _write_route_sparse_fixture(tmp_path / "route_data")
    output_dir = tmp_path / "pilot"

    report = build_large_scale_3dgs_pilot(
        LargeScale3DGSPilotOptions(
            data_dir=data_dir,
            output_dir=output_dir,
            axes="xy",
            tile_size=10,
            overlap=1,
            min_images=1,
            pilot_chunks=3,
            route_start_image=2,
            target_images_per_chunk=1,
            iterations=11,
            config=None,
            link_mode="copy",
        )
    )
    report_path, plan_path = write_large_scale_3dgs_pilot(report, output_dir)
    text = format_large_scale_3dgs_pilot_text(report, report_path, plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert report["type"] == "large-scale-3dgs-pilot"
    assert report["summary"]["selectedChunkCount"] == 3
    assert report["selection"]["selectedChunkIds"] == ["tile_x002_y000", "tile_x003_y000", "tile_x004_y000"]
    assert plan["type"] == "large-scale-3dgs-plan"
    assert plan["summary"]["chunkCount"] == 3
    assert plan["summary"]["sourceChunkCount"] == 8
    assert [chunk["id"] for chunk in plan["chunks"]] == report["selection"]["selectedChunkIds"]
    assert "--iterations 11" in plan["chunks"][0]["trainCommand"]
    assert "--config" not in plan["chunks"][0]["trainCommand"]
    assert "--route-start-image 2" in report["next"]["shellCommand"]
    assert "--tile-size 10" in report["next"]["shellCommand"]
    assert (output_dir / "chunks" / "tile_x002_y000" / "images" / "route_002.jpg").read_bytes() == b"jpg"
    assert not (output_dir / "chunks" / "tile_x001_y000").exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["selection"]["routeStartImage"] == 2
    assert "Real continuous 3DGS pilot" in text
    assert "next run: gs-mapper large-scale-3dgs-run" in text


def test_build_large_scale_3dgs_plan_materializes_chunk_sparse_and_images(tmp_path: Path) -> None:
    data_dir = _write_sparse_fixture(tmp_path / "data")
    output_dir = tmp_path / "large"

    plan = build_large_scale_3dgs_plan(
        LargeScale3DGSOptions(
            data_dir=data_dir,
            output_dir=output_dir,
            tile_size=20,
            overlap=0,
            axes="xy",
            min_images=1,
            materialize=True,
            link_mode="copy",
        )
    )

    first_chunk = plan["chunks"][0]
    chunk_dir = Path(first_chunk["dataDir"])
    images_txt = (chunk_dir / "sparse" / "0" / "images.txt").read_text(encoding="utf-8")
    points_txt = (chunk_dir / "sparse" / "0" / "points3D.txt").read_text(encoding="utf-8")

    assert "frame_000.jpg" in images_txt
    assert "frame_001.jpg" in images_txt
    assert "frame_002.jpg" not in images_txt
    assert "1 2 0 0" in points_txt
    assert (chunk_dir / "images" / "frame_000.jpg").read_bytes() == b"jpg"


def test_format_large_scale_3dgs_shell_skips_underfilled_chunks(tmp_path: Path) -> None:
    data_dir = _write_sparse_fixture(tmp_path / "data")
    plan = build_large_scale_3dgs_plan(
        LargeScale3DGSOptions(
            data_dir=data_dir,
            output_dir=tmp_path / "large",
            tile_size=20,
            overlap=0,
            axes="xy",
            min_images=2,
        )
    )

    shell = format_large_scale_3dgs_shell(plan)

    assert "gs-mapper train" in shell
    assert "# skip tile_x001_y000: too-few-images" in shell


def test_run_large_scale_3dgs_plan_dry_run_writes_report(tmp_path: Path) -> None:
    data_dir = _write_sparse_fixture(tmp_path / "data")
    output_dir = tmp_path / "large"
    plan = build_large_scale_3dgs_plan(
        LargeScale3DGSOptions(
            data_dir=data_dir,
            output_dir=output_dir,
            tile_size=20,
            overlap=0,
            axes="xy",
            min_images=1,
        )
    )
    plan_path = write_large_scale_3dgs_plan(plan, output_dir)

    report = run_large_scale_3dgs_plan(LargeScale3DGSRunOptions(plan_path=plan_path, dry_run=True, max_chunks=1))

    assert report["summary"]["selectedChunkCount"] == 1
    assert report["summary"]["plannedCount"] == 1
    assert Path(report["reportPath"]).exists()
    assert report["chunks"][0]["status"] == "planned"


def test_run_large_scale_3dgs_plan_executes_train_and_export_with_fake_runner(tmp_path: Path) -> None:
    data_dir = _write_sparse_fixture(tmp_path / "data")
    output_dir = tmp_path / "large"
    plan = build_large_scale_3dgs_plan(
        LargeScale3DGSOptions(
            data_dir=data_dir,
            output_dir=output_dir,
            tile_size=20,
            overlap=0,
            axes="xy",
            min_images=1,
        )
    )
    plan_path = write_large_scale_3dgs_plan(plan, output_dir)
    seen: list[list[str]] = []

    def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
        seen.append(args)
        if args[1] == "train":
            output_dir_arg = Path(args[args.index("--output") + 1])
            output_dir_arg.mkdir(parents=True, exist_ok=True)
            (output_dir_arg / "point_cloud.ply").write_bytes(b"ply")
        if args[1] == "export":
            output_file = Path(args[args.index("--output") + 1])
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(b"splat")
        return subprocess.CompletedProcess(args, 0)

    report = run_large_scale_3dgs_plan(
        LargeScale3DGSRunOptions(plan_path=plan_path, max_chunks=1),
        command_runner=fake_runner,
    )

    assert report["summary"]["doneCount"] == 1
    assert [args[1] for args in seen] == ["train", "export"]
    assert Path(report["chunks"][0]["splatOutput"]).read_bytes() == b"splat"


def test_run_large_scale_3dgs_plan_resume_skips_existing_splat(tmp_path: Path) -> None:
    data_dir = _write_sparse_fixture(tmp_path / "data")
    output_dir = tmp_path / "large"
    plan = build_large_scale_3dgs_plan(
        LargeScale3DGSOptions(
            data_dir=data_dir,
            output_dir=output_dir,
            tile_size=20,
            overlap=0,
            axes="xy",
            min_images=1,
        )
    )
    first_splat = Path(plan["chunks"][0]["splatOutput"])
    first_splat.parent.mkdir(parents=True, exist_ok=True)
    first_splat.write_bytes(b"existing")
    plan_path = write_large_scale_3dgs_plan(plan, output_dir)

    def fail_runner(args: list[str]) -> subprocess.CompletedProcess:
        raise AssertionError(f"runner should not be called: {args}")

    report = run_large_scale_3dgs_plan(
        LargeScale3DGSRunOptions(plan_path=plan_path, max_chunks=1),
        command_runner=fail_runner,
    )

    assert report["summary"]["skippedCount"] == 1
    assert report["chunks"][0]["reason"] == "splat-exists"


def test_default_command_runner_falls_back_to_python_module_for_checkout(monkeypatch) -> None:
    from gs_sim2real.train import large_scale_3dgs as module

    calls: list[list[str]] = []

    def fake_run(args, check=False):
        del check
        command = list(args)
        calls.append(command)
        if command[0] == "gs-mapper":
            raise FileNotFoundError(command[0])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = _default_command_runner(["gs-mapper", "train", "--help"])

    assert result.returncode == 0
    assert calls[0] == ["gs-mapper", "train", "--help"]
    assert calls[1][:3] == [module.sys.executable, "-m", "gs_sim2real.cli"]
    assert calls[1][3:] == ["train", "--help"]


def test_build_large_scale_3dgs_catalog_copies_ready_splats_to_public_root(tmp_path: Path) -> None:
    data_dir = _write_sparse_fixture(tmp_path / "data")
    output_dir = tmp_path / "large"
    plan = build_large_scale_3dgs_plan(
        LargeScale3DGSOptions(
            data_dir=data_dir,
            output_dir=output_dir,
            tile_size=20,
            overlap=0,
            axes="xy",
            min_images=1,
        )
    )
    first_chunk = plan["chunks"][0]
    first_splat = Path(first_chunk["splatOutput"])
    first_splat.parent.mkdir(parents=True, exist_ok=True)
    first_splat.write_bytes(b"splat")
    plan_path = write_large_scale_3dgs_plan(plan, output_dir)
    run_report_path = output_dir / "large_scale_3dgs_run_report.json"
    run_report_path.write_text(
        json.dumps(
            {
                "type": "large-scale-3dgs-run-report",
                "chunks": [{"id": first_chunk["id"], "status": "done"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    options = LargeScale3DGSCatalogOptions(
        plan_path=plan_path,
        output_path=tmp_path / "catalog.json",
        run_report_path=run_report_path,
        scene_id="Demo Scene 01",
        label="Demo Scene",
        public_root=tmp_path / "web-public",
        public_url_prefix="/assets/splats",
        link_mode="copy",
    )
    catalog = build_large_scale_3dgs_catalog(options)
    catalog_path = write_large_scale_3dgs_catalog(catalog, options)

    assert catalog["type"] == "large-scale-3dgs-tile-catalog"
    assert catalog["sceneId"] == "demo-scene-01"
    assert catalog["summary"] == {
        "tileCount": 2,
        "readyTileCount": 1,
        "missingSplatTileCount": 1,
    }
    assert catalog_path == tmp_path / "catalog.json"
    assert json.loads(catalog_path.read_text(encoding="utf-8"))["sceneId"] == "demo-scene-01"
    assert catalog["tiles"][0]["runStatus"] == "done"
    assert catalog["tiles"][0]["status"] == "ready"
    assert catalog["tiles"][0]["splatUrl"] == "/assets/splats/demo-scene-01/tile_x000_y000.splat"
    assert Path(catalog["tiles"][0]["publicPath"]).read_bytes() == b"splat"
    assert catalog["tiles"][1]["status"] == "missing-splat"


def test_build_large_scale_3dgs_catalog_can_require_existing_splats(tmp_path: Path) -> None:
    data_dir = _write_sparse_fixture(tmp_path / "data")
    output_dir = tmp_path / "large"
    plan = build_large_scale_3dgs_plan(
        LargeScale3DGSOptions(
            data_dir=data_dir,
            output_dir=output_dir,
            tile_size=20,
            overlap=0,
            axes="xy",
            min_images=1,
        )
    )
    first_splat = Path(plan["chunks"][0]["splatOutput"])
    first_splat.parent.mkdir(parents=True, exist_ok=True)
    first_splat.write_bytes(b"splat")
    plan_path = write_large_scale_3dgs_plan(plan, output_dir)

    catalog = build_large_scale_3dgs_catalog(
        LargeScale3DGSCatalogOptions(
            plan_path=plan_path,
            public_root=tmp_path / "web-public",
            link_mode="copy",
            require_splats=True,
        )
    )

    assert catalog["summary"]["tileCount"] == 1
    assert [tile["id"] for tile in catalog["tiles"]] == ["tile_x000_y000"]


def _write_grid_catalog_fixture(catalog_path: Path) -> None:
    tiles = []
    for tile_x, tile_z in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]:
        tile_id = f"tile_x{tile_x:03d}_z{tile_z:03d}"
        min_x = tile_x * 10.0
        min_z = tile_z * 10.0
        tiles.append(
            {
                "id": tile_id,
                "label": tile_id,
                "status": "ready",
                "runStatus": "done",
                "splatUrl": f"/splats/demo-grid/{tile_id}.splat",
                "coreBounds": {
                    "minX": min_x,
                    "maxX": min_x + 10.0,
                    "minZ": min_z,
                    "maxZ": min_z + 10.0,
                },
                "expandedBounds": {
                    "minX": min_x,
                    "maxX": min_x + 10.0,
                    "minZ": min_z,
                    "maxZ": min_z + 10.0,
                },
                "tileIndex": {
                    "x": tile_x,
                    "z": tile_z,
                },
                "axes": "xz",
            }
        )

    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        json.dumps(
            {
                "version": 1,
                "type": "large-scale-3dgs-tile-catalog",
                "sceneId": "demo-grid",
                "label": "Demo Grid",
                "tiling": {
                    "axes": "xz",
                    "tileSize": 10,
                    "overlap": 0,
                },
                "summary": {
                    "tileCount": len(tiles),
                    "readyTileCount": len(tiles),
                    "missingSplatTileCount": 0,
                },
                "tiles": tiles,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_large_scale_3dgs_route_generates_spiral_robot_route(tmp_path: Path) -> None:
    public_root = tmp_path / "apps" / "dreamwalker-web" / "public"
    catalog_path = public_root / "manifests" / "demo-grid-catalog.json"
    _write_grid_catalog_fixture(catalog_path)
    options = LargeScale3DGSRouteOptions(catalog_path=catalog_path, order="spiral")

    route = build_large_scale_3dgs_route(options)
    route_path = write_large_scale_3dgs_route(route, options)
    text = format_large_scale_3dgs_route_text(route, route_path)

    assert route_path == public_root / "robot-routes" / "demo-grid-route.json"
    assert route["protocol"] == "dreamwalker-robot-route/v1"
    assert route["label"] == "Demo Grid Route"
    assert route["route"] == [
        [5.0, 0.0, 5.0],
        [5.0, 0.0, 15.0],
        [15.0, 0.0, 15.0],
        [25.0, 0.0, 15.0],
        [25.0, 0.0, 5.0],
        [15.0, 0.0, 5.0],
    ]
    assert route["tileSequence"] == [
        "tile_x000_z000",
        "tile_x000_z001",
        "tile_x001_z001",
        "tile_x002_z001",
        "tile_x002_z000",
        "tile_x001_z000",
    ]
    assert route["pose"] == {
        "position": [15.0, 0.0, 5.0],
        "yawDegrees": 90,
    }
    assert route["sourceCatalog"]["order"] == "spiral"
    assert json.loads(route_path.read_text(encoding="utf-8"))["tileSequence"] == route["tileSequence"]
    assert "points: 6" in text
    assert "tile_x000_z000 -> tile_x000_z001" in text


def test_build_large_scale_3dgs_web_runbook_links_catalog_to_dreamwalker(tmp_path: Path) -> None:
    public_root = tmp_path / "apps" / "dreamwalker-web" / "public"
    catalog_path = public_root / "manifests" / "demo-catalog.json"
    options = LargeScale3DGSCatalogOptions(
        plan_path=tmp_path / "large_scale_3dgs_plan.json",
        output_path=catalog_path,
        public_root=public_root,
        web_app_dir=tmp_path / "apps" / "dreamwalker-web",
        site_url="/dreamwalker?fragment=residency",
        tile_preload="cache",
    )

    runbook = build_large_scale_3dgs_web_runbook(catalog_path, options)

    assert runbook["catalogUrl"] == "/manifests/demo-catalog.json"
    assert "validate:dynamic-map-catalog" in runbook["validateCommand"]
    assert "--public-root" in runbook["validateCommand"]
    assert str(public_root) in runbook["validateCommand"]
    assert (
        runbook["launchUrl"]
        == "/dreamwalker?fragment=residency&tileCatalog=%2Fmanifests%2Fdemo-catalog.json&tilePreload=cache"
    )


def test_build_large_scale_3dgs_web_runbook_includes_robot_route_playback(tmp_path: Path) -> None:
    public_root = tmp_path / "apps" / "dreamwalker-web" / "public"
    catalog_path = public_root / "manifests" / "demo-catalog.json"
    route_path = public_root / "robot-routes" / "demo-route.json"
    options = LargeScale3DGSCatalogOptions(
        plan_path=tmp_path / "large_scale_3dgs_plan.json",
        output_path=catalog_path,
        public_root=public_root,
        web_app_dir=tmp_path / "apps" / "dreamwalker-web",
        site_url="/dreamwalker?fragment=residency",
        tile_preload="cache",
        route_path=route_path,
        route_playback=True,
        route_playback_ms=800,
        route_playback_loop=True,
    )

    runbook = build_large_scale_3dgs_web_runbook(catalog_path, options)

    assert runbook["routeUrl"] == "/robot-routes/demo-route.json"
    assert f"--route {route_path}" in runbook["validateCommand"]
    assert "--route-playback 1" in runbook["validateCommand"]
    assert "--route-playback-ms 800" in runbook["validateCommand"]
    assert "--route-playback-loop 1" in runbook["validateCommand"]
    assert (
        runbook["launchUrl"] == "/dreamwalker?fragment=residency"
        "&tileCatalog=%2Fmanifests%2Fdemo-catalog.json"
        "&tilePreload=cache"
        "&robotRoute=%2Frobot-routes%2Fdemo-route.json"
        "&robotRoutePlayback=1"
        "&robotRoutePlaybackMs=800"
        "&robotRoutePlaybackLoop=1"
    )


def test_format_large_scale_3dgs_catalog_text_includes_web_follow_up(tmp_path: Path) -> None:
    public_root = tmp_path / "web-public"
    catalog_path = public_root / "manifests" / "demo-catalog.json"
    catalog = {
        "sceneId": "demo-scene",
        "label": "Demo Scene",
        "summary": {
            "readyTileCount": 2,
            "tileCount": 3,
        },
    }
    options = LargeScale3DGSCatalogOptions(
        plan_path=tmp_path / "large_scale_3dgs_plan.json",
        output_path=catalog_path,
        public_root=public_root,
        web_app_dir=Path("apps/dreamwalker-web"),
        site_url="http://localhost:5173/",
        tile_preload="metadata",
    )

    text = format_large_scale_3dgs_catalog_text(catalog, catalog_path, options)

    assert "validate: npm --prefix apps/dreamwalker-web run validate:dynamic-map-catalog" in text
    assert "--preload-mode metadata" in text
    assert "launch: http://localhost:5173/?tileCatalog=%2Fmanifests%2Fdemo-catalog.json&tilePreload=metadata" in text
