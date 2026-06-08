from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_large_scale_3dgs_fixture.py"
SPEC = importlib.util.spec_from_file_location("build_large_scale_3dgs_fixture", SCRIPT_PATH)
assert SPEC is not None
fixture_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = fixture_module
SPEC.loader.exec_module(fixture_module)


def _write_splat(path: Path, *, source_index: int, count: int = 6) -> None:
    payload = bytearray(count * fixture_module.SPLAT_RECORD_BYTES)
    for record_index in range(count):
        offset = record_index * fixture_module.SPLAT_RECORD_BYTES
        struct.pack_into(
            "<fff",
            payload,
            offset,
            float(record_index),
            float(source_index) * 0.1,
            float(record_index % 3),
        )
        payload[offset + 24 : offset + 28] = bytes(
            (
                40 + source_index,
                90 + record_index,
                130 + source_index,
                220,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_build_source_placements_expands_beyond_original_source_count() -> None:
    placements = fixture_module.build_source_placements(grid_width=5, grid_height=5)

    assert len(placements) == 25
    assert len({placement.source for placement in placements}) == len(fixture_module.PRODUCTION_SOURCES)
    assert placements[0].grid_x == 0
    assert placements[-1].grid_z == 4


def test_build_fixture_writes_regional_mosaic_report(tmp_path: Path) -> None:
    asset_dir = tmp_path / "assets"
    for source_index, source in enumerate(fixture_module.PRODUCTION_SOURCES):
        _write_splat(asset_dir / source.source, source_index=source_index)

    report = fixture_module.build_fixture(
        asset_dir=asset_dir,
        output=tmp_path / "regional.splat",
        report_path=tmp_path / "regional.report.json",
        grid_width=4,
        grid_height=3,
        max_splats_per_cell=4,
        spacing=25.0,
        seed=7,
    )

    assert report["sourceCount"] == len(fixture_module.PRODUCTION_SOURCES)
    assert report["placementCount"] == 12
    assert report["compositeSplatCount"] == 48
    assert report["grid"]["columns"] == 4
    assert report["grid"]["rows"] == 3
    assert report["grid"]["worldBounds"]["maxX"] - report["grid"]["worldBounds"]["minX"] > 70
    assert report["grid"]["worldBounds"]["maxZ"] - report["grid"]["worldBounds"]["minZ"] > 45
    assert len({placement["sourceIndex"] for placement in report["placements"]}) > 1
    assert json.loads((tmp_path / "regional.report.json").read_text(encoding="utf-8"))["placementCount"] == 12
