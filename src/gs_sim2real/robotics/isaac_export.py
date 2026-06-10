"""Export trained 3DGS maps to Isaac Sim (Omniverse NuRec USDZ).

Isaac Sim 5.0+ renders 3D Gaussian Splatting scenes natively through NuRec
volume prims (a ``UsdVol::Volume`` extension). NVIDIA's open-source
`3dgrut <https://github.com/nv-tlabs/3dgrut>`_ ships the official converter
(``threedgrut.export.scripts.transcode``); this module wraps it so a
live-mapping session round (or any standard 3DGS PLY) becomes a drag-and-drop
USDZ for Isaac Sim.

Only the transcode path of 3dgrut is used — it needs no CUDA build, just a
clone plus a few pip packages (see ``_CLONE_HINT``; torch/numpy already come
with this package).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

THREEDGRUT_ENV_VAR = "THREEDGRUT_ROOT"
TRANSCODE_MODULE = "threedgrut.export.scripts.transcode"
EXPORT_FORMATS = ("nurec", "lightfield")
_CLONE_HINT = (
    "3dgrut is required for USDZ export. Clone it and point --threedgrut-root "
    f"(or ${THREEDGRUT_ENV_VAR}) at the clone:\n"
    "  git clone https://github.com/nv-tlabs/3dgrut.git\n"
    '  pip install plyfile msgpack usd-core "nvidia-ncore>=19.0.0" simplejpeg tensorboard'
)


def find_threedgrut_root(explicit: Path | None = None) -> Path | None:
    """Locate a 3dgrut checkout: explicit flag > env var > importable package."""
    candidates = [explicit, os.environ.get(THREEDGRUT_ENV_VAR)]
    for candidate in candidates:
        if candidate is None:
            continue
        root = Path(candidate).expanduser()
        if (root / "threedgrut" / "export").is_dir():
            return root
        raise FileNotFoundError(f"not a 3dgrut checkout (threedgrut/export missing): {root}")
    spec = importlib.util.find_spec("threedgrut")
    if spec is not None and spec.origin is not None:
        return Path(spec.origin).resolve().parent.parent
    return None


def build_transcode_command(
    ply_path: Path,
    output_path: Path,
    *,
    export_format: str = "nurec",
    python_executable: str | None = None,
) -> list[str]:
    """Command line for 3dgrut's PLY -> USDZ transcode script."""
    if export_format not in EXPORT_FORMATS:
        raise ValueError(f"unknown export format {export_format!r}; choose from {EXPORT_FORMATS}")
    return [
        python_executable or sys.executable,
        "-m",
        TRANSCODE_MODULE,
        str(ply_path),
        "-o",
        str(output_path),
        "--format",
        export_format,
    ]


def export_usdz(
    ply_path: Path,
    output_path: Path,
    *,
    threedgrut_root: Path | None = None,
    export_format: str = "nurec",
    python_executable: str | None = None,
) -> Path:
    """Convert a standard 3DGS PLY into an Isaac Sim-ready USDZ via 3dgrut."""
    # resolve before building the command: the converter runs with cwd=3dgrut
    ply_path = Path(ply_path).resolve()
    output_path = Path(output_path).resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(f"input PLY not found: {ply_path}")

    root = find_threedgrut_root(threedgrut_root)
    if root is None:
        raise RuntimeError(_CLONE_HINT)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_transcode_command(
        ply_path, output_path, export_format=export_format, python_executable=python_executable
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    logger.info("running 3dgrut transcode: %s", " ".join(command))
    result = subprocess.run(command, env=env, cwd=str(root), capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout or "").strip().splitlines()[-15:])
        raise RuntimeError(f"3dgrut transcode failed (exit {result.returncode}):\n{tail}")
    if not output_path.is_file():
        raise RuntimeError(f"3dgrut transcode reported success but wrote no file: {output_path}")
    return output_path
