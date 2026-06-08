"""Stage a trained PLY into the Dynamic Map Viewer public directory.

Copies the PLY file and updates the asset manifest so the Dynamic Map Viewer
loads it as the splat source for the target fragment.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

DREAMWALKER_WEB = Path(__file__).resolve().parents[3] / "apps" / "dreamwalker-web"
MANIFEST_REL = Path("public") / "manifests" / "dreamwalker-live.assets.json"
SPLATS_REL = Path("public") / "splats"


def stage_ply(
    ply_path: str | Path,
    fragment: str = "residency",
    dreamwalker_root: str | Path | None = None,
) -> dict[str, str]:
    """Copy *ply_path* into Dynamic Map Viewer public/splats/ and update the manifest.

    Returns a dict with ``splat_dest``, ``manifest``, and ``launch_url``.
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    root = Path(dreamwalker_root) if dreamwalker_root else DREAMWALKER_WEB
    splats_dir = root / SPLATS_REL
    splats_dir.mkdir(parents=True, exist_ok=True)

    dest_name = f"{fragment}-main.ply"
    dest = splats_dir / dest_name
    shutil.copy2(ply_path, dest)

    manifest_path = root / MANIFEST_REL
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"version": 1, "label": "Local Dynamic Map Asset Manifest", "fragments": {}}

    if fragment not in manifest.get("fragments", {}):
        manifest.setdefault("fragments", {})[fragment] = {
            "label": f"{fragment.title()} (auto-staged)",
        }

    manifest["fragments"][fragment]["splatUrl"] = f"/splats/{dest_name}"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    launch_url = f"http://localhost:5173/?fragment={fragment}"

    return {
        "splat_dest": str(dest),
        "manifest": str(manifest_path),
        "launch_url": launch_url,
    }
