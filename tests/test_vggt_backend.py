"""Tests for the VGGT feedforward backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_images(image_dir: Path, count: int = 3) -> None:
    import cv2

    image_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        frame = np.full((48, 64, 3), fill_value=index * 40, dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"frame_{index:03d}.png"), frame)


class TestVggtBackend:
    def test_run_vggt_inference_requires_clone(self, tmp_path: Path) -> None:
        from gs_sim2real.preprocess.vggt_backend import run_vggt_inference

        image_dir = tmp_path / "images"
        _write_images(image_dir)
        frames = sorted(image_dir.glob("*.png"))

        with pytest.raises(FileNotFoundError, match="VGGT clone not found"):
            run_vggt_inference(
                frames,
                tmp_path / "out",
                vggt_root=tmp_path / "missing_vggt",
                device="cpu",
            )

    def test_pose_free_processor_routes_vggt(self, tmp_path: Path, monkeypatch) -> None:
        from gs_sim2real.preprocess.pose_free import PoseFreeProcessor

        image_dir = tmp_path / "images"
        _write_images(image_dir)
        output_dir = tmp_path / "sparse_out"
        calls: list[tuple] = []

        def fake_run_vggt_inference(image_paths, out_dir, **kwargs):
            calls.append((list(image_paths), out_dir, kwargs))
            sparse = out_dir / "sparse" / "0"
            sparse.mkdir(parents=True, exist_ok=True)
            (sparse / "cameras.txt").write_text("# stub\n", encoding="utf-8")
            (sparse / "images.txt").write_text("# stub\n", encoding="utf-8")
            (sparse / "points3D.txt").write_text("# stub\n", encoding="utf-8")
            return sparse

        monkeypatch.setattr(
            "gs_sim2real.preprocess.vggt_backend.run_vggt_inference",
            fake_run_vggt_inference,
        )

        processor = PoseFreeProcessor(method="vggt", num_frames=2, device="cpu")
        sparse_dir = processor.estimate_poses(image_dir, output_dir)

        assert len(calls) == 1
        assert len(calls[0][0]) == 2
        assert calls[0][2]["device"] == "cpu"
        assert Path(sparse_dir).name == "0"
        assert (output_dir / "vggt_frame_count.npy").is_file()
