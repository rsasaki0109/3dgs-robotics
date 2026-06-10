"""GS Mapper zero-install demo: photos or a short video -> browser 3DGS .splat.

Runs on Hugging Face Spaces (ZeroGPU-aware) and locally:

    python app.py
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path

import gradio as gr

import pipeline as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gs_mapper_space")

GPU_DURATION = int(os.environ.get("GS_MAPPER_GPU_DURATION", "240"))
DEFAULT_METHOD = os.environ.get("GS_MAPPER_DEMO_METHOD", "dust3r")
MAX_FRAMES = 16

REPO_URL = "https://github.com/rsasaki0109/3dgs-robotics"
VIEWER_URL = "https://rsasaki0109.github.io/3dgs-robotics/splat.html"

try:  # ZeroGPU decorator; falls back to identity off-Spaces.
    import spaces

    gpu = spaces.GPU(duration=GPU_DURATION)
except Exception:  # pragma: no cover - local runs

    def gpu(fn):
        return fn


def _warmup() -> None:
    try:
        if DEFAULT_METHOD == "dust3r":
            pl.ensure_dust3r()
            pl.prefetch_checkpoint()
    except Exception as exc:  # network failures retried lazily on first run
        logger.warning("warmup failed: %s", exc)


@gpu
def generate(photos, video, num_frames, iterations, progress=gr.Progress()):
    if not photos and not video:
        raise gr.Error("Upload at least 2 photos or one short video.")

    workdir = Path(tempfile.mkdtemp(prefix="gs_mapper_"))
    staged = workdir / "images"
    try:
        if video:
            count = pl.extract_video_frames(Path(video), staged, max_frames=int(num_frames))
            logger.info("extracted %d frames from video", count)
        else:
            paths = [Path(f.name if hasattr(f, "name") else f) for f in photos]
            pl.prepare_images(paths, staged)

        result = pl.run_pipeline(
            staged,
            workdir / "out",
            method=DEFAULT_METHOD,
            num_frames=int(num_frames),
            iterations=int(iterations),
            progress=lambda fraction, message: progress(fraction, desc=message),
        )
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("pipeline failed")
        raise gr.Error(f"Reconstruction failed: {exc}") from exc

    size_mb = result.splat_path.stat().st_size / 1e6
    summary = (
        f"{result.num_used_frames} frames -> {size_mb:.1f} MB .splat "
        f"in {result.elapsed_sec:.0f}s ({result.method}). "
        f"Tip: drag the downloaded .splat onto the [GS Mapper web viewer]({VIEWER_URL}) "
        f"or self-host it — see the [repo]({REPO_URL})."
    )
    splat = str(result.splat_path)
    return splat, splat, summary


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="GS Mapper — Photos to 3D Gaussian Splat") as demo:
        gr.Markdown(
            "# GS Mapper — Photos to 3D Gaussian Splat\n"
            "Turn **a handful of photos or a short walkaround video** into a browser-ready "
            "3D Gaussian Splat. Pose-free (DUSt3R) — no COLMAP, no install.\n\n"
            f"[GitHub]({REPO_URL}) · [Live scene gallery]({VIEWER_URL}) · "
            "Best results: 8–16 photos orbiting one subject with ~70% overlap."
        )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("Photos"):
                    photos = gr.File(
                        label="Photos (2–16 images, jpg/png)",
                        file_count="multiple",
                        file_types=["image"],
                    )
                with gr.Tab("Video"):
                    video = gr.Video(label="Short walkaround video (frames are sampled evenly)")
                num_frames = gr.Slider(4, MAX_FRAMES, value=10, step=1, label="Frames to use")
                iterations = gr.Slider(500, 4000, value=2000, step=100, label="Training iterations")
                run_btn = gr.Button("Build my splat", variant="primary")
            with gr.Column(scale=2):
                viewer = gr.Model3D(label="Result (drag to orbit)", height=520)
                splat_file = gr.File(label="Download .splat")
                status = gr.Markdown()

        run_btn.click(
            generate,
            inputs=[photos, video, num_frames, iterations],
            outputs=[viewer, splat_file, status],
            concurrency_limit=1,
        )

        examples_dir = Path(__file__).parent / "examples" / "campus"
        if examples_dir.is_dir():
            example_images = sorted(str(p) for p in examples_dir.glob("*.jpg"))
            if len(example_images) >= 2:
                gr.Examples(examples=[[example_images]], inputs=[photos], label="Example photo set")
    return demo


threading.Thread(target=_warmup, daemon=True).start()
demo = build_ui()

if __name__ == "__main__":
    demo.launch()
