"""Streamlit demo app for the gs-sim2real 3DGS pipeline.

Provides a browser-based interface for uploading images, running COLMAP
preprocessing, training 3D Gaussian Splatting models, viewing 3D point
clouds, and exporting results.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="3DGS Playground", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_PATH = PROJECT_ROOT / "docs" / "training_metrics.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_demo_metrics() -> dict | None:
    """Load training metrics JSON used for the demo / preview charts."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def _list_images(directory: Path) -> list[Path]:
    """Return sorted list of image files in *directory*."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _total_size_mb(paths: list[Path]) -> float:
    return sum(p.stat().st_size for p in paths) / (1024 * 1024)


def _generate_demo_point_cloud(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Generate a colourful demo point cloud (sphere + torus)."""
    rng = np.random.default_rng(42)

    # Sphere
    n_sphere = n // 2
    phi = rng.uniform(0, 2 * np.pi, n_sphere)
    costheta = rng.uniform(-1, 1, n_sphere)
    theta = np.arccos(costheta)
    r = 1.0 + rng.normal(0, 0.02, n_sphere)
    xs = r * np.sin(theta) * np.cos(phi)
    ys = r * np.sin(theta) * np.sin(phi)
    zs = r * np.cos(theta)
    sphere = np.column_stack([xs, ys, zs])

    # Torus
    n_torus = n - n_sphere
    u = rng.uniform(0, 2 * np.pi, n_torus)
    v = rng.uniform(0, 2 * np.pi, n_torus)
    R, rr = 2.5, 0.5
    xt = (R + rr * np.cos(v)) * np.cos(u)
    yt = (R + rr * np.cos(v)) * np.sin(u)
    zt = rr * np.sin(v)
    torus = np.column_stack([xt, yt, zt])

    positions = np.vstack([sphere, torus]).astype(np.float32)

    # Colour by height (z)
    z_norm = (positions[:, 2] - positions[:, 2].min()) / (positions[:, 2].ptp() + 1e-8)
    colors = np.column_stack(
        [
            z_norm,
            0.4 * np.ones(len(z_norm)),
            1.0 - z_norm,
        ]
    ).clip(0, 1)

    return positions, colors


def _load_ply_for_plotly(ply_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to load positions and colours from a PLY file."""
    try:
        from gs_sim2real.viewer.web_viewer import load_ply

        data = load_ply(ply_path)
        return data.positions, data.colors
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("3DGS Playground")

st.sidebar.header("Data Source")
data_mode = st.sidebar.radio(
    "Choose input method",
    ["Upload images", "Download sample images", "Local directory path"],
    label_visibility="collapsed",
)

uploaded_files: list | None = None
sample_download_clicked = False
local_dir_path = ""

if data_mode == "Upload images":
    uploaded_files = st.sidebar.file_uploader(
        "Upload images (jpg/png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
elif data_mode == "Download sample images":
    sample_download_clicked = st.sidebar.button("Download sample images")
else:
    local_dir_path = st.sidebar.text_input("Path to local image directory", value="")

st.sidebar.header("Pipeline Settings")
run_colmap = st.sidebar.checkbox("Run COLMAP preprocessing", value=True)
training_method = st.sidebar.selectbox("Training method", ["gsplat", "nerfstudio"])
num_iterations = st.sidebar.slider("Num iterations", min_value=100, max_value=30000, value=1000, step=100)
use_gpu = st.sidebar.checkbox("Use GPU", value=True)

run_pipeline = st.sidebar.button("Run Pipeline", type="primary")

# ---------------------------------------------------------------------------
# Resolve working image directory
# ---------------------------------------------------------------------------
work_dir: Path | None = None

if "work_dir" not in st.session_state:
    st.session_state["work_dir"] = None

# Handle uploads
if uploaded_files:
    tmp = Path(tempfile.mkdtemp(prefix="gs_app_"))
    img_dir = tmp / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded_files:
        (img_dir / uf.name).write_bytes(uf.getvalue())
    st.session_state["work_dir"] = str(img_dir)

# Handle sample download
if sample_download_clicked:
    with st.spinner("Downloading sample images..."):
        try:
            from gs_sim2real.common.download import download_sample_images

            tmp = Path(tempfile.mkdtemp(prefix="gs_app_sample_"))
            img_dir = download_sample_images(tmp, num_images=10)
            st.session_state["work_dir"] = str(img_dir)
            st.success(f"Sample images downloaded to {img_dir}")
        except Exception as exc:
            st.error(f"Download failed: {exc}")

# Handle local path
if local_dir_path:
    p = Path(local_dir_path)
    if p.is_dir():
        st.session_state["work_dir"] = str(p)
    else:
        st.sidebar.warning("Directory does not exist.")

work_dir = Path(st.session_state["work_dir"]) if st.session_state["work_dir"] else None

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if run_pipeline and work_dir is not None:
    output_root = Path(tempfile.mkdtemp(prefix="gs_app_out_"))
    st.session_state["output_root"] = str(output_root)

    colmap_dir = output_root / "colmap"
    train_dir = output_root / "train"

    # -- COLMAP --
    if run_colmap:
        st.session_state["colmap_status"] = "running"
        try:
            from gs_sim2real.preprocess.colmap import run_colmap as _run_colmap

            sparse_dir = _run_colmap(
                image_dir=work_dir,
                output_dir=colmap_dir,
                use_gpu=use_gpu,
            )
            st.session_state["colmap_status"] = "complete"
            st.session_state["sparse_dir"] = str(sparse_dir)
        except Exception as exc:
            st.session_state["colmap_status"] = f"error: {exc}"

    # -- Training --
    st.session_state["train_status"] = "running"
    data_for_train = colmap_dir if run_colmap else work_dir
    try:
        if training_method == "gsplat":
            from gs_sim2real.train.gsplat_trainer import train_gsplat

            ply = train_gsplat(
                data_dir=data_for_train,
                output_dir=train_dir,
                num_iterations=num_iterations,
            )
            st.session_state["ply_path"] = str(ply)
        else:
            from gs_sim2real.train.nerfstudio_trainer import train_nerfstudio

            train_nerfstudio(
                data_dir=data_for_train,
                output_dir=train_dir,
            )
        st.session_state["train_status"] = "complete"
    except Exception as exc:
        st.session_state["train_status"] = f"error: {exc}"

elif run_pipeline and work_dir is None:
    st.sidebar.error("Please provide images first.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_input, tab_preprocess, tab_train, tab_viewer, tab_export, tab_teleop = st.tabs(
    ["Input Images", "Preprocessing", "Training Progress", "3D Viewer", "Export", "Robot Teleop"]
)

# ---- Tab 1: Input Images ----
with tab_input:
    st.header("Input Images")
    if work_dir and work_dir.exists():
        images = _list_images(work_dir)
        if images:
            st.write(f"**{len(images)}** images  |  **{_total_size_mb(images):.1f} MB** total")
            cols = st.columns(min(len(images), 5))
            for idx, img_path in enumerate(images):
                cols[idx % len(cols)].image(str(img_path), caption=img_path.name, use_container_width=True)
        else:
            st.info("No images found in the selected directory.")
    else:
        st.info("Upload images, download samples, or enter a local path in the sidebar.")

# ---- Tab 2: Preprocessing ----
with tab_preprocess:
    st.header("COLMAP Structure-from-Motion")
    colmap_status = st.session_state.get("colmap_status", "not yet run")

    if colmap_status == "not yet run":
        st.info("COLMAP has not been run yet. Click **Run Pipeline** in the sidebar.")
    elif colmap_status == "running":
        st.warning("COLMAP is running...")
    elif colmap_status == "complete":
        st.success("COLMAP reconstruction complete.")
        sparse_dir = Path(st.session_state.get("sparse_dir", ""))
        if sparse_dir.exists():
            # Count registered images and 3D points
            n_images = 0
            n_points = 0
            images_txt = sparse_dir / "images.txt"
            points_txt = sparse_dir / "points3D.txt"
            images_bin = sparse_dir / "images.bin"
            points_bin = sparse_dir / "points3D.bin"

            if images_txt.exists():
                with open(images_txt) as f:
                    n_images = sum(1 for line in f if line.strip() and not line.startswith("#")) // 2
            elif images_bin.exists():
                import struct

                with open(images_bin, "rb") as f:
                    n_images = struct.unpack("<Q", f.read(8))[0]

            if points_txt.exists():
                with open(points_txt) as f:
                    n_points = sum(1 for line in f if line.strip() and not line.startswith("#"))
            elif points_bin.exists():
                import struct

                with open(points_bin, "rb") as f:
                    n_points = struct.unpack("<Q", f.read(8))[0]

            col1, col2 = st.columns(2)
            col1.metric("Registered images", n_images)
            col2.metric("3D points", f"{n_points:,}")
            st.write(f"Sparse reconstruction directory: `{sparse_dir}`")
    else:
        st.error(colmap_status)

# ---- Tab 3: Training Progress ----
with tab_train:
    st.header("Training Progress")
    train_status = st.session_state.get("train_status", "not yet run")

    if train_status == "not yet run":
        st.info("Training has not been run yet. Showing demo metrics preview below.")
    elif train_status == "running":
        st.warning("Training is in progress...")
    elif train_status == "complete":
        st.success("Training complete.")
    elif train_status.startswith("error"):
        st.error(train_status)

    # Show metrics (demo or real)
    metrics = load_demo_metrics()
    if metrics:
        scene_names = list(metrics.keys())
        selected_scene = st.selectbox("Scene", scene_names)
        scene = metrics[selected_scene]

        iters = scene.get("iterations", [])
        col1, col2 = st.columns(2)

        with col1:
            if "loss" in scene:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=iters, y=scene["loss"], mode="lines", name="Loss"))
                fig_loss.update_layout(title="Training Loss", xaxis_title="Iteration", yaxis_title="Loss", height=350)
                st.plotly_chart(fig_loss, use_container_width=True)

        with col2:
            if "psnr" in scene:
                fig_psnr = go.Figure()
                fig_psnr.add_trace(
                    go.Scatter(x=iters, y=scene["psnr"], mode="lines", name="PSNR", line={"color": "green"})
                )
                fig_psnr.update_layout(title="PSNR", xaxis_title="Iteration", yaxis_title="PSNR (dB)", height=350)
                st.plotly_chart(fig_psnr, use_container_width=True)

        if "num_gaussians" in scene:
            fig_ng = go.Figure()
            fig_ng.add_trace(
                go.Scatter(x=iters, y=scene["num_gaussians"], mode="lines", name="Gaussians", line={"color": "orange"})
            )
            fig_ng.update_layout(
                title="Number of Gaussians",
                xaxis_title="Iteration",
                yaxis_title="Count",
                height=300,
            )
            st.plotly_chart(fig_ng, use_container_width=True)

        if train_status == "not yet run":
            st.caption("These are pre-recorded demo metrics from docs/training_metrics.json.")
    else:
        st.write("No training metrics available.")

# ---- Tab 4: 3D Viewer ----
with tab_viewer:
    st.header("3D Viewer")

    ply_path_str = st.session_state.get("ply_path")
    positions: np.ndarray | None = None
    colors: np.ndarray | None = None

    if ply_path_str and Path(ply_path_str).exists():
        result = _load_ply_for_plotly(Path(ply_path_str))
        if result is not None:
            positions, colors = result
            st.write(f"Loaded **{len(positions):,}** points from `{ply_path_str}`")
    else:
        positions, colors = _generate_demo_point_cloud(2000)
        st.info("No trained model available. Showing a generated demo point cloud.")

    # Controls
    c1, c2 = st.columns(2)
    point_size = c1.slider("Point size", 1, 10, 3)
    color_mode = c2.selectbox("Color mode", ["RGB", "Height (Z)", "Depth (Y)"])

    if positions is not None:
        # Subsample for performance
        max_display = 50000
        if len(positions) > max_display:
            idx = np.random.default_rng(0).choice(len(positions), max_display, replace=False)
            positions = positions[idx]
            colors = colors[idx]

        if color_mode == "Height (Z)":
            z = positions[:, 2]
            z_norm = (z - z.min()) / (z.ptp() + 1e-8)
            plot_colors = [f"rgb({int(v * 255)},100,{int((1 - v) * 255)})" for v in z_norm]
        elif color_mode == "Depth (Y)":
            y = positions[:, 1]
            y_norm = (y - y.min()) / (y.ptp() + 1e-8)
            plot_colors = [f"rgb({int((1 - v) * 255)},{int(v * 200)},100)" for v in y_norm]
        else:
            plot_colors = [f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})" for r, g, b in colors]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode="markers",
                    marker={
                        "size": point_size,
                        "color": plot_colors,
                        "opacity": 0.8,
                    },
                )
            ]
        )
        fig.update_layout(
            scene={
                "xaxis_title": "X",
                "yaxis_title": "Y",
                "zaxis_title": "Z",
                "aspectmode": "data",
            },
            height=600,
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info("For the full interactive viewer, use the CLI command: `gs-sim2real view --model <path.ply>`")

# ---- Tab 5: Export ----
with tab_export:
    st.header("Export")

    ply_path_str = st.session_state.get("ply_path")
    if ply_path_str and Path(ply_path_str).exists():
        ply_path = Path(ply_path_str)
        st.subheader("Trained Model (.ply)")
        with open(ply_path, "rb") as f:
            st.download_button(
                label="Download point_cloud.ply",
                data=f,
                file_name=ply_path.name,
                mime="application/octet-stream",
            )

        # Check for training log
        train_log = ply_path.parent / "training.log"
        if train_log.exists():
            st.subheader("Training Log")
            with open(train_log, "rb") as f:
                st.download_button(
                    label="Download training.log",
                    data=f,
                    file_name="training.log",
                    mime="text/plain",
                )
    else:
        st.info("No trained model available for export yet. Run the pipeline first.")

    st.subheader("CLI Equivalent Commands")
    img_dir_display = st.session_state.get("work_dir", "<IMAGE_DIR>")
    gpu_flag = "" if use_gpu else " --no-gpu"
    st.code(
        f"""\
# Preprocess with COLMAP
gs-sim2real preprocess --images {img_dir_display} --output outputs/colmap{gpu_flag}

# Train 3DGS
gs-sim2real train --data outputs/colmap --method {training_method} --iterations {num_iterations}

# View the result
gs-sim2real view --model outputs/train/point_cloud.ply
""",
        language="bash",
    )

# ---- Tab 6: Robot Teleop ----
with tab_teleop:
    st.header("Robot Teleop in Dynamic Map Viewer")

    ply_path_str = st.session_state.get("ply_path")
    has_ply = ply_path_str and Path(ply_path_str).exists()

    if has_ply:
        st.success(f"Trained model ready: `{ply_path_str}`")
        fragment = st.text_input("Fragment name", value="residency")

        if st.button("Stage for Dynamic Map Viewer", type="primary"):
            try:
                from gs_sim2real.demo.stage_for_dreamwalker import stage_ply

                result = stage_ply(ply_path_str, fragment=fragment)
                st.session_state["dreamwalker_staged"] = result
                st.success(f"Staged: {result['splat_dest']}")
            except Exception as exc:
                st.error(f"Staging failed: {exc}")

        staged = st.session_state.get("dreamwalker_staged")
        if staged:
            st.markdown(f"**Launch URL**: [{staged['launch_url']}]({staged['launch_url']})")
            st.info("Run `cd apps/dreamwalker-web && npm run dev` in another terminal, then open the URL above.")
    else:
        st.info("No trained model available. Run the pipeline first, or use the CLI:")

    st.subheader("CLI One-Command Demo")
    st.code(
        """\
# From images (full pipeline)
gs-sim2real demo --images <IMAGE_DIR> --iterations 1000

# From an existing PLY
gs-sim2real demo --ply outputs/train/point_cloud.ply
""",
        language="bash",
    )

    st.subheader("Controls")
    st.markdown(
        """\
| Key | Action |
|-----|--------|
| WASD | Move robot |
| Mouse | Look around |
| R | Toggle robot mode |
| 1/2/3 | Front / Chase / Top camera |
| Space | Place waypoint |
"""
    )
