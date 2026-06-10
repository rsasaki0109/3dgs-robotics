# Isaac Sim integration (NuRec USDZ export)

Isaac Sim 5.0+ renders 3D Gaussian Splatting scenes natively through
[Omniverse NuRec](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html)
volume prims. Any map this repo produces — photos, a video, a rosbag, or a
live-mapping session — can be exported to a NuRec USDZ and dropped into Isaac
Sim as a photorealistic environment: real place in, simulated robot out.

```
rosbag / video / photos ──> 3DGS map (PLY) ──> export-isaac ──> USDZ ──> Isaac Sim 5.0+
```

No Isaac Sim at hand? `3dgs-robotics-camera-sim` simulates a camera in the
same map with nothing but ROS 2 — see
[live-mapping.md](live-mapping.md#ros-2-gs-camera-simulator-node).

## Setup (one time)

The export wraps NVIDIA's official converter from
[nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut). Only its transcode
path is used — no CUDA build, just a clone and a few pip packages:

```bash
git clone https://github.com/nv-tlabs/3dgrut.git ~/3dgrut
pip install plyfile msgpack usd-core "nvidia-ncore>=19.0.0" simplejpeg tensorboard
export THREEDGRUT_ROOT=~/3dgrut   # or pass --threedgrut-root each time
```

## Export

```bash
# a live-mapping session (last successful round; pick one with --round)
3dgs-robotics export-isaac --map outputs/live_mapping/session --output scene.usdz

# any standard 3DGS PLY (gsplat / INRIA layout)
3dgs-robotics export-isaac --ply outputs/my_scene/train/point_cloud.ply
```

`--format nurec` (default) targets Isaac Sim / Omniverse; `--format
lightfield` writes the newer `ParticleField3DGaussianSplat` USD schema.

## Import into Isaac Sim

1. Isaac Sim 5.0+ → **File > Import** (or drag-and-drop) the `.usdz` —
   the scene renders as a NuRec volume via RTX.
2. Add a **ground plane** (Create > Physics > Ground Plane): NuRec volumes
   are visual-only, so robots need collision geometry. Align it with the
   map's ground; box/mesh colliders for walls and obstacles follow the same
   pattern.
3. Drop in a robot from the Isaac Sim asset library and drive it through
   your reconstruction.

See NVIDIA's
[smartphone-to-Isaac-Sim walkthrough](https://developer.nvidia.com/blog/reconstruct-a-scene-in-nvidia-isaac-sim-using-only-a-smartphone/)
for the same flow with screenshots.

## Caveats

- **Scale is not metric.** Pose-free monocular maps live in an arbitrary
  reconstruction gauge — measure a known distance in the scene and scale the
  prim in Isaac Sim (or map with metric poses). Orientation is likewise
  arbitrary; rotate so the ground is z-up.
- **Draft rounds are blurry.** Live-mapping rebuilds default to 1500 gsplat
  iterations for latency. For a simulation-quality scene retrain the final
  round at 7k–15k iterations first (see
  [live-mapping.md](live-mapping.md#3dgs-localization) for the retrain
  recipe), or build the map one-shot with `--quality high`.
- Clean floaters before export: `3dgs-robotics splat-filter` on the PLY's
  `.splat` sibling does not touch the PLY — filter at training time or
  re-export after retraining instead.
