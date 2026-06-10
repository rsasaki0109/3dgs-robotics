# 3DGS Robotics Launch Kit

Real robot data to Gaussian-splat Physical AI benchmarks.

3DGS Robotics turns photos, robotics logs, and MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / LoGeR artifacts into browser-viewable Gaussian splats, scene contracts, route-policy benchmarks, and CI review bundles for Physical AI evaluation.

## Links

- [Project page](https://rsasaki0109.github.io/3dgs-robotics/) - First-stop page with the 3DGS Robotics pitch and viewer entry points.
- [Live splat viewer](https://rsasaki0109.github.io/3dgs-robotics/splat.html) - WebGL scene picker with nine bundled comparison splats.
- [Spark mobile / VR viewer](https://rsasaki0109.github.io/3dgs-robotics/splat_spark.html) - Spark viewer for mobile and WebXR-capable devices.
- [WebGPU viewer](https://rsasaki0109.github.io/3dgs-robotics/splat_webgpu.html) - GPU-sort viewer for Chrome, Edge, and WebGPU-enabled browsers.
- [Physical AI simulation contract](https://github.com/rsasaki0109/3dgs-robotics/blob/main/docs/physical-ai-sim.md) - Scene contract, sensor noise profiles, dynamic obstacles, and scenario CI reference.
- [GitHub repository](https://github.com/rsasaki0109/3dgs-robotics) - Source, Physical AI benchmark docs, external SLAM import docs, and tests.

## Where To Post

### X / Twitter

- URL: https://x.com/intent/tweet
- Audience: 3DGS, SLAM, robotics, and Physical AI builders who want a fast demo link.
- Angle: Lead with the live viewer, then mention policy benchmarks and CI artifacts.
- Copy block: `short-social`

### Hacker News Show HN

- URL: https://news.ycombinator.com/submit
- Audience: Graphics, mapping, robotics, simulation, and developer-tool readers who inspect repos.
- Angle: Frame it as an open-source bridge from real robot data to reproducible evaluation artifacts.
- Copy block: `community-post`

### LinkedIn

- URL: https://www.linkedin.com/feed/
- Audience: Robotics, autonomy, geospatial, and simulation engineers.
- Angle: Emphasize policy regression, reviewable simulation artifacts, and SLAM-to-benchmark handoff.
- Copy block: `technical-social`

### Reddit communities

- URL: https://www.reddit.com/search/?q=3D%20Gaussian%20Splatting%20SLAM&type=communities
- Audience: Subreddits around Gaussian Splatting, photogrammetry, 3D scanning, robotics, and ML systems.
- Angle: Pick one relevant community, check its rules, and post the community copy with screenshots.
- Copy block: `community-post`

### GitHub awesome lists

- URL: https://github.com/search?q=awesome+3d+gaussian+splatting&type=repositories
- Audience: Maintainers of curated 3DGS, SLAM, robotics, simulation, and Physical AI resource lists.
- Angle: Open a small PR with the awesome-list entry and link directly to the live demo and README.
- Copy block: `awesome-list`

### Japanese robotics channels

- URL: https://rsasaki0109.github.io/3dgs-robotics/
- Audience: Japanese robotics, mapping, autonomy, and computer-vision builders.
- Angle: Use the Japanese announcement and point people to the live splat viewer and benchmark docs.
- Copy block: `japanese`


## Copy Blocks

### Short social post (251/280 chars)

```text
3DGS Robotics turns robot logs and MASt3R-SLAM / VGGT-SLAM outputs into browser-viewable 3D Gaussian Splats plus CI-friendly Physical AI policy benchmarks.

Live demos: https://rsasaki0109.github.io/3dgs-robotics/splat.html
#3DGS #Robotics #PhysicalAI
```

### Technical social post

```text
I released 3DGS Robotics: an open-source pipeline for turning real outdoor robot data, DUSt3R / MASt3R pose-free reconstructions, and MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / LoGeR artifacts into Gaussian-splat scenes and Physical AI policy benchmark artifacts.

It ships nine public comparison scenes, external SLAM dry-run manifests, route-policy benchmarks with pose and raw camera / depth / LiDAR noise profiles, dynamic-obstacle timelines with multi-agent observation features, scenario matrix expansion, CI sharding, generated workflow validation, activation guards, and review bundles.

https://github.com/rsasaki0109/3dgs-robotics
```

### Community post

```text
3DGS Robotics is an open-source pipeline for turning real robot data into Gaussian-splat Physical AI evaluation artifacts. It accepts image folders, robotics logs, and external SLAM artifacts, then produces browser-viewable .splat files, scene contracts, policy benchmark reports, scenario shards, and review bundles.

The current demo set compares supervised GNSS + LiDAR, DUSt3R, MASt3R, VGGT-SLAM 2.0, MASt3R-SLAM, and Pi3X outputs on outdoor robotics scenes. The benchmark stack keeps dataset, policy registry, scenario matrix, CI manifest, workflow validation, activation, and review publishing as separate testable artifacts, plus partial-information knobs (pose / goal / heading noise, raw camera / depth / LiDAR noise, and dynamic obstacles with nearest + second-nearest features) for reactive-policy studies.

Live demo: https://rsasaki0109.github.io/3dgs-robotics/splat.html
Repo: https://github.com/rsasaki0109/3dgs-robotics
```

### Awesome-list entry

```text
- [3DGS Robotics](https://github.com/rsasaki0109/3dgs-robotics) - Converts photos, robotics logs, and MASt3R-SLAM / VGGT-SLAM / Pi3 / LoGeR artifacts into browser-viewable Gaussian splats, Physical AI route-policy benchmarks, and CI review bundles.
```

### Japanese announcement

```text
3DGS Robotics を公開しました。写真フォルダ、ロボティクスログ、MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / LoGeR の出力を、Web で見られる 3D Gaussian Splatting、Physical AI policy benchmark、scenario CI review bundle につなぐ OSS です。

Live demo: https://rsasaki0109.github.io/3dgs-robotics/splat.html
GitHub: https://github.com/rsasaki0109/3dgs-robotics
```

## Topics

`3d-gaussian-splatting`, `3dgs`, `slam`, `visual-slam`, `robotics`, `physical-ai`, `simulation`, `benchmark`, `autonomous-driving`, `mast3r`, `dust3r`, `vggt-slam`, `gsplat`, `scenario-ci`, `route-policy-benchmark`, `webgl`, `webgpu`
