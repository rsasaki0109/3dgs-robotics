# 屋外 3D Gaussian Splatting / Physical AI Simulation 開発計画

更新日: 2026-04-29（Pi3X production comparison asset 反映）

この文書は、GS Mapper の屋外 3DGS パイプラインと、その上に載せる Physical AI simulation / policy benchmark / scenario CI の現行計画をまとめる長めの handoff です。

古い PR ごとの transcript、`tuhh_day_04` の誤判定、MCD calibration 探索、実走ログ、個別コマンドの長い出力は [archive snapshot](archive/plan_outdoor_gs_2026_04_full_handoff.md) に残しています。本書は「次にどこへ進むか」を判断するための source of truth として更新します。

## 0. 読み方

1. まず **TL;DR** と **現在の主戦場** を読む。
2. 実装に入る前に **System Map** と **Scenario CI Pipeline** を確認する。
3. 屋外データ / viewer / external SLAM だけ触るなら **Outdoor 3DGS Track** を読む。
4. Physical AI / policy benchmark / CI を触るなら **Physical AI Simulation Track** を読む。
5. 古い実験値、MCD calibration、`ntu_day_02` 実走値、PR #55〜#80 の履歴が必要な場合だけ archive を読む。

## 1. TL;DR

- GS Mapper は、写真フォルダ、Autoware / MCD の robotics logs、MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / LoGeR などの external SLAM artifacts を、3D Gaussian Splatting training / export / browser viewer へつなぐ repo。
- Public demo は GitHub Pages で公開済み。`docs/scenes-list.json` が README table / preview capture / hero GIF / viewer picker の source of truth。
- Production viewer picker は 9 scenes。2 supervised、4 pose-free、3 external-SLAM comparison。
- MCD `tuhh_day_04` の supervised GNSS 成功扱いは撤回済み。`/vn200/GPS` が all-zero なので production picker には入れない。
- Valid GNSS supervised MCD demo は `ntu_day_02`。production asset は `docs/assets/outdoor-demo/mcd-ntu-day02-supervised.splat`。
- External SLAM import は VGGT-SLAM 2.0 / MASt3R-SLAM / Pi3X comparison splat まで実走済み。LoGeR profile も artifact resolver 側に候補追加済み。
- 2026-04-24 時点では、屋外 3DGS だけでなく **Physical AI simulation benchmark environment** を目指す方向へ拡張中。
- Route policy benchmark 系は、dataset / imitation / registry / benchmark / history / scenario-set / matrix / sharding / CI manifest / workflow materialization / validation / activation / review bundle / workflow trigger promotion gate / promotion-backed trigger adoption / adoption-aware review bundle まで分割済み。
- 最新の merged Pages refresh は `ee42f7a`。Tier 2 chain (#121–#134) で env-hardening + correlation gate plumbing が完成し、Pages landing は outdoor GS demo-first に刷新済み。local full pytest / GitHub Actions CI / Pages deploy は green。
- adoption step + CLI (`gs-mapper route-policy-scenario-ci-workflow-adopt`) + adoption-aware review bundle まで実装済み。review には `--adoption-report` を渡すと Pages の `review.{json,md,html}` に trigger mode / branches / manual vs adopted YAML の unified diff が乗る。
- Pages `/reviews/` には synthetic smoke fixture 由来の `docs/reviews/smoke-route-policy-ci/` sample bundle を置く方針に変更。`scripts/build_pages_sample_review_bundle.py` が smoke chain を回し、temp path を self-contained な `sample-artifacts/` 相対リンクへ書き換えて index も再生成する。
- 2026-04-25 〜 26 の Tier 2 rollup で、real-vs-sim correlation library (#113/#115) → scenario-set run report への attach (#121) → review bundle への surface (#125) → regression gate (#126) → per-bag overrides (#128) → translation/heading pair-distribution gates (#129/#130) → time stratification (#131/#132) + equal-pair-count mode (#133) + per-window stats (#134) まで一気に完成。`gs-mapper route-policy-scenario-ci-review` の correlation gate は実用 production rollout で使える状態。
- 同時に env-hardening 側も IMU finite-diff renderer (#111) → ObstaclePolicy protocol (#112) → IMU + peer-aware features を gym adapter feature dict へ surface (#122/#123) → query_collision / score_trajectory に per-step peer cache を threading (#124/#127) で multi-agent サポートが整った。

## 2. 現在の主戦場

今の大きな方向転換は、単なる「屋外 3DGS のデモ生成」から、次のような **Physical AI 用 simulation / evaluation environment** に寄せることです。

1. Real-world robotics logs から 3DGS scene を作る。
2. Browser / local renderer / headless environment で観測を返す。
3. Route policy / navigation policy / query policy を benchmark する。
4. Scenario matrix を小さな shard に分け、CI で回す。
5. CI workflow 自体も生成、検証、activation、review publishing の段階に分ける。
6. Review bundle を GitHub Pages に出し、workflow trigger を広げる前に人間が inspected artifact を見られるようにする。
7. Promotion report で PR / branch trigger へ広げてよいかを記録し、trigger-enabled workflow の adoption を分離する。

この構成にした理由は、開発がスケールすると「1 個の巨大 E2E が落ちる」よりも、「小さい scenario / shard / validation / activation / review gate がどこで落ちたか分かる」方が速いからです。ユーザーが求めていた「モジュール分割、関数分割、クラス分割、依存の局所化、テスト単位の分離」「影響範囲を閉じ込め、検証単位を細かく設計する」は、この route policy scenario CI chain の設計方針そのものです。

## 3. Recent Commits / 現在地

直近の主な流れ (2026-04-25 〜 26 の Tier 2 chain):

| Commit | 内容 |
| --- | --- |
| `2262f22` | Per-window correlation stats (mean/p95/max/heading + bag-time span) を review bundle の Markdown / HTML に surface。 |
| `95f1ea4` | `pair_distribution_strata_mode` で `equal-pair-count` を選べるように。スパース bag でも各 window を統計的に成立させる。 |
| `cef2659` | aggregate-statistic (mean/p95/max/heading-mean) を per-window 評価に切り替え可能に。stratified 時は aggregate tag を suppress。 |
| `2abd640` | `pair_distribution_strata` で per-pair 分布ゲートを N 等分時間 window に分けて評価。 |
| `4629835` | heading 版 per-pair 分布ゲートを追加。heading-bearing subset を分母に使う。 |
| `0683f91` | translation per-pair 分布ゲート (`max_pair_translation_error_meters` + fraction) を追加。 |
| `3762717` | per-bag-topic correlation threshold overrides を `--correlation-thresholds-config` JSON で受け付ける。 |
| `6db678e` | `score_trajectory` に per-step peer cache を threading。hypothetical trajectory でも policy obstacle が peer を見える。 |
| `5a6edfd` | correlation regression gate (mean/p95/max/heading-mean) を `gs-mapper route-policy-scenario-ci-review` に追加、`--fail-on-review` で exit 2。 |
| `ab8fbbe` | scenario CI review bundle (Markdown + HTML) に Real-vs-sim correlation セクションを追加。 |
| `ea1b5f8` | `HeadlessPhysicalAIEnvironment.query_collision` に per-step peer cache を threading。 |
| `a038b61` | `RoutePolicyGymAdapter` に peer-aware obstacle features (`peer-min-separation-meters` 等) + `step_positions` 解決を導入。 |
| `9bb15d2` | `RoutePolicyGymAdapter` の feature dict に IMU 7 軸 (step_dt + ang_vel + lin_acc) を出力。 |
| `b40e4a3` | scenario-set run report に correlation reports を attach (`--correlation-report`)、Markdown サマリ surface。 |
| `5195130` | rosbag correlator に IMU orientation merge を追加 (`merge_navsat_with_imu_orientation`)。 |
| `9e3be8b` | `ObstaclePolicy` protocol + 4 reference impl (Waypoint / Chase / Flee / MaintainSeparation) を導入。 |
| `8bf29b1` | env に IMU kinematic finite-diff renderer を baked-in。`imu-proxy` sensor が `ready-via-kinematic-finite-diff`。 |
| `7127641` | real-vs-sim rosbag correlation library + CLI (`scripts/run_rosbag_correlation.py`) を新設。 |

この chain で **correlation gate plumbing** と **multi-agent obstacle plumbing** が両方とも production rollout に使える状態に到達した。

### 3.1 Claude handoff snapshot

- 基準にする pushed state は `main @ 2262f22`。
- working tree は clean、origin/main と同期済み。Claude に渡す前の doc 更新で差分があるなら、まず `docs/plan_outdoor_gs.md` のみか確認する。
- GitHub Actions:
  - CI green on 2026-04-26 (PR #134 まで)
  - Deploy to GitHub Pages green on 2026-04-26
- local validation snapshot:
  - `python3 -m ruff check src/ tests/ scripts/`
  - `python3 -m ruff format --check src/ tests/ scripts/`
  - `git diff --check`
  - `python3 -m pytest tests/ -q --ignore=tests/e2e` => `738 passed`
- mypy note:
  - `python3 -m mypy src/gs_sim2real/sim/policy_scenario_ci_promotion.py` は pass。
  - `src/gs_sim2real/cli.py` を含む mypy は Waymo / MCD 周辺の既知型不整合で落ちる。Tier 2 chain は regression を入れていない。
- Tier 1 MCD rerun (`scripts/plan_mcd_quality_runs.py`) の 2/3 profile (`single_400_depth_long` L1=0.1951 / `single_800_ba` L1=0.2699) が gate pass。Profile 3 (`multi_3cam_300each_ba`) は手元 bag に `d455t` / `d435i` topics が無いので data-blocked。
- Claude が次に触るときの推奨 starting point（2026-05-15 GPT pro consultation 後の優先順位、詳細は §17 / `plan_review_bundle_provenance.md`）:
  1. **Sprint 1 PR A — production-review-bundle-manifest** (§17.2): `RoutePolicyScenarioCIReviewProvenance` 追加で synthetic / production を first-class field 化。Pages index に kind 列。後方互換のため `provenance is None` 時は既存 v1 JSON 等価。
  2. **Sprint 1 PR A2 — initial production review bundle 公開**: `outdoor-demo` 系 production scenario run の bundle を `docs/reviews/<run-id>/` に置き、README / Pages landing に導線追加。
  3. **Sprint 2 PR B — event-aligned stratification** (§17.3、旧 #133 OOS): `CorrelationEventWindow` schema を Sprint 2 で固めることで Sprint 3 の policy trace event を後方互換で乗せられる。
  4. **Sprint 3 PR C — policy trace events** (§17.4): real-vs-sim 比較を policy 行動レベルへ拡張する入り口。
  5. **Sprint 4 — multi-agent Tier 3 production matrix** (§17.5): 2 → 4 → 16 → 32+ agent の段階拡張。
  6. その後の backlog: LoGeR production comparison (§12.3)、MCD `ntu_day_02` `multi_3cam_300each_ba` (data-blocked)、Applanix `read_gsof_ins_pose_stream` (vendor 依存)。

## 4. System Map

### 4.1 層構造

| Layer | 目的 | 主な files |
| --- | --- | --- |
| Data / assets | public demo assets、scene manifests、Pages viewer | `docs/scenes-list.json`, `docs/sim-scenes.json`, `docs/assets/outdoor-demo/`, `docs/splat.html`, `docs/index.html` |
| Preprocess | image / video / rosbag / external SLAM artifact から COLMAP sparse 相当を作る | `src/gs_sim2real/datasets/`, `src/gs_sim2real/preprocess/`, `src/gs_sim2real/preprocess/external_slam_artifacts/` |
| Train / export | gsplat / nerfstudio training、`.splat` / scene bundle export | `src/gs_sim2real/train/`, `src/gs_sim2real/viewer/web_export.py`, `src/gs_sim2real/cli.py` |
| Physical AI sim contract | scene environment、sensor rig、headless env、observations/actions | `src/gs_sim2real/sim/contract.py`, `interfaces.py`, `headless.py`, `gym_adapter.py`, `occupancy.py`, `costmap.py` |
| Policy benchmark | dataset、imitation、registry、benchmark、history gates | `policy_dataset.py`, `policy_imitation.py`, `policy_benchmark.py`, `policy_benchmark_history.py` |
| Scenario execution | scenario-set、matrix expansion、sharding、merge | `policy_scenario_set.py`, `policy_scenario_matrix.py`, `policy_scenario_sharding.py` |
| Scenario CI | CI manifest、workflow materialization、validation、activation、review publishing | `policy_scenario_ci_manifest.py`, `policy_scenario_ci_workflow.py`, `policy_scenario_ci_activation.py`, `policy_scenario_ci_review.py` |
| Experiment labs | design seams の比較実験と docs 生成 | `src/gs_sim2real/experiments/`, `docs/experiments.md`, `docs/experiments.generated.md` |

### 4.2 分割の基本方針

- 外部依存の重い front-end は repo 内で import しない。MASt3R-SLAM / VGGT-SLAM / Pi3 / LoGeR は「artifact を吐いた後」に importer が受ける。
- Generated artifact は必ず versioned JSON / Markdown / HTML のどれかにする。
- CI workflow は手書きではなく manifest から生成する。
- Generated workflow はすぐ active path に置かず、validation → activation → review publishing を通す。
- Physical AI benchmark は single huge run にせず、scenario-set → matrix → shard → merge に分ける。
- Public Pages に出すものは `docs/` 配下だけ。実データ、rosbag、calibration YAML、training output は commit しない。

## 5. Production Assets / Viewer Contract

`docs/scenes-list.json` の production scene list:

1. `assets/outdoor-demo/outdoor-demo.splat` — Autoware 6-bag supervised default
2. `assets/outdoor-demo/outdoor-demo-dust3r.splat` — bag6 DUSt3R pose-free
3. `assets/outdoor-demo/mcd-tuhh-day04.splat` — MCD `tuhh_day_04` DUSt3R pose-free
4. `assets/outdoor-demo/bag6-mast3r.splat` — bag6 MAST3R pose-free metric
5. `assets/outdoor-demo/bag6-vggt-slam-20-15k.splat` — bag6 VGGT-SLAM 2.0 comparison
6. `assets/outdoor-demo/bag6-mast3r-slam-20-15k.splat` — bag6 MASt3R-SLAM comparison
7. `assets/outdoor-demo/bag6-pi3x-20-15k.splat` — bag6 Pi3X comparison
8. `assets/outdoor-demo/mcd-tuhh-day04-mast3r.splat` — MCD `tuhh_day_04` MAST3R pose-free metric
9. `assets/outdoor-demo/mcd-ntu-day02-supervised.splat` — MCD `ntu_day_02` supervised valid-GNSS demo

重要:

- `assets/outdoor-demo/mcd-tuhh-day04-supervised.splat` は diagnostic artifact として存在してもよいが、production picker / benchmark table に追加しない。
- production scene を増やしたら、README table、viewer picker 3 種、preview PNG、hero GIF の source of truth は `docs/scenes-list.json` に揃える。
- Drift は `tests/test_pages_assets.py` が検出する。

## 6. Outdoor 3DGS Track

### 6.1 目的

屋外 robotics data から 3DGS を作り、Pages viewer で公開できる `.splat` / bundle にする。

### 6.2 対応済み

- Autoware bag 系 supervised pipeline。
- DUSt3R / MASt3R pose-free preprocessing。
- MCD rosbag image / lidar / IMU / GNSS extraction。
- MCD static calibration downloader と TF handling。
- MCD GNSS zero guard。
- MCD CameraInfo 欠落時の PINHOLE 合成。
- MCD single-camera colorize / sparse depth export。
- IMU orientation CSV normalization。
- Angular-velocity yaw fallback。
- External SLAM artifact import facade。
- VGGT-SLAM 2.0 / MASt3R-SLAM / Pi3X comparison splat 実走。
- Pi3 / LoGeR profile / resolver candidate patterns。
- Pi3 / LoGeR smoke は archive に記録済み。
- README / Pages launch-kit / docs assets 整理。

### 6.3 未完

| Priority | Task | 状態 |
| --- | --- | --- |
| A | BYO photos / CoVLA mini 自己実証 | 外部入力待ち。ユーザ写真または HF access 承認が必要。 |
| A | 8-scene viewer smoke 継続運用 | `docs/scenes-list.json` source of truth 化済み。pre-PR で `pytest tests/test_pages_assets.py -q` を通す。 |
| B | Waymo 実データ E2E | code path / prereq script はあるが、実データと Python 3.10 環境が必要。 |
| B | Pi3 / LoGeR comparison production asset | Smoke は済み。production quality の full run は未実施。 |
| C | `ntu_day_02` quality push | `scripts/plan_mcd_quality_runs.py` と collector はある。実データ再実走は未実施。 |
| C | depth / appearance / sky の比較評価 | `outdoor-training-features` experiment lab はある。real metric run は未実施。 |

### 6.4 Outdoor 実装の読みどころ

| Area | Files | Notes |
| --- | --- | --- |
| MCD calibration / static TF | `src/gs_sim2real/datasets/ros_tf.py`, `scripts/download_mcd_calibration.sh` | MCDVIRAL official calibration YAML を downloader 経由で取得。YAML は CC BY-NC-SA なので repo に commit しない。 |
| MCD supervised sparse import | `src/gs_sim2real/cli.py`, `src/gs_sim2real/datasets/mcd.py` | `--mcd-static-calibration`、single-camera colorize/depth、CameraInfo 欠落時 PINHOLE 合成、zero-GNSS guard、IMU yaw fallback。 |
| MCD quality run planning | `src/gs_sim2real/experiments/mcd_quality_plan.py`, `scripts/plan_mcd_quality_runs.py`, `scripts/collect_mcd_quality_runs.py` | `ntu_day_02` baseline / single-camera BA / multi-camera BA の commands と summary。 |
| External SLAM import | `src/gs_sim2real/preprocess/external_slam.py`, `src/gs_sim2real/preprocess/external_slam_artifacts/` | facade + profile/resolver/materializer/importer/manifest 分割。artifact 未配置でも structured error manifest を出す。 |
| External SLAM planning | `scripts/plan_external_slam_imports.py`, `scripts/collect_external_slam_imports.py` | MASt3R-SLAM / VGGT-SLAM / Pi3 / LoGeR の dry-run gate matrix と collector。 |
| Outdoor feature comparison | `src/gs_sim2real/experiments/outdoor_training_features_lab.py` | depth supervision、appearance embedding、pose refinement、sky-mask profile 比較。 |
| Pages scene contract | `docs/scenes-list.json`, `scripts/pages_scene_manifest.py`, `tests/test_pages_assets.py` | README table、preview capture、hero GIF、viewer picker を manifest に揃える。 |
| README preview capture | `scripts/capture_readme_splat_previews.py` | WebGL は headed Chromium 推奨。headless では黒 canvas になることがある。 |
| Hero GIF | `scripts/record_demo_gif.py` | `docs/scenes-list.json` の production scenes を順に cycle する。 |

## 7. External SLAM Track

### 7.1 現行方針

MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / LoGeR を直接 dependency として repo に抱えない。各 front-end は repo 外で実行し、出力された trajectory / pose tensor / point cloud を GS Mapper に渡す。

GS Mapper 側の責務:

- candidate artifact path を探索する。
- TUM / KITTI / NMEA / tensor pose を一時 trajectory に materialize する。
- point tensor / PLY / PCD / NPY を point cloud として読む。
- image directory と pose count を align する。
- dry-run manifest に selected artifact、candidate trace、missing reason、gate result を残す。
- COLMAP sparse に変換し、既存 training path に渡す。

### 7.2 Profile 状態

| System | 状態 | Notes |
| --- | --- | --- |
| MASt3R-SLAM | production comparison 実走済み | `bag6-mast3r-slam-20-15k.splat` |
| VGGT-SLAM 2.0 | production comparison 実走済み | `bag6-vggt-slam-20-15k.splat` |
| Pi3 / Pi3X | production comparison 実走済み | `bag6-pi3x-20-15k.splat`。camera_poses tensor + dense points/colors/confidence を `external-slam` importer で materialize。 |
| LoGeR | smoke 済み、profile 候補追加済み | `--output_txt` trajectory と `.pt` artifact 候補。production asset は未実走。 |

### 7.3 Dry-run examples

```bash
gs-mapper preprocess --method external-slam --images <images-dir> \
  --external-slam-system vggt-slam --external-slam-output <slam-output-dir> \
  --external-slam-dry-run --external-slam-manifest-format json \
  --external-slam-fail-on-dry-run-gate
```

```bash
python3 scripts/plan_external_slam_imports.py --format markdown
python3 scripts/plan_external_slam_imports.py --format shell
python3 scripts/collect_external_slam_imports.py --format markdown
```

## 8. Physical AI Simulation Track

### 8.1 North Star

GS Mapper を「3DGS demo generator」で止めず、Physical AI policy を検証できる simulation environment にする。

最小の完成形:

1. Real outdoor scene を 3DGS asset として持つ。
2. Scene metadata、bounds、sensor rig、coordinate frame を stable JSON contract として持つ。
3. Headless environment が pose / observation / collision / reward を返す。
4. Route policy baseline と imitation policy を同じ benchmark interface で評価できる。
5. Scenario matrix を生成し、CI shard で実行できる。
6. Workflow 生成から review bundle まで、自動化の各段階を小さく検証できる。

### 8.2 既存モジュール

| Module | Role |
| --- | --- |
| `contract.py` | `SimulationCatalog`, `SceneEnvironment`, `SensorRig`, `TrajectoryEpisode` などの contract。 |
| `interfaces.py` | `PhysicalAIEnvironment`, `Observation`, `AgentAction`, `Pose3D`, `TrajectoryScore`。 |
| `headless.py` | Headless environment。bounds / occupancy / trajectory scoring。 |
| `gym_adapter.py` | Route policy を gym-like interface で動かす adapter。 |
| `occupancy.py` | LiDAR observation から occupancy grid を作る utility。 |
| `costmap.py` | Collision query summary。 |
| `footprint.py` | Robot footprint。point collision ではなく body radius / height を見る。 |
| `planning.py`, `route_planning.py` | occupancy planning / candidate route / replanning。 |
| `observation_renderer.py`, `splat_renderer.py` | observation / splat render integration。 |

### 8.3 Policy benchmark modules

| Module | Role |
| --- | --- |
| `policy_dataset.py` | Route policy dataset collection / JSON / transitions JSONL。 |
| `policy_imitation.py` | Imitation model / action decoder / fit / evaluation。 |
| `policy_feedback.py` | Observation / reward / sample building。 |
| `policy_quality.py` | Dataset quality / baseline evaluation。 |
| `policy_replay.py` | Replay batches / feature schema / transition table。 |
| `policy_benchmark.py` | Goal suite / registry / benchmark report。 |
| `policy_benchmark_history.py` | Benchmark snapshots / regression gates / history report。 |

### 8.4 まだ弱いところ

| Area | 課題 |
| --- | --- |
| Observation realism | 現在は lightweight contract 中心。camera image / depth / splat render の統合を強める必要がある。 |
| Dynamics | Headless env は policy evaluation の最小実装。real robot dynamics / latency / actuation constraints は薄い。 |
| Sensor noise | Pose / goal position / heading は `RoutePolicySensorNoiseProfile` で scenario config に落ちた。LiDAR / camera / IMU raw noise はまだ扱っていない。 |
| Multi-agent / moving obstacles | 単一の moving obstacle は `DynamicObstacleTimeline` で scenario config に入った (`step_index` に対して線形補間する waypointed sphere)。gym adapter の feature dict に `dynamic-obstacle-count` / `nearest-dynamic-obstacle-distance-meters` / `nearest-dynamic-obstacle-bearing-radians / -x / -y` を追加し、learned policy が signal を拾えるように。Multi-agent 相互作用 / reactive policy 側の連携は今後。 |
| Real benchmark correlation | 実機 / rosbag replay と sim benchmark の相関検証は未実施。 |

## 9. Route Policy Scenario CI Pipeline

この chain が 2026-04-23 時点の最重要な進捗です。巨大な benchmark を一発で回すのではなく、設定生成、sharding、CI workflow 生成、検証、activation、review publishing を分割します。

### 9.1 Pipeline overview

```text
registry + scenes + goal suites + configs
  -> scenario matrix
  -> scenario sets
  -> shard plan
  -> shard run JSONs
  -> shard merge report + history gate
  -> CI manifest
  -> generated workflow YAML (manual-only)
  -> workflow validation report
  -> workflow activation report (manual-only active path)
  -> Pages review bundle
  -> trigger promotion report
  -> trigger-enabled adoption (re-materialize + re-validate + re-activate to a distinct active path)
```

### 9.2 Modules

| Stage | Module | CLI | Output |
| --- | --- | --- | --- |
| Scenario set execution | `policy_scenario_set.py` | `route-policy-scenario-set` | scenario-set run JSON / Markdown |
| Matrix expansion | `policy_scenario_matrix.py` | `route-policy-scenario-matrix` | scenario matrix expansion JSON |
| Sharding | `policy_scenario_sharding.py` | `route-policy-scenario-shards` | shard plan JSON / shard scenario-set files |
| Shard merge | `policy_scenario_sharding.py` | `route-policy-scenario-shard-merge` | shard merge JSON / history JSON |
| CI manifest | `policy_scenario_ci_manifest.py` | `route-policy-scenario-ci-manifest` | CI manifest JSON |
| Workflow materialization | `policy_scenario_ci_workflow.py` | `route-policy-scenario-ci-workflow` | generated YAML / workflow index JSON |
| Workflow validation | `policy_scenario_ci_workflow.py` | `route-policy-scenario-ci-workflow-validate` | validation JSON / Markdown |
| Workflow activation | `policy_scenario_ci_activation.py` | `route-policy-scenario-ci-workflow-activate` | activation JSON / Markdown / active workflow YAML |
| Review publishing | `policy_scenario_ci_review.py` | `route-policy-scenario-ci-review` | review JSON / Markdown / HTML bundle |
| Workflow trigger promotion | `policy_scenario_ci_promotion.py` | `route-policy-scenario-ci-workflow-promote` | promotion JSON / Markdown |
| Trigger-enabled adoption | `policy_scenario_ci_adoption.py` | `route-policy-scenario-ci-workflow-adopt` | adoption JSON / Markdown / adopted YAML under `.github/workflows/<id>-adopted.yml` |

### 9.3 Important contracts

- `RoutePolicyScenarioCIManifest` は shard jobs と merge job を構造化する。
- `RoutePolicyScenarioCIWorkflowMaterialization` は generated YAML と config を保持する。
- `RoutePolicyScenarioCIWorkflowValidationReport` は YAML parse / text checks / payload checks / manifest consistency を保持する。
- `RoutePolicyScenarioCIWorkflowActivationReport` は validation PASS、source path、destination path、content equality、overwrite を gate 化する。
- `RoutePolicyScenarioCIReviewArtifact` は shard merge / validation / activation を Pages 向け review bundle にまとめる。
- `RoutePolicyScenarioCIWorkflowPromotionReport` は review PASS、history PASS、review URL、trigger mode、allowed branches を gate 化する。
- `RoutePolicyScenarioCIWorkflowAdoptionReport` は promotion PASS、manifest / workflow id 一致、manual path と distinct な adopted active path、adopted YAML の trigger block / branch literal 出力、再 validation / activation の PASS を gate 化する。
- `RoutePolicyScenarioCIReviewAdoption` は review artifact の任意 sub-record で、adoption id / trigger mode / adopted active path / push・pull request branches / manual vs adopted YAML の unified diff を Pages 向けに保持する。review の `passed` gate 自体は変えず、purely additive presentation。

### 9.4 Example commands

Scenario matrix:

```bash
gs-mapper route-policy-scenario-matrix \
  --matrix path/to/matrix.json \
  --output-dir runs/scenarios/generated \
  --output runs/scenarios/matrix-expansion.json \
  --markdown-output runs/scenarios/matrix-expansion.md
```

Shard plan:

```bash
gs-mapper route-policy-scenario-shards \
  --expansion runs/scenarios/matrix-expansion.json \
  --output-dir runs/scenarios/shards \
  --max-scenarios-per-shard 4 \
  --shard-plan-id outdoor-demo-shards \
  --index-output runs/scenarios/shard-plan.json \
  --markdown-output runs/scenarios/shard-plan.md
```

Shard merge:

```bash
gs-mapper route-policy-scenario-shard-merge \
  --run runs/scenarios/ci/runs/shard-001.json \
  --run runs/scenarios/ci/runs/shard-002.json \
  --merge-id outdoor-demo-shard-merge \
  --output runs/scenarios/ci/shard-merge.json \
  --history-output runs/scenarios/ci/shard-history.json \
  --history-markdown-output runs/scenarios/ci/shard-history.md \
  --fail-on-regression
```

CI manifest:

```bash
gs-mapper route-policy-scenario-ci-manifest \
  --shard-plan runs/scenarios/shard-plan.json \
  --manifest-id outdoor-demo-ci \
  --report-dir runs/scenarios/ci/reports \
  --run-output-dir runs/scenarios/ci/runs \
  --history-output-dir runs/scenarios/ci/histories \
  --merge-id outdoor-demo-shard-merge \
  --merge-output runs/scenarios/ci/shard-merge.json \
  --merge-history-output runs/scenarios/ci/shard-history.json \
  --cache-key-prefix outdoor-demo-policy \
  --fail-on-regression \
  --output runs/scenarios/ci-manifest.json \
  --markdown-output runs/scenarios/ci-manifest.md
```

Workflow materialization:

```bash
gs-mapper route-policy-scenario-ci-workflow \
  --manifest runs/scenarios/ci-manifest.json \
  --workflow-id outdoor-demo-policy-shards \
  --workflow-name "Outdoor Demo Policy Shards" \
  --artifact-root runs/scenarios/ci \
  --workflow-output .github/workflows/outdoor-demo-policy-shards.generated.yml \
  --index-output runs/scenarios/ci-workflow.json \
  --markdown-output runs/scenarios/ci-workflow.md
```

Workflow validation:

```bash
gs-mapper route-policy-scenario-ci-workflow-validate \
  --manifest runs/scenarios/ci-manifest.json \
  --workflow-index runs/scenarios/ci-workflow.json \
  --workflow .github/workflows/outdoor-demo-policy-shards.generated.yml \
  --output runs/scenarios/ci-workflow-validation.json \
  --markdown-output runs/scenarios/ci-workflow-validation.md \
  --fail-on-validation
```

Workflow activation:

```bash
gs-mapper route-policy-scenario-ci-workflow-activate \
  --workflow-index runs/scenarios/ci-workflow.json \
  --validation-report runs/scenarios/ci-workflow-validation.json \
  --workflow .github/workflows/outdoor-demo-policy-shards.generated.yml \
  --active-workflow-output .github/workflows/outdoor-demo-policy-shards.yml \
  --output runs/scenarios/ci-workflow-activation.json \
  --markdown-output runs/scenarios/ci-workflow-activation.md \
  --fail-on-activation
```

Review bundle:

```bash
gs-mapper route-policy-scenario-ci-review \
  --shard-merge runs/scenarios/ci/shard-merge.json \
  --validation-report runs/scenarios/ci-workflow-validation.json \
  --activation-report runs/scenarios/ci-workflow-activation.json \
  --review-id outdoor-demo-policy-review \
  --pages-base-url https://rsasaki0109.github.io/gs-mapper/reviews/outdoor-demo-policy/ \
  --bundle-dir docs/reviews/outdoor-demo-policy \
  --fail-on-review
```

Workflow promotion:

```bash
gs-mapper route-policy-scenario-ci-workflow-promote \
  --review runs/scenarios/ci-review.json \
  --review-url https://rsasaki0109.github.io/gs-mapper/reviews/outdoor-demo-policy/ \
  --trigger-mode pull-request \
  --pull-request-branch main \
  --output runs/scenarios/ci-workflow-promotion.json \
  --markdown-output runs/scenarios/ci-workflow-promotion.md \
  --fail-on-promotion
```

### 9.5 Current next step: promotion-backed workflow adoption

目的:

- Promotion report が PASS したあとに、trigger-enabled workflow を再 materialize / validate / activate する手順を固定する。
- tiny fixture で matrix expansion から promotion までを一周する smoke recipe を追加する。
- adoption 手順は active workflow YAML を直接 mutation せず、manual-only workflow と trigger-enabled workflow の差分が review できる形にする。

実装済み API:

```python
promotion = promote_route_policy_scenario_ci_workflow(
    review_artifact,
    trigger_mode="pull-request",
    pull_request_branches=("main",),
    review_url="https://rsasaki0109.github.io/gs-mapper/reviews/outdoor-demo-policy/",
)
write_route_policy_scenario_ci_workflow_promotion_json(
    "runs/scenarios/ci-workflow-promotion.json",
    promotion,
)
```

Promotion checks:

- review artifact が PASS。
- validation が PASS。
- activation が ACTIVE。
- shard merge が PASS。
- history gate が PASS。
- review URL が absolute http(s) URL。
- trigger mode が allowed set。
- trigger mode に必要な branches が空でない。
- branches が literal branch name policy を満たす。
- active workflow path が `.github/workflows/*.yml` / `.yaml` に閉じている。

### 9.6 Scenario CI smoke recipe

`scripts/smoke_route_policy_scenario_ci.py` が tiny 1-scene / 1-policy fixture で `scenario matrix -> shard plan -> scenario-set run -> shard merge -> CI manifest -> workflow materialization -> validation -> activation -> review -> promotion -> adoption` を一周する。各 gate に `[PASS]/[FAIL] <name>` を出し、落ちた gate で `GateFailure` を上げて non-zero exit する。

狙い:

- chain 全体の integration smoke を、巨大 E2E ではなく 1 分未満で回せる形にする。
- workflow activation / adoption は `<tmpdir>/.github/workflows/...` に閉じ、repo 本物の `.github/workflows/` には触らない。
- review bundle / promotion / adoption report の JSON / Markdown / HTML を tmpdir に吐き、目視レビューしたいときは `--keep` / `--root <path>` で保持できる。

回帰検出:

- `tests/test_smoke_route_policy_scenario_ci.py` が `run_smoke()` を importlib で叩き、全 gate の PASS ログ、artifact path、promotion trigger config、manual vs adopted YAML の差分 (`workflow_dispatch` のみ vs `pull_request:` 追加) を snapshot-assert する。

### 9.7 Promotion-backed trigger adoption

`adopt_route_policy_scenario_ci_workflow` が promotion report PASS を受けて、manual-only workflow YAML を触らずに trigger-enabled 版を別ファイルとして生成する。

- 入力: `RoutePolicyScenarioCIWorkflowPromotionReport`、同じ `RoutePolicyScenarioCIManifest`、manual-only の materialization。
- 出力: `.github/workflows/<id>-adopted.yml`（活性化された trigger-enabled YAML）、`ci-workflow-adoption.json`（gate report）、同 Markdown レンダリング。
- 失敗時は materialize も write もせずに blocked report を返すので、manual path を絶対に上書きしない。
- Gate: `promotion-promoted`, `manifest-id`, `workflow-id`, `adopted-path-distinct-from-manual`, `adopted-source-path-distinct`, trigger block (`workflow-dispatch-retained`, `push-trigger-emitted`, `pull-request-trigger-emitted`), per-branch literal check (`push-branch:<name>` / `pull-request-branch:<name>`), `adopted-validation-passed`, `adopted-activation-active`。

CLI surface は `gs-mapper route-policy-scenario-ci-workflow-adopt` として追加済み。manifest / workflow index / promotion JSON と adopted source / active path を渡せば同じ gate を経由する。

### 9.8 Adoption-aware review bundle

review bundle は adoption の結果を任意で取り込める。`build_route_policy_scenario_ci_review_artifact(..., adoption=RoutePolicyScenarioCIReviewAdoption)`、または CLI の `--adoption-report` を渡すと、以下を追加で Pages に出す:

- `adoption` sub-record に adoption_id / trigger_mode / adopted active path / push・pull_request branches を埋める。
- manual-only と adopted YAML の unified diff (`difflib.unified_diff`) を `workflow_diff` として保持。
- Markdown renderer は `## Adopted Workflow` セクション + \`\`\`diff ブロックを追加。
- HTML renderer は "Adopted Workflow" セクションに trigger mode / branches / 色分け diff (`<pre class="diff">` + add / del / hunk span) を描く。

review の `passed` gate 自体は shard merge / validation / activation / history のままで変わらない。adoption は purely additive presentation。

smoke script は promotion + adoption 完了後に review を再 build して bundle を上書きするので、`<tmpdir>/pages/<review-id>/review.{json,md,html}` は最終 run で adoption 情報入りになる。

Pages `docs/reviews/` index は `scripts/build_pages_reviews_index.py` で生成済み。`docs/reviews/index.html` / `docs/reviews/index.json` は、review bundle が未公開でも "no review bundles published yet" placeholder を出すので Pages `/reviews/` が 404 にならない。公開済み bundle が増えたら次のコマンドで index を再生成する:

```bash
PYTHONPATH=src python3 scripts/build_pages_reviews_index.py \
  --reviews-dir docs/reviews \
  --html-output docs/reviews/index.html \
  --json-output docs/reviews/index.json
```

Public sample として `docs/reviews/smoke-route-policy-ci/` を生成する場合は次を使う。これは `scripts/smoke_route_policy_scenario_ci.py` の synthetic fixture から作るため production benchmark ではないが、review bundle / adoption diff / index discovery の Pages contract を実物として確認できる。

```bash
PYTHONPATH=src python3 scripts/build_pages_sample_review_bundle.py
```

生成後の bundle は `review.json` / `review.md` / `index.html` と、リンク先の `sample-artifacts/` を含む。`/tmp/...` や `https://example.test/...` は commit しない。

Production benchmark run を Pages `/reviews/` に公開する場合は `scripts/publish_production_review_bundle.py` を使う。`route-policy-scenario-ci-review` が外部の production 実行で吐いた `review.json`(`provenance.kind="production"`)を `docs/reviews/<bundle-id>/` に bundle 化し、`docs/reviews/index.{html,json}` も再生成する。

```bash
PYTHONPATH=src python3 scripts/publish_production_review_bundle.py \
  --review-json runs/outdoor-demo/ci-review.json \
  --bundle-id outdoor-demo-direct-baseline-001
```

`provenance.kind` が `production` 以外だと exit 2 で reject される。bundle id は lowercase kebab-case のみ。

## 10. Public / Launch Track

### 10.1 現状

- README に CI / Pages badge がある。
- GitHub Pages live demo がある。
- `docs/index.html` は GS Mapper の public landing として整備済み。
- `docs/launch-kit.md` / `docs/launch-kit.json` に external announcement 素材がある。
- README 冒頭に MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / LoGeR updates への star/watch callout がある。

### 10.2 Star を増やすために効く方向

コード機能よりも「初見で何がすごいか分かる」ことが重要。

優先順:

1. README top の live demo preview を安定させる。
2. External SLAM comparison table を維持する。
3. `docs/launch-kit.md` の copy を短くする。
4. Pi3 / LoGeR production comparison asset を足す。
5. Review bundle を Pages に出して、CI / benchmark の信頼性を見せる。
6. 使い方を `photos-to-splat` / `external-slam import` / `physical-ai benchmark` の 3 入口に分ける。

### 10.3 ただし今の主目的

「告知機能」だけを作りすぎない。現在の主目的は Physical AI simulation environment の品質を上げること。外向けの整備は、実装された実体を見せるためにやる。

## 11. Verification Commands

### 11.1 通常 pre-PR

```bash
ruff format --check src/ tests/ scripts/
ruff check src/ tests/ scripts/
PYTHONPATH=src pytest tests/ -q --ignore=tests/e2e
```

現行環境では `python` がない場合があるので `python3` を使う。

### 11.2 Full local validation

```bash
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
mypy src/gs_sim2real/sim/policy_scenario_ci_review.py \
  src/gs_sim2real/sim/policy_scenario_ci_activation.py \
  src/gs_sim2real/sim/policy_scenario_ci_promotion.py \
  src/gs_sim2real/sim/policy_scenario_ci_workflow.py \
  src/gs_sim2real/sim/__init__.py
python3 -m compileall -q src/gs_sim2real tests
pytest -q
git diff --check
```

`src/gs_sim2real/cli.py` を含む mypy full pass は、現状では Waymo / MCD loader 周辺の既知型エラーが残っている。scenario CI slice の型確認は module 単位で切る。

### 11.3 Outdoor / Pages まわり

```bash
PYTHONPATH=src pytest \
  tests/test_pages_assets.py \
  tests/test_viewer.py \
  tests/test_mcd.py \
  tests/test_mcd_gnss_preflight.py \
  tests/test_external_slam.py \
  -q
```

Viewer assets だけなら:

```bash
PYTHONPATH=src pytest tests/test_pages_assets.py tests/test_viewer.py -q
```

### 11.4 Physical AI / scenario CI まわり

```bash
pytest tests/test_physical_ai_policy_benchmark.py tests/test_cli.py -q
```

絞り込み:

```bash
pytest tests/test_physical_ai_policy_benchmark.py -q -k "scenario_ci_workflow"
pytest tests/test_physical_ai_policy_benchmark.py -q -k "scenario_ci_review"
pytest tests/test_cli.py -q -k "scenario_ci"
```

### 11.5 Preview / GIF

README preview PNG:

```bash
export DISPLAY=:0
python3 scripts/capture_readme_splat_previews.py
python3 scripts/enhance_demo_sweep_previews.py --hero-gif
```

Hero GIF:

```bash
python3 scripts/record_demo_gif.py
python3 scripts/enhance_demo_sweep_previews.py --hero-gif
```

### 11.6 MCD quality planning

```bash
python3 scripts/check_mcd_gnss.py <session-dir> --gnss-topic /vn200/GPS
python3 scripts/plan_mcd_quality_runs.py --format markdown
python3 scripts/collect_mcd_quality_runs.py --format markdown
python3 scripts/collect_mcd_quality_runs.py --format benchmark
python3 scripts/collect_mcd_quality_runs.py --format gate --fail-on-gate
```

## 12. Backlog

### 12.1 A: Immediate next

新しい優先順位は §17 Roadmap に集約済み。本セクションは status トラッキング用に残す。

| Task | Why | Suggested slice |
| --- | --- | --- |
| Review bundle sample under docs | 完了。Pages `/reviews/` が空ではなく scenario CI review / adoption diff の形を見せられる | `docs/reviews/smoke-route-policy-ci/` を `scripts/build_pages_sample_review_bundle.py` で生成。synthetic smoke fixture であり production benchmark ではないことを bundle 内に明示。 |
| Real review bundle from production scenario CI | **進行中 (Sprint 1 / §17.2)**。GPT pro consultation で synthetic vs production の区別を `RoutePolicyScenarioCIReviewProvenance` で first-class 化し、`docs/reviews/<run-id>/` に production bundle を公開する PR A + PR A2 に分割。 | PR A: contract / CLI / Pages index 骨格 (`plan_review_bundle_provenance.md §1`). PR A2: 実 production run の `gs-mapper route-policy-scenario-ci-review --kind production --bundle-dir docs/reviews/<id>` 実行と index 再生成。 |

### 12.2 B: Physical AI env hardening

| Task | Status (2026-04-26) |
| --- | --- |
| Observation renderer integration | ✅ 完了。`RoutePolicyGymAdapter` の feature dict に IMU 7 軸 (#122) と peer-aware obstacle features (#123) を surface。残課題は scene bundle 側の input sensor を増やすこと (depth / LiDAR fan-out) — このセッション以降の別チケット。 |
| Sensor noise profiles (raw sensors) | ✅ 完了。env-side noise + IMU kinematic finite-diff renderer (#111) が実装され、gym adapter feature dict に流れる (#122) ので route policy benchmark から observation 経由で σ が乗る。physics / rosbag-replay 由来の IMU renderer は引き続き OOS。 |
| Dynamic obstacles (multi-agent) | ✅ 完了。`ObstaclePolicy` protocol + 4 reference impls (#112)、env / gym adapter に per-step peer cache (#123/#124/#127)、`MaintainSeparationObstaclePolicy` 等の policy obstacle が rollout 中に peer を観測可能。残課題は Pi3-style 大規模 multi-agent scenario の production 配信 — Tier 3 候補。 |
| Route policy replay viewer | 引き続き OOS。Policy trajectory と scene を Pages で inspect する viewer は未着手。 |
| Real-vs-sim correlation report | ✅ 完了。`scripts/run_rosbag_correlation.py` (#113/#115) → scenario-set run report への attach (#121) → review bundle への surface + regression gate (#125/#126) → per-bag overrides (#128) → translation/heading per-pair distribution + time stratification (#129〜#134) まで実装済み。`gs-mapper route-policy-scenario-ci-review --max-correlation-* --correlation-thresholds-config --correlation-pair-distribution-strata` が production rollout で使える。残課題は event-aligned stratification (#133 OOS、外部 event timestamp が必要)。 |

### 12.3 B: Outdoor asset quality

| Task | Status (2026-04-26) |
| --- | --- |
| Pi3 production comparison | 完了。Pi3X VO 20 frames → `external-slam` import → gsplat 15k → `docs/assets/outdoor-demo/bag6-pi3x-20-15k.splat`。 |
| LoGeR production comparison | 引き続き OOS。External SLAM comparison の説得力が増す。要 GPU run。 |
| MCD `ntu_day_02` quality reruns | 部分完了。`single_400_depth_long` (L1=0.1951) と `single_800_ba` (L1=0.2699) は gate pass。`multi_3cam_300each_ba` は手元の bag に `d455t` / `d435i` topics が無く data-blocked、要 MCDVIRAL の追加 download。 |
| Waymo E2E | high-value だが dataset access と env blocker がある。 |

#### 12.3.1 MCD quality gate targets

Production rerun は `scripts/collect_mcd_quality_runs.py --format gate --fail-on-gate` が通る状態を目標にする。Gate 本体は `src/gs_sim2real/experiments/mcd_quality_gate.py` の `MCDQualityGatePolicy` で、default は:

| Check | Default threshold | Notes |
| --- | --- | --- |
| `artifacts` | `require_complete_artifacts=True` | plan の `expected_artifacts` が全部そろっている |
| `frames` | `min_frame_fraction=0.95` | 取れた image 数 / planned `max_frames` |
| `depth` | `min_depth_fraction=0.95` | depth map 数 / image 数 (depth export 有効時) |
| `registered` | `min_registered_fraction=0.90` | COLMAP `images.txt` の登録行数 / image 数 |
| `sparse_points` | `min_sparse_points=1` | `points3D.txt` の行数下限 |
| `trained_gaussians` | `min_trained_gaussians=1` | `point_cloud.ply` の vertex 数 |
| `splat_gaussians` | `min_splat_gaussians=1` | `.splat` byte / 32 |
| `final_l1` | `require_final_l1=True` | train log に final L1 が残っている |
| `final_l1_max` | `max_final_l1=None` | 数値上限が必要なときだけ set する |

`ntu_day_02` rerun profile (`ntu_day02_single_400_depth_long` / `ntu_day02_single_800_ba` / `ntu_day02_multi_3cam_300each_ba`) は `scripts/plan_mcd_quality_runs.py` が生成。production 実行後は上記 gate を全 profile で満たす ことが完了条件。`max_final_l1` は baseline run の実測が出るまで `None` のままにしておく (regression guard として後から絞る)。

### 12.4 C: Public launch polish

| Task | Why |
| --- | --- |
| Launch kit cleanup | Star を増やすには短い copy と画像が必要。Env-hardening (pose + raw sensor noise / multi-agent dynamic obstacles) を technical / community copy に反映、Physical AI docs link + topics (`gsplat` / `scenario-ci` / `route-policy-benchmark`) 追加済み。残りは実スクリーンショット / 動画素材の差し替え。 |
| Demo preview refresh | 完了。`scripts/enhance_demo_sweep_previews.py` で 9 production preview PNG を 1280x720 のまま foreground crop / punch-up し、`hero.gif` も production scene preview 由来の軽量 loop に更新。Pages landing は Outdoor GS capability / production scene wall を前面化済み。 |

## 13. Scope Boundaries

- Python package path `gs_sim2real` は compatibility のため維持する。屋外 pipeline work の一部として rename しない。
- Legacy `gs-sim2real` CLI alias は dedicated deprecation pass まで残す。
- Downloaded MCD calibration YAML、rosbag data、Waymo tfrecords、generated training outputs は commit しない。
- External SLAM implementation 本体を repo に vendor しない。artifact importer だけを持つ。
- `docs/splat-viewer/main.js` など vendored viewer code は、compatibility fix 以外で大きく触らない。
- Generated workflow は直接 `.github/workflows/` に置かず、validation / activation / review flow を通す。
- `docs/scenes-list.json` の production scene 追加は README / viewer / tests とセットで扱う。

## 14. 既知の落とし穴

- MCD topic は `/vn200/GPS` の大文字 `GPS`。`/vn200/gps` ではない。
- `tuhh_day_04` の `/vn200/GPS` は all-zero。supervised GNSS demo には使わない。
- MCD calibration YAML は公式 Download page から取得できるが、license 上 repo に YAML を commit しない。
- IMU orientation CSV は zero-length / non-finite quaternion を無視し、全 identity のときだけ姿勢なし扱いにする。一定の non-identity mount orientation は有効な姿勢として残す。
- Orientation が全 identity でも `angular_velocity_z` が非ゼロなら yaw-only fallback として積分する。
- `capture_readme_splat_previews.py` は headless だと WebGL canvas が黒になることがある。CI では静的 contract test、実 capture は headed smoke。
- Waymo は code path があっても実データ E2E 未検証。Python 3.10 venv と dataset agreement を先に確認する。
- Review bundle は CI workflow の信頼性を示す artifact であり、benchmark の実行そのものを代替しない。
- Activation report の `activated=True` は workflow file が guardrail を通ったという意味。GitHub 上で workflow が成功したという意味ではない。

## 15. Archive Map

古い詳細は [archive snapshot](archive/plan_outdoor_gs_2026_04_full_handoff.md) に残しています。

| Need | Archive section |
| --- | --- |
| PR #55〜#80 の時系列 | `## 15`, `## 15.1`, `## 15.2` |
| `tuhh_day_04` supervised 誤判定の詳細 | `## 15.3`, `## 15.4` |
| `ntu_day_02` valid-GNSS 実走値 | `## 15.5` |
| MCD calibration YAML discovery / Drive ID | `## 4.3.3.a`, `## 4.3.3.c`, `## 15.1` |
| 8-scene viewer smoke transcript | `## 15.3` |
| Pi3 / LoGeR smoke details | External SLAM sections near `Pi3X official model` and `LoGeR official reimplementation` |
| Legacy command blocks / one-off output paths | `## 9`, `## 15.*` |

## 16. Related Documents

| File | Role |
| --- | --- |
| `README.md` | Public-facing overview, live demo, benchmark table |
| `CONTRIBUTING.md` | Development workflow and issue/PR expectations |
| `docs/physical-ai-sim.md` | Physical AI simulation contract and route policy benchmark docs |
| `docs/experiments.md` | Public experiment-process index |
| `docs/experiments.generated.md` | Generated detailed experiment comparison tables |
| `docs/decisions.md` | Accepted/deferred design decisions |
| `docs/interfaces.md` | Stable interfaces that production code may depend on |
| `docs/launch-kit.md` | Public announcement / launch material |
| `docs/plan_review_bundle_provenance.md` | PR A / PR B (production review bundle provenance + event-aligned stratification) の contract 差分メモ |
| `docs/archive/plan_outdoor_gs_2026_04_full_handoff.md` | Full historical outdoor-GS handoff snapshot |

## 17. Roadmap (2026-05〜, GPT pro consultation 後)

2026-05-15 の GPT pro consultation の結論を 4 sprint に分解。優先順位の根拠は「実屋外 scene を使った Physical AI policy evaluation が CI artifact として継続的に出ること」が現状の最大の説得力ボトルネックである、という判断。詳細な contract 差分は [`plan_review_bundle_provenance.md`](plan_review_bundle_provenance.md)。

### 17.1 優先順位

| 優先 | Sprint | Task | 狙い | 状態 |
| ---: | --- | --- | --- | --- |
| 1 | Sprint 1 | Real production review bundle 公開 (PR A + PR A2) | 外向け説得力 | 未着手（本書 §17.2） |
| 2 | Sprint 2 | Event-aligned stratification (PR B) | 評価品質 / policy 行動レベル correlation の土台 | 未着手（本書 §17.3） |
| 3 | Sprint 3 | Policy trace events (PR C) | デバッグ・説明力、Sprint 4 viewer の入力 | 未着手（本書 §17.4） |
| 4 | Sprint 4 | Multi-agent Tier 3 production matrix (PR D 系) | env hardening、Tier 3 候補の本丸 | 未着手（本書 §17.5） |
| 5 | — | LoGeR production comparison asset | asset 比較の厚み | §12.3 既存 backlog |
| 6 | — | MCD `ntu_day_02` `multi_3cam_300each_ba` 追加 download | asset 品質補完（data-blocked） | §12.3 既存 backlog |
| 7 | — | Applanix `read_gsof_ins_pose_stream` | data input 拡張（vendor 依存） | §3.1 OOS |

「splat を 1 個増やす」より「実屋外 scene を使った policy evaluation が CI artifact として継続的に出る」方が GS Mapper の現在地では効く、という判断。

### 17.2 Sprint 1: production-review-bundle-manifest

Goal: production benchmark run が `RoutePolicyScenarioCIReviewArtifact` として first-class に区別され、Pages `/reviews/` index で synthetic / production が一目で分かる。

主な追加:

- `RoutePolicyScenarioCIReviewProvenance` dataclass（`kind`, `generated_at`, `git_commit`, `scene_id`, `scenario_set_id`, `matrix_hash`, `policy_version`, `env_contract_version`, `correlation_threshold_profile`, `asset_source`, `extra`）。
- `RoutePolicyScenarioCIReviewArtifact.provenance` optional field。`provenance is None` 時は既存 v1 JSON とバイト等価。
- CLI に `--kind {synthetic,production}` 他 9 個のフラグ。`--kind production` は他 provenance フィールドの指定を warning で促す。
- Pages index (`scripts/build_pages_reviews_index.py`) に `Kind` / `Scene` / `Generated` 列、schema v2 へ bump。

PR 分割:

- **PR A**: contract / CLI / Pages index 骨格 + sample bundle を `kind=synthetic` に更新。
- **PR A2**: 実 production scenario run の bundle を `docs/reviews/<run-id>/` に置き、index 再生成。README / Pages landing に導線を追加。

詳細フィールド一覧、後方互換ルール、テストケースは [`plan_review_bundle_provenance.md §1`](plan_review_bundle_provenance.md)。

### 17.3 Sprint 2: event-aligned stratification

Goal: correlation gate を「等間隔時間 window」「等 pair 数 window」に加え「外部 event timestamp window」で評価できる。

主な追加:

- `CorrelationEventWindow` dataclass（`name`, `start_time`, `end_time`, `tags`, `source ∈ {"external","policy_trace"}`）。**`source` を Sprint 2 で先に入れる**ことで Sprint 3 の policy trace event を後方互換で乗せられる。
- `_PAIR_DISTRIBUTION_STRATA_MODES` に `"event-aligned"` 追加。
- `RealVsSimCorrelationThresholds.event_windows_path` optional field。
- fallback chain: event-aligned 指定で windows が読めない / 0 window のときは **explicit fallback** to `equal-pair-count`、review bundle metadata に `correlationStratificationFallback` を立て stderr warning を出す（silent fallback はしない）。
- `RealVsSimCorrelationWindowStats` に optional `event_name` / `event_tags` / `event_source` を追加（未指定なら JSON 出力しない、v1 互換）。

詳細は [`plan_review_bundle_provenance.md §2`](plan_review_bundle_provenance.md)。

### 17.4 Sprint 3: policy trace events

Goal: route / imitation policy が rollout 中に `goal_reached` / `near_obstacle_slowdown` / `collision` / `near_miss` / `route_deviation` 等の event を吐き、Sprint 2 の event-aligned correlation の `source = "policy_trace"` 経路にそのまま流す。

Sprint 2 で schema を先に固めているので、Sprint 3 では event 検出ロジックと bag time 対応付けだけ実装すれば良い。Real-vs-sim 比較は GPT pro 提案の段階化（occurrence → order → timing → event-local pose → segment-level trajectory）で進める。

### 17.5 Sprint 4: multi-agent Tier 3

Goal: seeded multi-agent scenario を production matrix に載せる。既存 `DynamicObstacleTimeline` / `ObstaclePolicy` protocol / per-step peer cache の上に、scenario contract の追加 dimension として `agents` / `population` / `interaction_metrics` を載せる。

scenario matrix の段階拡張:

1. 2-agent deterministic crossing
2. 4-agent route conflict
3. 16-agent seeded population
4. 32+ agent Pi3-style dense

各段階で review bundle / shard merge gate が安定することを確認してから次の規模へ。最初の public scenario は 4〜8 agent 程度で良い（CI / shard / review contract が安定してから Pi3-style dense に拡張する方が安全）。

### 17.6 Sprint 1 完了後に有効化する CI 自動化

| CI | 頻度 | 目的 | 起点 |
| --- | --- | --- | --- |
| PR smoke | every PR | contract drift / unit regression 検出 | 既存 `pytest tests/ -q --ignore=tests/e2e` |
| Nightly production review | scheduled | `outdoor-demo` 系 production scene で scenario CI を回し `docs/reviews/<run-id>-<date>/` を生成 (commit + push) | PR A2 後 |
| Manual promotion | `workflow_dispatch` | Pages に載せたい review bundle を picker から選んで promote | PR A2 後 |

Sprint 1 完了前に nightly を走らせると synthetic と production を区別する手段が無いので、**必ず PR A → PR A2 → CI 自動化** の順で進める。Sprint 1 の `provenance.extra["runTrigger"]` に `nightly` / `manual` / `pr` を入れれば、`kind=production` 内でも nightly / manual / pr を区別できる。
