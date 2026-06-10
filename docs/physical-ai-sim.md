# Physical AI Simulation Contract

GS Mapper is moving from a 3DGS demo repository toward a scene environment layer for Physical AI agents. The first contract is intentionally small: it standardizes what each reconstructed scene exposes before adding a full physics engine or renderer backend.

Generated catalog: [`docs/sim-scenes.json`](sim-scenes.json)

Generator:

```bash
python3 scripts/generate_sim_catalog.py --output docs/sim-scenes.json
```

## Contract

Each `SceneEnvironment` describes:

- `sceneId`, `label`, `summary`, and links to the `.splat`, preview image, and live viewer deep link
- coordinate frame: unit, up axis, gravity vector, handedness, and scale status
- world bounds with `declared`, `estimated`, or `placeholder` confidence
- sensor rig: RGB, depth-proxy, and LiDAR-ray-proxy contracts
- train/eval trajectory episodes
- evaluation tasks for localization, viewpoint planning, waypoint navigation, and mapping coverage

The Python API lives in `gs_sim2real.sim`:

```python
from pathlib import Path

from gs_sim2real.sim import load_simulation_catalog_from_scene_picker

catalog = load_simulation_catalog_from_scene_picker(Path("docs/scenes-list.json"))
scene = catalog.scene_by_id("bag6-mast3r")

print(scene.coordinate_frame.scale_status)
print(scene.viewer_url)
print([task.task_id for task in scene.evaluation_tasks])
```

## Current Readiness

`rgb-forward` is ready through the existing splat viewers and the local `.splat` observation renderer. `depth-proxy` is backed by the same local rasterizer and returns float32 depth plus a validity mask. `lidar-ray-proxy` samples that depth image into LiDAR-like ranges and world-frame points. Those ray points can be converted into a sparse voxel occupancy grid for geometry-aware collision checks.

Metric and estimated-metric scenes expose navigation and mapping tasks. Relative-scale scenes expose localization and viewpoint-planning tasks, but avoid waypoint-navigation scoring until a metric alignment is provided.

## Scene Sources

The simulation catalog is derived from the existing public scene picker at `docs/scenes-list.json`. This keeps the Physical AI environment list synchronized with the nine public bundled splats.

The supervised default `outdoor-demo` scene reads declared bounds from `docs/assets/outdoor-demo/scene.json`. Other bundled splats currently use README trajectory extents as estimated bounds. This is explicit in each `bounds.confidence` value and prevents downstream agents from treating estimated boxes as surveyed collision geometry.

## Environment Interface

Concrete environments should implement `PhysicalAIEnvironment`:

- `reset(scene_id, seed=None)`
- `step(action)`
- `render_observation(request)`
- `query_collision(pose)`
- `sample_goal(scene_id, seed=None)`
- `score_trajectory(scene_id, trajectory)`

Payload dataclasses are available for stable integration:

- `Pose3D`
- `AgentAction`
- `ObservationRequest`
- `Observation`
- `CollisionQuery`
- `TrajectoryScore`

## Headless Backend

`HeadlessPhysicalAIEnvironment` is the first executable backend. It does not simulate dynamics yet; it gives agents a deterministic bounds-based environment that can reset scenes, sample goals, apply simple actions, reject out-of-bounds poses, and return observations. Without an injected renderer, observations stay metadata-only.

```python
from pathlib import Path

from gs_sim2real.sim import (
    AgentAction,
    HeadlessPhysicalAIEnvironment,
    ObservationRequest,
    load_simulation_catalog_from_scene_picker,
)

catalog = load_simulation_catalog_from_scene_picker(Path("docs/scenes-list.json"))
env = HeadlessPhysicalAIEnvironment(catalog)

reset = env.reset("outdoor-demo")
goal = env.sample_goal("outdoor-demo", seed=42)
transition = env.step(AgentAction("twist", {"linearX": 0.5}, duration_seconds=1.0))
collision = env.query_collision(env.state.pose)
observation = env.render_observation(ObservationRequest(pose=env.state.pose, sensor_id="rgb-forward"))
```

For local renderer-backed RGB, depth, and ray proxies, inject `SplatAssetObservationRenderer`. It reads the same bundled `.splat` assets as the public viewer and returns JPEG-backed `rgb-forward` observations, float32 `depth-proxy` observations with validity masks, or LiDAR-like `lidar-ray-proxy` ranges and points.

```python
from pathlib import Path

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    ObservationRequest,
    SplatAssetObservationRenderer,
    SplatRenderConfig,
    load_simulation_catalog_from_scene_picker,
)

docs_root = Path("docs")
catalog = load_simulation_catalog_from_scene_picker(docs_root / "scenes-list.json")
renderer = SplatAssetObservationRenderer(
    docs_root,
    config=SplatRenderConfig(width=320, height=240, far_clip=80.0, point_radius=1),
)
env = HeadlessPhysicalAIEnvironment(catalog, observation_renderer=renderer)

env.reset("outdoor-demo")
observation = env.render_observation(ObservationRequest(pose=env.state.pose, sensor_id="rgb-forward"))
jpeg_base64 = observation.outputs["rgb"]["jpegBase64"]

depth = env.render_observation(
    ObservationRequest(
        pose=env.state.pose,
        sensor_id="depth-proxy",
        outputs=("depth", "validity-mask"),
    )
)
depth_base64 = depth.outputs["depth"]["depthBase64"]
mask_base64 = depth.outputs["validityMask"]["maskBase64"]

lidar = env.render_observation(
    ObservationRequest(
        pose=env.state.pose,
        sensor_id="lidar-ray-proxy",
        outputs=("ranges", "points"),
    )
)
ranges_base64 = lidar.outputs["ranges"]["rangesBase64"]
points_base64 = lidar.outputs["points"]["pointsBase64"]
```

To turn those ray points into lightweight collision geometry, build an occupancy grid and inject it into the headless environment. The backend can query either the pose point or a conservative circular robot footprint.

```python
from gs_sim2real.sim import (
    OccupancyPlanningContext,
    RobotFootprint,
    RouteCandidate,
    RoutePolicyEnvConfig,
    RoutePolicyGoalSpec,
    RoutePolicyGoalSuite,
    RoutePolicyGymAdapter,
    RoutePolicyImitationFitConfig,
    RoutePolicyBenchmarkRegressionThresholds,
    RoutePolicyMatrixConfigSpec,
    RoutePolicyMatrixGoalSuiteSpec,
    RoutePolicyMatrixRegistrySpec,
    RoutePolicyMatrixSceneSpec,
    RoutePolicyQualityThresholds,
    RoutePolicyRegistry,
    RoutePolicyRegistryEntry,
    RoutePolicyScenarioMatrix,
    RoutePolicyScenarioSet,
    RoutePolicyScenarioSpec,
    RoutePolicyScenarioCIReviewArtifact,
    RoutePolicyScenarioCIWorkflowConfig,
    RoutePolicyScenarioCIWorkflowActivationReport,
    RoutePolicyScenarioCIWorkflowPromotionReport,
    RoutePolicyScenarioCIWorkflowValidationReport,
    activate_route_policy_scenario_ci_workflow,
    build_route_policy_scenario_ci_manifest,
    build_route_policy_scenario_ci_review_artifact,
    build_route_policy_scenario_shard_plan,
    build_route_policy_benchmark_history,
    build_occupancy_grid_from_lidar_observation,
    build_route_policy_replay_batch,
    build_route_policy_replay_schema,
    build_route_policy_sample,
    collect_route_policy_dataset,
    evaluate_route_policy_baselines,
    evaluate_route_policy_dataset_quality,
    evaluate_route_policy_imitation_model,
    expand_route_policy_scenario_matrix_to_directory,
    fit_route_policy_imitation_model,
    load_route_policy_benchmark_history_json,
    load_route_policy_goal_suite_json,
    load_route_policy_imitation_model_json,
    load_route_policy_registry_json,
    load_route_policy_scenario_ci_manifest_json,
    load_route_policy_scenario_ci_review_json,
    load_route_policy_scenario_ci_workflow_activation_json,
    load_route_policy_scenario_ci_workflow_json,
    load_route_policy_scenario_ci_workflow_promotion_json,
    load_route_policy_scenario_ci_workflow_validation_json,
    load_route_policy_scenario_matrix_json,
    load_route_policy_scenario_set_json,
    load_route_policy_scenario_set_run_json,
    load_route_policy_scenario_shard_plan_json,
    iter_route_policy_replay_batches,
    load_route_policy_transitions_jsonl,
    materialize_route_policy_scenario_ci_workflow,
    merge_route_policy_scenario_shard_run_jsons,
    promote_route_policy_scenario_ci_workflow,
    replan_after_blocked_rollout,
    render_route_policy_benchmark_history_markdown,
    render_route_policy_benchmark_markdown,
    render_route_policy_quality_markdown,
    render_route_policy_scenario_ci_manifest_markdown,
    render_route_policy_scenario_ci_review_html,
    render_route_policy_scenario_ci_review_markdown,
    render_route_policy_scenario_ci_workflow_activation_markdown,
    render_route_policy_scenario_ci_workflow_markdown,
    render_route_policy_scenario_ci_workflow_promotion_markdown,
    render_route_policy_scenario_ci_workflow_validation_markdown,
    render_route_policy_scenario_matrix_markdown,
    render_route_policy_scenario_set_markdown,
    render_route_policy_scenario_shard_merge_markdown,
    render_route_policy_scenario_shard_plan_markdown,
    run_route_policy_imitation_benchmark,
    run_route_policy_scenario_set,
    validate_route_policy_scenario_ci_workflow,
    rollout_route,
    rollout_route_with_replanning,
    select_best_route,
    write_route_policy_benchmark_history_json,
    write_route_policy_benchmark_report_json,
    write_route_policy_dataset_json,
    write_route_policy_goal_suite_json,
    write_route_policy_imitation_model_json,
    write_route_policy_registry_json,
    write_route_policy_scenario_ci_manifest_json,
    write_route_policy_scenario_ci_review_bundle,
    write_route_policy_scenario_ci_review_json,
    write_route_policy_scenario_ci_workflow_activation_json,
    write_route_policy_scenario_ci_workflow_json,
    write_route_policy_scenario_ci_workflow_promotion_json,
    write_route_policy_scenario_ci_workflow_validation_json,
    write_route_policy_scenario_ci_workflow_yaml,
    write_route_policy_scenario_matrix_expansion_json,
    write_route_policy_scenario_matrix_json,
    write_route_policy_scenario_set_json,
    write_route_policy_scenario_set_run_json,
    write_route_policy_scenario_shard_merge_json,
    write_route_policy_scenario_shard_plan_json,
    write_route_policy_scenario_shards_from_expansion,
    write_route_policy_transitions_jsonl,
)

occupancy = build_occupancy_grid_from_lidar_observation(
    lidar,
    voxel_size_meters=0.5,
    inflation_radius_meters=0.5,
)
env.set_occupancy_grid(occupancy)
env.set_robot_footprint(RobotFootprint(radius_meters=0.45, height_meters=1.2))

collision = env.query_collision(env.state.pose)
```

For repeated route checks, use `OccupancyPlanningContext` to cache the LiDAR-to-occupancy result by scene, viewpoint, and voxel settings:

```python
planning = OccupancyPlanningContext(voxel_size_meters=0.5, inflation_radius_meters=0.5)
occupancy = planning.set_environment_occupancy(env, scene_id="outdoor-demo", pose=env.state.pose)
cache_info = planning.cache_info()
```

For route-level planning, pass candidate trajectories through the same scoring path. The selected route is ranked by pass/fail, collision rate, collision count, clearance, and path length.

```python
plan = select_best_route(
    env,
    scene_id="outdoor-demo",
    candidates=(
        RouteCandidate("nominal", (env.state.pose, goal)),
        RouteCandidate("detour", (env.state.pose, waypoint, goal)),
    ),
    planning_context=planning,
)
best_route = plan.selected.candidate
```

To execute the selected route through the environment step contract, roll it out as either absolute `teleport` steps or fixed-duration `twist` segments. The rollout records every transition plus applied/collision status for each action.

```python
rollout = rollout_route(
    env,
    plan.selected,
    action_type="teleport",
    stop_on_collision=True,
)
route_passed = rollout.passed
rollout_metrics = rollout.metrics()
```

For closed-loop recovery, feed a blocked rollout into replanning. Candidate continuations are automatically anchored at the last applied pose before being scored and optionally executed.

```python
replan = replan_after_blocked_rollout(
    env,
    scene_id="outdoor-demo",
    rollout=rollout,
    candidates=(
        RouteCandidate("recover-left", (left_waypoint, goal)),
        RouteCandidate("recover-right", (right_waypoint, goal)),
    ),
    planning_context=planning,
    execute=True,
)

closed_loop = rollout_route_with_replanning(
    env,
    scene_id="outdoor-demo",
    initial_route=plan.selected,
    replan_candidate_batches=((RouteCandidate("detour", (waypoint, goal)),),),
)
```

For policy integration, compact the planning or rollout record into numeric features plus a scalar reward. This keeps agent-facing observations stable while preserving full debug records separately.

```python
sample = build_route_policy_sample(closed_loop)
agent_observation = sample.observation.features
agent_reward = sample.reward.reward
agent_terminal = sample.reward.terminal
```

For learned policy loops, wrap the same route feedback with the Gymnasium-style adapter. It avoids a hard `gymnasium` dependency, but follows the same `reset()` and `step()` return shape.

```python
policy_env = RoutePolicyGymAdapter(
    env,
    RoutePolicyEnvConfig(scene_id="outdoor-demo", max_steps=64),
)

observation, info = policy_env.reset(seed=7, goal=goal)
observation, reward, terminated, truncated, info = policy_env.step(
    {"routeId": "agent-waypoint", "target": goal.to_dict()}
)
```

To build replay data for imitation learning or offline RL, collect adapter episodes into a stable dataset envelope. JSON preserves full episodes; JSONL flattens transitions for streaming training jobs.

```python
def direct_goal_policy(observation, info):
    return {"routeId": f"direct-{info['stepIndex']}", "target": info["goal"]}


dataset = collect_route_policy_dataset(
    (policy_env,),
    direct_goal_policy,
    episode_count=16,
    dataset_id="outdoor-demo-policy-rollouts",
    seed_start=100,
)
write_route_policy_dataset_json("runs/outdoor-policy-rollouts.json", dataset)
write_route_policy_transitions_jsonl("runs/outdoor-policy-transitions.jsonl", dataset)
```

Gate those rollouts before they enter training. The quality report keeps pass/fail checks and metrics separate from the raw replay data, so CI can reject weak datasets without changing the collector.

```python
quality = evaluate_route_policy_dataset_quality(
    dataset,
    thresholds=RoutePolicyQualityThresholds(
        min_success_rate=0.95,
        max_collision_rate=0.01,
        max_truncation_rate=0.05,
        min_scene_count=1,
        min_episode_count=16,
    ),
)
print(render_route_policy_quality_markdown(quality))
assert quality.passed
```

For simple policy comparisons, collect each named baseline against the same adapter, seed range, and goals. The returned evaluation keeps each policy dataset and its QA report isolated.

```python
baselines = evaluate_route_policy_baselines(
    (policy_env,),
    {
        "direct": direct_goal_policy,
        "agent": learned_policy,
    },
    episode_count=16,
    evaluation_id="outdoor-demo-baselines",
    seed_start=100,
)
print(baselines.best_policy_name)
```

For offline training, load JSONL transitions back into a flat table and pin a numeric feature schema. Missing columns are filled with `0.0`, which lets older replay files keep working while the schema evolves.

```python
transition_table = load_route_policy_transitions_jsonl("runs/outdoor-policy-transitions.jsonl")
schema = build_route_policy_replay_schema(
    transition_table,
    action_keys=("payload.target.position.0", "payload.target.position.1", "payload.target.position.2"),
)

for batch in iter_route_policy_replay_batches(transition_table, batch_size=32, schema=schema, shuffle=True, seed=7):
    trainer.step(
        observations=batch.observation_matrix,
        actions=batch.action_matrix,
        rewards=batch.reward_vector,
        dones=batch.done_vector,
    )

full_batch = build_route_policy_replay_batch(transition_table, schema=schema)
```

For a dependency-free reference baseline, fit the built-in k-nearest-neighbor imitation model from the same batch. It decodes the predicted action vector back into an adapter-compatible route action and can be evaluated through the same baseline gate as hand-written policies.

```python
imitation_model = fit_route_policy_imitation_model(
    full_batch,
    config=RoutePolicyImitationFitConfig(neighbor_count=3),
    metadata={"source": "outdoor-demo-policy-rollouts"},
)

imitation_eval = evaluate_route_policy_imitation_model(
    (policy_env,),
    imitation_model,
    episode_count=16,
    evaluation_id="outdoor-demo-imitation",
    seed_start=100,
    thresholds=RoutePolicyQualityThresholds(min_success_rate=0.8),
)
print(imitation_eval.best_policy_name)
```

Persist fitted imitation models when the benchmark needs to run in a separate job from replay collection. The saved model contains the replay schema and training vectors, so it can be loaded without the original JSONL file.

```python
write_route_policy_imitation_model_json("runs/outdoor-imitation-model.json", imitation_model)
loaded_model = load_route_policy_imitation_model_json("runs/outdoor-imitation-model.json")

benchmark_report = run_route_policy_imitation_benchmark(
    (policy_env,),
    loaded_model,
    episode_count=16,
    benchmark_id="outdoor-demo-policy-benchmark",
    include_direct_baseline=True,
    seed_start=100,
)
write_route_policy_benchmark_report_json("runs/outdoor-policy-benchmark.json", benchmark_report)
print(render_route_policy_benchmark_markdown(benchmark_report))
```

The same flow is available from the CLI:

```bash
gs-mapper route-policy-benchmark \
  --transitions-jsonl runs/outdoor-policy-transitions.jsonl \
  --scene-catalog docs/scenes-list.json \
  --scene-id outdoor-demo \
  --action-keys payload.target.position.0 payload.target.position.1 payload.target.position.2 \
  --episode-count 16 \
  --include-direct-baseline \
  --model-output runs/outdoor-imitation-model.json \
  --output runs/outdoor-policy-benchmark.json \
  --markdown-output runs/outdoor-policy-benchmark.md
```

For repeatable comparisons, move fixed goals and saved policy paths into versioned JSON artifacts. Registry model paths are resolved relative to the registry file, so the bundle can move as one directory.

```python
goal_suite = RoutePolicyGoalSuite(
    suite_id="outdoor-demo-fixed-goals",
    scene_id="outdoor-demo",
    goals=(
        RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),
        RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),
    ),
)
write_route_policy_goal_suite_json("runs/outdoor-goals.json", goal_suite)

registry = RoutePolicyRegistry(
    registry_id="outdoor-demo-policies",
    policies=(
        RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),
        RoutePolicyRegistryEntry(
            policy_name="imitation-k3",
            policy_type="imitation-model",
            model_path="outdoor-imitation-model.json",
        ),
    ),
)
write_route_policy_registry_json("runs/outdoor-policies.json", registry)
```

```bash
gs-mapper route-policy-benchmark \
  --policy-registry runs/outdoor-policies.json \
  --goal-suite runs/outdoor-goals.json \
  --scene-catalog docs/scenes-list.json \
  --episode-count 16 \
  --output runs/outdoor-policy-registry-benchmark.json \
  --markdown-output runs/outdoor-policy-registry-benchmark.md
```

Once single-run reports are stable, collect them into a history artifact. The history keeps only compact per-policy metric snapshots, so CI jobs can compare commits or datasets without rebuilding rollout datasets.

```python
history = build_route_policy_benchmark_history(
    (
        "runs/commit-a/outdoor-policy-registry-benchmark.json",
        "runs/commit-b/outdoor-policy-registry-benchmark.json",
    ),
    baseline_report="runs/baseline/outdoor-policy-registry-benchmark.json",
    history_id="outdoor-demo-policy-history",
    thresholds=RoutePolicyBenchmarkRegressionThresholds(
        max_success_rate_drop=0.05,
        max_collision_rate_increase=0.01,
        max_truncation_rate_increase=0.02,
        max_mean_reward_drop=0.25,
    ),
)
write_route_policy_benchmark_history_json("runs/outdoor-policy-history.json", history)
print(render_route_policy_benchmark_history_markdown(history))

loaded_history = load_route_policy_benchmark_history_json("runs/outdoor-policy-history.json")
assert loaded_history.passed
```

The same regression gate is available from the CLI. Add `--fail-on-regression` in CI to return exit status 2 when a current report falls outside the blessed baseline envelope.

```bash
gs-mapper route-policy-benchmark-history \
  --report runs/commit-a/outdoor-policy-registry-benchmark.json \
  --report runs/commit-b/outdoor-policy-registry-benchmark.json \
  --baseline-report runs/baseline/outdoor-policy-registry-benchmark.json \
  --history-id outdoor-demo-policy-history \
  --max-success-rate-drop 0.05 \
  --max-collision-rate-increase 0.01 \
  --max-truncation-rate-increase 0.02 \
  --max-mean-reward-drop 0.25 \
  --output runs/outdoor-policy-history.json \
  --markdown-output runs/outdoor-policy-history.md \
  --fail-on-regression
```

To scale beyond one scene, define a scenario set. Each scenario pins a scene catalog, optional scene id, optional goal suite, and optional simulator/evaluation overrides. The runner executes the same policy registry for every scenario, writes one benchmark report per scenario, and sends those reports into the history gate. Relative scenario paths and the embedded `policyRegistryPath` resolve from the scenario-set JSON directory.

```python
scenario_set = RoutePolicyScenarioSet(
    scenario_set_id="outdoor-demo-scenarios",
    policy_registry_path="outdoor-policies.json",
    episode_count=16,
    seed_start=100,
    max_steps=64,
    scenarios=(
        RoutePolicyScenarioSpec(
            scenario_id="short-goals",
            scene_catalog="../docs/scenes-list.json",
            scene_id="outdoor-demo",
            goal_suite_path="outdoor-short-goals.json",
        ),
        RoutePolicyScenarioSpec(
            scenario_id="long-goals",
            scene_catalog="../docs/scenes-list.json",
            scene_id="outdoor-demo",
            goal_suite_path="outdoor-long-goals.json",
            max_steps=96,
        ),
    ),
)
write_route_policy_scenario_set_json("runs/scenarios/outdoor-scenarios.json", scenario_set)

loaded_scenarios = load_route_policy_scenario_set_json("runs/scenarios/outdoor-scenarios.json")
registry = load_route_policy_registry_json("runs/scenarios/outdoor-policies.json")
scenario_report = run_route_policy_scenario_set(
    loaded_scenarios,
    registry,
    report_dir="runs/scenarios/reports",
    scenario_set_base_path="runs/scenarios",
    registry_base_path="runs/scenarios",
    policy_registry_path="runs/scenarios/outdoor-policies.json",
    history_output="runs/scenarios/history.json",
)
write_route_policy_scenario_set_run_json("runs/scenarios/scenario-run.json", scenario_report)
print(render_route_policy_scenario_set_markdown(scenario_report))

loaded_run = load_route_policy_scenario_set_run_json("runs/scenarios/scenario-run.json")
assert loaded_run.passed
```

The CLI keeps the same artifact boundaries:

```bash
gs-mapper route-policy-scenario-set \
  --scenario-set runs/scenarios/outdoor-scenarios.json \
  --report-dir runs/scenarios/reports \
  --output runs/scenarios/scenario-run.json \
  --markdown-output runs/scenarios/scenario-run.md \
  --history-output runs/scenarios/history.json \
  --history-markdown-output runs/scenarios/history.md \
  --baseline-report runs/baseline/outdoor-policy-registry-benchmark.json \
  --max-success-rate-drop 0.05 \
  --max-collision-rate-increase 0.01 \
  --max-truncation-rate-increase 0.02 \
  --fail-on-regression
```

For broader coverage, create a scenario matrix and expand it into one scenario-set JSON per policy registry. The matrix is a Cartesian product over scene catalogs, goal suites, and simulator/evaluation configs, while keeping policy registries as separate scenario-set artifacts. Matrix paths are authored relative to the matrix JSON directory; generated scenario-set files are rewritten so their paths resolve from the generated file directory.

```python
matrix = RoutePolicyScenarioMatrix(
    matrix_id="outdoor-demo-matrix",
    registries=(
        RoutePolicyMatrixRegistrySpec("direct", "outdoor-policies-direct.json"),
        RoutePolicyMatrixRegistrySpec("imitation", "outdoor-policies-imitation.json"),
    ),
    scenes=(
        RoutePolicyMatrixSceneSpec(
            "outdoor-demo",
            "../docs/scenes-list.json",
            scene_id="outdoor-demo",
        ),
    ),
    goal_suites=(
        RoutePolicyMatrixGoalSuiteSpec("short", "outdoor-short-goals.json"),
        RoutePolicyMatrixGoalSuiteSpec("long", "outdoor-long-goals.json"),
    ),
    configs=(
        RoutePolicyMatrixConfigSpec("fast", episode_count=8, seed_start=100, max_steps=64),
        RoutePolicyMatrixConfigSpec("stress", episode_count=32, seed_start=1000, max_steps=96),
    ),
    episode_count=16,
    seed_start=100,
)
write_route_policy_scenario_matrix_json("runs/scenarios/outdoor-matrix.json", matrix)

loaded_matrix = load_route_policy_scenario_matrix_json("runs/scenarios/outdoor-matrix.json")
expansion = expand_route_policy_scenario_matrix_to_directory(
    loaded_matrix,
    "runs/scenarios/generated",
    matrix_base_path="runs/scenarios",
)
write_route_policy_scenario_matrix_expansion_json("runs/scenarios/matrix-expansion.json", expansion)
print(render_route_policy_scenario_matrix_markdown(expansion))
```

```bash
gs-mapper route-policy-scenario-matrix \
  --matrix runs/scenarios/outdoor-matrix.json \
  --output-dir runs/scenarios/generated \
  --index-output runs/scenarios/matrix-expansion.json \
  --markdown-output runs/scenarios/matrix-expansion.md
```

### Sensor noise profiles

Real robots observe their pose, goal, and sensor readings with non-zero uncertainty. `RoutePolicySensorNoiseProfile` attaches a small Gaussian noise budget (pose position σ, pose heading σ, goal position σ) to a scenario so the gym adapter perturbs the *observed* pose/goal before handing features to the policy. The true simulator state is unchanged — noise is purely a feature-layer transform — so collision checks and trajectory scoring stay honest.

Profiles are JSON artifacts referenced by path from the scenario spec (or the matrix config axis, which threads the reference into every generated scenario):

```python
write_route_policy_sensor_noise_profile_json(
    "runs/scenarios/sensor-noise/outdoor-gnss.json",
    RoutePolicySensorNoiseProfile(
        profile_id="outdoor-gnss",
        pose_position_std_meters=0.25,
        pose_heading_std_radians=0.02,
        goal_position_std_meters=0.15,
    ),
)
```

Reference the profile from a scenario or matrix config:

```python
scenario = RoutePolicyScenarioSpec(
    scenario_id="outdoor-near-short",
    scene_catalog="scenes.json",
    scene_id="outdoor-demo",
    goal_suite_path="near-goals.json",
    sensor_noise_profile_path="sensor-noise/outdoor-gnss.json",
)
# Or, for every scenario sharing a matrix config axis:
config = RoutePolicyMatrixConfigSpec(
    config_id="noisy-short",
    episode_count=1,
    seed_start=0,
    max_steps=8,
    sensor_noise_profile_path="sensor-noise/outdoor-gnss.json",
)
```

Determinism: the noise RNG is derived from `sha256(base_seed | profile_id | episode_index | step_index | kind)` so the same scenario replay always produces identical noisy observations across Python interpreter restarts. Setting every σ to `0.0` (the default) returns the identity profile — the adapter short-circuits to the true pose.

### Raw sensor noise profiles

`RoutePolicySensorNoiseProfile` perturbs features the policy already sees. `RawSensorNoiseProfile` is its sibling at the observation renderer boundary — it adds Gaussian noise to RGB pixels (clipped to `[0, 255]` before JPEG re-encode), float32 depth maps (clamped to the advertised far-clip horizon and non-negative), and LiDAR ranges (clamped non-negative). The decoded arrays inside an `Observation.outputs` dict are perturbed and re-encoded in place, so downstream consumers keep reading the same base64 fields.

```python
from gs_sim2real.sim import (
    RawSensorNoiseProfile,
    write_raw_sensor_noise_profile_json,
)

write_raw_sensor_noise_profile_json(
    "runs/scenarios/raw-noise/outdoor-sensor.json",
    RawSensorNoiseProfile(
        profile_id="outdoor-sensor",
        rgb_intensity_std=3.0,        # 0-255 scale
        depth_range_std_meters=0.10,  # per-pixel float32 σ
        lidar_range_std_meters=0.05,  # per-ray σ
    ),
)
```

Reference the profile from a scenario spec or a matrix config axis the same way the pose-facing profile does — the field is `raw_sensor_noise_profile_path`:

```python
scenario = RoutePolicyScenarioSpec(
    scenario_id="outdoor-near-raw-noise",
    scene_catalog="scenes.json",
    scene_id="outdoor-demo",
    goal_suite_path="near-goals.json",
    raw_sensor_noise_profile_path="raw-noise/outdoor-sensor.json",
)
config = RoutePolicyMatrixConfigSpec(
    config_id="raw-noise-short",
    episode_count=1,
    seed_start=0,
    max_steps=8,
    raw_sensor_noise_profile_path="raw-noise/outdoor-sensor.json",
)
```

Application seam: `HeadlessPhysicalAIEnvironment(..., raw_sensor_noise_profile=…)` stores the profile and, whenever `render_observation` reaches a base `ObservationRenderer` that `can_render` the request, the result is routed through `apply_raw_sensor_noise_to_observation` before being returned. The noise RNG is seeded from `sha256(reset_seed | profile_id | sensor_id | render_request_index | "obs")`, and an internal counter advances per render call so consecutive queries at the same pose still draw distinct noise. When no observation renderer is attached the env falls back to its metadata-only response and the profile is a no-op.

IMU readings come from a kinematic finite-difference renderer baked into `HeadlessPhysicalAIEnvironment`. After every `step` the env stores a `KinematicState` derived from the pose delta and the action's `duration_seconds`: linear velocity in the world frame (`(p_next - p_prev) / dt`), linear acceleration as the velocity finite difference rotated into the agent body frame, and body-frame angular velocity from the body-frame delta quaternion `q_prev⁻¹ ⊗ q_next` divided by `dt`. `render_observation(ObservationRequest(..., sensor_id="imu-proxy"))` reads that state and emits `angular-velocity` (`angularVelocityBase64`, `rad/s`) and `linear-acceleration` (`linearAccelerationBase64`, `m/s²`) blocks in the float32-le-xyz format that `RawSensorNoiseProfile` already perturbs. Gravity is intentionally not modelled — this is a kinematic accelerometer, not an inertial-frame IMU. After `reset` and after any `teleport` action the kinematic state is the zero state (`stepDtSeconds == 0`); a downstream physics or rosbag-replay layer can replace `_render_imu_observation` if you need a richer model.

`RoutePolicyGymAdapter` decodes that observation into the policy's feature dict on every reset and step: `imu-step-dt-seconds` carries the kinematic-state freshness gate (`0.0` after reset / teleport), and three pairs of axis-resolved scalars — `imu-angular-velocity-{x,y,z}` (rad/s) and `imu-linear-acceleration-{x,y,z}` (m/s²) — surface the body-frame readings. Because the adapter goes through `env.render_observation(ObservationRequest(sensor_id="imu-proxy", outputs=("angular-velocity", "linear-acceleration")))`, the same `RawSensorNoiseProfile` σ values that perturb the env-side outputs flow into the policy features automatically (the IMU RNG is seeded from `(reset_seed | profile_id | "imu-proxy" | render_request_index | "obs")`, so a feature query at a given step is deterministic). The block is omitted entirely when the wrapped environment does not expose `render_observation` or the scene's sensor rig has no `imu-proxy` sensor — adapters built around a stub env therefore keep their existing feature dict unchanged.

### Dynamic obstacles

Static occupancy grids collapse the gap between trivial direct-goal policies and policies that have to react to the world. `DynamicObstacleTimeline` layers on top of the static scene: each `DynamicObstacle` carries a sorted list of `(step_index, position)` waypoints plus a sphere radius, and the environment consults the timeline inside every collision query. Positions are linearly interpolated between bracketing waypoints (clamped outside the range), so a single pair of waypoints gives a constant-velocity moving obstacle for free.

```python
from gs_sim2real.sim import (
    DynamicObstacle,
    DynamicObstacleTimeline,
    DynamicObstacleWaypoint,
    write_route_policy_dynamic_obstacle_timeline_json,
)

timeline = DynamicObstacleTimeline(
    timeline_id="outdoor-cross-traffic",
    obstacles=(
        DynamicObstacle(
            obstacle_id="cyclist",
            waypoints=(
                DynamicObstacleWaypoint(step_index=0, position=(-1.0, 0.0, 0.0)),
                DynamicObstacleWaypoint(step_index=8, position=(1.0, 0.0, 0.0)),
            ),
            radius_meters=0.25,
        ),
    ),
)
write_route_policy_dynamic_obstacle_timeline_json(
    "runs/scenarios/obstacles/outdoor-cross-traffic.json",
    timeline,
)
```

Scenario specs and matrix configs reference the timeline JSON via a new optional `dynamic_obstacles_path` — the same shape the sensor noise profile uses:

```python
scenario = RoutePolicyScenarioSpec(
    scenario_id="outdoor-near-cross-traffic",
    scene_catalog="scenes.json",
    scene_id="outdoor-demo",
    goal_suite_path="near-goals.json",
    dynamic_obstacles_path="obstacles/outdoor-cross-traffic.json",
)
config = RoutePolicyMatrixConfigSpec(
    config_id="cross-traffic",
    episode_count=2,
    seed_start=0,
    max_steps=8,
    dynamic_obstacles_path="obstacles/outdoor-cross-traffic.json",
)
```

When the environment's collision check sees that a query pose sits inside any obstacle's sphere at the current step, it reports `dynamic-obstacle:<obstacle_id>` and lets the scenario CI chain record it the same way static occupancy collisions are recorded. Trajectory scoring steps through the trajectory pose-by-pose so each step of a multi-pose rollout is checked against the obstacle's interpolated position at that step — an obstacle crossing the path between step 3 and step 4 blocks step 3 but not step 5.

**Reactive chase and flee obstacles**: a `DynamicObstacle` can also ignore later waypoints and react to the queried agent position at a fixed speed. `chase_target_agent=True` walks from `waypoints[0]` toward the agent at `waypoints[0] + direction_to_agent * min(step * chase_speed_m_per_step, distance_to_agent)` — clamped once the obstacle reaches the agent. `flee_from_agent=True` uses the same speed magnitude but walks *away* from the agent along the `agent → waypoints[0]` direction with no upper bound, so the obstacle keeps retreating every step. The two modes are mutually exclusive and both are pure functions of the current agent position and the step index, so replays stay deterministic (no agent-pose history is retained). A reactive obstacle queried without an agent position (e.g. headless Markdown rendering) stays pinned at its first waypoint.

```python
hunter = DynamicObstacle(
    obstacle_id="hunter",
    waypoints=(DynamicObstacleWaypoint(step_index=0, position=(3.0, 0.0, 0.0)),),
    radius_meters=0.25,
    chase_target_agent=True,
    chase_speed_m_per_step=0.5,  # metres per scenario step
)
runner = DynamicObstacle(
    obstacle_id="runner",
    waypoints=(DynamicObstacleWaypoint(step_index=0, position=(1.0, 0.0, 0.0)),),
    radius_meters=0.25,
    flee_from_agent=True,
    chase_speed_m_per_step=0.5,  # same magnitude, sign flipped
)
```

**Observation features**: when the adapter's environment has a non-empty `DynamicObstacleTimeline`, the gym-style feature dict gains an obstacle-awareness block so learned policies can react without needing full scene rendering. Distances and bearings are measured from the same observed pose the policy already sees — so a sensor-noise profile perturbs obstacle observations in lock-step with pose / goal observations, keeping the feature block consistent under partial-information benchmarks. The new keys:

- `dynamic-obstacle-count` — timeline cardinality (always equal to `len(timeline.obstacles)`).
- `nearest-dynamic-obstacle-distance-meters` — minimum clearance (Euclidean distance minus radius, floored at 0) from the observed pose to any obstacle at the current step.
- `nearest-dynamic-obstacle-bearing-radians` — XY-plane bearing from the observed pose to the nearest obstacle's centre, in `[-π, π]`.
- `nearest-dynamic-obstacle-bearing-x`, `nearest-dynamic-obstacle-bearing-y` — unit-vector components of the same bearing (both `0.0` when the obstacle coincides with the pose).
- `second-nearest-dynamic-obstacle-distance-meters`, `second-nearest-dynamic-obstacle-bearing-radians`, `second-nearest-dynamic-obstacle-bearing-x`, `second-nearest-dynamic-obstacle-bearing-y` — same four features for the second-closest obstacle. Emitted only when at least two obstacles are on the timeline so multi-agent scenarios let a policy tell apart a single-threat lane from a flanked one; stays omitted when only one obstacle is configured.
- `nearest-dynamic-obstacle-reactive-mode` and (when the second slot is emitted) `second-nearest-dynamic-obstacle-reactive-mode` — scalar reactive indicator for each surfaced obstacle: `+1.0` when the obstacle chases the agent (`chase_target_agent`), `-1.0` when it flees (`flee_from_agent`), `0.0` for static waypoint obstacles. Lets a policy condition on the threat mode directly instead of inferring it from distance derivatives.
- `peer-min-separation-meters` — minimum sphere-aware clearance over all obstacle pairs (`max(0, distance(centre_i, centre_j) - radius_i - radius_j)`). Emitted alongside the second-nearest block so a policy can tell apart a tightly-clumped swarm from a spread-out one even when both share the same nearest-obstacle distance; stays omitted when only one obstacle is configured. The adapter resolves obstacle positions through `DynamicObstacleTimeline.step_positions(step_index, agent_position=..., previous_positions=cache)` and threads the previous step's resolved positions back in so policy-driven obstacles (e.g. `MaintainSeparationObstaclePolicy`) actually see their peers in the `ObstaclePolicyContext`.

**Obstacle policies (multi-agent seam)**: a `DynamicObstacle` can carry an opt-in `policy: ObstaclePolicy` callable (runtime-only — *not* serialised in the timeline JSON, so v1 scenario CI artifacts keep loading unchanged). When set, `position_at_step` builds an `ObstaclePolicyContext` (own id, step index, default-fallback position, agent position, peer obstacles' previous-step positions) and uses the policy's `next_position` instead of the chase / flee / waypoint logic. Four reference implementations ship: `WaypointInterpolationObstaclePolicy`, `ChaseAgentObstaclePolicy`, and `FleeAgentObstaclePolicy` replay the existing inline behaviours bit-for-bit; `MaintainSeparationObstaclePolicy` wraps an inner policy and pushes the result outward when any peer is closer than `min_separation_meters`. To resolve a multi-obstacle step in one pass without dependency cycles, call `timeline.step_positions(step_index, agent_position=..., previous_positions=...)` — each obstacle sees its peers' resolved positions from the previous step. `timeline.blocking_obstacle(...)` accepts the same `peer_positions` keyword for collision queries that already know the prior step's layout.

```python
from gs_sim2real.sim import (
    ChaseAgentObstaclePolicy,
    DynamicObstacle,
    DynamicObstacleTimeline,
    DynamicObstacleWaypoint,
    MaintainSeparationObstaclePolicy,
)

inner = ChaseAgentObstaclePolicy(start_position=(3.0, 0.0, 0.0), speed_m_per_step=0.5)
chaser = DynamicObstacle(
    obstacle_id="chaser",
    waypoints=(DynamicObstacleWaypoint(step_index=0, position=(3.0, 0.0, 0.0)),),
    radius_meters=0.3,
    chase_target_agent=True,
    chase_speed_m_per_step=0.5,
    policy=MaintainSeparationObstaclePolicy(inner, min_separation_meters=1.0),
)
peer = DynamicObstacle(
    obstacle_id="peer",
    waypoints=(DynamicObstacleWaypoint(step_index=0, position=(2.5, 0.0, 0.0)),),
    radius_meters=0.3,
)
timeline = DynamicObstacleTimeline(timeline_id="duo", obstacles=(chaser, peer))

state = timeline.step_positions(0, agent_position=(0.0, 0.0, 0.0))
state = timeline.step_positions(1, agent_position=(0.0, 0.0, 0.0), previous_positions=state)
```

`HeadlessPhysicalAIEnvironment` keeps a per-step peer-position cache so `query_collision` automatically threads `peer_positions` into `blocking_obstacle` and the per-obstacle `position_at_step`. The cache is seeded inside `reset` (`step_positions(0, agent_position=initial_pose, previous_positions={})`), refreshed inside `step` after the new pose commits (`step_positions(step_index+1, agent_position=next_pose, previous_positions=cache)`), and rebuilt inside `set_dynamic_obstacles` whenever the timeline is swapped. Policy-driven obstacles therefore observe up-to-date peer state during real collision queries on a rollout, not just inside the gym adapter's feature dict. `score_trajectory` rebuilds an isolated peer cache locally (one `step_positions` call per pose, threading the previous step's resolved layout in via `previous_positions`) so a hypothetical trajectory's per-pose collision queries also see policy-driven obstacles consulting their siblings — without mutating the env's stepwise cache.

The entire block is omitted when no timeline is configured, so existing scenario-set fixtures keep their exact feature dict.

For CI-sized execution, split the generated scenario sets into shard scenario-set files. Each shard is still a normal `RoutePolicyScenarioSet`, so CI jobs can run shards with the existing `route-policy-scenario-set` command. The final merge step reads the shard run JSON files, collects every per-scenario benchmark report, and rebuilds one global history gate.

```python
shard_plan = write_route_policy_scenario_shards_from_expansion(
    expansion,
    "runs/scenarios/shards",
    max_scenarios_per_shard=4,
    shard_plan_id="outdoor-demo-shards",
)
write_route_policy_scenario_shard_plan_json("runs/scenarios/shard-plan.json", shard_plan)
print(render_route_policy_scenario_shard_plan_markdown(shard_plan))

# Run each shard independently with run_route_policy_scenario_set(...) or the CLI below.
# Then merge all shard run JSON files:
merge = merge_route_policy_scenario_shard_run_jsons(
    (
        "runs/scenarios/shard-runs/outdoor-matrix-direct-shard-001.json",
        "runs/scenarios/shard-runs/outdoor-matrix-direct-shard-002.json",
    ),
    merge_id="outdoor-demo-shard-merge",
    history_output="runs/scenarios/shard-history.json",
    history_markdown_output="runs/scenarios/shard-history.md",
)
write_route_policy_scenario_shard_merge_json("runs/scenarios/shard-merge.json", merge)
print(render_route_policy_scenario_shard_merge_markdown(merge))
```

```bash
gs-mapper route-policy-scenario-shards \
  --expansion runs/scenarios/matrix-expansion.json \
  --max-scenarios-per-shard 4 \
  --shard-plan-id outdoor-demo-shards \
  --output-dir runs/scenarios/shards \
  --index-output runs/scenarios/shard-plan.json \
  --markdown-output runs/scenarios/shard-plan.md

gs-mapper route-policy-scenario-set \
  --scenario-set runs/scenarios/shards/outdoor-matrix-direct-shard-001.json \
  --report-dir runs/scenarios/shard-reports/001 \
  --output runs/scenarios/shard-runs/outdoor-matrix-direct-shard-001.json \
  --history-output runs/scenarios/shard-runs/outdoor-matrix-direct-shard-001-history.json

gs-mapper route-policy-scenario-shard-merge \
  --run runs/scenarios/shard-runs/outdoor-matrix-direct-shard-001.json \
  --run runs/scenarios/shard-runs/outdoor-matrix-direct-shard-002.json \
  --merge-id outdoor-demo-shard-merge \
  --history-output runs/scenarios/shard-history.json \
  --history-markdown-output runs/scenarios/shard-history.md \
  --output runs/scenarios/shard-merge.json \
  --markdown-output runs/scenarios/shard-merge.md \
  --fail-on-regression
```

To make CI matrix jobs stable, generate a manifest from the shard plan. The manifest includes a `matrix.include` list for shard jobs, expected report paths, cache keys, generated CLI commands, and the merge job dependencies.

```python
ci_manifest = build_route_policy_scenario_ci_manifest(
    shard_plan,
    manifest_id="outdoor-demo-ci",
    report_dir="runs/scenarios/ci/reports",
    run_output_dir="runs/scenarios/ci/runs",
    history_output_dir="runs/scenarios/ci/histories",
    merge_id="outdoor-demo-shard-merge",
    merge_output="runs/scenarios/ci/shard-merge.json",
    merge_history_output="runs/scenarios/ci/shard-history.json",
    cache_key_prefix="outdoor-demo-policy",
    fail_on_regression=True,
)
write_route_policy_scenario_ci_manifest_json("runs/scenarios/ci-manifest.json", ci_manifest)
print(render_route_policy_scenario_ci_manifest_markdown(ci_manifest))
```

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

The manifest can also materialize a GitHub Actions workflow. By default the workflow is manual-only (`workflow_dispatch`) so generated YAML can be reviewed before enabling push or pull-request triggers.

```python
workflow = materialize_route_policy_scenario_ci_workflow(
    ci_manifest,
    config=RoutePolicyScenarioCIWorkflowConfig(
        workflow_id="outdoor-demo-policy-shards",
        workflow_name="Outdoor Demo Policy Shards",
        artifact_root="runs/scenarios/ci",
        push_branches=("main",),
        pull_request_branches=("main",),
    ),
)
write_route_policy_scenario_ci_workflow_yaml(
    ".github/workflows/outdoor-demo-policy-shards.generated.yml",
    workflow,
)
write_route_policy_scenario_ci_workflow_json("runs/scenarios/ci-workflow.json", workflow)
print(render_route_policy_scenario_ci_workflow_markdown(workflow))
```

```bash
gs-mapper route-policy-scenario-ci-workflow \
  --manifest runs/scenarios/ci-manifest.json \
  --workflow-id outdoor-demo-policy-shards \
  --workflow-name "Outdoor Demo Policy Shards" \
  --artifact-root runs/scenarios/ci \
  --push-branch main \
  --pull-request-branch main \
  --workflow-output .github/workflows/outdoor-demo-policy-shards.generated.yml \
  --index-output runs/scenarios/ci-workflow.json \
  --markdown-output runs/scenarios/ci-workflow.md
```

Validate the generated YAML before enabling the workflow. The validator parses the YAML, checks materialization metadata, verifies shard matrix entries and commands against the manifest, and writes a JSON report that can be used as a review gate.

```python
validation = validate_route_policy_scenario_ci_workflow(
    ci_manifest,
    workflow,
    workflow_path=".github/workflows/outdoor-demo-policy-shards.generated.yml",
)
write_route_policy_scenario_ci_workflow_validation_json(
    "runs/scenarios/ci-workflow-validation.json",
    validation,
)
print(render_route_policy_scenario_ci_workflow_validation_markdown(validation))
```

```bash
gs-mapper route-policy-scenario-ci-workflow-validate \
  --manifest runs/scenarios/ci-manifest.json \
  --workflow-index runs/scenarios/ci-workflow.json \
  --workflow .github/workflows/outdoor-demo-policy-shards.generated.yml \
  --output runs/scenarios/ci-workflow-validation.json \
  --markdown-output runs/scenarios/ci-workflow-validation.md \
  --fail-on-validation
```

After validation passes, activate the workflow into a real GitHub Actions path. Activation refuses to write unless the validation report passed, the validated source path matches the source file being activated, the source YAML matches the materialization index, and the destination is under `.github/workflows/`.

```python
activation = activate_route_policy_scenario_ci_workflow(
    workflow,
    validation,
    source_workflow_path=".github/workflows/outdoor-demo-policy-shards.generated.yml",
    active_workflow_path=".github/workflows/outdoor-demo-policy-shards.yml",
)
write_route_policy_scenario_ci_workflow_activation_json(
    "runs/scenarios/ci-workflow-activation.json",
    activation,
)
print(render_route_policy_scenario_ci_workflow_activation_markdown(activation))
```

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

Publish a review bundle for GitHub Pages once shard runs have been merged. The bundle contains `review.json`, `review.md`, and an `index.html` page that summarizes shard status, workflow validation, workflow activation, and the active workflow path.

```python
review = build_route_policy_scenario_ci_review_artifact(
    shard_merge,
    validation,
    activation,
    review_id="outdoor-demo-policy-review",
    pages_base_url="https://example.github.io/3dgs-robotics/reviews/outdoor-demo-policy/",
)
write_route_policy_scenario_ci_review_bundle(
    "docs/reviews/outdoor-demo-policy",
    review,
)
print(render_route_policy_scenario_ci_review_markdown(review))
```

```bash
gs-mapper route-policy-scenario-ci-review \
  --shard-merge runs/scenarios/ci/shard-merge.json \
  --validation-report runs/scenarios/ci-workflow-validation.json \
  --activation-report runs/scenarios/ci-workflow-activation.json \
  --review-id outdoor-demo-policy-review \
  --pages-base-url https://example.github.io/3dgs-robotics/reviews/outdoor-demo-policy/ \
  --bundle-dir docs/reviews/outdoor-demo-policy \
  --fail-on-review
```

Before widening from manual dispatch to repository triggers, write a promotion report. The gate requires the review to pass, validation and activation to be green, shard merge and history checks to pass, the active workflow path to stay under `.github/workflows/`, a published review URL, and literal branch names for the requested trigger mode.

```python
promotion = promote_route_policy_scenario_ci_workflow(
    review,
    trigger_mode="pull-request",
    pull_request_branches=("main",),
    review_url="https://example.github.io/3dgs-robotics/reviews/outdoor-demo-policy/",
)
write_route_policy_scenario_ci_workflow_promotion_json(
    "runs/scenarios/ci-workflow-promotion.json",
    promotion,
)
print(render_route_policy_scenario_ci_workflow_promotion_markdown(promotion))
```

```bash
gs-mapper route-policy-scenario-ci-workflow-promote \
  --review runs/scenarios/ci-review.json \
  --review-url https://example.github.io/3dgs-robotics/reviews/outdoor-demo-policy/ \
  --trigger-mode pull-request \
  --pull-request-branch main \
  --output runs/scenarios/ci-workflow-promotion.json \
  --markdown-output runs/scenarios/ci-workflow-promotion.md \
  --fail-on-promotion
```

After the promotion report passes, the adoption step re-materializes the scenario CI manifest with the promoted trigger mode / branches, re-runs validation and activation against a **separate** active workflow path, and records a per-gate adoption report. The manual-only active workflow file is never overwritten — adoption always writes a parallel YAML so the two files can be diffed side by side before flipping the repo over to the trigger-enabled workflow.

```python
from gs_sim2real.sim import (
    adopt_route_policy_scenario_ci_workflow,
    write_route_policy_scenario_ci_workflow_adoption_json,
)

adoption = adopt_route_policy_scenario_ci_workflow(
    promotion,
    manifest,
    materialization,  # the manual-only materialization that produced the activated YAML
    adopted_source_workflow_path="runs/scenarios/ci-workflow-adopted.generated.yml",
    adopted_active_workflow_path=".github/workflows/outdoor-demo-policy-shards-adopted.yml",
)
write_route_policy_scenario_ci_workflow_adoption_json(
    "runs/scenarios/ci-workflow-adoption.json",
    adoption,
)
```

Adoption gates:

- `promotion-promoted`: only adopt from a PROMOTED report.
- `manifest-id` / `workflow-id`: the adoption must target the same manifest and workflow the promotion report was built from.
- `adopted-path-distinct-from-manual`: the adopted active workflow path must differ from the promoted `active_workflow_path`; collisions fail pre-materialization so the manual YAML is never touched.
- `adopted-source-path-distinct`: the staged YAML source path must also be separate from the manual active path.
- `workflow-dispatch-retained`: adopted YAML keeps `workflow_dispatch` so on-demand reruns still work.
- `push-trigger-emitted` / `pull-request-trigger-emitted`: required trigger blocks appear when the promoted `trigger_mode` demands them.
- Per-branch gates (`push-branch:<name>` / `pull-request-branch:<name>`): each promoted branch literally appears in the adopted YAML.
- `adopted-validation-passed` / `adopted-activation-active`: the re-run validation and activation reports must themselves pass.

The same flow is also exposed as a CLI command:

```bash
gs-mapper route-policy-scenario-ci-workflow-adopt \
  --manifest runs/scenarios/ci-manifest.json \
  --workflow-index runs/scenarios/ci-workflow.json \
  --promotion runs/scenarios/ci-workflow-promotion.json \
  --adopted-workflow-output runs/scenarios/ci-workflow-adopted.generated.yml \
  --adopted-active-workflow-output .github/workflows/outdoor-demo-policy-shards-adopted.yml \
  --adoption-id outdoor-demo-policy-adoption \
  --output runs/scenarios/ci-workflow-adoption.json \
  --markdown-output runs/scenarios/ci-workflow-adoption.md \
  --fail-on-adoption
```

After the adoption lands, the review bundle can be re-published with the adoption info attached so reviewers on Pages see the trigger mode, branches, and a unified diff between the manual-only and the adopted YAMLs without checking out the branch. Pass `--adoption-report` to the review CLI (and, if needed, `--manual-workflow` / `--adopted-workflow` to override which YAMLs are read):

```bash
gs-mapper route-policy-scenario-ci-review \
  --shard-merge runs/scenarios/ci/shard-merge.json \
  --validation-report runs/scenarios/ci-workflow-validation.json \
  --activation-report runs/scenarios/ci-workflow-activation.json \
  --adoption-report runs/scenarios/ci-workflow-adoption.json \
  --review-id outdoor-demo-policy-review \
  --pages-base-url https://example.github.io/3dgs-robotics/reviews/outdoor-demo-policy/ \
  --bundle-dir docs/reviews/outdoor-demo-policy \
  --fail-on-review
```

The resulting `review.json` / `review.md` / `index.html` gain an "Adopted Workflow" section with the trigger mode, push/pull_request branches, and a unified diff block. The diff is produced from the manual and adopted YAMLs that live under `.github/workflows/` so no extra build step is needed — by default the CLI reads the activation report's active path for the manual side and the adoption report's adopted active path for the adopted side.

A minimal end-to-end recipe that walks matrix expansion all the way through the adoption-enriched review bundle lives at `scripts/smoke_route_policy_scenario_ci.py`.

When multiple review bundles live side-by-side under `docs/reviews/`, rebuild the Pages index so the bundles become discoverable from a single entry point:

```bash
PYTHONPATH=src python3 scripts/build_pages_reviews_index.py \
  --reviews-dir docs/reviews \
  --html-output docs/reviews/index.html \
  --json-output docs/reviews/index.json
```

The script scans the target directory for sub-directories that contain a `review.json`, loads each bundle, and writes `index.html` plus a structured `index.json` alongside them. Missing or non-bundle sub-directories are skipped; an empty reviews directory still produces a stable "no review bundles published yet" placeholder so Pages deploys never 404. Each index row carries a PASS / FAIL pill, shard / scenario / report counts, and (when the bundle was produced with `--adoption-report`) an ADOPTED / BLOCKED pill plus the promoted trigger mode.

Supported actions:

- `twist`: `linearX`, `linearY`, `linearZ` or `vx`, `vy`, `vz`
- `teleport`: absolute `x`, `y`, `z` plus optional `qx`, `qy`, `qz`, `qw`

The backend always blocks poses outside `SceneEnvironment.bounds`. When a `VoxelOccupancyGrid` is set, in-bounds collision checks also reject poses that fall into occupied voxels. When a `RobotFootprint` is set, the occupancy query checks every voxel touched by the circular body radius and height instead of only the pose point. `score_trajectory()` uses the same collision path and reports `collision-rate`, `collision-count`, clearance metrics, and per-reason notes.

## Partial-information benchmark recipe

A common task is "evaluate a route policy against noisy pose + obstacles that react to it". The individual primitives are documented above; this recipe stitches them together into one scenario-set run so a reader can see the whole partial-information surface in one place.

1. **Author the noise profile(s).** `RoutePolicySensorNoiseProfile` perturbs pose / goal / heading that the policy observes; `RawSensorNoiseProfile` perturbs rendered camera / depth / LiDAR / IMU arrays at the observation renderer boundary. Persist each as JSON — the scenario spec references them by path. Ready-to-copy references live at [`docs/fixtures/sensor-noise/outdoor-gnss.json`](fixtures/sensor-noise/outdoor-gnss.json) (pose-facing) and [`docs/fixtures/raw-noise/outdoor-sensor.json`](fixtures/raw-noise/outdoor-sensor.json) (raw-sensor), both pinned by `tests/test_bundled_sensor_noise_fixtures.py`.

   ```python
   from gs_sim2real.sim import (
       RawSensorNoiseProfile,
       RoutePolicySensorNoiseProfile,
       write_raw_sensor_noise_profile_json,
       write_route_policy_sensor_noise_profile_json,
   )

   write_route_policy_sensor_noise_profile_json(
       "runs/scenarios/sensor-noise/outdoor-gnss.json",
       RoutePolicySensorNoiseProfile(
           profile_id="outdoor-gnss",
           pose_position_std_meters=0.25,
           pose_heading_std_radians=0.02,
           goal_position_std_meters=0.15,
       ),
   )
   write_raw_sensor_noise_profile_json(
       "runs/scenarios/raw-noise/outdoor-sensor.json",
       RawSensorNoiseProfile(
           profile_id="outdoor-sensor",
           depth_range_std_meters=0.10,
           lidar_range_std_meters=0.05,
       ),
   )
   ```

2. **Author the reactive obstacle timeline.** Combine chase + flee + static waypoint obstacles on one `DynamicObstacleTimeline`. The gym adapter surfaces the top-two closest obstacles with a `reactive-mode` scalar (`+1`/`-1`/`0`), so the policy can condition on threat mode directly.

   ```python
   from gs_sim2real.sim import (
       DynamicObstacle,
       DynamicObstacleTimeline,
       DynamicObstacleWaypoint,
       write_route_policy_dynamic_obstacle_timeline_json,
   )

   write_route_policy_dynamic_obstacle_timeline_json(
       "runs/scenarios/obstacles/mixed-reactive.json",
       DynamicObstacleTimeline(
           timeline_id="mixed-reactive",
           obstacles=(
               DynamicObstacle(
                   obstacle_id="hunter",
                   waypoints=(DynamicObstacleWaypoint(step_index=0, position=(3.0, 0.0, 0.0)),),
                   radius_meters=0.25,
                   chase_target_agent=True,
                   chase_speed_m_per_step=0.5,
               ),
               DynamicObstacle(
                   obstacle_id="runner",
                   waypoints=(DynamicObstacleWaypoint(step_index=0, position=(0.0, 1.0, 0.0)),),
                   radius_meters=0.25,
                   flee_from_agent=True,
                   chase_speed_m_per_step=0.5,
               ),
               DynamicObstacle(
                   obstacle_id="bollard",
                   waypoints=(DynamicObstacleWaypoint(step_index=0, position=(0.0, -2.0, 0.0)),),
                   radius_meters=0.25,
               ),
           ),
       ),
   )
   ```

   Use `python3 scripts/show_dynamic_obstacle_timeline.py runs/scenarios/obstacles/mixed-reactive.json` to eyeball the Markdown summary (reactive mode + speed columns per obstacle). The same shape ships as a ready-to-copy reference at [`docs/fixtures/dynamic-obstacles/mixed-reactive.json`](fixtures/dynamic-obstacles/mixed-reactive.json), loadable with `load_route_policy_dynamic_obstacle_timeline_json` and pinned by `tests/test_bundled_mixed_reactive_fixture.py`.

3. **Wire all three paths into the scenario spec.** `sensor_noise_profile_path`, `raw_sensor_noise_profile_path`, and `dynamic_obstacles_path` each carry independently — a scenario can set any subset.

   ```python
   from gs_sim2real.sim import (
       RoutePolicyScenarioSet,
       RoutePolicyScenarioSpec,
       write_route_policy_scenario_set_json,
   )

   scenario_set = RoutePolicyScenarioSet(
       scenario_set_id="partial-information-outdoor",
       policy_registry_path="registries/outdoor.json",
       scenarios=(
           RoutePolicyScenarioSpec(
               scenario_id="outdoor-near-partial",
               scene_catalog="scenes.json",
               scene_id="outdoor-demo",
               goal_suite_path="near-goals.json",
               sensor_noise_profile_path="sensor-noise/outdoor-gnss.json",
               raw_sensor_noise_profile_path="raw-noise/outdoor-sensor.json",
               dynamic_obstacles_path="obstacles/mixed-reactive.json",
               episode_count=8,
               max_steps=16,
           ),
       ),
   )
   write_route_policy_scenario_set_json(
       "runs/scenarios/partial-information-outdoor.json", scenario_set
   )
   ```

4. **Run through the standard scenario-set runner.** `run_route_policy_scenario_set` loads every profile and timeline, constructs `HeadlessPhysicalAIEnvironment` + `RoutePolicyGymAdapter` with the pose-facing noise on the adapter and the raw-sensor noise on the env, then evaluates the registry. The same CLI + scenario-shard + review-bundle chain (`gs-mapper route-policy-scenario-set`, `scripts/smoke_route_policy_scenario_ci.py`, etc.) keeps working — partial-information knobs only affect inputs, not the pipeline shape.

Determinism stays intact across all three knobs: each noise profile's RNG is seeded from `(reset_seed | profile_id | episode_index | step_index | kind)`, and each reactive obstacle is a pure function of the current agent position and the step index. A scenario rerun under the same seeds produces bit-identical observations and bit-identical feature dicts.

## Real-vs-sim correlation

`gs_sim2real.robotics.rosbag_correlation` closes the loop between a headless rollout and the recorded rosbag2 it was meant to model. `read_navsat_pose_stream(bag_paths, *, topic=None, reference_origin_wgs84=None)` reuses the same `rosbags`/`AnyReader` machinery that `MCDLoader` already depends on (so zstd-compressed sqlite3 bags work without decompression) and converts the chosen `sensor_msgs/NavSatFix` topic into a metric local-ENU `BagPoseStream`. The first valid fix anchors the ENU origin unless `reference_origin_wgs84=(lat, lon, alt)` pins one explicitly so multiple bags share the same frame. Placeholder fixes at `latitude == longitude == 0` are dropped during ingest.

`correlate_against_sim_trajectory(bag_stream, sim_samples, *, max_match_dt_seconds=0.05)` performs a nearest-timestamp pairing between the bag samples and the sim trajectory (a sequence of `SimPoseSample` produced by a benchmark / scenario runner) and reduces per-pair translation errors into min / mean / max / p50 / p95 statistics inside `RealVsSimCorrelationReport`. Pairs whose clock skew exceeds `max_match_dt_seconds` are discarded so a stale-bag rollout cannot artificially inflate the matched-pair count. Heading-error means and maxima are emitted whenever the bag stream carries orientation (`read_navsat_pose_stream` leaves orientations `None` because NavSatFix has no attitude — a future `read_gsof_pose_stream` / `read_imu_pose_stream` slots in without changing the report shape).

The CLI ships at `scripts/run_rosbag_correlation.py`:

```bash
python3 scripts/run_rosbag_correlation.py \
    --bag data/autoware_leo_drive_bag1 \
    --sim-rollout artifacts/rollout/bag1.jsonl \
    --output artifacts/correlation/bag1.json \
    --markdown artifacts/correlation/bag1.md \
    --max-match-dt-seconds 0.05
```

The sim-rollout JSONL is one record per line with `timestampSeconds`, `position` (3-element list), and `orientationXyzw` (4-element list). The script writes the JSON report (with up to `--max-pairs-kept` evenly-strided `CorrelatedPosePair` entries embedded — pass `--no-pairs` to drop them entirely) and an optional Markdown summary suitable for PR / scenario-CI artifact display.

Pass `--imu-topic <topic>` to merge a `sensor_msgs/Imu` orientation stream onto the NavSatFix positions: each NavSatFix sample picks up the nearest-timestamp IMU quaternion within `--imu-pair-dt-seconds` (default 0.05 s), and the correlator's heading-error mean / max fields populate from the resulting `BagPoseStream`. The same composition is available as a library:

```python
from gs_sim2real.robotics import (
    merge_navsat_with_imu_orientation,
    read_imu_orientation_stream,
    read_navsat_pose_stream,
)

bag = ["data/autoware_leo_drive_bag1"]
navsat = read_navsat_pose_stream(bag)
imu = read_imu_orientation_stream(bag)  # /sensing/imu/imu_data on the Autoware Leo Drive bags
fused = merge_navsat_with_imu_orientation(navsat, imu, max_pair_dt_seconds=0.05)
```

NavSatFix samples whose nearest IMU quaternion is more than `max_pair_dt_seconds` away keep `orientation_xyzw=None` so the correlator skips them when reducing heading errors — the mean / max are computed over the matched pairs only.

Pre-computed correlation reports can be attached to a scenario-set run so the headless-vs-bag drift travels alongside the benchmark history report through the same scenario CI artifact channel. Pass one or more `--correlation-report <path>` flags to `gs-mapper route-policy-scenario-set` (repeatable) — each path is loaded via `load_real_vs_sim_correlation_report_json`, embedded into `RoutePolicyScenarioSetRunReport.correlation_reports`, surfaced in `render_route_policy_scenario_set_markdown` as a "Real-vs-sim correlation" table, and round-tripped through `write_route_policy_scenario_set_run_json` / `load_route_policy_scenario_set_run_json`. Library callers can pass the same paths via `run_route_policy_scenario_set(..., correlation_report_paths=[...])`. The report-shape on the wire is `BagPoseStreamMetadata` (the round-trippable summary view of `BagPoseStream`); `BagPoseStream.metadata()` builds it from a live stream and `bag_pose_stream_metadata_from_dict` recovers it from a JSON payload.

`gs-mapper route-policy-scenario-ci-review` lifts those reports up into the Pages-hosted review bundle automatically: after loading the shard merge report, the CLI walks each `RoutePolicyScenarioShardRunSummary.run_path`, reads the embedded `correlation_reports` from each shard's run JSON, and threads them into `build_route_policy_scenario_ci_review_artifact(..., correlation_reports=..., correlation_report_paths=...)`. The review's Markdown grows a `## Real-vs-sim correlation` section and the HTML grows a matching `<section>` table so reviewers see headless-vs-bag drift next to the validation / activation / merge gate without checking out the branch. Pass `--no-correlation-reports` to skip the auto-collect step (useful in unit tests with fictional shard `run_path`s, or to keep the review JSON shape unchanged for an existing fixture). Shard summaries whose `run_path` is missing from disk are skipped silently, so a partially-published merge keeps building.

The review CLI also exposes optional regression thresholds backed by `RealVsSimCorrelationThresholds` and `evaluate_real_vs_sim_correlation_thresholds(report, thresholds)`. Pass any subset of `--max-correlation-translation-mean-meters`, `--max-correlation-translation-p95-meters`, `--max-correlation-translation-max-meters`, or `--max-correlation-heading-mean-radians` to `gs-mapper route-policy-scenario-ci-review` to fail the review (and trip the existing `--fail-on-review` exit) when any embedded correlation report exceeds that bound. Failed gates are recorded as `(report_index, bag_topic, failed_checks)` triples on `RoutePolicyScenarioCIReviewArtifact.correlation_failed_reports` and surfaced in the Markdown's `### Correlation gate failures` and HTML's `<h3>` block; `RoutePolicyScenarioCIReviewArtifact.passed` now also requires `correlation_passed`. Reports without heading data skip the heading-mean check (so a NavSatFix-only correlator stays quiet), and unset thresholds default to "do not check this stat" — the review JSON stays bit-identical to a pre-#125 artifact unless at least one bound is populated.

For multi-bag rollouts where one topic should be held to a tighter standard than another, pass `--correlation-thresholds-config <path>` to load a flat `{"<bag_source_topic>": {<thresholds>}}` JSON. Topics that match an override use that override's bounds; everything else falls through to the scalar `--max-correlation-*` defaults. Overrides round-trip through `RoutePolicyScenarioCIReviewArtifact.correlation_threshold_overrides` (default empty) and surface in the Markdown / HTML gate descriptor (e.g. `Gate FAIL  (default: …; per-topic overrides: 1)`). Empty entries inside the JSON are silently dropped on load (they would behave identically to falling through to the default), and the JSON shape is the same one produced by `correlation_threshold_overrides_to_dict` so config files can be programmatically generated and replayed bit-for-bit.

`RealVsSimCorrelationThresholds` also exposes a per-pair distribution gate via `max_pair_translation_error_meters` paired with `max_exceeding_translation_pair_fraction`. When *both* are set the evaluator walks the report's `CorrelatedPosePair` list, counts how many pairs exceed the per-pair bound, and fails with a `translation-pair-distribution` tag when the exceeding fraction is above the allowed limit (e.g. `max_pair_translation_error_meters=0.5` + `max_exceeding_translation_pair_fraction=0.05` rejects any report where more than 5% of pairs sit above 0.5 m). Setting only one of the two fields is treated as "do not check this stat" so partial configurations stay quiet, and reports whose pair list was dropped (e.g. correlator called with `keep_pairs=False`) skip the gate. The CLI flags `--max-correlation-pair-translation-meters` and `--max-correlation-pair-fraction` mirror this on the review side; the Markdown / HTML gate descriptor reads `pair distribution: ≤ 0.05 fraction of pairs above 0.5 m` when the gate is active.

The same shape is mirrored for heading errors via `max_pair_heading_error_radians` + `max_exceeding_heading_pair_fraction` (paired with the `--max-correlation-pair-heading-radians` / `--max-correlation-heading-pair-fraction` CLI flags). Pairs whose `heading_error_radians` is `None` (i.e. NavSatFix-only correlator with no orientation merge) are filtered out of the denominator so the fraction is computed against the heading-bearing subset only — when no pair carries heading data the gate skips silently. The dedicated failure tag is `heading-pair-distribution` and the descriptor reads `heading pair distribution: ≤ 0.05 fraction of pairs above 0.1 rad`.

Set `pair_distribution_strata=N` (CLI: `--correlation-pair-distribution-strata N`) to evaluate the per-pair distribution gates against `N` equal-duration time windows over the bag's `bag_timestamp_seconds` range. Each window's gate runs independently and emits `translation-pair-distribution-window-{i}` / `heading-pair-distribution-window-{i}` failure tags so the review can pinpoint where drift is concentrated (e.g. clean for the first half, then a step-change in the second half). Empty windows skip silently, and `strata=1` (or `None`) keeps the existing aggregate tag for backwards compatibility.

`pair_distribution_strata` also stratifies the aggregate-statistic gates: `mean` / `p95` / `max` are recomputed per window from the strided pair sample (and `heading-mean` from the heading-bearing subset of each window), with the report-level aggregate tags suppressed in favour of `translation-mean-window-{i}` / `translation-p95-window-{i}` / `translation-max-window-{i}` / `heading-mean-window-{i}`. The denominator is the strided pair sample held in `report.pairs` (so per-window aggregates are approximate when `max_pairs_kept` is small relative to the matched pair count) but the per-window stratification is enough to detect drift that begins partway through a bag and would otherwise hide behind a clean aggregate. Windows whose pair list is empty (or whose heading-bearing subset is empty for the heading-mean check) skip silently.

`pair_distribution_strata_mode` (CLI: `--correlation-pair-distribution-strata-mode`) selects how pairs are partitioned. The default `equal-duration` splits by `bag_timestamp_seconds` over the bag's full time range; `equal-pair-count` splits the strided pair list by index so each window holds a near-equal number of pairs (the first `len(pairs) % strata` windows hold one extra pair). Use `equal-pair-count` when the bag has uneven sample density — e.g. a long quiet stretch followed by a burst of activity — and `equal-duration` would leave most windows nearly empty while one window carries the bulk of the pairs. Both modes round-trip through JSON; the mode is omitted from the payload when it equals the default.

When stratification is configured, the review builder also computes per-window aggregate stats via `compute_per_window_correlation_stats(report, strata=…, mode=…)` and stores them on `RoutePolicyScenarioCIReviewArtifact.correlation_per_window_stats` as `(report_index, bag_topic, tuple[RealVsSimCorrelationWindowStats, …])`. Each `RealVsSimCorrelationWindowStats` carries the window index, pair count, bag-time span (`bag_time_start_seconds` / `bag_time_end_seconds`), the translation mean / p95 / max, and the heading mean (or `None` when no pair in the window has heading data). The Markdown gains a `### Per-window correlation stats` section and the HTML grows a matching `<h3>` table per topic so reviewers can see *which* window is drifting and how much, not just whether a window failed the gate. Empty windows are omitted from the table; the field defaults to an empty tuple so existing review fixtures without stratification keep their exact JSON shape.

## Next Implementation Layer

The scenario CI chain from matrix expansion through promotion-backed adoption is now covered by `scripts/smoke_route_policy_scenario_ci.py`, with both library API (`adopt_route_policy_scenario_ci_workflow`) and CLI surface (`gs-mapper route-policy-scenario-ci-workflow-adopt`). The review bundle is adoption-aware: passing `--adoption-report` to the review CLI (or `adoption=` to `build_route_policy_scenario_ci_review_artifact`) embeds the trigger mode, branches, and unified manual-vs-adopted YAML diff into the Pages-hosted bundle. The next useful layer is surfacing the reviews on the `/reviews/` Pages index so discovery no longer requires knowing the bundle URL in advance.
