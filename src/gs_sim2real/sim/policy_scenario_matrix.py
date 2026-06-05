"""Matrix expansion for route policy scenario-set artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
import json
from os import path as os_path
from pathlib import Path
import re
from typing import Any

from .contract import DEFAULT_SITE_URL
from .policy_quality import RoutePolicyQualityThresholds
from .policy_scenario_multi_agent import (
    AgentRoleSpec,
    InteractionMetricsSpec,
    PopulationSpec,
    agent_role_spec_from_dict,
    interaction_metrics_spec_from_dict,
    population_spec_from_dict,
)
from .policy_scenario_set import (
    RoutePolicyScenarioSet,
    RoutePolicyScenarioSpec,
    route_policy_scenario_set_from_dict,
    write_route_policy_scenario_set_json,
)


ROUTE_POLICY_SCENARIO_MATRIX_VERSION = "gs-mapper-route-policy-scenario-matrix/v1"
ROUTE_POLICY_SCENARIO_MATRIX_EXPANSION_VERSION = "gs-mapper-route-policy-scenario-matrix-expansion/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyMatrixRegistrySpec:
    """One policy registry axis value in a scenario matrix."""

    registry_id: str
    policy_registry_path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.registry_id):
            raise ValueError("registry_id must not be empty")
        if not str(self.policy_registry_path):
            raise ValueError("policy_registry_path must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-matrix-registry",
            "registryId": self.registry_id,
            "policyRegistryPath": self.policy_registry_path,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyMatrixSceneSpec:
    """One scene axis value in a scenario matrix.

    ``agents`` / ``population`` / ``interaction_metrics`` are optional
    Sprint 4 (PR D / D2) extensions for multi-agent scenarios. Empty /
    ``None`` means a legacy ego-only scenario, which keeps the matrix
    expansion behaviour identical to pre-Sprint-4. ``agents`` and
    ``population`` are mutually exclusive: the former declares peers
    one-by-one, the latter samples them from a distribution.
    """

    scene_key: str
    scene_catalog: str
    scene_id: str | None = None
    site_url: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    agents: tuple[AgentRoleSpec, ...] = ()
    population: PopulationSpec | None = None
    interaction_metrics: InteractionMetricsSpec | None = None

    def __post_init__(self) -> None:
        if not str(self.scene_key):
            raise ValueError("scene_key must not be empty")
        if not str(self.scene_catalog):
            raise ValueError("scene_catalog must not be empty")
        if self.agents and self.population is not None:
            raise ValueError("agents and population are mutually exclusive on a scene spec")
        if self.agents:
            agent_ids = tuple(spec.agent_id for spec in self.agents)
            if len(set(agent_ids)) != len(agent_ids):
                raise ValueError("agents must have unique agent_id values")
            ego_count = sum(1 for spec in self.agents if spec.role == "ego")
            if ego_count > 1:
                raise ValueError("scene spec may declare at most one ego agent")

    @property
    def is_multi_agent(self) -> bool:
        """True when the scene declares peers via agents or population."""

        if self.population is not None:
            return True
        return any(spec.role != "ego" for spec in self.agents)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "recordType": "route-policy-matrix-scene",
            "sceneKey": self.scene_key,
            "sceneCatalog": self.scene_catalog,
            "sceneId": self.scene_id,
            "siteUrl": self.site_url,
            "metadata": _json_mapping(self.metadata),
        }
        if self.agents:
            payload["agents"] = [spec.to_dict() for spec in self.agents]
        if self.population is not None:
            payload["population"] = self.population.to_dict()
        if self.interaction_metrics is not None:
            payload["interactionMetrics"] = self.interaction_metrics.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class RoutePolicyMatrixGoalSuiteSpec:
    """One goal-suite axis value in a scenario matrix."""

    goal_suite_key: str
    goal_suite_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.goal_suite_key):
            raise ValueError("goal_suite_key must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-matrix-goal-suite",
            "goalSuiteKey": self.goal_suite_key,
            "goalSuitePath": self.goal_suite_path,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyMatrixConfigSpec:
    """One simulator/evaluation config axis value in a scenario matrix."""

    config_id: str
    episode_count: int | None = None
    seed_start: int | None = None
    max_steps: int | None = None
    thresholds: RoutePolicyQualityThresholds | None = None
    sensor_noise_profile_path: str | None = None
    raw_sensor_noise_profile_path: str | None = None
    dynamic_obstacles_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.config_id):
            raise ValueError("config_id must not be empty")
        if self.episode_count is not None:
            _positive_int(self.episode_count, "episode_count")
        if self.seed_start is not None:
            _non_negative_int(self.seed_start, "seed_start")
        if self.max_steps is not None:
            _positive_int(self.max_steps, "max_steps")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-matrix-config",
            "configId": self.config_id,
            "episodeCount": self.episode_count,
            "seedStart": self.seed_start,
            "maxSteps": self.max_steps,
            "thresholds": None if self.thresholds is None else self.thresholds.to_dict(),
            "sensorNoiseProfilePath": self.sensor_noise_profile_path,
            "rawSensorNoiseProfilePath": self.raw_sensor_noise_profile_path,
            "dynamicObstaclesPath": self.dynamic_obstacles_path,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioMatrix:
    """Compact Cartesian product that expands into one scenario set per registry."""

    matrix_id: str
    registries: tuple[RoutePolicyMatrixRegistrySpec, ...]
    scenes: tuple[RoutePolicyMatrixSceneSpec, ...]
    goal_suites: tuple[RoutePolicyMatrixGoalSuiteSpec, ...]
    configs: tuple[RoutePolicyMatrixConfigSpec, ...]
    episode_count: int = 16
    seed_start: int = 100
    max_steps: int | None = None
    include_direct_baseline: bool = False
    site_url: str = DEFAULT_SITE_URL
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_MATRIX_VERSION

    def __post_init__(self) -> None:
        if not str(self.matrix_id):
            raise ValueError("matrix_id must not be empty")
        _require_unique("registry_id", tuple(registry.registry_id for registry in self.registries))
        _require_unique("scene_key", tuple(scene.scene_key for scene in self.scenes))
        _require_unique("goal_suite_key", tuple(goal.goal_suite_key for goal in self.goal_suites))
        _require_unique("config_id", tuple(config.config_id for config in self.configs))
        _require_unique("registry slug", tuple(_slug(registry.registry_id) for registry in self.registries))
        _require_unique("scene slug", tuple(_slug(scene.scene_key) for scene in self.scenes))
        _require_unique("goal suite slug", tuple(_slug(goal.goal_suite_key) for goal in self.goal_suites))
        _require_unique("config slug", tuple(_slug(config.config_id) for config in self.configs))
        _positive_int(self.episode_count, "episode_count")
        _non_negative_int(self.seed_start, "seed_start")
        if self.max_steps is not None:
            _positive_int(self.max_steps, "max_steps")

    @property
    def scenario_set_count(self) -> int:
        return len(self.registries)

    @property
    def scenario_count_per_set(self) -> int:
        suite_config = len(self.goal_suites) * len(self.configs)
        scene_seed_total = sum(len(_scene_population_seeds(scene)) for scene in self.scenes)
        return scene_seed_total * suite_config

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-matrix",
            "version": self.version,
            "matrixId": self.matrix_id,
            "scenarioSetCount": self.scenario_set_count,
            "scenarioCountPerSet": self.scenario_count_per_set,
            "episodeCount": self.episode_count,
            "seedStart": self.seed_start,
            "maxSteps": self.max_steps,
            "includeDirectBaseline": self.include_direct_baseline,
            "siteUrl": self.site_url,
            "registries": [registry.to_dict() for registry in self.registries],
            "scenes": [scene.to_dict() for scene in self.scenes],
            "goalSuites": [goal_suite.to_dict() for goal_suite in self.goal_suites],
            "configs": [config.to_dict() for config in self.configs],
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioMatrixOutput:
    """One scenario-set file written by a matrix expansion."""

    scenario_set_id: str
    registry_id: str
    policy_registry_path: str
    scenario_count: int
    scenario_set_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-matrix-output",
            "scenarioSetId": self.scenario_set_id,
            "registryId": self.registry_id,
            "policyRegistryPath": self.policy_registry_path,
            "scenarioCount": self.scenario_count,
            "scenarioSetPath": self.scenario_set_path,
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioMatrixExpansionReport:
    """Versioned expansion report for generated scenario-set artifacts."""

    matrix_id: str
    scenario_sets: tuple[RoutePolicyScenarioSet, ...]
    outputs: tuple[RoutePolicyScenarioMatrixOutput, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_MATRIX_EXPANSION_VERSION

    @property
    def scenario_set_count(self) -> int:
        return len(self.scenario_sets)

    @property
    def scenario_count(self) -> int:
        return sum(scenario_set.scenario_count for scenario_set in self.scenario_sets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-matrix-expansion",
            "version": self.version,
            "matrixId": self.matrix_id,
            "scenarioSetCount": self.scenario_set_count,
            "scenarioCount": self.scenario_count,
            "outputs": [output.to_dict() for output in self.outputs],
            "scenarioSets": [scenario_set.to_dict() for scenario_set in self.scenario_sets],
            "metadata": _json_mapping(self.metadata),
        }


def expand_route_policy_scenario_matrix(matrix: RoutePolicyScenarioMatrix) -> tuple[RoutePolicyScenarioSet, ...]:
    """Expand a compact matrix into one scenario set per policy registry.

    When a scene carries a :class:`PopulationSpec`, the expander fans the
    scene out across ``population.seed_count`` distinct seeds, producing
    one scenario per ``(scene, goal_suite, config, seed)`` tuple. Each
    fanned scenario id carries a ``-seed-<n>`` suffix; legacy ego-only
    scenes (``population is None``) emit a single seedless id as before.
    """

    scenario_sets: list[RoutePolicyScenarioSet] = []
    for registry in matrix.registries:
        scenarios = tuple(
            _scenario_from_axes(
                matrix,
                scene=scene,
                goal_suite=goal_suite,
                config=config,
                registry=registry,
                population_seed=seed,
            )
            for scene, goal_suite, config in product(matrix.scenes, matrix.goal_suites, matrix.configs)
            for seed in _scene_population_seeds(scene)
        )
        scenario_sets.append(
            RoutePolicyScenarioSet(
                scenario_set_id=f"{matrix.matrix_id}-{_slug(registry.registry_id)}",
                policy_registry_path=registry.policy_registry_path,
                episode_count=matrix.episode_count,
                seed_start=matrix.seed_start,
                max_steps=matrix.max_steps,
                include_direct_baseline=matrix.include_direct_baseline,
                site_url=matrix.site_url,
                scenarios=scenarios,
                metadata={
                    "matrixId": matrix.matrix_id,
                    "registryId": registry.registry_id,
                    "registryMetadata": _json_mapping(registry.metadata),
                    **_json_mapping(matrix.metadata),
                },
            )
        )
    return tuple(scenario_sets)


def expand_route_policy_scenario_matrix_to_directory(
    matrix: RoutePolicyScenarioMatrix,
    output_dir: str | Path,
    *,
    matrix_base_path: str | Path | None = None,
) -> RoutePolicyScenarioMatrixExpansionReport:
    """Expand a scenario matrix and write each scenario-set JSON into a directory."""

    directory = Path(output_dir)
    source_base = Path(matrix_base_path) if matrix_base_path is not None else Path(".")
    directory.mkdir(parents=True, exist_ok=True)
    scenario_sets = tuple(
        _rebase_scenario_set_paths(
            scenario_set,
            source_base=source_base,
            target_base=directory,
        )
        for scenario_set in expand_route_policy_scenario_matrix(matrix)
    )
    outputs: list[RoutePolicyScenarioMatrixOutput] = []
    for registry, scenario_set in zip(matrix.registries, scenario_sets, strict=True):
        path = directory / f"{_slug(scenario_set.scenario_set_id)}.json"
        write_route_policy_scenario_set_json(path, scenario_set)
        outputs.append(
            RoutePolicyScenarioMatrixOutput(
                scenario_set_id=scenario_set.scenario_set_id,
                registry_id=registry.registry_id,
                policy_registry_path=registry.policy_registry_path,
                scenario_count=scenario_set.scenario_count,
                scenario_set_path=path.as_posix(),
            )
        )
    return RoutePolicyScenarioMatrixExpansionReport(
        matrix_id=matrix.matrix_id,
        scenario_sets=scenario_sets,
        outputs=tuple(outputs),
        metadata={"outputDir": directory.as_posix()},
    )


def write_route_policy_scenario_matrix_json(path: str | Path, matrix: RoutePolicyScenarioMatrix) -> Path:
    """Write a route policy scenario matrix JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(matrix.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_scenario_matrix_json(path: str | Path) -> RoutePolicyScenarioMatrix:
    """Load a route policy scenario matrix JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_scenario_matrix_from_dict(_mapping(payload, "scenarioMatrix"))


def write_route_policy_scenario_matrix_expansion_json(
    path: str | Path,
    expansion: RoutePolicyScenarioMatrixExpansionReport,
) -> Path:
    """Write a scenario matrix expansion report as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(expansion.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_scenario_matrix_expansion_json(path: str | Path) -> RoutePolicyScenarioMatrixExpansionReport:
    """Load a scenario matrix expansion report JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_scenario_matrix_expansion_from_dict(_mapping(payload, "scenarioMatrixExpansion"))


def route_policy_scenario_matrix_from_dict(payload: Mapping[str, Any]) -> RoutePolicyScenarioMatrix:
    """Rebuild a scenario matrix from JSON."""

    _record_type(payload, "route-policy-scenario-matrix")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_MATRIX_VERSION))
    if version != ROUTE_POLICY_SCENARIO_MATRIX_VERSION:
        raise ValueError(f"unsupported route policy scenario matrix version: {version}")
    return RoutePolicyScenarioMatrix(
        matrix_id=str(payload["matrixId"]),
        registries=tuple(
            route_policy_matrix_registry_spec_from_dict(_mapping(item, "registry"))
            for item in _sequence(payload.get("registries", ()), "registries")
        ),
        scenes=tuple(
            route_policy_matrix_scene_spec_from_dict(_mapping(item, "scene"))
            for item in _sequence(payload.get("scenes", ()), "scenes")
        ),
        goal_suites=tuple(
            route_policy_matrix_goal_suite_spec_from_dict(_mapping(item, "goalSuite"))
            for item in _sequence(payload.get("goalSuites", ()), "goalSuites")
        ),
        configs=tuple(
            route_policy_matrix_config_spec_from_dict(_mapping(item, "config"))
            for item in _sequence(payload.get("configs", ()), "configs")
        ),
        episode_count=int(payload.get("episodeCount", 16)),
        seed_start=int(payload.get("seedStart", 100)),
        max_steps=None if payload.get("maxSteps") is None else int(payload["maxSteps"]),
        include_direct_baseline=bool(payload.get("includeDirectBaseline", False)),
        site_url=DEFAULT_SITE_URL if payload.get("siteUrl") is None else str(payload["siteUrl"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def route_policy_matrix_registry_spec_from_dict(payload: Mapping[str, Any]) -> RoutePolicyMatrixRegistrySpec:
    """Rebuild a matrix registry axis value from JSON."""

    _record_type(payload, "route-policy-matrix-registry")
    return RoutePolicyMatrixRegistrySpec(
        registry_id=str(payload["registryId"]),
        policy_registry_path=str(payload["policyRegistryPath"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_matrix_scene_spec_from_dict(payload: Mapping[str, Any]) -> RoutePolicyMatrixSceneSpec:
    """Rebuild a matrix scene axis value from JSON."""

    _record_type(payload, "route-policy-matrix-scene")
    agents_payload = payload.get("agents") or ()
    if isinstance(agents_payload, (str, bytes, bytearray)) or not isinstance(agents_payload, Sequence):
        raise ValueError("scene 'agents' must be a list of agent role specs")
    population_payload = payload.get("population")
    interaction_metrics_payload = payload.get("interactionMetrics")
    return RoutePolicyMatrixSceneSpec(
        scene_key=str(payload["sceneKey"]),
        scene_catalog=str(payload["sceneCatalog"]),
        scene_id=None if payload.get("sceneId") is None else str(payload["sceneId"]),
        site_url=None if payload.get("siteUrl") is None else str(payload["siteUrl"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        agents=tuple(agent_role_spec_from_dict(_mapping(item, "agents[i]")) for item in agents_payload),
        population=None
        if population_payload is None
        else population_spec_from_dict(_mapping(population_payload, "population")),
        interaction_metrics=None
        if interaction_metrics_payload is None
        else interaction_metrics_spec_from_dict(_mapping(interaction_metrics_payload, "interactionMetrics")),
    )


def route_policy_matrix_goal_suite_spec_from_dict(payload: Mapping[str, Any]) -> RoutePolicyMatrixGoalSuiteSpec:
    """Rebuild a matrix goal-suite axis value from JSON."""

    _record_type(payload, "route-policy-matrix-goal-suite")
    return RoutePolicyMatrixGoalSuiteSpec(
        goal_suite_key=str(payload["goalSuiteKey"]),
        goal_suite_path=None if payload.get("goalSuitePath") is None else str(payload["goalSuitePath"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_matrix_config_spec_from_dict(payload: Mapping[str, Any]) -> RoutePolicyMatrixConfigSpec:
    """Rebuild a matrix config axis value from JSON."""

    _record_type(payload, "route-policy-matrix-config")
    thresholds_payload = payload.get("thresholds")
    return RoutePolicyMatrixConfigSpec(
        config_id=str(payload["configId"]),
        episode_count=None if payload.get("episodeCount") is None else int(payload["episodeCount"]),
        seed_start=None if payload.get("seedStart") is None else int(payload["seedStart"]),
        max_steps=None if payload.get("maxSteps") is None else int(payload["maxSteps"]),
        thresholds=None
        if thresholds_payload is None
        else _quality_thresholds_from_dict(_mapping(thresholds_payload, "thresholds")),
        sensor_noise_profile_path=None
        if payload.get("sensorNoiseProfilePath") is None
        else str(payload["sensorNoiseProfilePath"]),
        raw_sensor_noise_profile_path=None
        if payload.get("rawSensorNoiseProfilePath") is None
        else str(payload["rawSensorNoiseProfilePath"]),
        dynamic_obstacles_path=None
        if payload.get("dynamicObstaclesPath") is None
        else str(payload["dynamicObstaclesPath"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_matrix_expansion_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioMatrixExpansionReport:
    """Rebuild a matrix expansion report from JSON."""

    _record_type(payload, "route-policy-scenario-matrix-expansion")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_MATRIX_EXPANSION_VERSION))
    if version != ROUTE_POLICY_SCENARIO_MATRIX_EXPANSION_VERSION:
        raise ValueError(f"unsupported route policy scenario matrix expansion version: {version}")
    return RoutePolicyScenarioMatrixExpansionReport(
        matrix_id=str(payload["matrixId"]),
        scenario_sets=tuple(
            route_policy_scenario_set_from_dict(_mapping(item, "scenarioSet"))
            for item in _sequence(payload.get("scenarioSets", ()), "scenarioSets")
        ),
        outputs=tuple(
            route_policy_scenario_matrix_output_from_dict(_mapping(item, "output"))
            for item in _sequence(payload.get("outputs", ()), "outputs")
        ),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def route_policy_scenario_matrix_output_from_dict(payload: Mapping[str, Any]) -> RoutePolicyScenarioMatrixOutput:
    """Rebuild one matrix output entry from JSON."""

    _record_type(payload, "route-policy-scenario-matrix-output")
    return RoutePolicyScenarioMatrixOutput(
        scenario_set_id=str(payload["scenarioSetId"]),
        registry_id=str(payload["registryId"]),
        policy_registry_path=str(payload["policyRegistryPath"]),
        scenario_count=int(payload["scenarioCount"]),
        scenario_set_path=None if payload.get("scenarioSetPath") is None else str(payload["scenarioSetPath"]),
    )


def render_route_policy_scenario_matrix_markdown(expansion: RoutePolicyScenarioMatrixExpansionReport) -> str:
    """Render a compact Markdown summary for a matrix expansion."""

    lines = [
        f"# Route Policy Scenario Matrix: {expansion.matrix_id}",
        f"- Scenario sets: {expansion.scenario_set_count}",
        f"- Scenarios: {expansion.scenario_count}",
        "",
        "| Scenario set | Registry | Scenarios | Path |",
        "| --- | --- | ---: | --- |",
    ]
    for output in expansion.outputs:
        lines.append(
            "| "
            f"{output.scenario_set_id} | "
            f"{output.registry_id} | "
            f"{output.scenario_count} | "
            f"{output.scenario_set_path or 'n/a'} |"
        )
    return "\n".join(lines) + "\n"


def run_cli(args: Any) -> None:
    """Run the route policy scenario-matrix CLI."""

    matrix_path = Path(getattr(args, "matrix"))
    matrix = load_route_policy_scenario_matrix_json(matrix_path)
    expansion = expand_route_policy_scenario_matrix_to_directory(
        matrix,
        getattr(args, "output_dir"),
        matrix_base_path=matrix_path.parent,
    )
    write_route_policy_scenario_matrix_expansion_json(getattr(args, "index_output"), expansion)
    markdown = render_route_policy_scenario_matrix_markdown(expansion)
    if getattr(args, "markdown_output", None):
        output_path = Path(getattr(args, "markdown_output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Scenario matrix expansion saved to: {getattr(args, 'index_output')}")


def _scene_population_seeds(scene: RoutePolicyMatrixSceneSpec) -> tuple[int | None, ...]:
    """Return the seed dimension implied by a scene's optional population."""

    if scene.population is None:
        return (None,)
    return scene.population.seeds()


def _scenario_from_axes(
    matrix: RoutePolicyScenarioMatrix,
    *,
    scene: RoutePolicyMatrixSceneSpec,
    goal_suite: RoutePolicyMatrixGoalSuiteSpec,
    config: RoutePolicyMatrixConfigSpec,
    registry: RoutePolicyMatrixRegistrySpec,
    population_seed: int | None,
) -> RoutePolicyScenarioSpec:
    scenario_id_parts = [_slug(scene.scene_key), _slug(goal_suite.goal_suite_key), _slug(config.config_id)]
    if population_seed is not None:
        scenario_id_parts.append(f"seed-{population_seed}")
    scenario_id = "-".join(scenario_id_parts)
    metadata: dict[str, Any] = {
        "matrixId": matrix.matrix_id,
        "registryId": registry.registry_id,
        "sceneKey": scene.scene_key,
        "goalSuiteKey": goal_suite.goal_suite_key,
        "configId": config.config_id,
        "sceneMetadata": _json_mapping(scene.metadata),
        "goalSuiteMetadata": _json_mapping(goal_suite.metadata),
        "configMetadata": _json_mapping(config.metadata),
    }
    if scene.agents:
        metadata["agents"] = [agent.to_dict() for agent in scene.agents]
    if scene.population is not None:
        metadata["population"] = scene.population.to_dict()
    if scene.interaction_metrics is not None:
        metadata["interactionMetrics"] = scene.interaction_metrics.to_dict()
    if population_seed is not None:
        metadata["populationSeed"] = int(population_seed)
    return RoutePolicyScenarioSpec(
        scenario_id=scenario_id,
        scene_catalog=scene.scene_catalog,
        scene_id=scene.scene_id,
        goal_suite_path=goal_suite.goal_suite_path,
        episode_count=config.episode_count,
        seed_start=config.seed_start,
        max_steps=config.max_steps,
        site_url=scene.site_url,
        thresholds=config.thresholds,
        sensor_noise_profile_path=config.sensor_noise_profile_path,
        raw_sensor_noise_profile_path=config.raw_sensor_noise_profile_path,
        dynamic_obstacles_path=config.dynamic_obstacles_path,
        metadata=metadata,
    )


def _rebase_scenario_set_paths(
    scenario_set: RoutePolicyScenarioSet,
    *,
    source_base: Path,
    target_base: Path,
) -> RoutePolicyScenarioSet:
    return RoutePolicyScenarioSet(
        scenario_set_id=scenario_set.scenario_set_id,
        policy_registry_path=_rebase_optional_path(scenario_set.policy_registry_path, source_base, target_base),
        episode_count=scenario_set.episode_count,
        seed_start=scenario_set.seed_start,
        max_steps=scenario_set.max_steps,
        include_direct_baseline=scenario_set.include_direct_baseline,
        site_url=scenario_set.site_url,
        scenarios=tuple(
            RoutePolicyScenarioSpec(
                scenario_id=scenario.scenario_id,
                scene_catalog=_rebase_path(scenario.scene_catalog, source_base, target_base),
                scene_id=scenario.scene_id,
                goal_suite_path=_rebase_optional_path(scenario.goal_suite_path, source_base, target_base),
                episode_count=scenario.episode_count,
                seed_start=scenario.seed_start,
                max_steps=scenario.max_steps,
                site_url=scenario.site_url,
                thresholds=scenario.thresholds,
                sensor_noise_profile_path=_rebase_optional_path(
                    scenario.sensor_noise_profile_path, source_base, target_base
                ),
                raw_sensor_noise_profile_path=_rebase_optional_path(
                    scenario.raw_sensor_noise_profile_path, source_base, target_base
                ),
                dynamic_obstacles_path=_rebase_optional_path(scenario.dynamic_obstacles_path, source_base, target_base),
                metadata=scenario.metadata,
            )
            for scenario in scenario_set.scenarios
        ),
        metadata=scenario_set.metadata,
        version=scenario_set.version,
    )


def _rebase_optional_path(path_value: str | None, source_base: Path, target_base: Path) -> str | None:
    if path_value is None:
        return None
    return _rebase_path(path_value, source_base, target_base)


def _rebase_path(path_value: str, source_base: Path, target_base: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return path.as_posix()
    resolved = (source_base / path).resolve()
    return Path(os_path.relpath(resolved, start=target_base.resolve())).as_posix()


def _quality_thresholds_from_dict(payload: Mapping[str, Any]) -> RoutePolicyQualityThresholds:
    return RoutePolicyQualityThresholds(
        min_success_rate=float(payload.get("minSuccessRate", 0.8)),
        max_collision_rate=float(payload.get("maxCollisionRate", 0.05)),
        max_truncation_rate=float(payload.get("maxTruncationRate", 0.1)),
        min_mean_reward=None if payload.get("minMeanReward") is None else float(payload["minMeanReward"]),
        min_scene_count=int(payload.get("minSceneCount", 1)),
        min_episode_count=int(payload.get("minEpisodeCount", 1)),
        min_transition_count=int(payload.get("minTransitionCount", 1)),
    )


def _require_unique(field_name: str, values: Sequence[str]) -> None:
    if not values:
        raise ValueError(f"{field_name} values must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} values must be unique")


def _record_type(payload: Mapping[str, Any], expected: str) -> None:
    record_type = payload.get("recordType")
    if record_type != expected:
        raise ValueError(f"expected {expected!r}, got {record_type!r}")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a mapping")


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError(f"{field_name} must be a sequence")


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _positive_int(value: int, field_name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{field_name} must be positive")


def _non_negative_int(value: int, field_name: str) -> None:
    if int(value) < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return slug or "unnamed"
