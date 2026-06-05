"""Policy trace events emitted by route / imitation policies.

A *policy trace* is a JSONL stream of point-in-time events (one
``PolicyTraceEvent`` per line) such as ``goal_reached``, ``collision``,
or ``near_miss``. These are independent of any bag recording; they
describe what the policy *observed* during a rollout.

The trace is convertible to ``CorrelationEventWindow`` records consumed
by the ``event-aligned`` real-vs-sim correlation stratification gate
(see :mod:`gs_sim2real.robotics.rosbag_correlation` and
``--correlation-event-windows``). This module owns the policy-side of
that handoff; the rosbag correlation module already reserves
``source="policy_trace"`` on :class:`CorrelationEventWindow` for it.

Sprint 3 / PR C scope: emit trace events from an *existing*
:class:`RoutePolicyDatasetExport` (terminal goal_reached / collision /
truncated, plus optional per-step near_miss thresholded on a clearance
feature), and ship the JSONL <-> event windows conversion. The
emission path is offline / post-hoc on purpose; live emission inside
the rollout loop is a follow-up.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from types import TracebackType
from typing import IO, Any, Protocol, runtime_checkable

from ..robotics.rosbag_correlation import (
    CORRELATION_EVENT_WINDOWS_VERSION,
    CorrelationEventWindow,
)
from .policy_dataset import (
    RoutePolicyDatasetExport,
    RoutePolicyEpisodeRecord,
    RoutePolicyTransitionRecord,
)
from .policy_replay import load_route_policy_dataset_json


ROUTE_POLICY_TRACE_EVENT_VERSION = "gs-mapper-route-policy-trace-event/v1"

_BUILTIN_POLICY_TRACE_EVENT_NAMES: frozenset[str] = frozenset(
    {"goal_reached", "collision", "near_miss", "truncated", "terminated"}
)


@dataclass(frozen=True, slots=True)
class PolicyTraceEvent:
    """One point-in-time event emitted by a route / imitation policy.

    ``timestamp_seconds`` is policy-local (simulation) time. Convert to
    bag time at window-build time via ``time_offset_seconds``; the
    offset is intentionally kept out of the stored event so the same
    trace can be aligned to multiple bags.

    ``event_name`` is free-form so policies can emit custom event kinds,
    but the well-known names in :data:`_BUILTIN_POLICY_TRACE_EVENT_NAMES`
    are what the default conversion path treats specially.
    """

    event_name: str
    timestamp_seconds: float
    episode_id: str
    episode_index: int
    step_index: int
    bag_timestamp_seconds: float | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.event_name)
        if not name:
            raise ValueError("PolicyTraceEvent event_name must not be empty")
        object.__setattr__(self, "event_name", name)
        timestamp = float(self.timestamp_seconds)
        if not math.isfinite(timestamp):
            raise ValueError("PolicyTraceEvent timestamp_seconds must be finite")
        object.__setattr__(self, "timestamp_seconds", timestamp)
        episode_id = str(self.episode_id)
        if not episode_id:
            raise ValueError("PolicyTraceEvent episode_id must not be empty")
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "episode_index", int(self.episode_index))
        object.__setattr__(self, "step_index", int(self.step_index))
        if self.bag_timestamp_seconds is not None:
            bag_ts = float(self.bag_timestamp_seconds)
            if not math.isfinite(bag_ts):
                raise ValueError("PolicyTraceEvent bag_timestamp_seconds must be finite")
            object.__setattr__(self, "bag_timestamp_seconds", bag_ts)
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "recordType": "route-policy-trace-event",
            "version": ROUTE_POLICY_TRACE_EVENT_VERSION,
            "eventName": self.event_name,
            "timestampSeconds": float(self.timestamp_seconds),
            "episodeId": self.episode_id,
            "episodeIndex": int(self.episode_index),
            "stepIndex": int(self.step_index),
        }
        if self.bag_timestamp_seconds is not None:
            payload["bagTimestampSeconds"] = float(self.bag_timestamp_seconds)
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def policy_trace_event_from_dict(payload: Mapping[str, Any]) -> PolicyTraceEvent:
    """Rebuild :class:`PolicyTraceEvent` from a JSON payload."""

    record_type = payload.get("recordType")
    if record_type is not None and record_type != "route-policy-trace-event":
        raise ValueError(
            f"unexpected policy trace event recordType: {record_type!r} (expected 'route-policy-trace-event')"
        )
    version = payload.get("version")
    if version is not None and version != ROUTE_POLICY_TRACE_EVENT_VERSION:
        raise ValueError(
            f"unsupported policy trace event version: {version!r} (expected {ROUTE_POLICY_TRACE_EVENT_VERSION!r})"
        )
    tags_payload = payload.get("tags") or ()
    if isinstance(tags_payload, (str, bytes, bytearray)):
        raise ValueError("PolicyTraceEvent tags must be a list of strings, not a string")
    metadata_payload = payload.get("metadata") or {}
    if not isinstance(metadata_payload, Mapping):
        raise ValueError("PolicyTraceEvent metadata must be a mapping")
    bag_ts_payload = payload.get("bagTimestampSeconds")
    return PolicyTraceEvent(
        event_name=str(payload["eventName"]),
        timestamp_seconds=float(payload["timestampSeconds"]),
        episode_id=str(payload["episodeId"]),
        episode_index=int(payload["episodeIndex"]),
        step_index=int(payload["stepIndex"]),
        bag_timestamp_seconds=None if bag_ts_payload is None else float(bag_ts_payload),
        tags=tuple(str(tag) for tag in tags_payload),
        metadata=dict(metadata_payload),
    )


def write_policy_trace_jsonl(
    path: str | Path,
    events: Iterable[PolicyTraceEvent],
) -> Path:
    """Write a policy trace as JSONL (one event per line, stable key order)."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for event in events:
        lines.append(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False))
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def load_policy_trace_jsonl(path: str | Path) -> tuple[PolicyTraceEvent, ...]:
    """Load a policy trace JSONL into PolicyTraceEvent records."""

    text = Path(path).read_text(encoding="utf-8")
    events: list[PolicyTraceEvent] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, Mapping):
            raise ValueError("policy trace JSONL lines must be JSON objects")
        events.append(policy_trace_event_from_dict(payload))
    return tuple(events)


def extract_policy_trace_events_from_dataset(
    dataset: RoutePolicyDatasetExport,
    *,
    segment_duration_seconds: float = 1.0,
    near_miss_clearance_meters: float | None = None,
    near_miss_feature_key: str = "nearest-dynamic-obstacle-clearance-meters",
    time_offset_seconds: float = 0.0,
    base_episode_offsets_seconds: Mapping[str, float] | None = None,
) -> tuple[PolicyTraceEvent, ...]:
    """Extract trace events from an existing benchmark dataset export.

    Emits, per episode:

    - exactly one terminal event named after the final transition's
      termination reason (``goal_reached`` / ``collision`` / ``truncated``
      / ``terminated``); falls back to ``terminated`` for unknown reasons,
    - one ``near_miss`` event per transition whose
      ``next_observation[near_miss_feature_key]`` is below
      ``near_miss_clearance_meters`` (when both are set).

    Timestamps are synthesized as
    ``time_offset_seconds + (base_episode_offsets_seconds[episode_id] or 0)
    + (step_index + 1) * segment_duration_seconds`` so the trace lives on
    a stable timeline. Callers that need bag-aligned timestamps can pass
    ``base_episode_offsets_seconds`` (episode_id -> bag-time offset of
    that episode's first step), or shift the whole trace via
    ``time_offset_seconds``.
    """

    if segment_duration_seconds <= 0.0 or not math.isfinite(segment_duration_seconds):
        raise ValueError("segment_duration_seconds must be positive and finite")
    offsets: Mapping[str, float] = base_episode_offsets_seconds or {}
    events: list[PolicyTraceEvent] = []
    for episode in dataset.episodes:
        episode_offset = float(offsets.get(episode.episode_id, 0.0))
        for transition in episode.transitions:
            if near_miss_clearance_meters is not None and _maybe_below_threshold(
                transition.next_observation, near_miss_feature_key, near_miss_clearance_meters
            ):
                events.append(
                    _build_event(
                        event_name="near_miss",
                        episode=episode,
                        transition=transition,
                        segment_duration_seconds=segment_duration_seconds,
                        time_offset_seconds=time_offset_seconds + episode_offset,
                        metadata={
                            "clearanceMeters": float(transition.next_observation[near_miss_feature_key]),
                            "thresholdMeters": float(near_miss_clearance_meters),
                            "featureKey": near_miss_feature_key,
                        },
                    )
                )
        if episode.transitions:
            terminal_event_name = _terminal_event_name(episode)
            if terminal_event_name is not None:
                events.append(
                    _build_event(
                        event_name=terminal_event_name,
                        episode=episode,
                        transition=episode.transitions[-1],
                        segment_duration_seconds=segment_duration_seconds,
                        time_offset_seconds=time_offset_seconds + episode_offset,
                        metadata=_terminal_metadata(episode),
                    )
                )
    return tuple(events)


def convert_policy_trace_events_to_event_windows(
    events: Iterable[PolicyTraceEvent],
    *,
    half_width_seconds: float = 0.5,
    time_offset_seconds: float = 0.0,
    name_template: str = "{event_name}-{episode_id}-{step_index}",
) -> tuple[CorrelationEventWindow, ...]:
    """Convert point trace events into CorrelationEventWindow records.

    Each event becomes a single window
    ``[t - half_width, t + half_width]`` where ``t`` is
    ``bag_timestamp_seconds`` when set, else
    ``timestamp_seconds + time_offset_seconds``. ``source`` is forced to
    ``policy_trace`` so the event windows can be merged with externally
    sourced windows without losing provenance. ``tags`` is the event's
    tags plus a ``policy-trace`` marker plus the raw event name (so
    threshold overrides keyed on tags can target a specific event kind).

    For PR C we keep one-window-per-event semantics; contiguous near_miss
    merging is a follow-up refinement.
    """

    if half_width_seconds <= 0.0 or not math.isfinite(half_width_seconds):
        raise ValueError("half_width_seconds must be positive and finite")
    windows: list[CorrelationEventWindow] = []
    for event in events:
        if event.bag_timestamp_seconds is not None:
            centre = float(event.bag_timestamp_seconds)
        else:
            centre = float(event.timestamp_seconds) + float(time_offset_seconds)
        name = name_template.format(
            event_name=event.event_name,
            episode_id=event.episode_id,
            episode_index=event.episode_index,
            step_index=event.step_index,
        )
        tag_set: tuple[str, ...] = (
            "policy-trace",
            f"event:{event.event_name}",
            *event.tags,
        )
        windows.append(
            CorrelationEventWindow(
                name=name,
                start_time=centre - half_width_seconds,
                end_time=centre + half_width_seconds,
                tags=tag_set,
                source="policy_trace",
            )
        )
    return tuple(windows)


def write_correlation_event_windows_json(
    path: str | Path,
    windows: Iterable[CorrelationEventWindow],
) -> Path:
    """Write CorrelationEventWindow records as a v1 event-windows JSON file.

    Provided here (rather than in :mod:`gs_sim2real.robotics.rosbag_correlation`)
    so callers building windows from a policy trace have a one-stop module
    for the conversion; the rosbag side stays read-only for now.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recordType": CORRELATION_EVENT_WINDOWS_VERSION,
        "windows": [window.to_dict() for window in windows],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def run_dataset_to_trace_cli(args: argparse.Namespace) -> None:
    """Handle the ``route-policy-dataset-to-trace`` subcommand."""

    dataset_path = Path(getattr(args, "dataset"))
    dataset = load_route_policy_dataset_json(dataset_path)
    near_miss_threshold: float | None = (
        None
        if getattr(args, "near_miss_clearance_meters", None) is None
        else float(getattr(args, "near_miss_clearance_meters"))
    )
    events = extract_policy_trace_events_from_dataset(
        dataset,
        segment_duration_seconds=float(getattr(args, "segment_duration_seconds")),
        near_miss_clearance_meters=near_miss_threshold,
        near_miss_feature_key=str(getattr(args, "near_miss_feature_key")),
        time_offset_seconds=float(getattr(args, "time_offset_seconds")),
    )
    output_path = write_policy_trace_jsonl(getattr(args, "output"), events)
    print(
        f"Wrote {len(events)} policy trace event(s) to {output_path}",
        file=sys.stdout,
    )


def run_trace_to_event_windows_cli(args: argparse.Namespace) -> None:
    """Handle the ``route-policy-trace-to-event-windows`` subcommand."""

    events = load_policy_trace_jsonl(getattr(args, "trace"))
    windows = convert_policy_trace_events_to_event_windows(
        events,
        half_width_seconds=float(getattr(args, "half_width_seconds")),
        time_offset_seconds=float(getattr(args, "time_offset_seconds")),
        name_template=str(getattr(args, "name_template")),
    )
    output_path = write_correlation_event_windows_json(getattr(args, "output"), windows)
    print(
        f"Wrote {len(windows)} correlation event window(s) to {output_path}",
        file=sys.stdout,
    )


def _build_event(
    *,
    event_name: str,
    episode: RoutePolicyEpisodeRecord,
    transition: RoutePolicyTransitionRecord,
    segment_duration_seconds: float,
    time_offset_seconds: float,
    metadata: Mapping[str, Any],
) -> PolicyTraceEvent:
    timestamp = time_offset_seconds + float(transition.step_index + 1) * segment_duration_seconds
    return PolicyTraceEvent(
        event_name=event_name,
        timestamp_seconds=timestamp,
        episode_id=transition.episode_id,
        episode_index=transition.episode_index,
        step_index=transition.step_index,
        tags=(f"scene:{episode.scene_id}",),
        metadata=dict(metadata),
    )


def _terminal_event_name(episode: RoutePolicyEpisodeRecord) -> str | None:
    if not episode.transitions:
        return None
    summary = episode.summary()
    if bool(summary.get("goalReached", False)):
        return "goal_reached"
    if bool(summary.get("blocked", False)):
        return "collision"
    if episode.truncated:
        return "truncated"
    if episode.terminated:
        return "terminated"
    return None


def _terminal_metadata(episode: RoutePolicyEpisodeRecord) -> dict[str, Any]:
    summary = episode.summary()
    metadata: dict[str, Any] = {
        "stepCount": int(summary.get("stepCount", episode.step_count)),
        "totalReward": float(summary.get("totalReward", episode.total_reward)),
    }
    reason = summary.get("terminationReason")
    if reason is not None:
        metadata["terminationReason"] = str(reason)
    return metadata


def _maybe_below_threshold(
    observation: Mapping[str, Any],
    key: str,
    threshold: float,
) -> bool:
    value = observation.get(key)
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(numeric):
        return False
    return numeric <= threshold


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
        if not math.isfinite(value):
            raise ValueError("JSON float values must be finite")
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


@runtime_checkable
class PolicyTraceEventStream(Protocol):
    """Sink for live :class:`PolicyTraceEvent` records.

    Implementations are expected to be append-only and to flush eagerly so
    that crashes mid-rollout still leave a usable trace on disk. ``close``
    must be idempotent.
    """

    def emit(self, event: PolicyTraceEvent) -> None: ...

    def close(self) -> None: ...


class JsonlPolicyTraceEventStream(AbstractContextManager["JsonlPolicyTraceEventStream"]):
    """JSONL-backed :class:`PolicyTraceEventStream`.

    One line per event, ``json.dumps(..., sort_keys=True)`` for stable diff
    output, ``flush`` after every write so a crashed rollout still produces
    a parseable JSONL prefix. Re-opening an existing file appends.
    """

    def __init__(self, path: str | Path, *, mode: str = "a") -> None:
        if mode not in ("a", "w"):
            raise ValueError("JsonlPolicyTraceEventStream mode must be 'a' or 'w'")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] | None = self._path.open(mode, encoding="utf-8")
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event: PolicyTraceEvent) -> None:
        if self._closed or self._handle is None:
            raise RuntimeError("JsonlPolicyTraceEventStream is closed")
        self._handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class PolicyTraceEmissionConfig:
    """Configuration for :class:`RoutePolicyTraceEmitter`.

    The defaults match :func:`extract_policy_trace_events_from_dataset` so
    a live trace and a post-hoc trace of the same rollout produce the same
    event timeline. ``near_miss_feature_key`` defaults to the key the
    Gym adapter actually emits (``nearest-dynamic-obstacle-distance-meters``).
    """

    segment_duration_seconds: float = 1.0
    near_miss_clearance_meters: float | None = None
    near_miss_feature_key: str = "nearest-dynamic-obstacle-distance-meters"
    near_miss_edge_only: bool = True
    time_offset_seconds: float = 0.0
    episode_id_template: str = "{scene_id}-episode-{episode_index}"

    def __post_init__(self) -> None:
        if self.segment_duration_seconds <= 0.0 or not math.isfinite(self.segment_duration_seconds):
            raise ValueError("segment_duration_seconds must be positive and finite")
        if self.near_miss_clearance_meters is not None:
            threshold = float(self.near_miss_clearance_meters)
            if not math.isfinite(threshold):
                raise ValueError("near_miss_clearance_meters must be finite")
            object.__setattr__(self, "near_miss_clearance_meters", threshold)
        if not str(self.near_miss_feature_key):
            raise ValueError("near_miss_feature_key must not be empty")
        if "{episode_index}" not in self.episode_id_template:
            raise ValueError("episode_id_template must reference {episode_index} to stay unique per episode")


class RoutePolicyTraceEmitter:
    """State machine for emitting trace events from a *live* rollout.

    The emitter is intentionally decoupled from the Gym adapter: the
    adapter (or any custom rollout loop) calls :meth:`begin_episode` once
    per ``reset`` and :meth:`record_step` once per ``step``. The emitter
    handles terminal-event dispatch, near-miss edge detection, sim-time
    synthesis, and stream forwarding so callers stay simple.

    Returned events are also pushed to ``stream`` (when one is set); the
    return value is exposed mainly so tests can assert on emission without
    needing a stream sink.
    """

    def __init__(
        self,
        stream: PolicyTraceEventStream | None = None,
        config: PolicyTraceEmissionConfig | None = None,
    ) -> None:
        self._stream = stream
        self._config = config or PolicyTraceEmissionConfig()
        self._near_miss_below: bool = False
        self._episode_open: bool = False
        self._episode_id: str | None = None

    @property
    def config(self) -> PolicyTraceEmissionConfig:
        return self._config

    def begin_episode(self, *, scene_id: str, episode_index: int) -> str:
        """Mark the start of a new episode and return its synthesized id."""

        self._near_miss_below = False
        self._episode_open = True
        episode_id = self._config.episode_id_template.format(
            scene_id=str(scene_id),
            episode_index=int(episode_index),
        )
        self._episode_id = episode_id
        return episode_id

    def record_step(
        self,
        *,
        scene_id: str,
        episode_index: int,
        step_index: int,
        next_observation: Mapping[str, Any],
        blocked: bool,
        goal_reached: bool,
        truncated: bool,
        terminated: bool,
        extra_tags: Sequence[str] = (),
    ) -> tuple[PolicyTraceEvent, ...]:
        """Emit any events for the just-completed step, return them in order."""

        if not self._episode_open or self._episode_id is None:
            raise RuntimeError("RoutePolicyTraceEmitter.record_step called without begin_episode")
        events: list[PolicyTraceEvent] = []
        config = self._config
        timestamp = config.time_offset_seconds + float(step_index + 1) * config.segment_duration_seconds
        clearance_payload = next_observation.get(config.near_miss_feature_key)
        clearance_numeric = _coerce_finite_float(clearance_payload)
        threshold = config.near_miss_clearance_meters
        is_below_now = threshold is not None and clearance_numeric is not None and clearance_numeric <= threshold
        should_emit_near_miss = (
            threshold is not None and is_below_now and (not config.near_miss_edge_only or not self._near_miss_below)
        )
        if should_emit_near_miss:
            assert threshold is not None  # narrowed above
            assert clearance_numeric is not None
            events.append(
                PolicyTraceEvent(
                    event_name="near_miss",
                    timestamp_seconds=timestamp,
                    episode_id=self._episode_id,
                    episode_index=int(episode_index),
                    step_index=int(step_index),
                    tags=(f"scene:{scene_id}", *extra_tags),
                    metadata={
                        "clearanceMeters": float(clearance_numeric),
                        "thresholdMeters": float(threshold),
                        "featureKey": config.near_miss_feature_key,
                    },
                )
            )
        self._near_miss_below = bool(is_below_now)

        terminal_name = _live_terminal_event_name(
            blocked=blocked,
            goal_reached=goal_reached,
            truncated=truncated,
            terminated=terminated,
        )
        if terminal_name is not None:
            events.append(
                PolicyTraceEvent(
                    event_name=terminal_name,
                    timestamp_seconds=timestamp,
                    episode_id=self._episode_id,
                    episode_index=int(episode_index),
                    step_index=int(step_index),
                    tags=(f"scene:{scene_id}", *extra_tags),
                    metadata={
                        "terminationReason": _termination_reason_from_flags(
                            blocked=blocked,
                            goal_reached=goal_reached,
                            truncated=truncated,
                        ),
                    },
                )
            )
            self._episode_open = False

        if self._stream is not None:
            for event in events:
                self._stream.emit(event)
        return tuple(events)

    def close(self) -> None:
        """Close the underlying stream (idempotent)."""

        if self._stream is not None:
            self._stream.close()


def _live_terminal_event_name(
    *,
    blocked: bool,
    goal_reached: bool,
    truncated: bool,
    terminated: bool,
) -> str | None:
    if goal_reached:
        return "goal_reached"
    if blocked:
        return "collision"
    if truncated:
        return "truncated"
    if terminated:
        return "terminated"
    return None


def _termination_reason_from_flags(
    *,
    blocked: bool,
    goal_reached: bool,
    truncated: bool,
) -> str:
    if goal_reached:
        return "goal-reached"
    if blocked:
        return "blocked-route"
    if truncated:
        return "max-steps"
    return "terminated"


def _coerce_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


__all__ = [
    "ROUTE_POLICY_TRACE_EVENT_VERSION",
    "JsonlPolicyTraceEventStream",
    "PolicyTraceEmissionConfig",
    "PolicyTraceEvent",
    "PolicyTraceEventStream",
    "RoutePolicyTraceEmitter",
    "convert_policy_trace_events_to_event_windows",
    "extract_policy_trace_events_from_dataset",
    "load_policy_trace_jsonl",
    "policy_trace_event_from_dict",
    "run_dataset_to_trace_cli",
    "run_trace_to_event_windows_cli",
    "write_correlation_event_windows_json",
    "write_policy_trace_jsonl",
]
