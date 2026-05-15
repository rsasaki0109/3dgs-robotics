"""Real-vs-sim trajectory correlation against rosbag2 ground truth.

The headless Physical AI environment can already produce a deterministic
sim trajectory (a sequence of agent poses) for any registered scene
backed by an MCD / Autoware bag. What it could not previously do is
*close the loop* against the original bag — answering the obvious
question "how far does our headless rollout drift from the recorded
real-world pose?". This module fills that gap.

Inputs:

* a rosbag2 directory readable by :mod:`rosbags` — the same machinery
  :class:`~gs_sim2real.datasets.mcd.MCDLoader` already relies on, so
  zstd-compressed sqlite3 bags work out of the box.
* a sim trajectory: an ordered sequence of
  :class:`SimPoseSample` with monotonic ``timestamp_seconds``. Most
  callers will load this from a JSONL written by a benchmark / scenario
  runner.
* an optional reference origin ``(latitude, longitude, altitude)`` so
  multiple bags can be expressed in a shared local ENU frame; otherwise
  the first valid bag fix anchors the frame.

The library reads the bag's GNSS topic into a :class:`BagPoseStream`
in metric local-ENU coordinates, then performs a nearest-timestamp
temporal alignment against the sim samples (rejecting matches whose
clock skew exceeds ``max_match_dt_seconds``). Per-pair translation
errors are reduced into min / mean / max / p50 / p95 statistics
inside :class:`RealVsSimCorrelationReport`, and the report is
JSON-serialisable so scenario CI artifacts can carry it alongside
the benchmark report.

Heading-error reduction is included when the bag stream and sim
samples both expose orientation, but the GNSS-only pose stream that
:func:`read_navsat_pose_stream` returns leaves orientations ``None``
(NavSatFix has no attitude). A future ``read_gsof_pose_stream`` /
``read_imu_pose_stream`` can populate orientations and the
heading-error fields fall out automatically.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import bisect
import json
import math
from pathlib import Path
from typing import Any


REAL_VS_SIM_CORRELATION_REPORT_VERSION = "gs-mapper-real-vs-sim-correlation-report/v1"
CORRELATION_EVENT_WINDOWS_VERSION = "gs-mapper-correlation-event-windows/v1"

_PAIR_DISTRIBUTION_STRATA_MODES: frozenset[str] = frozenset({"equal-duration", "equal-pair-count", "event-aligned"})
_CORRELATION_EVENT_SOURCES: frozenset[str] = frozenset({"external", "policy_trace"})


@dataclass(frozen=True, slots=True)
class CorrelationEventWindow:
    """One scenario-phase / event-aligned bag-time window.

    Used by the ``event-aligned`` pair distribution stratification mode to
    bucket :class:`CorrelatedPosePair` instances by which scenario event
    (turn-left, slowdown, obstacle-avoid, …) their bag timestamp falls in.

    ``source`` is included from day one so Sprint 3 can feed
    policy-trace-derived event timelines through the same path without a
    schema change. ``tags`` is a free-form set of labels (e.g. ``corner``,
    ``dynamic``); per-tag threshold overrides are deliberately out of
    scope for the MVP and may be added later via the threshold overrides
    config.
    """

    name: str
    start_time: float
    end_time: float
    tags: tuple[str, ...] = ()
    source: str = "external"

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name:
            raise ValueError("CorrelationEventWindow name must not be empty")
        object.__setattr__(self, "name", name)
        start = float(self.start_time)
        end = float(self.end_time)
        if end <= start:
            raise ValueError(f"CorrelationEventWindow end_time ({end}) must be greater than start_time ({start})")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        source = str(self.source)
        if source not in _CORRELATION_EVENT_SOURCES:
            raise ValueError(
                f"CorrelationEventWindow source must be one of {sorted(_CORRELATION_EVENT_SOURCES)}; got {source!r}"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "startTime": float(self.start_time),
            "endTime": float(self.end_time),
        }
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.source != "external":
            payload["source"] = self.source
        return payload


def correlation_event_window_from_dict(payload: Mapping[str, Any]) -> CorrelationEventWindow:
    """Rebuild :class:`CorrelationEventWindow` from a JSON payload."""

    tags_payload = payload.get("tags") or ()
    if isinstance(tags_payload, (str, bytes, bytearray)):
        raise ValueError("CorrelationEventWindow tags must be a list of strings, not a string")
    return CorrelationEventWindow(
        name=str(payload["name"]),
        start_time=float(payload["startTime"]),
        end_time=float(payload["endTime"]),
        tags=tuple(str(tag) for tag in tags_payload),
        source=str(payload.get("source") or "external"),
    )


def load_correlation_event_windows_json(
    path: str | Path,
) -> tuple[CorrelationEventWindow, ...]:
    """Load a ``gs-mapper-correlation-event-windows/v1`` JSON file."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("correlation event windows JSON root must be an object")
    record_type = raw.get("recordType")
    if record_type != CORRELATION_EVENT_WINDOWS_VERSION:
        raise ValueError(
            f"unexpected correlation event windows recordType: {record_type!r} "
            f"(expected {CORRELATION_EVENT_WINDOWS_VERSION!r})"
        )
    windows_payload = raw.get("windows")
    if windows_payload is None:
        return ()
    if isinstance(windows_payload, (str, bytes, bytearray)) or not isinstance(windows_payload, Sequence):
        raise ValueError("correlation event windows JSON 'windows' must be a list")
    return tuple(correlation_event_window_from_dict(item) for item in windows_payload if isinstance(item, Mapping))


@dataclass(frozen=True, slots=True)
class BagPoseSample:
    """One pose extracted from a rosbag2 topic in local ENU coordinates."""

    timestamp_seconds: float
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestampSeconds": float(self.timestamp_seconds),
            "position": list(self.position),
        }
        if self.orientation_xyzw is not None:
            payload["orientationXyzw"] = list(self.orientation_xyzw)
        return payload


@dataclass(frozen=True, slots=True)
class BagPoseStreamMetadata:
    """Round-trippable provenance + sizing metadata for a bag pose stream.

    The stream's actual samples are too heavy to ship inside a CI
    artifact, so :class:`RealVsSimCorrelationReport` keeps only the
    metadata view. :meth:`BagPoseStream.metadata` builds one of these
    from a live stream; :func:`bag_pose_stream_metadata_from_dict`
    recovers it from a JSON payload.
    """

    frame_id: str
    source_topic: str
    source_msgtype: str
    sample_count: int
    duration_seconds: float
    reference_origin_wgs84: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frameId": self.frame_id,
            "sourceTopic": self.source_topic,
            "sourceMsgtype": self.source_msgtype,
            "sampleCount": int(self.sample_count),
            "durationSeconds": float(self.duration_seconds),
        }
        if self.reference_origin_wgs84 is not None:
            payload["referenceOriginWgs84"] = list(self.reference_origin_wgs84)
        return payload


@dataclass(frozen=True, slots=True)
class BagPoseStream:
    """Ordered bag pose samples plus provenance metadata."""

    samples: tuple[BagPoseSample, ...]
    frame_id: str
    source_topic: str
    source_msgtype: str
    reference_origin_wgs84: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("BagPoseStream must contain at least one sample")
        timestamps = [sample.timestamp_seconds for sample in self.samples]
        if any(b < a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("BagPoseStream samples must be sorted by timestamp")

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def duration_seconds(self) -> float:
        return float(self.samples[-1].timestamp_seconds - self.samples[0].timestamp_seconds)

    def metadata(self) -> BagPoseStreamMetadata:
        """Return the round-trippable metadata view of this stream."""

        return BagPoseStreamMetadata(
            frame_id=self.frame_id,
            source_topic=self.source_topic,
            source_msgtype=self.source_msgtype,
            sample_count=self.sample_count,
            duration_seconds=self.duration_seconds,
            reference_origin_wgs84=self.reference_origin_wgs84,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.metadata().to_dict()


@dataclass(frozen=True, slots=True)
class SimPoseSample:
    """One sim trajectory pose. Orientation is required (renderers always have it)."""

    timestamp_seconds: float
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestampSeconds": float(self.timestamp_seconds),
            "position": list(self.position),
            "orientationXyzw": list(self.orientation_xyzw),
        }


@dataclass(frozen=True, slots=True)
class CorrelatedPosePair:
    """One nearest-timestamp pose match between the bag and sim streams."""

    bag_timestamp_seconds: float
    sim_timestamp_seconds: float
    bag_position: tuple[float, float, float]
    sim_position: tuple[float, float, float]
    translation_error_meters: float
    heading_error_radians: float | None = None

    @property
    def time_offset_seconds(self) -> float:
        return float(self.sim_timestamp_seconds - self.bag_timestamp_seconds)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "bagTimestampSeconds": float(self.bag_timestamp_seconds),
            "simTimestampSeconds": float(self.sim_timestamp_seconds),
            "timeOffsetSeconds": self.time_offset_seconds,
            "bagPosition": list(self.bag_position),
            "simPosition": list(self.sim_position),
            "translationErrorMeters": float(self.translation_error_meters),
        }
        if self.heading_error_radians is not None:
            payload["headingErrorRadians"] = float(self.heading_error_radians)
        return payload


@dataclass(frozen=True, slots=True)
class RealVsSimCorrelationReport:
    """Aggregate statistics for one real-vs-sim correlation run."""

    bag_source: BagPoseStreamMetadata
    sim_sample_count: int
    matched_pair_count: int
    matched_seconds: float
    translation_error_min_meters: float
    translation_error_mean_meters: float
    translation_error_max_meters: float
    translation_error_p50_meters: float
    translation_error_p95_meters: float
    heading_error_mean_radians: float | None = None
    heading_error_max_radians: float | None = None
    pairs: tuple[CorrelatedPosePair, ...] = field(default_factory=tuple)
    version: str = REAL_VS_SIM_CORRELATION_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "recordType": "real-vs-sim-correlation-report",
            "version": self.version,
            "bagSource": self.bag_source.to_dict(),
            "simSampleCount": int(self.sim_sample_count),
            "matchedPairCount": int(self.matched_pair_count),
            "matchedSeconds": float(self.matched_seconds),
            "translationErrorMeters": {
                "min": float(self.translation_error_min_meters),
                "mean": float(self.translation_error_mean_meters),
                "max": float(self.translation_error_max_meters),
                "p50": float(self.translation_error_p50_meters),
                "p95": float(self.translation_error_p95_meters),
            },
            "pairs": [pair.to_dict() for pair in self.pairs],
        }
        if self.heading_error_mean_radians is not None:
            payload["headingErrorRadians"] = {
                "mean": float(self.heading_error_mean_radians),
                "max": float(self.heading_error_max_radians or 0.0),
            }
        return payload


@dataclass(frozen=True, slots=True)
class RealVsSimCorrelationThresholds:
    """Optional regression thresholds for a real-vs-sim correlation gate.

    Each threshold is a hard upper bound on the corresponding statistic
    of one :class:`RealVsSimCorrelationReport`. ``None`` (the default)
    means \"do not check this stat\". When a report exceeds any
    populated threshold, :func:`evaluate_real_vs_sim_correlation_thresholds`
    returns ``passed=False`` along with a list of failure tags
    (``translation-mean`` / ``translation-p95`` / ``translation-max`` /
    ``heading-mean`` / ``translation-pair-distribution``) so callers can
    route the failure into a CI gate or a review bundle annotation.

    The pair-distribution check is active only when both
    ``max_pair_translation_error_meters`` and
    ``max_exceeding_translation_pair_fraction`` are set: the evaluator
    walks the report's :class:`CorrelatedPosePair` list, counts how many
    pairs exceed the per-pair bound, and fails with
    ``translation-pair-distribution`` when the exceeding fraction is
    above the allowed limit. Reports with no pairs (e.g. those produced
    with ``keep_pairs=False`` on the correlator) skip this check.
    """

    max_translation_error_mean_meters: float | None = None
    max_translation_error_p95_meters: float | None = None
    max_translation_error_max_meters: float | None = None
    max_heading_error_mean_radians: float | None = None
    max_pair_translation_error_meters: float | None = None
    max_exceeding_translation_pair_fraction: float | None = None
    max_pair_heading_error_radians: float | None = None
    max_exceeding_heading_pair_fraction: float | None = None
    pair_distribution_strata: int | None = None
    pair_distribution_strata_mode: str = "equal-duration"
    event_windows_path: str | None = None

    def __post_init__(self) -> None:
        if self.pair_distribution_strata is not None:
            if int(self.pair_distribution_strata) < 1:
                raise ValueError("pair_distribution_strata must be >= 1 when set")
            object.__setattr__(self, "pair_distribution_strata", int(self.pair_distribution_strata))
        mode = str(self.pair_distribution_strata_mode)
        if mode not in _PAIR_DISTRIBUTION_STRATA_MODES:
            raise ValueError(
                f"pair_distribution_strata_mode must be one of {sorted(_PAIR_DISTRIBUTION_STRATA_MODES)}; got {mode!r}"
            )
        object.__setattr__(self, "pair_distribution_strata_mode", mode)

    @property
    def is_empty(self) -> bool:
        return (
            self.max_translation_error_mean_meters is None
            and self.max_translation_error_p95_meters is None
            and self.max_translation_error_max_meters is None
            and self.max_heading_error_mean_radians is None
            and self.max_pair_translation_error_meters is None
            and self.max_exceeding_translation_pair_fraction is None
            and self.max_pair_heading_error_radians is None
            and self.max_exceeding_heading_pair_fraction is None
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.max_translation_error_mean_meters is not None:
            payload["maxTranslationErrorMeanMeters"] = float(self.max_translation_error_mean_meters)
        if self.max_translation_error_p95_meters is not None:
            payload["maxTranslationErrorP95Meters"] = float(self.max_translation_error_p95_meters)
        if self.max_translation_error_max_meters is not None:
            payload["maxTranslationErrorMaxMeters"] = float(self.max_translation_error_max_meters)
        if self.max_heading_error_mean_radians is not None:
            payload["maxHeadingErrorMeanRadians"] = float(self.max_heading_error_mean_radians)
        if self.max_pair_translation_error_meters is not None:
            payload["maxPairTranslationErrorMeters"] = float(self.max_pair_translation_error_meters)
        if self.max_exceeding_translation_pair_fraction is not None:
            payload["maxExceedingTranslationPairFraction"] = float(self.max_exceeding_translation_pair_fraction)
        if self.max_pair_heading_error_radians is not None:
            payload["maxPairHeadingErrorRadians"] = float(self.max_pair_heading_error_radians)
        if self.max_exceeding_heading_pair_fraction is not None:
            payload["maxExceedingHeadingPairFraction"] = float(self.max_exceeding_heading_pair_fraction)
        if self.pair_distribution_strata is not None and self.pair_distribution_strata > 1:
            payload["pairDistributionStrata"] = int(self.pair_distribution_strata)
            if self.pair_distribution_strata_mode != "equal-duration":
                payload["pairDistributionStrataMode"] = self.pair_distribution_strata_mode
        if self.event_windows_path is not None:
            payload["eventWindowsPath"] = str(self.event_windows_path)
        return payload


def real_vs_sim_correlation_thresholds_from_dict(payload: Mapping[str, Any]) -> RealVsSimCorrelationThresholds:
    """Rebuild :class:`RealVsSimCorrelationThresholds` from its JSON payload."""

    def _optional(key: str) -> float | None:
        value = payload.get(key)
        return None if value is None else float(value)

    strata_value = payload.get("pairDistributionStrata")
    mode_value = payload.get("pairDistributionStrataMode")
    event_windows_value = payload.get("eventWindowsPath")
    return RealVsSimCorrelationThresholds(
        max_translation_error_mean_meters=_optional("maxTranslationErrorMeanMeters"),
        max_translation_error_p95_meters=_optional("maxTranslationErrorP95Meters"),
        max_translation_error_max_meters=_optional("maxTranslationErrorMaxMeters"),
        max_heading_error_mean_radians=_optional("maxHeadingErrorMeanRadians"),
        max_pair_translation_error_meters=_optional("maxPairTranslationErrorMeters"),
        max_exceeding_translation_pair_fraction=_optional("maxExceedingTranslationPairFraction"),
        max_pair_heading_error_radians=_optional("maxPairHeadingErrorRadians"),
        max_exceeding_heading_pair_fraction=_optional("maxExceedingHeadingPairFraction"),
        pair_distribution_strata=None if strata_value is None else int(strata_value),
        pair_distribution_strata_mode="equal-duration" if mode_value is None else str(mode_value),
        event_windows_path=None if event_windows_value is None else str(event_windows_value),
    )


def correlation_threshold_overrides_from_dict(
    payload: Mapping[str, Any],
) -> dict[str, RealVsSimCorrelationThresholds]:
    """Rebuild per-topic correlation threshold overrides from JSON.

    The payload is a flat ``{<bag_source_topic>: {<thresholds>}}`` mapping.
    Topics whose payload is empty (no bounds set) are dropped — they would
    behave identically to falling through to the default thresholds.
    """

    overrides: dict[str, RealVsSimCorrelationThresholds] = {}
    for topic, item in payload.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"correlation thresholds for topic {topic!r} must be a mapping")
        thresholds = real_vs_sim_correlation_thresholds_from_dict(item)
        if thresholds.is_empty:
            continue
        overrides[str(topic)] = thresholds
    return overrides


def load_correlation_threshold_overrides_json(
    path: str | Path,
) -> dict[str, RealVsSimCorrelationThresholds]:
    """Load per-topic correlation threshold overrides from a JSON file.

    File shape: a flat ``{<bag_source_topic>: {<thresholds>}}`` object.
    Empty entries are dropped on load (treated as \"no override for this
    topic\").
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("correlation threshold overrides JSON root must be an object")
    return correlation_threshold_overrides_from_dict(raw)


def correlation_threshold_overrides_to_dict(
    overrides: Mapping[str, RealVsSimCorrelationThresholds],
) -> dict[str, dict[str, Any]]:
    """Serialise per-topic overrides into the JSON payload shape."""

    return {str(topic): thresholds.to_dict() for topic, thresholds in overrides.items()}


def evaluate_real_vs_sim_correlation_thresholds(
    report: RealVsSimCorrelationReport,
    thresholds: RealVsSimCorrelationThresholds,
    *,
    event_windows: Sequence[CorrelationEventWindow] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Return ``(passed, failed_checks)`` for ``report`` against ``thresholds``.

    Empty thresholds always pass with no failed checks. Otherwise each
    populated bound is compared against the corresponding statistic; the
    failure tag (``translation-mean`` / ``translation-p95`` /
    ``translation-max`` / ``heading-mean``) is added to ``failed_checks``
    when the report exceeds the bound. Reports without heading data
    skip the ``heading-mean`` check (no failure recorded).

    When ``thresholds.pair_distribution_strata_mode == "event-aligned"``
    the caller is expected to pre-resolve ``event_windows`` (e.g. via
    :func:`load_correlation_event_windows_json`). When ``event_windows``
    is empty or omitted, the gate falls back to ``equal-pair-count``
    inside :func:`_split_pairs` — failure tags still include the window
    index but no longer correspond to a named event.
    """

    failed: list[str] = []
    strata = thresholds.pair_distribution_strata
    strata_mode = thresholds.pair_distribution_strata_mode
    aggregate_stratified = strata is not None and strata > 1 and bool(report.pairs)
    if aggregate_stratified:
        # Stratified mode replaces the report-level aggregate checks with
        # per-window aggregates computed from report.pairs (the strided
        # sample). Empty windows skip silently. Aggregates from heading
        # use the heading-bearing subset of the window.
        for window_index, window_pairs in enumerate(
            _split_pairs(report.pairs, int(strata), strata_mode, event_windows=event_windows)
        ):
            if not window_pairs:
                continue
            translation_errors = [float(pair.translation_error_meters) for pair in window_pairs]
            window_mean = sum(translation_errors) / len(translation_errors)
            window_max = max(translation_errors)
            window_p95 = _percentile(translation_errors, 95.0)
            if (
                thresholds.max_translation_error_mean_meters is not None
                and window_mean > thresholds.max_translation_error_mean_meters
            ):
                failed.append(f"translation-mean-window-{window_index}")
            if (
                thresholds.max_translation_error_p95_meters is not None
                and window_p95 > thresholds.max_translation_error_p95_meters
            ):
                failed.append(f"translation-p95-window-{window_index}")
            if (
                thresholds.max_translation_error_max_meters is not None
                and window_max > thresholds.max_translation_error_max_meters
            ):
                failed.append(f"translation-max-window-{window_index}")
            if thresholds.max_heading_error_mean_radians is not None:
                heading_errors = [
                    float(pair.heading_error_radians) for pair in window_pairs if pair.heading_error_radians is not None
                ]
                if heading_errors:
                    window_heading_mean = sum(heading_errors) / len(heading_errors)
                    if window_heading_mean > thresholds.max_heading_error_mean_radians:
                        failed.append(f"heading-mean-window-{window_index}")
    else:
        if (
            thresholds.max_translation_error_mean_meters is not None
            and float(report.translation_error_mean_meters) > thresholds.max_translation_error_mean_meters
        ):
            failed.append("translation-mean")
        if (
            thresholds.max_translation_error_p95_meters is not None
            and float(report.translation_error_p95_meters) > thresholds.max_translation_error_p95_meters
        ):
            failed.append("translation-p95")
        if (
            thresholds.max_translation_error_max_meters is not None
            and float(report.translation_error_max_meters) > thresholds.max_translation_error_max_meters
        ):
            failed.append("translation-max")
        if (
            thresholds.max_heading_error_mean_radians is not None
            and report.heading_error_mean_radians is not None
            and float(report.heading_error_mean_radians) > thresholds.max_heading_error_mean_radians
        ):
            failed.append("heading-mean")
    if (
        thresholds.max_pair_translation_error_meters is not None
        and thresholds.max_exceeding_translation_pair_fraction is not None
        and report.pairs
    ):
        bound = float(thresholds.max_pair_translation_error_meters)
        limit = float(thresholds.max_exceeding_translation_pair_fraction)
        if strata is None or strata <= 1:
            exceeding = sum(1 for pair in report.pairs if float(pair.translation_error_meters) > bound)
            if exceeding / len(report.pairs) > limit:
                failed.append("translation-pair-distribution")
        else:
            for window_index, window_pairs in enumerate(
                _split_pairs(report.pairs, strata, strata_mode, event_windows=event_windows)
            ):
                if not window_pairs:
                    continue
                exceeding = sum(1 for pair in window_pairs if float(pair.translation_error_meters) > bound)
                if exceeding / len(window_pairs) > limit:
                    failed.append(f"translation-pair-distribution-window-{window_index}")
    if (
        thresholds.max_pair_heading_error_radians is not None
        and thresholds.max_exceeding_heading_pair_fraction is not None
        and report.pairs
    ):
        bound = float(thresholds.max_pair_heading_error_radians)
        limit = float(thresholds.max_exceeding_heading_pair_fraction)
        if strata is None or strata <= 1:
            with_heading = [pair for pair in report.pairs if pair.heading_error_radians is not None]
            if with_heading:
                exceeding = sum(1 for pair in with_heading if float(pair.heading_error_radians) > bound)
                if exceeding / len(with_heading) > limit:
                    failed.append("heading-pair-distribution")
        else:
            for window_index, window_pairs in enumerate(
                _split_pairs(report.pairs, strata, strata_mode, event_windows=event_windows)
            ):
                with_heading = [pair for pair in window_pairs if pair.heading_error_radians is not None]
                if not with_heading:
                    continue
                exceeding = sum(1 for pair in with_heading if float(pair.heading_error_radians) > bound)
                if exceeding / len(with_heading) > limit:
                    failed.append(f"heading-pair-distribution-window-{window_index}")
    return (not failed, tuple(failed))


@dataclass(frozen=True, slots=True)
class RealVsSimCorrelationWindowStats:
    """Per-window aggregate stats for one stratified correlation report.

    Computed transiently from ``report.pairs`` (the strided pair sample)
    so values are approximate when ``max_pairs_kept`` is small relative
    to the matched pair count. ``bag_time_start_seconds`` and
    ``bag_time_end_seconds`` describe the window's temporal extent for
    the equal-duration mode; for the equal-pair-count mode they describe
    the actual bag-time span of the pairs that landed in the window.
    ``heading_error_mean_radians`` is ``None`` when no pair in the window
    carries heading data.

    ``event_name`` / ``event_tags`` / ``event_source`` are populated only
    for the ``event-aligned`` mode. They roundtrip through JSON only when
    non-empty so v1 payloads from the equal-duration / equal-pair-count
    modes stay byte-equivalent.
    """

    window_index: int
    pair_count: int
    bag_time_start_seconds: float
    bag_time_end_seconds: float
    translation_error_mean_meters: float
    translation_error_p95_meters: float
    translation_error_max_meters: float
    heading_error_mean_radians: float | None = None
    event_name: str | None = None
    event_tags: tuple[str, ...] = ()
    event_source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_tags", tuple(str(tag) for tag in self.event_tags))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "windowIndex": int(self.window_index),
            "pairCount": int(self.pair_count),
            "bagTimeStartSeconds": float(self.bag_time_start_seconds),
            "bagTimeEndSeconds": float(self.bag_time_end_seconds),
            "translationErrorMeanMeters": float(self.translation_error_mean_meters),
            "translationErrorP95Meters": float(self.translation_error_p95_meters),
            "translationErrorMaxMeters": float(self.translation_error_max_meters),
        }
        if self.heading_error_mean_radians is not None:
            payload["headingErrorMeanRadians"] = float(self.heading_error_mean_radians)
        if self.event_name is not None:
            payload["eventName"] = self.event_name
        if self.event_tags:
            payload["eventTags"] = list(self.event_tags)
        if self.event_source is not None:
            payload["eventSource"] = self.event_source
        return payload


def real_vs_sim_correlation_window_stats_from_dict(
    payload: Mapping[str, Any],
) -> RealVsSimCorrelationWindowStats:
    """Rebuild :class:`RealVsSimCorrelationWindowStats` from JSON."""

    heading_value = payload.get("headingErrorMeanRadians")
    event_tags_payload = payload.get("eventTags") or ()
    return RealVsSimCorrelationWindowStats(
        window_index=int(payload["windowIndex"]),
        pair_count=int(payload["pairCount"]),
        bag_time_start_seconds=float(payload["bagTimeStartSeconds"]),
        bag_time_end_seconds=float(payload["bagTimeEndSeconds"]),
        translation_error_mean_meters=float(payload["translationErrorMeanMeters"]),
        translation_error_p95_meters=float(payload["translationErrorP95Meters"]),
        translation_error_max_meters=float(payload["translationErrorMaxMeters"]),
        heading_error_mean_radians=None if heading_value is None else float(heading_value),
        event_name=None if payload.get("eventName") is None else str(payload["eventName"]),
        event_tags=tuple(str(tag) for tag in event_tags_payload),
        event_source=None if payload.get("eventSource") is None else str(payload["eventSource"]),
    )


def compute_per_window_correlation_stats(
    report: RealVsSimCorrelationReport,
    *,
    strata: int,
    mode: str = "equal-duration",
    event_windows: Sequence[CorrelationEventWindow] | None = None,
) -> tuple[RealVsSimCorrelationWindowStats, ...]:
    """Return per-window aggregate stats for ``report.pairs``.

    Empty windows (no pairs landed) are dropped from the returned tuple
    for the equal-duration / equal-pair-count modes. For ``event-aligned``
    mode empty windows are *retained* so a missing event (e.g. policy
    skipped the turn) is still surfaced in the review bundle with
    pair_count=0 and zeroed stats.

    Heading mean is ``None`` when no pair in the window carries heading
    data. The bag-time span fields describe the actual min/max
    ``bag_timestamp_seconds`` of the pairs in each window for non-event
    modes; for ``event-aligned`` empty windows they fall back to the
    event's declared ``[start_time, end_time]``.

    When ``mode == "event-aligned"`` and ``event_windows`` is non-empty,
    each emitted stat carries the event's ``name`` / ``tags`` / ``source``
    so the review bundle can label rows by event.
    """

    use_event_mode = mode == "event-aligned" and bool(event_windows)
    if not use_event_mode:
        if strata <= 1 or not report.pairs:
            return ()
        windows = _split_pairs(report.pairs, strata, mode)
        stats: list[RealVsSimCorrelationWindowStats] = []
        for window_index, window_pairs in enumerate(windows):
            if not window_pairs:
                continue
            stats.append(_build_window_stats(window_index, window_pairs))
        return tuple(stats)

    assert event_windows is not None  # narrowed by use_event_mode
    windows = _split_pairs_by_events(report.pairs, event_windows)
    stats = []
    for window_index, window_pairs in enumerate(windows):
        event = event_windows[window_index]
        if window_pairs:
            stat = _build_window_stats(window_index, window_pairs)
        else:
            stat = RealVsSimCorrelationWindowStats(
                window_index=window_index,
                pair_count=0,
                bag_time_start_seconds=event.start_time,
                bag_time_end_seconds=event.end_time,
                translation_error_mean_meters=0.0,
                translation_error_p95_meters=0.0,
                translation_error_max_meters=0.0,
                heading_error_mean_radians=None,
            )
        stats.append(
            RealVsSimCorrelationWindowStats(
                window_index=stat.window_index,
                pair_count=stat.pair_count,
                bag_time_start_seconds=stat.bag_time_start_seconds,
                bag_time_end_seconds=stat.bag_time_end_seconds,
                translation_error_mean_meters=stat.translation_error_mean_meters,
                translation_error_p95_meters=stat.translation_error_p95_meters,
                translation_error_max_meters=stat.translation_error_max_meters,
                heading_error_mean_radians=stat.heading_error_mean_radians,
                event_name=event.name,
                event_tags=event.tags,
                event_source=event.source,
            )
        )
    return tuple(stats)


def _build_window_stats(
    window_index: int,
    window_pairs: Sequence[CorrelatedPosePair],
) -> RealVsSimCorrelationWindowStats:
    """Compute the aggregate window stats for a non-empty bucket of pairs."""

    translation_errors = [float(pair.translation_error_meters) for pair in window_pairs]
    timestamps = [float(pair.bag_timestamp_seconds) for pair in window_pairs]
    heading_errors = [
        float(pair.heading_error_radians) for pair in window_pairs if pair.heading_error_radians is not None
    ]
    return RealVsSimCorrelationWindowStats(
        window_index=window_index,
        pair_count=len(window_pairs),
        bag_time_start_seconds=min(timestamps),
        bag_time_end_seconds=max(timestamps),
        translation_error_mean_meters=sum(translation_errors) / len(translation_errors),
        translation_error_p95_meters=_percentile(translation_errors, 95.0),
        translation_error_max_meters=max(translation_errors),
        heading_error_mean_radians=(sum(heading_errors) / len(heading_errors) if heading_errors else None),
    )


def _split_pairs_by_time(
    pairs: Sequence[CorrelatedPosePair],
    strata: int,
) -> list[list[CorrelatedPosePair]]:
    """Partition ``pairs`` into ``strata`` equal-duration time windows.

    The window boundaries span ``[min(bag_timestamp_seconds), max(...)]``.
    Pairs at the upper boundary go to the last window. When all pairs share
    the same timestamp (zero duration) every pair lands in window 0 and the
    other windows are empty.
    """

    if strata <= 1 or not pairs:
        return [list(pairs)]
    timestamps = [float(pair.bag_timestamp_seconds) for pair in pairs]
    start = min(timestamps)
    end = max(timestamps)
    duration = end - start
    windows: list[list[CorrelatedPosePair]] = [[] for _ in range(strata)]
    if duration <= 0.0:
        windows[0].extend(pairs)
        return windows
    for pair, ts in zip(pairs, timestamps, strict=True):
        offset = ts - start
        index = int(offset / duration * strata)
        if index >= strata:
            index = strata - 1
        windows[index].append(pair)
    return windows


def _split_pairs_by_count(
    pairs: Sequence[CorrelatedPosePair],
    strata: int,
) -> list[list[CorrelatedPosePair]]:
    """Partition ``pairs`` into ``strata`` near-equal-count groups by index order.

    The first ``len(pairs) % strata`` windows hold one extra pair so the
    total population is preserved (e.g. 10 pairs across 4 windows yields
    counts ``[3, 3, 2, 2]``). Useful when the bag has uneven sample
    density (long quiet stretches followed by bursts) and equal-duration
    windows would leave most windows nearly empty.
    """

    if strata <= 1 or not pairs:
        return [list(pairs)]
    total = len(pairs)
    base = total // strata
    extras = total % strata
    windows: list[list[CorrelatedPosePair]] = []
    cursor = 0
    for window_index in range(strata):
        size = base + (1 if window_index < extras else 0)
        windows.append(list(pairs[cursor : cursor + size]))
        cursor += size
    return windows


def _split_pairs_by_events(
    pairs: Sequence[CorrelatedPosePair],
    event_windows: Sequence[CorrelationEventWindow],
) -> list[list[CorrelatedPosePair]]:
    """Partition ``pairs`` into one bucket per :class:`CorrelationEventWindow`.

    Each pair lands in the *first* event whose ``[start_time, end_time)``
    range contains its ``bag_timestamp_seconds``; pairs that fall outside
    every event window are dropped (the windows define the universe of
    interest). When ``event_windows`` is empty the caller is expected to
    fall back to a non-event mode — this function returns an empty list
    of windows in that case rather than implicitly bucketing all pairs.
    """

    if not event_windows:
        return []
    windows: list[list[CorrelatedPosePair]] = [[] for _ in event_windows]
    if not pairs:
        return windows
    for pair in pairs:
        ts = float(pair.bag_timestamp_seconds)
        for index, event in enumerate(event_windows):
            if event.start_time <= ts < event.end_time:
                windows[index].append(pair)
                break
    return windows


def _split_pairs(
    pairs: Sequence[CorrelatedPosePair],
    strata: int,
    mode: str,
    *,
    event_windows: Sequence[CorrelationEventWindow] | None = None,
) -> list[list[CorrelatedPosePair]]:
    """Dispatch to the requested stratification splitter.

    For ``event-aligned`` mode the caller must pass ``event_windows`` that
    are non-empty; an empty / missing sequence falls back to
    ``equal-pair-count`` to keep the gate evaluable rather than silently
    skipping the per-window check.
    """

    if mode == "event-aligned":
        if event_windows:
            return _split_pairs_by_events(pairs, event_windows)
        # Fallback: keep gate functional when caller forgot to plumb windows.
        return _split_pairs_by_count(pairs, strata)
    if mode == "equal-pair-count":
        return _split_pairs_by_count(pairs, strata)
    return _split_pairs_by_time(pairs, strata)


_NAVSAT_MSGTYPES: frozenset[str] = frozenset({"sensor_msgs/msg/NavSatFix", "sensor_msgs/NavSatFix"})
_IMU_MSGTYPES: frozenset[str] = frozenset({"sensor_msgs/msg/Imu", "sensor_msgs/Imu"})
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E_SQ = _WGS84_F * (2.0 - _WGS84_F)


def wgs84_to_ecef(latitude: float, longitude: float, altitude: float) -> tuple[float, float, float]:
    """Convert WGS84 (deg, deg, m) to ECEF (m)."""

    lat = math.radians(float(latitude))
    lon = math.radians(float(longitude))
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E_SQ * sin_lat * sin_lat)
    h = float(altitude)
    x = (n + h) * cos_lat * math.cos(lon)
    y = (n + h) * cos_lat * math.sin(lon)
    z = (n * (1.0 - _WGS84_E_SQ) + h) * sin_lat
    return (x, y, z)


def wgs84_to_local_enu(
    latitude: float,
    longitude: float,
    altitude: float,
    *,
    origin_latitude: float,
    origin_longitude: float,
    origin_altitude: float,
) -> tuple[float, float, float]:
    """Convert WGS84 (lat, lon, alt) to local ENU metres relative to an origin."""

    px, py, pz = wgs84_to_ecef(latitude, longitude, altitude)
    ox, oy, oz = wgs84_to_ecef(origin_latitude, origin_longitude, origin_altitude)
    dx = px - ox
    dy = py - oy
    dz = pz - oz
    lat0 = math.radians(float(origin_latitude))
    lon0 = math.radians(float(origin_longitude))
    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return (east, north, up)


def read_navsat_pose_stream(
    bag_paths: Sequence[Path],
    *,
    topic: str | None = None,
    reference_origin_wgs84: tuple[float, float, float] | None = None,
    skip_zero_fixes: bool = True,
) -> BagPoseStream:
    """Read a ``sensor_msgs/NavSatFix`` topic into a local-ENU pose stream.

    The first valid fix anchors the ENU origin unless
    ``reference_origin_wgs84`` is provided. Placeholder fixes at
    ``latitude == longitude == 0`` are dropped when ``skip_zero_fixes``
    is true (the public Autoware Leo Drive bags occasionally publish
    these during sensor warm-up).
    """

    paths = [Path(item) for item in bag_paths]
    if not paths:
        raise ValueError("read_navsat_pose_stream requires at least one bag path")

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    samples: list[BagPoseSample] = []
    chosen_topic: str | None = None
    chosen_msgtype: str | None = None
    origin = reference_origin_wgs84

    with AnyReader(paths, default_typestore=typestore) as reader:
        connection = _select_navsat_connection(reader.topics, topic)
        if connection is None:
            raise FileNotFoundError("no sensor_msgs/NavSatFix topic found in the supplied bag paths")
        chosen_topic = connection.topic
        chosen_msgtype = connection.msgtype
        for _, timestamp_ns, rawdata in reader.messages(connections=[connection]):
            ts = float(timestamp_ns) * 1e-9
            msg = reader.deserialize(rawdata, connection.msgtype)
            lat = float(getattr(msg, "latitude", float("nan")))
            lon = float(getattr(msg, "longitude", float("nan")))
            alt = float(getattr(msg, "altitude", 0.0))
            if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(alt)):
                continue
            if skip_zero_fixes and abs(lat) < 1e-12 and abs(lon) < 1e-12:
                continue
            if origin is None:
                origin = (lat, lon, alt)
            east, north, up = wgs84_to_local_enu(
                lat,
                lon,
                alt,
                origin_latitude=origin[0],
                origin_longitude=origin[1],
                origin_altitude=origin[2],
            )
            samples.append(BagPoseSample(timestamp_seconds=ts, position=(east, north, up)))

    if not samples:
        raise ValueError("no valid NavSatFix samples were extracted from the supplied bag paths")
    samples.sort(key=lambda sample: sample.timestamp_seconds)
    assert chosen_topic is not None and chosen_msgtype is not None
    return BagPoseStream(
        samples=tuple(samples),
        frame_id="enu",
        source_topic=chosen_topic,
        source_msgtype=chosen_msgtype,
        reference_origin_wgs84=origin,
    )


def read_imu_orientation_stream(
    bag_paths: Sequence[Path],
    *,
    topic: str | None = None,
) -> tuple[tuple[float, tuple[float, float, float, float]], ...]:
    """Read a ``sensor_msgs/Imu`` topic into ``(timestamp, xyzw)`` pairs.

    Imu has no position field, so the result is a flat list of
    timestamped quaternions sorted ascending by timestamp. Pair the
    result onto a NavSatFix-derived :class:`BagPoseStream` using
    :func:`merge_navsat_with_imu_orientation`. Identity-or-zero
    quaternions (norm below ``1e-6``) are dropped — public bags
    occasionally publish placeholder identity samples during sensor
    warm-up.
    """

    paths = [Path(item) for item in bag_paths]
    if not paths:
        raise ValueError("read_imu_orientation_stream requires at least one bag path")

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    samples: list[tuple[float, tuple[float, float, float, float]]] = []

    with AnyReader(paths, default_typestore=typestore) as reader:
        connection = _select_imu_connection(reader.topics, topic)
        if connection is None:
            raise FileNotFoundError("no sensor_msgs/Imu topic found in the supplied bag paths")
        for _, timestamp_ns, rawdata in reader.messages(connections=[connection]):
            ts = float(timestamp_ns) * 1e-9
            msg = reader.deserialize(rawdata, connection.msgtype)
            orientation = getattr(msg, "orientation", None)
            if orientation is None:
                continue
            try:
                qx = float(orientation.x)
                qy = float(orientation.y)
                qz = float(orientation.z)
                qw = float(orientation.w)
            except AttributeError:
                continue
            if not all(math.isfinite(component) for component in (qx, qy, qz, qw)):
                continue
            norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            if norm <= 1e-6:
                continue
            inv_norm = 1.0 / norm
            samples.append((ts, (qx * inv_norm, qy * inv_norm, qz * inv_norm, qw * inv_norm)))

    if not samples:
        raise ValueError("no valid Imu orientation samples were extracted from the supplied bag paths")
    samples.sort(key=lambda item: item[0])
    return tuple(samples)


def merge_navsat_with_imu_orientation(
    navsat_stream: BagPoseStream,
    imu_orientations: Sequence[tuple[float, tuple[float, float, float, float]]],
    *,
    max_pair_dt_seconds: float = 0.05,
) -> BagPoseStream:
    """Pair each NavSatFix sample with the nearest-timestamp IMU orientation.

    Returns a new :class:`BagPoseStream` whose samples carry the
    NavSatFix position plus the matched IMU quaternion. Samples whose
    nearest IMU orientation is more than ``max_pair_dt_seconds`` away
    keep ``orientation_xyzw=None`` so the correlator's heading-error
    aggregation continues to skip them, matching the behaviour of a
    NavSatFix-only stream. The output preserves the source bag's frame
    id, topic, msgtype, and reference origin.
    """

    if max_pair_dt_seconds < 0.0 or not math.isfinite(max_pair_dt_seconds):
        raise ValueError("max_pair_dt_seconds must be finite and non-negative")
    if not imu_orientations:
        raise ValueError("imu_orientations must contain at least one sample")

    imu_timestamps = [ts for ts, _ in imu_orientations]
    if any(b < a for a, b in zip(imu_timestamps, imu_timestamps[1:])):
        raise ValueError("imu_orientations must be sorted ascending by timestamp")

    fused: list[BagPoseSample] = []
    for sample in navsat_stream.samples:
        nearest = _nearest_bag_index(imu_timestamps, sample.timestamp_seconds)
        if nearest is None:
            fused.append(sample)
            continue
        imu_ts, quaternion = imu_orientations[nearest]
        if abs(imu_ts - sample.timestamp_seconds) > max_pair_dt_seconds:
            fused.append(sample)
            continue
        fused.append(
            BagPoseSample(
                timestamp_seconds=sample.timestamp_seconds,
                position=sample.position,
                orientation_xyzw=quaternion,
            )
        )

    return BagPoseStream(
        samples=tuple(fused),
        frame_id=navsat_stream.frame_id,
        source_topic=navsat_stream.source_topic,
        source_msgtype=navsat_stream.source_msgtype,
        reference_origin_wgs84=navsat_stream.reference_origin_wgs84,
    )


def correlate_against_sim_trajectory(
    bag_stream: BagPoseStream,
    sim_samples: Sequence[SimPoseSample],
    *,
    max_match_dt_seconds: float = 0.05,
    keep_pairs: bool = True,
    max_pairs_kept: int = 1024,
) -> RealVsSimCorrelationReport:
    """Pair sim samples with the nearest bag sample by timestamp and reduce errors.

    A sim sample is matched against the nearest-in-time bag sample; the
    pair is dropped when the time gap exceeds ``max_match_dt_seconds``.
    Aggregate translation-error statistics (min / mean / max / p50 /
    p95) are computed from the surviving pairs. When ``keep_pairs`` is
    true the report keeps an evenly-strided sample of up to
    ``max_pairs_kept`` :class:`CorrelatedPosePair` entries for
    inspection — pass ``False`` to suppress them entirely.
    """

    if max_match_dt_seconds < 0.0 or not math.isfinite(max_match_dt_seconds):
        raise ValueError("max_match_dt_seconds must be finite and non-negative")
    if max_pairs_kept <= 0:
        raise ValueError("max_pairs_kept must be positive")

    bag_timestamps = [sample.timestamp_seconds for sample in bag_stream.samples]
    sim_sample_count = len(sim_samples)
    pairs: list[CorrelatedPosePair] = []
    heading_errors: list[float] = []

    for sim_sample in sim_samples:
        nearest = _nearest_bag_index(bag_timestamps, sim_sample.timestamp_seconds)
        if nearest is None:
            continue
        bag_sample = bag_stream.samples[nearest]
        dt = abs(sim_sample.timestamp_seconds - bag_sample.timestamp_seconds)
        if dt > max_match_dt_seconds:
            continue
        translation = math.dist(sim_sample.position, bag_sample.position)
        heading_error: float | None = None
        if bag_sample.orientation_xyzw is not None:
            heading_error = _heading_error_radians(bag_sample.orientation_xyzw, sim_sample.orientation_xyzw)
            heading_errors.append(heading_error)
        pairs.append(
            CorrelatedPosePair(
                bag_timestamp_seconds=bag_sample.timestamp_seconds,
                sim_timestamp_seconds=sim_sample.timestamp_seconds,
                bag_position=bag_sample.position,
                sim_position=sim_sample.position,
                translation_error_meters=translation,
                heading_error_radians=heading_error,
            )
        )

    if not pairs:
        return RealVsSimCorrelationReport(
            bag_source=bag_stream.metadata(),
            sim_sample_count=sim_sample_count,
            matched_pair_count=0,
            matched_seconds=0.0,
            translation_error_min_meters=float("nan"),
            translation_error_mean_meters=float("nan"),
            translation_error_max_meters=float("nan"),
            translation_error_p50_meters=float("nan"),
            translation_error_p95_meters=float("nan"),
            pairs=(),
        )

    errors = [pair.translation_error_meters for pair in pairs]
    matched_seconds = pairs[-1].bag_timestamp_seconds - pairs[0].bag_timestamp_seconds
    heading_mean = sum(heading_errors) / len(heading_errors) if heading_errors else None
    heading_max = max(heading_errors) if heading_errors else None
    kept = _stride_sample(pairs, max_pairs_kept) if keep_pairs else ()

    return RealVsSimCorrelationReport(
        bag_source=bag_stream.metadata(),
        sim_sample_count=sim_sample_count,
        matched_pair_count=len(pairs),
        matched_seconds=float(matched_seconds),
        translation_error_min_meters=min(errors),
        translation_error_mean_meters=sum(errors) / len(errors),
        translation_error_max_meters=max(errors),
        translation_error_p50_meters=_percentile(errors, 50.0),
        translation_error_p95_meters=_percentile(errors, 95.0),
        heading_error_mean_radians=heading_mean,
        heading_error_max_radians=heading_max,
        pairs=kept,
    )


def write_real_vs_sim_correlation_report_json(
    path: str | Path,
    report: RealVsSimCorrelationReport,
) -> Path:
    """Persist a correlation report as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_real_vs_sim_correlation_report_json(path: str | Path) -> RealVsSimCorrelationReport:
    """Load a correlation report JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("correlation report root must be a JSON object")
    return real_vs_sim_correlation_report_from_dict(payload)


def real_vs_sim_correlation_report_from_dict(payload: Mapping[str, Any]) -> RealVsSimCorrelationReport:
    """Rebuild a :class:`RealVsSimCorrelationReport` from its JSON payload."""

    record_type = payload.get("recordType")
    if record_type != "real-vs-sim-correlation-report":
        raise ValueError(f"unexpected recordType for correlation report: {record_type!r}")
    version = str(payload.get("version", REAL_VS_SIM_CORRELATION_REPORT_VERSION))
    if version != REAL_VS_SIM_CORRELATION_REPORT_VERSION:
        raise ValueError(f"unsupported real-vs-sim correlation report version: {version}")
    bag_payload = payload.get("bagSource")
    if not isinstance(bag_payload, Mapping):
        raise ValueError("correlation report bagSource must be a mapping")
    translation_payload = payload.get("translationErrorMeters")
    if not isinstance(translation_payload, Mapping):
        raise ValueError("correlation report translationErrorMeters must be a mapping")
    heading_payload = payload.get("headingErrorRadians")
    heading_mean: float | None = None
    heading_max: float | None = None
    if isinstance(heading_payload, Mapping):
        heading_mean = float(heading_payload["mean"]) if "mean" in heading_payload else None
        heading_max = float(heading_payload["max"]) if "max" in heading_payload else None
    pair_items = payload.get("pairs", ())
    if not isinstance(pair_items, Sequence):
        raise ValueError("correlation report pairs must be a sequence")
    pairs = tuple(correlated_pose_pair_from_dict(item) for item in pair_items if isinstance(item, Mapping))
    return RealVsSimCorrelationReport(
        bag_source=bag_pose_stream_metadata_from_dict(bag_payload),
        sim_sample_count=int(payload["simSampleCount"]),
        matched_pair_count=int(payload["matchedPairCount"]),
        matched_seconds=float(payload["matchedSeconds"]),
        translation_error_min_meters=float(translation_payload["min"]),
        translation_error_mean_meters=float(translation_payload["mean"]),
        translation_error_max_meters=float(translation_payload["max"]),
        translation_error_p50_meters=float(translation_payload["p50"]),
        translation_error_p95_meters=float(translation_payload["p95"]),
        heading_error_mean_radians=heading_mean,
        heading_error_max_radians=heading_max,
        pairs=pairs,
        version=version,
    )


def bag_pose_stream_metadata_from_dict(payload: Mapping[str, Any]) -> BagPoseStreamMetadata:
    """Rebuild :class:`BagPoseStreamMetadata` from its JSON payload."""

    origin_payload = payload.get("referenceOriginWgs84")
    origin: tuple[float, float, float] | None = None
    if origin_payload is not None:
        origin_seq = tuple(float(component) for component in origin_payload)
        if len(origin_seq) != 3:
            raise ValueError("bag pose stream referenceOriginWgs84 must have three elements")
        origin = origin_seq
    return BagPoseStreamMetadata(
        frame_id=str(payload["frameId"]),
        source_topic=str(payload["sourceTopic"]),
        source_msgtype=str(payload["sourceMsgtype"]),
        sample_count=int(payload["sampleCount"]),
        duration_seconds=float(payload["durationSeconds"]),
        reference_origin_wgs84=origin,
    )


def correlated_pose_pair_from_dict(payload: Mapping[str, Any]) -> CorrelatedPosePair:
    """Rebuild a :class:`CorrelatedPosePair` from its JSON payload."""

    bag_position = tuple(float(component) for component in payload["bagPosition"])
    sim_position = tuple(float(component) for component in payload["simPosition"])
    if len(bag_position) != 3 or len(sim_position) != 3:
        raise ValueError("correlation pair positions must have three elements")
    heading = payload.get("headingErrorRadians")
    return CorrelatedPosePair(
        bag_timestamp_seconds=float(payload["bagTimestampSeconds"]),
        sim_timestamp_seconds=float(payload["simTimestampSeconds"]),
        bag_position=bag_position,
        sim_position=sim_position,
        translation_error_meters=float(payload["translationErrorMeters"]),
        heading_error_radians=None if heading is None else float(heading),
    )


def render_real_vs_sim_correlation_markdown(report: RealVsSimCorrelationReport) -> str:
    """Render a compact Markdown summary suitable for PR / CI artifact display."""

    bag = report.bag_source
    lines = [
        f"# Real-vs-sim correlation: `{bag.source_topic}`",
        f"- Bag source: `{bag.source_msgtype}` ({bag.sample_count} samples, {bag.duration_seconds:.2f} s)",
        f"- Sim samples: {report.sim_sample_count}",
        f"- Matched pairs: {report.matched_pair_count} ({report.matched_seconds:.2f} s span)",
        "",
        "| Translation error (m) | Value |",
        "| --- | ---: |",
        f"| min | {report.translation_error_min_meters:.4f} |",
        f"| mean | {report.translation_error_mean_meters:.4f} |",
        f"| p50 | {report.translation_error_p50_meters:.4f} |",
        f"| p95 | {report.translation_error_p95_meters:.4f} |",
        f"| max | {report.translation_error_max_meters:.4f} |",
    ]
    if report.heading_error_mean_radians is not None:
        lines.extend(
            [
                "",
                "| Heading error (rad) | Value |",
                "| --- | ---: |",
                f"| mean | {report.heading_error_mean_radians:.4f} |",
                f"| max | {report.heading_error_max_radians:.4f} |",
            ]
        )
    return "\n".join(lines) + "\n"


def load_sim_pose_samples_jsonl(path: str | Path) -> tuple[SimPoseSample, ...]:
    """Load sim pose samples from a JSONL file with one record per line.

    Each record must contain ``timestampSeconds``, ``position`` (3-element
    list), and ``orientationXyzw`` (4-element list).
    """

    samples: list[SimPoseSample] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
        samples.append(_sim_pose_sample_from_dict(payload, line_number))
    if not samples:
        raise ValueError(f"no sim pose samples found in {path}")
    samples.sort(key=lambda sample: sample.timestamp_seconds)
    return tuple(samples)


def _sim_pose_sample_from_dict(payload: Mapping[str, Any], line_number: int) -> SimPoseSample:
    try:
        ts = float(payload["timestampSeconds"])
        position = payload["position"]
        orientation = payload["orientationXyzw"]
    except KeyError as exc:
        raise ValueError(f"line {line_number}: missing required field {exc}") from exc
    if not isinstance(position, Iterable) or len(list(position)) != 3:
        raise ValueError(f"line {line_number}: position must be a 3-element list")
    if not isinstance(orientation, Iterable) or len(list(orientation)) != 4:
        raise ValueError(f"line {line_number}: orientationXyzw must be a 4-element list")
    return SimPoseSample(
        timestamp_seconds=ts,
        position=tuple(float(c) for c in payload["position"]),
        orientation_xyzw=tuple(float(c) for c in payload["orientationXyzw"]),
    )


def _select_navsat_connection(topics: Mapping[str, Any], requested: str | None):
    """Pick the first NavSatFix connection (preferring ``requested`` when supplied)."""

    return _select_typed_connection(topics, requested, _NAVSAT_MSGTYPES, "NavSatFix")


def _select_imu_connection(topics: Mapping[str, Any], requested: str | None):
    """Pick the first sensor_msgs/Imu connection (preferring ``requested`` when supplied)."""

    return _select_typed_connection(topics, requested, _IMU_MSGTYPES, "Imu")


def _select_typed_connection(
    topics: Mapping[str, Any],
    requested: str | None,
    allowed_msgtypes: frozenset[str],
    label: str,
):
    if requested:
        info = topics.get(requested)
        if info is None:
            raise ValueError(f"requested topic not found: {requested}")
        for connection in info.connections:
            if connection.msgtype in allowed_msgtypes:
                return connection
        raise ValueError(f"requested topic {requested} is not a {label} topic")
    for info in topics.values():
        for connection in info.connections:
            if connection.msgtype in allowed_msgtypes:
                return connection
    return None


def _nearest_bag_index(bag_timestamps: Sequence[float], target: float) -> int | None:
    if not bag_timestamps:
        return None
    index = bisect.bisect_left(bag_timestamps, target)
    if index == 0:
        return 0
    if index >= len(bag_timestamps):
        return len(bag_timestamps) - 1
    before = bag_timestamps[index - 1]
    after = bag_timestamps[index]
    return index - 1 if (target - before) <= (after - target) else index


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires non-empty values")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    fraction = rank - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _stride_sample(pairs: Sequence[CorrelatedPosePair], max_count: int) -> tuple[CorrelatedPosePair, ...]:
    if len(pairs) <= max_count:
        return tuple(pairs)
    stride = len(pairs) / max_count
    indices = sorted({int(round(i * stride)) for i in range(max_count)})
    indices = [index for index in indices if 0 <= index < len(pairs)]
    return tuple(pairs[index] for index in indices)


def _heading_error_radians(
    bag_xyzw: tuple[float, float, float, float],
    sim_xyzw: tuple[float, float, float, float],
) -> float:
    """Yaw difference (in radians) between two unit quaternions, wrapped to [0, π]."""

    bag_yaw = _quaternion_yaw(bag_xyzw)
    sim_yaw = _quaternion_yaw(sim_xyzw)
    diff = sim_yaw - bag_yaw
    wrapped = math.atan2(math.sin(diff), math.cos(diff))
    return abs(wrapped)


def _quaternion_yaw(xyzw: tuple[float, float, float, float]) -> float:
    qx, qy, qz, qw = xyzw
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)
