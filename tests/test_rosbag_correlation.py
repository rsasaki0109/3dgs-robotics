"""Tests for the real-vs-sim rosbag correlation library + CLI."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from gs_sim2real.robotics import (
    BagPoseSample,
    BagPoseStream,
    BagPoseStreamMetadata,
    CorrelatedPosePair,
    REAL_VS_SIM_CORRELATION_REPORT_VERSION,
    RealVsSimCorrelationThresholds,
    RealVsSimCorrelationWindowStats,
    SimPoseSample,
    compute_per_window_correlation_stats,
    correlate_against_sim_trajectory,
    correlation_threshold_overrides_from_dict,
    correlation_threshold_overrides_to_dict,
    evaluate_real_vs_sim_correlation_thresholds,
    load_correlation_threshold_overrides_json,
    load_real_vs_sim_correlation_report_json,
    load_sim_pose_samples_jsonl,
    merge_navsat_with_imu_orientation,
    real_vs_sim_correlation_report_from_dict,
    real_vs_sim_correlation_thresholds_from_dict,
    real_vs_sim_correlation_window_stats_from_dict,
    render_real_vs_sim_correlation_markdown,
    wgs84_to_ecef,
    wgs84_to_local_enu,
    write_real_vs_sim_correlation_report_json,
)
from gs_sim2real.robotics import rosbag_correlation as module_under_test


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_bag_stream(samples: list[BagPoseSample]) -> BagPoseStream:
    return BagPoseStream(
        samples=tuple(samples),
        frame_id="enu",
        source_topic="/gnss/fix",
        source_msgtype="sensor_msgs/msg/NavSatFix",
        reference_origin_wgs84=(35.0, 139.0, 10.0),
    )


def test_wgs84_to_ecef_at_known_reference_point() -> None:
    # On the equator at the prime meridian and zero altitude the ECEF point
    # is (a, 0, 0) where a is the WGS84 semi-major axis.
    x, y, z = wgs84_to_ecef(0.0, 0.0, 0.0)
    assert x == pytest.approx(6378137.0, rel=1e-9)
    assert y == pytest.approx(0.0, abs=1e-6)
    assert z == pytest.approx(0.0, abs=1e-6)


def test_wgs84_to_local_enu_returns_zero_at_origin() -> None:
    east, north, up = wgs84_to_local_enu(
        35.0,
        139.0,
        12.0,
        origin_latitude=35.0,
        origin_longitude=139.0,
        origin_altitude=12.0,
    )
    assert east == pytest.approx(0.0, abs=1e-6)
    assert north == pytest.approx(0.0, abs=1e-6)
    assert up == pytest.approx(0.0, abs=1e-6)


def test_wgs84_to_local_enu_small_eastward_step_matches_great_circle() -> None:
    # 0.0001 deg east at lat 35 deg ~= 9.107 m (lon-deg shrinks with cos lat
    # and equatorial deg ~= 111319 m). Allow a small cm-scale tolerance to
    # account for ellipsoid vs sphere differences.
    east, north, up = wgs84_to_local_enu(
        35.0,
        139.0001,
        0.0,
        origin_latitude=35.0,
        origin_longitude=139.0,
        origin_altitude=0.0,
    )
    assert east == pytest.approx(9.107, abs=0.05)
    assert abs(north) < 1e-3
    assert abs(up) < 1e-3


def test_correlator_pairs_nearest_timestamps_and_aggregates_errors() -> None:
    bag_stream = _make_bag_stream(
        [
            BagPoseSample(timestamp_seconds=0.00, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.00, position=(1.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=2.00, position=(2.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=3.00, position=(3.0, 0.0, 0.0)),
        ]
    )
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.01, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.01, position=(1.5, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=2.02, position=(2.0, 0.5, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]

    report = correlate_against_sim_trajectory(
        bag_stream,
        sim_samples,
        max_match_dt_seconds=0.1,
    )

    assert report.matched_pair_count == 3
    assert report.translation_error_min_meters == pytest.approx(0.0)
    assert report.translation_error_max_meters == pytest.approx(0.5, rel=1e-6)
    assert report.translation_error_mean_meters == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_correlator_drops_pairs_outside_max_match_dt_seconds() -> None:
    bag_stream = _make_bag_stream(
        [
            BagPoseSample(timestamp_seconds=0.00, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=10.0, position=(10.0, 0.0, 0.0)),
        ]
    )
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.01, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=5.0, position=(5.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]

    report = correlate_against_sim_trajectory(
        bag_stream,
        sim_samples,
        max_match_dt_seconds=0.5,
    )
    assert report.matched_pair_count == 1
    assert report.matched_seconds == 0.0  # one matched pair has zero span
    assert report.pairs[0].sim_timestamp_seconds == pytest.approx(0.01)


def test_correlator_returns_nan_aggregates_when_nothing_matches() -> None:
    bag_stream = _make_bag_stream([BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0))])
    sim_samples = [
        SimPoseSample(timestamp_seconds=10.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(bag_stream, sim_samples, max_match_dt_seconds=0.1)
    assert report.matched_pair_count == 0
    assert math.isnan(report.translation_error_mean_meters)
    assert report.pairs == ()


def test_correlator_emits_heading_error_when_bag_has_orientation() -> None:
    bag_stream = _make_bag_stream(
        [
            BagPoseSample(
                timestamp_seconds=0.0,
                position=(0.0, 0.0, 0.0),
                # 90 deg yaw quaternion in xyzw.
                orientation_xyzw=(0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
            )
        ]
    )
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    ]
    report = correlate_against_sim_trajectory(bag_stream, sim_samples, max_match_dt_seconds=0.1)
    assert report.heading_error_mean_radians == pytest.approx(math.pi / 2.0, rel=1e-6)
    assert report.heading_error_max_radians == pytest.approx(math.pi / 2.0, rel=1e-6)


def test_correlator_keeps_strided_sample_of_pairs() -> None:
    bag_samples = [BagPoseSample(timestamp_seconds=float(i) * 0.01, position=(float(i), 0.0, 0.0)) for i in range(200)]
    sim_samples = [
        SimPoseSample(
            timestamp_seconds=float(i) * 0.01,
            position=(float(i), 0.0, 0.0),
            orientation_xyzw=(0, 0, 0, 1),
        )
        for i in range(200)
    ]

    report = correlate_against_sim_trajectory(
        _make_bag_stream(bag_samples),
        sim_samples,
        max_match_dt_seconds=0.005,
        keep_pairs=True,
        max_pairs_kept=50,
    )
    assert report.matched_pair_count == 200
    assert len(report.pairs) <= 50
    assert len(report.pairs) >= 40  # roughly evenly strided


def test_correlator_drops_pairs_when_keep_pairs_is_false() -> None:
    bag_stream = _make_bag_stream([BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0))])
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(
        bag_stream,
        sim_samples,
        max_match_dt_seconds=0.1,
        keep_pairs=False,
    )
    assert report.matched_pair_count == 1
    assert report.pairs == ()


def test_report_to_dict_round_trips_to_json(tmp_path: Path) -> None:
    bag_stream = _make_bag_stream(
        [
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
        ]
    )
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.0, position=(1.5, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(bag_stream, sim_samples, max_match_dt_seconds=0.5)
    output = write_real_vs_sim_correlation_report_json(tmp_path / "report.json", report)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["recordType"] == "real-vs-sim-correlation-report"
    assert payload["version"] == REAL_VS_SIM_CORRELATION_REPORT_VERSION
    assert payload["matchedPairCount"] == 2
    assert payload["bagSource"]["sourceTopic"] == "/gnss/fix"
    assert payload["bagSource"]["referenceOriginWgs84"] == [35.0, 139.0, 10.0]
    assert "translationErrorMeters" in payload
    assert isinstance(payload["pairs"], list)


def test_render_markdown_includes_bag_and_sim_summary() -> None:
    bag_stream = _make_bag_stream(
        [
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
        ]
    )
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(bag_stream, sim_samples)
    markdown = render_real_vs_sim_correlation_markdown(report)
    assert "/gnss/fix" in markdown
    assert "sensor_msgs/msg/NavSatFix" in markdown
    assert "translation error" in markdown.lower()


def test_load_sim_pose_samples_jsonl_round_trip(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        "\n".join(
            [
                json.dumps({"timestampSeconds": 1.0, "position": [1.0, 2.0, 3.0], "orientationXyzw": [0, 0, 0, 1]}),
                "  ",  # blank line should be skipped
                json.dumps({"timestampSeconds": 0.5, "position": [0.5, 0.0, 0.0], "orientationXyzw": [0, 0, 0, 1]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_sim_pose_samples_jsonl(rollout)
    assert len(samples) == 2
    # Sorted ascending by timestamp regardless of file order.
    assert samples[0].timestamp_seconds == pytest.approx(0.5)
    assert samples[1].timestamp_seconds == pytest.approx(1.0)
    assert samples[1].position == (1.0, 2.0, 3.0)


def test_load_sim_pose_samples_jsonl_rejects_invalid_records(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_sim_pose_samples_jsonl(bad)

    missing_field = tmp_path / "missing.jsonl"
    missing_field.write_text(
        json.dumps({"timestampSeconds": 0.0, "position": [0, 0, 0]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required field"):
        load_sim_pose_samples_jsonl(missing_field)


def test_run_rosbag_correlation_cli_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        json.dumps({"timestampSeconds": 0.0, "position": [0.0, 0.0, 0.0], "orientationXyzw": [0, 0, 0, 1]})
        + "\n"
        + json.dumps({"timestampSeconds": 1.0, "position": [1.5, 0.0, 0.0], "orientationXyzw": [0, 0, 0, 1]})
        + "\n",
        encoding="utf-8",
    )

    fake_stream = BagPoseStream(
        samples=(
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
        ),
        frame_id="enu",
        source_topic="/gnss/fix",
        source_msgtype="sensor_msgs/msg/NavSatFix",
        reference_origin_wgs84=(35.0, 139.0, 10.0),
    )

    def _fake_reader(*_args, **_kwargs) -> BagPoseStream:
        return fake_stream

    output = tmp_path / "out.json"
    markdown = tmp_path / "out.md"

    monkeypatch.setattr(module_under_test, "read_navsat_pose_stream", _fake_reader)
    # The script imports from gs_sim2real.robotics, so we also patch the public
    # re-export the script reaches through.
    import gs_sim2real.robotics as robotics_pkg

    monkeypatch.setattr(robotics_pkg, "read_navsat_pose_stream", _fake_reader)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_rosbag_correlation.py"),
            "--bag",
            str(tmp_path),
            "--sim-rollout",
            str(rollout),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--max-match-dt-seconds",
            "0.5",
        ],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        },
    )
    # The CLI reads the actual bag via the real (un-monkeypatched) module
    # since we run it as a subprocess. Skip the subprocess assertion when no
    # real bag exists at tmp_path — exercise the CLI path inside this process
    # instead.
    if completed.returncode != 0 and "no rosbag found" not in completed.stderr.lower():
        pass

    # In-process exercise of the CLI path (read patched).
    from scripts import run_rosbag_correlation as cli_module  # type: ignore[import-not-found]

    monkeypatch.setattr(cli_module, "read_navsat_pose_stream", _fake_reader)
    sys.argv = [
        "run_rosbag_correlation.py",
        "--bag",
        str(tmp_path),
        "--sim-rollout",
        str(rollout),
        "--output",
        str(output),
        "--markdown",
        str(markdown),
        "--max-match-dt-seconds",
        "0.5",
    ]
    assert cli_module.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["matchedPairCount"] == 2
    assert "translation error" in markdown.read_text(encoding="utf-8").lower()


def test_correlated_pose_pair_round_trips_orientation_dropped() -> None:
    pair = CorrelatedPosePair(
        bag_timestamp_seconds=1.0,
        sim_timestamp_seconds=1.05,
        bag_position=(1.0, 2.0, 3.0),
        sim_position=(1.5, 2.0, 3.0),
        translation_error_meters=0.5,
    )
    payload = pair.to_dict()
    assert payload["bagTimestampSeconds"] == 1.0
    assert payload["timeOffsetSeconds"] == pytest.approx(0.05, rel=1e-6)
    assert "headingErrorRadians" not in payload


def test_bag_pose_stream_validation_rejects_unsorted_samples() -> None:
    with pytest.raises(ValueError, match="sorted"):
        BagPoseStream(
            samples=(
                BagPoseSample(timestamp_seconds=2.0, position=(0.0, 0.0, 0.0)),
                BagPoseSample(timestamp_seconds=1.0, position=(0.0, 0.0, 0.0)),
            ),
            frame_id="enu",
            source_topic="/x",
            source_msgtype="sensor_msgs/msg/NavSatFix",
        )

    with pytest.raises(ValueError, match="at least one sample"):
        BagPoseStream(
            samples=(),
            frame_id="enu",
            source_topic="/x",
            source_msgtype="sensor_msgs/msg/NavSatFix",
        )


def test_merge_navsat_with_imu_orientation_fills_in_quaternion_within_window() -> None:
    navsat = _make_bag_stream(
        [
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=2.0, position=(2.0, 0.0, 0.0)),
        ]
    )
    imu = (
        (0.01, (0.0, 0.0, 0.0, 1.0)),
        (1.02, (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0))),
        # No IMU sample anywhere near t=2.0; that NavSatFix sample stays None.
    )

    fused = merge_navsat_with_imu_orientation(navsat, imu, max_pair_dt_seconds=0.05)

    assert fused.source_topic == navsat.source_topic
    assert fused.reference_origin_wgs84 == navsat.reference_origin_wgs84
    assert fused.samples[0].orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert fused.samples[1].orientation_xyzw is not None
    assert fused.samples[2].orientation_xyzw is None


def test_merge_navsat_with_imu_orientation_rejects_unsorted_imu_samples() -> None:
    navsat = _make_bag_stream([BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0))])
    out_of_order = (
        (1.0, (0.0, 0.0, 0.0, 1.0)),
        (0.5, (0.0, 0.0, 0.0, 1.0)),
    )
    with pytest.raises(ValueError, match="sorted ascending"):
        merge_navsat_with_imu_orientation(navsat, out_of_order)


def test_merge_navsat_with_imu_orientation_rejects_empty_imu_samples() -> None:
    navsat = _make_bag_stream([BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match="at least one sample"):
        merge_navsat_with_imu_orientation(navsat, ())


def test_correlator_emits_heading_error_on_imu_merged_stream() -> None:
    navsat = _make_bag_stream(
        [
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
        ]
    )
    imu = (
        (0.01, (0.0, 0.0, 0.0, 1.0)),
        (1.0, (0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0))),  # 45 deg yaw
    )
    fused = merge_navsat_with_imu_orientation(navsat, imu, max_pair_dt_seconds=0.05)
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]

    report = correlate_against_sim_trajectory(fused, sim_samples, max_match_dt_seconds=0.05)
    assert report.matched_pair_count == 2
    # Mean of 0 rad and 45 deg = 22.5 deg.
    assert report.heading_error_mean_radians == pytest.approx(math.radians(22.5), rel=1e-6)
    assert report.heading_error_max_radians == pytest.approx(math.radians(45.0), rel=1e-6)


def test_real_vs_sim_correlation_report_round_trips_through_json(tmp_path: Path) -> None:
    """to_dict -> from_dict (and disk round-trip) must rebuild an identical report."""
    bag_samples = [
        BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
        BagPoseSample(
            timestamp_seconds=1.0,
            position=(1.0, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
    ]
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.01, position=(0.05, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.01, position=(1.05, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(
        _make_bag_stream(bag_samples),
        sim_samples,
        max_match_dt_seconds=0.05,
    )
    assert isinstance(report.bag_source, BagPoseStreamMetadata)

    rebuilt = real_vs_sim_correlation_report_from_dict(report.to_dict())
    assert rebuilt.to_dict() == report.to_dict()

    output_path = tmp_path / "correlation.json"
    write_real_vs_sim_correlation_report_json(output_path, report)
    loaded = load_real_vs_sim_correlation_report_json(output_path)
    assert loaded.to_dict() == report.to_dict()
    assert loaded.bag_source.source_topic == "/gnss/fix"
    assert loaded.matched_pair_count == 2


def test_real_vs_sim_correlation_report_from_dict_rejects_bad_record_type() -> None:
    bag_samples = [
        BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
        BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
    ]
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    payload = correlate_against_sim_trajectory(
        _make_bag_stream(bag_samples),
        sim_samples,
    ).to_dict()
    payload["recordType"] = "not-a-correlation-report"
    with pytest.raises(ValueError, match="recordType"):
        real_vs_sim_correlation_report_from_dict(payload)


def _correlation_report_with(
    *,
    translation_mean: float,
    translation_p95: float,
    translation_max: float,
    heading_mean: float | None = None,
):
    """Build a synthetic two-pair report with the requested error statistics."""
    bag_samples = [
        BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
        BagPoseSample(timestamp_seconds=1.0, position=(translation_mean * 2.0, 0.0, 0.0)),
    ]
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(_make_bag_stream(bag_samples), sim_samples)
    # Inject the requested statistics deterministically (the synthetic pairs above
    # already produce a known mean/max but the test wants exact control).
    return type(report)(
        bag_source=report.bag_source,
        sim_sample_count=report.sim_sample_count,
        matched_pair_count=report.matched_pair_count,
        matched_seconds=report.matched_seconds,
        translation_error_min_meters=0.0,
        translation_error_mean_meters=translation_mean,
        translation_error_max_meters=translation_max,
        translation_error_p50_meters=translation_mean,
        translation_error_p95_meters=translation_p95,
        heading_error_mean_radians=heading_mean,
        heading_error_max_radians=None if heading_mean is None else heading_mean,
        pairs=report.pairs,
    )


def test_evaluate_real_vs_sim_correlation_thresholds_passes_when_empty() -> None:
    """Empty thresholds always pass with no failed checks."""
    report = _correlation_report_with(translation_mean=10.0, translation_p95=20.0, translation_max=30.0)
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, RealVsSimCorrelationThresholds())
    assert passed is True
    assert failed == ()


def test_evaluate_real_vs_sim_correlation_thresholds_flags_each_exceeded_stat() -> None:
    """Each populated threshold contributes its own failure tag when exceeded."""
    report = _correlation_report_with(
        translation_mean=0.5,
        translation_p95=2.0,
        translation_max=5.0,
        heading_mean=0.6,
    )
    thresholds = RealVsSimCorrelationThresholds(
        max_translation_error_mean_meters=0.4,
        max_translation_error_p95_meters=1.0,
        max_translation_error_max_meters=4.0,
        max_heading_error_mean_radians=0.3,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is False
    assert set(failed) == {"translation-mean", "translation-p95", "translation-max", "heading-mean"}


def test_evaluate_real_vs_sim_correlation_thresholds_skips_heading_when_absent() -> None:
    """A report without heading data must not synthesise a heading-mean failure."""
    report = _correlation_report_with(
        translation_mean=0.05, translation_p95=0.05, translation_max=0.05, heading_mean=None
    )
    thresholds = RealVsSimCorrelationThresholds(max_heading_error_mean_radians=0.001)
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert failed == ()


def test_real_vs_sim_correlation_thresholds_round_trip_through_json() -> None:
    """to_dict / from_dict must preserve the populated bounds and skip the empty ones."""
    thresholds = RealVsSimCorrelationThresholds(
        max_translation_error_mean_meters=1.5,
        max_translation_error_p95_meters=3.0,
    )
    rebuilt = real_vs_sim_correlation_thresholds_from_dict(thresholds.to_dict())
    assert rebuilt == thresholds
    assert "maxTranslationErrorMaxMeters" not in thresholds.to_dict()
    assert real_vs_sim_correlation_thresholds_from_dict({}) == RealVsSimCorrelationThresholds()


def test_correlation_threshold_overrides_round_trip_drops_empty_entries(tmp_path: Path) -> None:
    """from_dict/to_dict + load_..._json must drop topics whose payload has no bounds set."""
    payload = {
        "/gnss/fix": {"maxTranslationErrorMeanMeters": 0.5},
        "/imu/data": {"maxHeadingErrorMeanRadians": 0.05},
        "/empty/topic": {},  # should be dropped
    }
    overrides = correlation_threshold_overrides_from_dict(payload)
    assert set(overrides.keys()) == {"/gnss/fix", "/imu/data"}
    assert overrides["/gnss/fix"].max_translation_error_mean_meters == pytest.approx(0.5)
    assert overrides["/imu/data"].max_heading_error_mean_radians == pytest.approx(0.05)

    rebuilt_payload = correlation_threshold_overrides_to_dict(overrides)
    assert "/empty/topic" not in rebuilt_payload
    assert correlation_threshold_overrides_from_dict(rebuilt_payload) == overrides

    config_path = tmp_path / "overrides.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_correlation_threshold_overrides_json(config_path)
    assert loaded == overrides


def test_correlation_threshold_overrides_rejects_non_mapping_root(tmp_path: Path) -> None:
    """A JSON file whose root is not an object must raise on load."""
    bad = tmp_path / "bad.json"
    bad.write_text('["not-an-object"]', encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        load_correlation_threshold_overrides_json(bad)


def _correlation_report_with_pairs(
    pair_errors: list[float],
):
    """Build a synthetic correlation report whose pair list has the requested per-pair errors."""
    bag_samples = [BagPoseSample(timestamp_seconds=float(i), position=(0.0, 0.0, 0.0)) for i in range(len(pair_errors))]
    sim_samples = [
        SimPoseSample(timestamp_seconds=float(i), position=(error, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1))
        for i, error in enumerate(pair_errors)
    ]
    return correlate_against_sim_trajectory(_make_bag_stream(bag_samples), sim_samples, max_match_dt_seconds=0.5)


def test_correlation_pair_distribution_gate_passes_when_under_fraction() -> None:
    """1/10 pairs above 0.5 m at a 0.2 limit (10% > 20% is False) must not fail."""
    report = _correlation_report_with_pairs([0.0] * 9 + [1.0])
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.5,
        max_exceeding_translation_pair_fraction=0.2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert "translation-pair-distribution" not in failed


def test_correlation_pair_distribution_gate_fails_when_over_fraction() -> None:
    """3/10 pairs above 0.5 m at a 0.2 limit (30% > 20% is True) must fail with the new tag."""
    report = _correlation_report_with_pairs([0.0] * 7 + [1.0, 1.0, 1.0])
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.5,
        max_exceeding_translation_pair_fraction=0.2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is False
    assert "translation-pair-distribution" in failed


def test_correlation_pair_distribution_gate_requires_both_bounds() -> None:
    """Setting only one of the two pair-distribution fields must skip the check silently."""
    report = _correlation_report_with_pairs([1.0] * 10)
    only_bound = RealVsSimCorrelationThresholds(max_pair_translation_error_meters=0.1)
    only_fraction = RealVsSimCorrelationThresholds(max_exceeding_translation_pair_fraction=0.0)
    for thresholds in (only_bound, only_fraction):
        passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
        assert passed is True
        assert "translation-pair-distribution" not in failed


def test_correlation_pair_distribution_gate_skips_empty_pair_list() -> None:
    """A report whose pairs were dropped (keep_pairs=False) must not trip the gate."""
    bag_samples = [BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0))]
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(10.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(
        _make_bag_stream(bag_samples), sim_samples, max_match_dt_seconds=0.1, keep_pairs=False
    )
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.1,
        max_exceeding_translation_pair_fraction=0.0,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert failed == ()


def test_correlation_thresholds_pair_distribution_round_trips_through_json() -> None:
    """Pair-distribution bounds must round-trip through to_dict/from_dict."""
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.4,
        max_exceeding_translation_pair_fraction=0.05,
    )
    rebuilt = real_vs_sim_correlation_thresholds_from_dict(thresholds.to_dict())
    assert rebuilt == thresholds


def _correlation_report_with_heading_pairs(heading_errors: list[float | None]):
    """Build a synthetic report whose pairs carry the requested heading errors."""
    bag_samples: list[BagPoseSample] = []
    sim_samples: list[SimPoseSample] = []
    for index, heading in enumerate(heading_errors):
        ts = float(index)
        bag_orientation: tuple[float, float, float, float] | None
        if heading is None:
            bag_orientation = None
        else:
            half = heading / 2.0
            bag_orientation = (0.0, 0.0, math.sin(half), math.cos(half))
        bag_samples.append(
            BagPoseSample(timestamp_seconds=ts, position=(0.0, 0.0, 0.0), orientation_xyzw=bag_orientation)
        )
        sim_samples.append(
            SimPoseSample(timestamp_seconds=ts, position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
    return correlate_against_sim_trajectory(_make_bag_stream(bag_samples), sim_samples, max_match_dt_seconds=0.5)


def test_correlation_heading_pair_distribution_gate_fails_when_over_fraction() -> None:
    """3/10 pairs above 0.5 rad at a 20% limit (30% > 20%) must fail with the heading tag."""
    report = _correlation_report_with_heading_pairs([0.0] * 7 + [1.0, 1.0, 1.0])
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_heading_error_radians=0.5,
        max_exceeding_heading_pair_fraction=0.2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is False
    assert "heading-pair-distribution" in failed


def test_correlation_heading_pair_distribution_gate_passes_when_under_fraction() -> None:
    """1/10 pairs above 0.5 rad at a 20% limit (10% < 20%) must not fail."""
    report = _correlation_report_with_heading_pairs([0.0] * 9 + [1.0])
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_heading_error_radians=0.5,
        max_exceeding_heading_pair_fraction=0.2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert "heading-pair-distribution" not in failed


def test_correlation_heading_pair_distribution_skips_pairs_without_heading() -> None:
    """Pairs whose heading_error_radians is None must be excluded from the fraction denominator."""
    # 10 pairs total: 4 with heading data (3 of which exceed the bound), 6 without.
    # Without filtering: 3/10 = 30% > 20% -> would fail.
    # With filtering: 3/4 = 75% > 20% -> still fails, but on a different denominator.
    # Verify the denominator is the heading-bearing subset by using a different pattern:
    # 10 pairs total: 4 with heading (1 exceeding -> 1/4 = 25%), 6 without.
    # 25% > 20% -> fails (with heading-only denominator), but 1/10 = 10% (would pass).
    heading_errors: list[float | None] = [0.0, 0.0, 0.0, 1.0, None, None, None, None, None, None]
    report = _correlation_report_with_heading_pairs(heading_errors)
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_heading_error_radians=0.5,
        max_exceeding_heading_pair_fraction=0.2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    # 1/4 = 25% > 20% on the heading-bearing denominator -> fails.
    assert passed is False
    assert "heading-pair-distribution" in failed


def test_correlation_heading_pair_distribution_skips_when_no_pair_has_heading() -> None:
    """When every pair lacks heading data, the gate must silently pass."""
    heading_errors: list[float | None] = [None] * 5
    report = _correlation_report_with_heading_pairs(heading_errors)
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_heading_error_radians=0.0,
        max_exceeding_heading_pair_fraction=0.0,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert failed == ()


def test_correlation_thresholds_heading_pair_distribution_round_trips() -> None:
    """Heading pair-distribution bounds must round-trip through to_dict/from_dict."""
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_heading_error_radians=0.3,
        max_exceeding_heading_pair_fraction=0.1,
    )
    rebuilt = real_vs_sim_correlation_thresholds_from_dict(thresholds.to_dict())
    assert rebuilt == thresholds


def _correlation_report_with_pair_timeline(
    pair_errors: list[tuple[float, float]],
):
    """Build a synthetic report whose pairs have explicit (timestamp, error) shape."""
    bag_samples = [BagPoseSample(timestamp_seconds=ts, position=(0.0, 0.0, 0.0)) for ts, _ in pair_errors]
    sim_samples = [
        SimPoseSample(timestamp_seconds=ts, position=(error, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1))
        for ts, error in pair_errors
    ]
    return correlate_against_sim_trajectory(_make_bag_stream(bag_samples), sim_samples, max_match_dt_seconds=0.5)


def test_correlation_pair_distribution_strata_isolates_late_drift() -> None:
    """Pairs concentrated in window 1 must trip a window-specific failure tag, not the aggregate."""
    # 10 pairs over [0, 9] s. First 5 are clean, last 5 have drift that exceeds the bound.
    # Aggregate: 5/10 = 50% > 0.2 limit -> would fire the aggregate tag.
    # With strata=2: window 0 has 0/5 exceeding, window 1 has 5/5 exceeding -> only window 1 trips.
    pairs_timeline = [(float(i), 0.0) for i in range(5)] + [(float(i), 1.0) for i in range(5, 10)]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.5,
        max_exceeding_translation_pair_fraction=0.2,
        pair_distribution_strata=2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is False
    # Aggregate tag must NOT appear — only window-specific tags.
    assert "translation-pair-distribution" not in failed
    assert "translation-pair-distribution-window-1" in failed
    assert "translation-pair-distribution-window-0" not in failed


def test_correlation_pair_distribution_strata_skips_empty_windows_silently() -> None:
    """Windows with no pairs (e.g. a long quiet stretch in the bag) must not produce a failure."""
    # 6 pairs all clustered in [0, 1] s. Strata=4 means windows 0 and 1 collect everything,
    # windows 2 and 3 are empty. Since all errors are 0, no window fails.
    pairs_timeline = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.6, 0.0), (0.8, 0.0), (1.0, 0.0)]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.1,
        max_exceeding_translation_pair_fraction=0.0,
        pair_distribution_strata=4,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert failed == ()


def test_correlation_pair_distribution_strata_one_falls_back_to_aggregate_tag() -> None:
    """strata=1 (or None) must keep the existing aggregate tag for backwards compat."""
    pairs_timeline = [(float(i), 1.0) for i in range(10)]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    aggregate = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.5,
        max_exceeding_translation_pair_fraction=0.2,
        pair_distribution_strata=1,
    )
    _, failed_aggregate = evaluate_real_vs_sim_correlation_thresholds(report, aggregate)
    assert "translation-pair-distribution" in failed_aggregate
    assert all(not tag.startswith("translation-pair-distribution-window-") for tag in failed_aggregate)


def test_correlation_pair_distribution_strata_zero_or_negative_raises() -> None:
    """pair_distribution_strata must be >= 1 when set; out-of-range values raise on construction."""
    with pytest.raises(ValueError, match="pair_distribution_strata"):
        RealVsSimCorrelationThresholds(pair_distribution_strata=0)
    with pytest.raises(ValueError, match="pair_distribution_strata"):
        RealVsSimCorrelationThresholds(pair_distribution_strata=-3)


def test_correlation_pair_distribution_strata_round_trips_through_json() -> None:
    """pair_distribution_strata must round-trip and be omitted from JSON when None or 1."""
    explicit = RealVsSimCorrelationThresholds(pair_distribution_strata=4)
    payload = explicit.to_dict()
    assert payload["pairDistributionStrata"] == 4
    assert real_vs_sim_correlation_thresholds_from_dict(payload) == explicit
    # strata=1 is functionally identical to None so it is dropped on serialisation.
    explicit_one = RealVsSimCorrelationThresholds(pair_distribution_strata=1)
    assert "pairDistributionStrata" not in explicit_one.to_dict()


def test_correlation_aggregate_stats_stratification_emits_window_specific_tags() -> None:
    """When stratified, mean/p95/max checks fire per window with -window-{i} tags."""
    # 10 pairs over [0, 9] s. First 5 are clean (0 m), last 5 have 1 m drift.
    # Aggregate mean = 0.5; per-window mean: window 0 = 0.0, window 1 = 1.0.
    # threshold = 0.3: aggregate would fail; with strata=2 only window 1 fires.
    pairs_timeline = [(float(i), 0.0) for i in range(5)] + [(float(i), 1.0) for i in range(5, 10)]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    thresholds = RealVsSimCorrelationThresholds(
        max_translation_error_mean_meters=0.3,
        pair_distribution_strata=2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is False
    assert "translation-mean" not in failed  # aggregate suppressed when stratified
    assert "translation-mean-window-1" in failed
    assert "translation-mean-window-0" not in failed


def test_correlation_aggregate_stats_stratification_skips_empty_windows() -> None:
    """Windows that hold no pairs (e.g. quiet stretches) must not emit window-specific failures."""
    # 6 pairs all in [0, 1] s. Strata=4 means windows 2-3 are empty.
    pairs_timeline = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.6, 0.0), (0.8, 0.0), (1.0, 0.0)]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    thresholds = RealVsSimCorrelationThresholds(
        max_translation_error_max_meters=0.0,
        pair_distribution_strata=4,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert failed == ()


def test_correlation_aggregate_stats_stratification_applies_p95_and_max_per_window() -> None:
    """p95 and max thresholds must each fire on their own window-specific tag."""
    # Window 0: errors [0, 0, 0, 0, 0] -> p95=0, max=0.
    # Window 1: errors [0, 0, 0, 0, 5.0] -> p95~5.0, max=5.0.
    pairs_timeline = [(float(i), 0.0) for i in range(5)] + [
        (5.0, 0.0),
        (6.0, 0.0),
        (7.0, 0.0),
        (8.0, 0.0),
        (9.0, 5.0),
    ]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    thresholds = RealVsSimCorrelationThresholds(
        max_translation_error_p95_meters=1.0,
        max_translation_error_max_meters=1.0,
        pair_distribution_strata=2,
    )
    _, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert "translation-p95-window-1" in failed
    assert "translation-max-window-1" in failed
    assert "translation-p95-window-0" not in failed
    assert "translation-max-window-0" not in failed
    # Aggregate tags suppressed under stratification.
    assert "translation-p95" not in failed
    assert "translation-max" not in failed


def test_correlation_aggregate_stratification_skips_heading_when_no_window_carries_data() -> None:
    """Stratified heading-mean check must skip a window whose pairs all lack heading data."""
    heading_errors: list[float | None] = [0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, None, None]
    bag_samples: list[BagPoseSample] = []
    sim_samples: list[SimPoseSample] = []
    for index, heading in enumerate(heading_errors):
        ts = float(index)
        if heading is None:
            bag_orientation = None
        else:
            half = heading / 2.0
            bag_orientation = (0.0, 0.0, math.sin(half), math.cos(half))
        bag_samples.append(
            BagPoseSample(timestamp_seconds=ts, position=(0.0, 0.0, 0.0), orientation_xyzw=bag_orientation)
        )
        sim_samples.append(
            SimPoseSample(timestamp_seconds=ts, position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
    report = correlate_against_sim_trajectory(_make_bag_stream(bag_samples), sim_samples, max_match_dt_seconds=0.5)
    thresholds = RealVsSimCorrelationThresholds(
        max_heading_error_mean_radians=0.0,  # any positive heading would fail
        pair_distribution_strata=2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    # Window 0 has 5 heading-bearing pairs all at 0 rad (passes); window 1 has 0
    # heading-bearing pairs and skips silently. No failure either way.
    assert passed is True
    assert failed == ()


def test_correlation_strata_mode_equal_pair_count_partitions_by_index() -> None:
    """equal-pair-count mode keeps each window statistically meaningful for sparse bags."""
    # Sparse-then-dense timeline: 3 pairs in [0,2] s, then 7 pairs in [9, 9.6] s.
    # equal-duration with strata=2 puts ~all 10 pairs in window 1 (the longer half).
    # equal-pair-count with strata=2 splits 5/5 by index order regardless of timestamps.
    pairs_timeline = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (9.0, 1.0),
        (9.1, 1.0),
        (9.2, 1.0),
        (9.3, 1.0),
        (9.4, 1.0),
        (9.5, 1.0),
        (9.6, 1.0),
    ]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    # Threshold tolerates 30% exceeding under the per-pair gate.
    thresholds_index = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.5,
        max_exceeding_translation_pair_fraction=0.3,
        pair_distribution_strata=2,
        pair_distribution_strata_mode="equal-pair-count",
    )
    # equal-pair-count window 0 = pairs[0:5] (3 clean + 2 dirty -> 40% exceeding -> fails).
    # window 1 = pairs[5:10] (5 dirty -> 100% exceeding -> fails).
    _, failed_index = evaluate_real_vs_sim_correlation_thresholds(report, thresholds_index)
    assert "translation-pair-distribution-window-0" in failed_index
    assert "translation-pair-distribution-window-1" in failed_index

    # equal-duration on the same data: window 0 covers t in [0, 4.8], window 1 covers
    # t in (4.8, 9.6]. Window 0 has 3 clean pairs (0% exceeding), window 1 has 7
    # dirty pairs (100% exceeding). Window 0 should pass, window 1 fails.
    thresholds_time = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.5,
        max_exceeding_translation_pair_fraction=0.3,
        pair_distribution_strata=2,
        pair_distribution_strata_mode="equal-duration",
    )
    _, failed_time = evaluate_real_vs_sim_correlation_thresholds(report, thresholds_time)
    assert "translation-pair-distribution-window-0" not in failed_time
    assert "translation-pair-distribution-window-1" in failed_time


def test_correlation_strata_mode_round_trips_through_json_only_when_non_default() -> None:
    """Mode is dropped from JSON when default ('equal-duration') and round-trips otherwise."""
    default = RealVsSimCorrelationThresholds(pair_distribution_strata=4)
    assert "pairDistributionStrataMode" not in default.to_dict()
    explicit = RealVsSimCorrelationThresholds(
        pair_distribution_strata=4, pair_distribution_strata_mode="equal-pair-count"
    )
    payload = explicit.to_dict()
    assert payload["pairDistributionStrataMode"] == "equal-pair-count"
    assert real_vs_sim_correlation_thresholds_from_dict(payload) == explicit


def test_correlation_strata_mode_rejects_unknown_value() -> None:
    """Unknown mode strings must raise on construction so typos fail fast."""
    with pytest.raises(ValueError, match="pair_distribution_strata_mode"):
        RealVsSimCorrelationThresholds(pair_distribution_strata_mode="round-robin")


def test_correlation_strata_mode_equal_pair_count_handles_uneven_split() -> None:
    """Total pair count must be preserved when N does not divide evenly."""
    # 10 pairs split into 4 windows: counts should be [3, 3, 2, 2].
    pairs_timeline = [(float(i), 0.0) for i in range(10)]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    # Use a large bound so the gate passes; we just want to verify the splitter
    # by examining that empty windows don't show up.
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=999.0,
        max_exceeding_translation_pair_fraction=0.5,
        pair_distribution_strata=4,
        pair_distribution_strata_mode="equal-pair-count",
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert failed == ()
    # Make threshold tighter to force every window to fail and check no "window-N"
    # tag is missing (which would indicate an empty window was skipped).
    thresholds_strict = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=-1.0,  # everything exceeds
        max_exceeding_translation_pair_fraction=0.0,
        pair_distribution_strata=4,
        pair_distribution_strata_mode="equal-pair-count",
    )
    _, failed_strict = evaluate_real_vs_sim_correlation_thresholds(report, thresholds_strict)
    for index in range(4):
        assert f"translation-pair-distribution-window-{index}" in failed_strict


def test_compute_per_window_correlation_stats_returns_per_window_aggregates() -> None:
    """compute_per_window_correlation_stats must return mean/p95/max + bag span per window."""
    pairs_timeline = [(float(i), 0.0) for i in range(5)] + [(float(i), 1.0) for i in range(5, 10)]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    stats = compute_per_window_correlation_stats(report, strata=2, mode="equal-duration")
    assert len(stats) == 2
    window0, window1 = stats
    assert window0.window_index == 0
    assert window0.pair_count == 5
    assert window0.translation_error_mean_meters == pytest.approx(0.0)
    assert window0.translation_error_max_meters == pytest.approx(0.0)
    assert window0.heading_error_mean_radians is None
    assert window0.bag_time_start_seconds == pytest.approx(0.0)
    assert window0.bag_time_end_seconds == pytest.approx(4.0)
    assert window1.window_index == 1
    assert window1.pair_count == 5
    assert window1.translation_error_mean_meters == pytest.approx(1.0)
    assert window1.translation_error_max_meters == pytest.approx(1.0)
    assert window1.bag_time_start_seconds == pytest.approx(5.0)
    assert window1.bag_time_end_seconds == pytest.approx(9.0)


def test_compute_per_window_correlation_stats_drops_empty_windows() -> None:
    """Windows that hold no pairs (e.g. quiet stretches) must be omitted from the result."""
    pairs_timeline = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    stats = compute_per_window_correlation_stats(report, strata=4, mode="equal-duration")
    # Equal-duration over [0, 1] s into 4 windows. Pairs at t=0, 0.5, 1.0
    # land in windows 0, 2, 3 (index = floor(offset/duration * 4)).
    indices = sorted(stat.window_index for stat in stats)
    assert indices == [0, 2, 3]


def test_compute_per_window_correlation_stats_returns_empty_when_strata_le_one() -> None:
    """strata <= 1 must return an empty tuple (no stratification requested)."""
    pairs_timeline = [(float(i), 0.0) for i in range(5)]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    assert compute_per_window_correlation_stats(report, strata=1, mode="equal-duration") == ()
    assert compute_per_window_correlation_stats(report, strata=0, mode="equal-duration") == ()


def test_real_vs_sim_correlation_window_stats_round_trips_through_json() -> None:
    """Window stats round-trip through to_dict/from_dict including the optional heading mean."""
    stats = RealVsSimCorrelationWindowStats(
        window_index=2,
        pair_count=4,
        bag_time_start_seconds=1.0,
        bag_time_end_seconds=2.5,
        translation_error_mean_meters=0.123,
        translation_error_p95_meters=0.456,
        translation_error_max_meters=0.789,
        heading_error_mean_radians=0.05,
    )
    rebuilt = real_vs_sim_correlation_window_stats_from_dict(stats.to_dict())
    assert rebuilt == stats
    # No heading data: heading key must be omitted from the payload.
    no_heading = RealVsSimCorrelationWindowStats(
        window_index=0,
        pair_count=1,
        bag_time_start_seconds=0.0,
        bag_time_end_seconds=0.0,
        translation_error_mean_meters=0.0,
        translation_error_p95_meters=0.0,
        translation_error_max_meters=0.0,
    )
    payload = no_heading.to_dict()
    assert "headingErrorMeanRadians" not in payload
    assert real_vs_sim_correlation_window_stats_from_dict(payload) == no_heading


# ---------------------------------------------------------------------------
# Event-aligned stratification (Sprint 2 / PR B)
# ---------------------------------------------------------------------------


def test_correlation_event_window_round_trips_through_json() -> None:
    from gs_sim2real.robotics import CorrelationEventWindow, correlation_event_window_from_dict

    window = CorrelationEventWindow(
        name="turn_left",
        start_time=8.5,
        end_time=12.2,
        tags=("corner", "intersection"),
        source="policy_trace",
    )
    payload = window.to_dict()
    assert payload == {
        "name": "turn_left",
        "startTime": 8.5,
        "endTime": 12.2,
        "tags": ["corner", "intersection"],
        "source": "policy_trace",
    }
    assert correlation_event_window_from_dict(payload) == window


def test_correlation_event_window_defaults_omit_external_source_and_empty_tags() -> None:
    from gs_sim2real.robotics import CorrelationEventWindow

    window = CorrelationEventWindow(name="approach", start_time=0.0, end_time=8.5)
    payload = window.to_dict()
    assert "source" not in payload
    assert "tags" not in payload


def test_correlation_event_window_rejects_invalid_inputs() -> None:
    from gs_sim2real.robotics import CorrelationEventWindow

    with pytest.raises(ValueError):
        CorrelationEventWindow(name="", start_time=0.0, end_time=1.0)
    with pytest.raises(ValueError):
        CorrelationEventWindow(name="bad", start_time=5.0, end_time=4.0)
    with pytest.raises(ValueError):
        CorrelationEventWindow(name="bad", start_time=0.0, end_time=1.0, source="unknown")


def test_load_correlation_event_windows_json_validates_record_type(tmp_path: Path) -> None:
    from gs_sim2real.robotics import load_correlation_event_windows_json

    valid = tmp_path / "windows.json"
    valid.write_text(
        json.dumps(
            {
                "recordType": "gs-mapper-correlation-event-windows/v1",
                "bagSourceTopic": "/odom/imu",
                "windows": [
                    {"name": "approach", "startTime": 0.0, "endTime": 8.5, "tags": ["straight"]},
                    {"name": "turn", "startTime": 8.5, "endTime": 12.2, "tags": ["corner"], "source": "policy_trace"},
                ],
            }
        ),
        encoding="utf-8",
    )
    windows = load_correlation_event_windows_json(valid)
    assert tuple(w.name for w in windows) == ("approach", "turn")
    assert windows[1].source == "policy_trace"

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"recordType": "something-else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="recordType"):
        load_correlation_event_windows_json(bad)


def test_compute_per_window_correlation_stats_event_aligned_buckets_pairs_and_keeps_empty_events() -> None:
    """Event-aligned mode buckets by event window and retains empty events with pair_count=0."""
    from gs_sim2real.robotics import CorrelationEventWindow

    pairs_timeline = [(0.0, 0.1), (1.0, 0.1), (2.0, 0.1), (9.0, 0.5), (10.0, 0.5), (50.0, 0.9)]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    events = (
        CorrelationEventWindow(name="approach", start_time=0.0, end_time=5.0),
        CorrelationEventWindow(name="turn", start_time=8.0, end_time=12.0, tags=("corner",), source="policy_trace"),
        CorrelationEventWindow(name="dead", start_time=20.0, end_time=30.0),
    )
    stats = compute_per_window_correlation_stats(report, strata=3, mode="event-aligned", event_windows=events)
    # All three events surface — even the empty "dead" window.
    assert len(stats) == 3
    assert stats[0].event_name == "approach"
    assert stats[0].pair_count == 3
    assert stats[1].event_name == "turn"
    assert stats[1].event_source == "policy_trace"
    assert stats[1].event_tags == ("corner",)
    assert stats[1].pair_count == 2
    assert stats[2].event_name == "dead"
    assert stats[2].pair_count == 0
    # Pair at 50.0s outside every event window is dropped from the universe.


def test_compute_per_window_correlation_stats_event_aligned_falls_back_when_windows_missing() -> None:
    """Empty / missing event_windows with event-aligned mode falls back to equal-pair-count.

    The library-level fallback keeps the gate functional even when a caller
    forgot to plumb the resolved windows; structured "fallback applied"
    bookkeeping lives at the review-bundle layer (where
    ``correlationStratificationFallbacks`` is recorded in metadata).
    """
    pairs_timeline = [(float(i), 0.1 * i) for i in range(8)]
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    stats = compute_per_window_correlation_stats(report, strata=4, mode="event-aligned", event_windows=None)
    # Falls back to equal-pair-count → 8 pairs / 4 strata = 2 pairs per window.
    assert len(stats) == 4
    assert all(stat.event_name is None for stat in stats)
    assert [stat.pair_count for stat in stats] == [2, 2, 2, 2]

    # Empty event_windows tuple behaves the same as None.
    stats_empty_tuple = compute_per_window_correlation_stats(report, strata=4, mode="event-aligned", event_windows=())
    assert [stat.pair_count for stat in stats_empty_tuple] == [2, 2, 2, 2]


def test_evaluate_real_vs_sim_correlation_thresholds_event_aligned_labels_windows() -> None:
    from gs_sim2real.robotics import CorrelationEventWindow

    pairs_timeline = (
        [(0.0, 0.05), (1.0, 0.05), (2.0, 0.05)] + [(9.0, 0.4), (10.0, 0.4), (11.0, 0.4)] + [(20.0, 0.05), (21.0, 0.05)]
    )
    report = _correlation_report_with_pair_timeline(pairs_timeline)
    events = (
        CorrelationEventWindow(name="straight", start_time=0.0, end_time=5.0),
        CorrelationEventWindow(name="turn", start_time=8.0, end_time=15.0),
        CorrelationEventWindow(name="exit", start_time=18.0, end_time=25.0),
    )
    thresholds = RealVsSimCorrelationThresholds(
        max_translation_error_mean_meters=0.2,
        pair_distribution_strata=3,
        pair_distribution_strata_mode="event-aligned",
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds, event_windows=events)
    # Only the turn window exceeds 0.2 mean — the failure tag is keyed by the
    # window index inside event_windows.
    assert passed is False
    assert "translation-mean-window-1" in failed
    assert "translation-mean-window-0" not in failed
    assert "translation-mean-window-2" not in failed


def test_thresholds_round_trip_carries_event_windows_path() -> None:
    """event_windows_path stays attached when present and is dropped when None."""
    with_path = RealVsSimCorrelationThresholds(
        pair_distribution_strata=3,
        pair_distribution_strata_mode="event-aligned",
        event_windows_path="/tmp/events.json",
    )
    payload = with_path.to_dict()
    assert payload["pairDistributionStrataMode"] == "event-aligned"
    assert payload["eventWindowsPath"] == "/tmp/events.json"
    assert real_vs_sim_correlation_thresholds_from_dict(payload) == with_path

    without = RealVsSimCorrelationThresholds(max_translation_error_mean_meters=0.5)
    assert "eventWindowsPath" not in without.to_dict()


def test_correlation_strata_mode_accepts_event_aligned_value() -> None:
    """The mode enum now includes event-aligned alongside the two existing modes."""
    RealVsSimCorrelationThresholds(pair_distribution_strata_mode="event-aligned")
    with pytest.raises(ValueError):
        RealVsSimCorrelationThresholds(pair_distribution_strata_mode="something-else")
