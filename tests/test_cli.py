"""Tests for CLI entry point and subcommand help."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gs_sim2real.cli import _resolve_photos_to_splat_quality, build_parser, main


class TestCLIHelp:
    """Verify that CLI --help for each subcommand exits cleanly."""

    def test_cli_help(self) -> None:
        """Running main with --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_cli_download_help(self) -> None:
        """Running download --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["download", "--help"])
        assert exc_info.value.code == 0

    def test_cli_preprocess_help(self) -> None:
        """Running preprocess --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["preprocess", "--help"])
        assert exc_info.value.code == 0

    def test_cli_train_help(self) -> None:
        """Running train --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["train", "--help"])
        assert exc_info.value.code == 0

    def test_cli_large_scale_3dgs_plan_help(self) -> None:
        """Running large-scale-3dgs-plan --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["large-scale-3dgs-plan", "--help"])
        assert exc_info.value.code == 0

    def test_cli_large_scale_3dgs_smoke_data_help(self) -> None:
        """Running large-scale-3dgs-smoke-data --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["large-scale-3dgs-smoke-data", "--help"])
        assert exc_info.value.code == 0

    def test_cli_large_scale_3dgs_preflight_help(self) -> None:
        """Running large-scale-3dgs-preflight --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["large-scale-3dgs-preflight", "--help"])
        assert exc_info.value.code == 0

    def test_cli_large_scale_3dgs_pilot_help(self) -> None:
        """Running large-scale-3dgs-pilot --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["large-scale-3dgs-pilot", "--help"])
        assert exc_info.value.code == 0

    def test_cli_large_scale_3dgs_preflight_write_plan_args(self) -> None:
        """large-scale-3dgs-preflight accepts plan materialization args."""
        args = build_parser().parse_args(
            [
                "large-scale-3dgs-preflight",
                "--data",
                "outputs/sparse",
                "--write-plan",
                "--link-mode",
                "copy",
            ]
        )

        assert args.write_plan is True
        assert args.link_mode == "copy"

    def test_cli_large_scale_3dgs_pilot_args(self) -> None:
        """large-scale-3dgs-pilot accepts route pilot args."""
        args = build_parser().parse_args(
            [
                "large-scale-3dgs-pilot",
                "--data",
                "outputs/sparse",
                "--output",
                "outputs/pilot",
                "--pilot-chunks",
                "4",
                "--route-start-image",
                "12",
                "--target-images-per-chunk",
                "24",
                "--link-mode",
                "copy",
                "--format",
                "shell",
            ]
        )

        assert args.output == "outputs/pilot"
        assert args.pilot_chunks == 4
        assert args.route_start_image == 12
        assert args.target_images_per_chunk == 24
        assert args.link_mode == "copy"
        assert args.format == "shell"

    def test_cli_large_scale_3dgs_run_help(self) -> None:
        """Running large-scale-3dgs-run --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["large-scale-3dgs-run", "--help"])
        assert exc_info.value.code == 0

    def test_cli_large_scale_3dgs_catalog_help(self) -> None:
        """Running large-scale-3dgs-catalog --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["large-scale-3dgs-catalog", "--help"])
        assert exc_info.value.code == 0

    def test_cli_large_scale_3dgs_route_help(self) -> None:
        """Running large-scale-3dgs-route --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["large-scale-3dgs-route", "--help"])
        assert exc_info.value.code == 0

    def test_cli_splat_tile_catalog_help(self) -> None:
        """Running splat-tile-catalog --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["splat-tile-catalog", "--help"])
        assert exc_info.value.code == 0

    def test_cli_large_scale_3dgs_catalog_route_args(self) -> None:
        """large-scale-3dgs-catalog accepts robot route playback runbook args."""
        args = build_parser().parse_args(
            [
                "large-scale-3dgs-catalog",
                "--plan",
                "large_scale_3dgs_plan.json",
                "--route",
                "public/robot-routes/demo-route.json",
                "--route-playback",
                "--route-playback-ms",
                "800",
                "--route-playback-loop",
            ]
        )

        assert args.route == "public/robot-routes/demo-route.json"
        assert args.route_playback is True
        assert args.route_playback_ms == 800
        assert args.route_playback_loop is True

    def test_cli_large_scale_3dgs_route_args(self) -> None:
        """large-scale-3dgs-route accepts route generation args."""
        args = build_parser().parse_args(
            [
                "large-scale-3dgs-route",
                "--catalog",
                "public/manifests/demo-catalog.json",
                "--output",
                "public/robot-routes/demo-route.json",
                "--label",
                "Demo Route",
                "--order",
                "snake",
                "--default-y",
                "1.25",
                "--include-missing-splats",
            ]
        )

        assert args.catalog == "public/manifests/demo-catalog.json"
        assert args.output == "public/robot-routes/demo-route.json"
        assert args.label == "Demo Route"
        assert args.order == "snake"
        assert args.default_y == 1.25
        assert args.include_missing_splats is True

    def test_cmd_train_gsplat_fails_fast_without_colmap(self, tmp_path: Path) -> None:
        """gsplat training should reject data dirs with no COLMAP sparse model."""
        from gs_sim2real import cli

        data = tmp_path / "empty"
        data.mkdir()
        args = build_parser().parse_args(
            [
                "train",
                "--data",
                str(data),
                "--output",
                str(tmp_path / "out"),
                "--method",
                "gsplat",
                "--iterations",
                "1",
            ]
        )
        with pytest.raises(FileNotFoundError, match="No COLMAP sparse"):
            cli.cmd_train(args)

    def test_cmd_train_gsplat_skip_data_check(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--skip-data-check should bypass preflight (train_gsplat may still fail later)."""
        from gs_sim2real import cli
        from gs_sim2real.train import gsplat_trainer as gsplat_module

        data = tmp_path / "empty"
        data.mkdir()
        out = tmp_path / "tout"
        out.mkdir()

        def fake_train_gsplat(data_dir, output_dir, config=None, num_iterations=30000):
            del data_dir, output_dir, config, num_iterations
            ply = out / "point_cloud.ply"
            ply.write_bytes(b"ply")
            return ply

        monkeypatch.setattr(gsplat_module, "train_gsplat", fake_train_gsplat)
        args = build_parser().parse_args(
            [
                "train",
                "--data",
                str(data),
                "--output",
                str(out),
                "--method",
                "gsplat",
                "--iterations",
                "1",
                "--skip-data-check",
            ]
        )
        cli.cmd_train(args)

    def test_cli_view_help(self) -> None:
        """Running view --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["view", "--help"])
        assert exc_info.value.code == 0

    def test_cli_run_help(self) -> None:
        """Running run --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--help"])
        assert exc_info.value.code == 0

    def test_cli_benchmark_help(self) -> None:
        """Running benchmark --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["benchmark", "--help"])
        assert exc_info.value.code == 0

    def test_cli_robotics_node_help(self) -> None:
        """Running robotics-node --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["robotics-node", "--help"])
        assert exc_info.value.code == 0

    def test_cli_sim2real_server_help(self) -> None:
        """Running sim2real-server --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["sim2real-server", "--help"])
        assert exc_info.value.code == 0

    def test_cli_sim2real_query_help(self) -> None:
        """Running sim2real-query --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["sim2real-query", "--help"])
        assert exc_info.value.code == 0

    def test_cli_sim2real_benchmark_images_help(self) -> None:
        """Running sim2real-benchmark-images --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["sim2real-benchmark-images", "--help"])
        assert exc_info.value.code == 0

    def test_cli_route_policy_benchmark_help(self) -> None:
        """Running route-policy-benchmark --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["route-policy-benchmark", "--help"])
        assert exc_info.value.code == 0

    def test_cli_route_policy_benchmark_history_help(self) -> None:
        """Running route-policy-benchmark-history --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["route-policy-benchmark-history", "--help"])
        assert exc_info.value.code == 0

    def test_cli_route_policy_scenario_set_help(self) -> None:
        """Running route-policy-scenario-set --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["route-policy-scenario-set", "--help"])
        assert exc_info.value.code == 0

    def test_cli_route_policy_scenario_matrix_help(self) -> None:
        """Running route-policy-scenario-matrix --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["route-policy-scenario-matrix", "--help"])
        assert exc_info.value.code == 0

    def test_cli_route_policy_scenario_shards_help(self) -> None:
        """Running route-policy-scenario-shards --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["route-policy-scenario-shards", "--help"])
        assert exc_info.value.code == 0

    def test_cli_route_policy_scenario_shard_merge_help(self) -> None:
        """Running route-policy-scenario-shard-merge --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["route-policy-scenario-shard-merge", "--help"])
        assert exc_info.value.code == 0

    def test_cli_route_policy_scenario_ci_manifest_help(self) -> None:
        """Running route-policy-scenario-ci-manifest --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["route-policy-scenario-ci-manifest", "--help"])
        assert exc_info.value.code == 0

    def test_cli_route_policy_scenario_ci_workflow_help(self) -> None:
        """Running route-policy-scenario-ci-workflow --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["route-policy-scenario-ci-workflow", "--help"])
        assert exc_info.value.code == 0

    def test_cli_route_policy_scenario_ci_workflow_validate_help(self) -> None:
        """Running route-policy-scenario-ci-workflow-validate --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["route-policy-scenario-ci-workflow-validate", "--help"])
        assert exc_info.value.code == 0

    def test_cli_route_policy_scenario_ci_workflow_activate_help(self) -> None:
        """Running route-policy-scenario-ci-workflow-activate --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["route-policy-scenario-ci-workflow-activate", "--help"])
        assert exc_info.value.code == 0

    def test_cli_route_policy_scenario_ci_review_help(self) -> None:
        """Running route-policy-scenario-ci-review --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["route-policy-scenario-ci-review", "--help"])
        assert exc_info.value.code == 0

    def test_cli_route_policy_scenario_ci_workflow_promote_help(self) -> None:
        """Running route-policy-scenario-ci-workflow-promote --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["route-policy-scenario-ci-workflow-promote", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_localization_alignment_help(self) -> None:
        """Running experiment-localization-alignment --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-localization-alignment", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_render_backend_selection_help(self) -> None:
        """Running experiment-render-backend-selection --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-render-backend-selection", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_outdoor_training_features_help(self) -> None:
        """Running experiment-outdoor-training-features --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-outdoor-training-features", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_localization_import_help(self) -> None:
        """Running experiment-localization-import --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-localization-import", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_transport_selection_help(self) -> None:
        """Running experiment-query-transport-selection --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-transport-selection", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_request_import_help(self) -> None:
        """Running experiment-query-request-import --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-request-import", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_live_localization_stream_import_help(self) -> None:
        """Running experiment-live-localization-stream-import --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-live-localization-stream-import", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_route_capture_import_help(self) -> None:
        """Running experiment-route-capture-import --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-route-capture-import", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_sim2real_websocket_protocol_help(self) -> None:
        """Running experiment-sim2real-websocket-protocol --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-sim2real-websocket-protocol", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_localization_review_bundle_import_help(self) -> None:
        """Running experiment-localization-review-bundle-import --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-localization-review-bundle-import", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_response_build_help(self) -> None:
        """Running experiment-query-response-build --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-response-build", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_timeout_policy_help(self) -> None:
        """Running experiment-query-timeout-policy --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-timeout-policy", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_queue_policy_help(self) -> None:
        """Running experiment-query-queue-policy --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-queue-policy", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_cancellation_policy_help(self) -> None:
        """Running experiment-query-cancellation-policy --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-cancellation-policy", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_coalescing_policy_help(self) -> None:
        """Running experiment-query-coalescing-policy --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-coalescing-policy", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_error_mapping_help(self) -> None:
        """Running experiment-query-error-mapping --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-error-mapping", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_source_identity_help(self) -> None:
        """Running experiment-query-source-identity --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-source-identity", "--help"])
        assert exc_info.value.code == 0

    def test_cli_robotics_node_enable_image_relay_flag(self) -> None:
        """robotics-node parser accepts the image relay toggle."""
        args = build_parser().parse_args(["robotics-node", "--enable-image-relay"])
        assert args.enable_image_relay is True

    def test_cli_sim2real_server_static_pose_flags(self) -> None:
        """sim2real-server parser accepts static render settings."""
        args = build_parser().parse_args(
            [
                "sim2real-server",
                "--ply",
                "scene.ply",
                "--pose-source",
                "static",
                "--static-position",
                "1.0",
                "2.0",
                "3.0",
                "--static-orientation",
                "0.0",
                "0.0",
                "0.0",
                "1.0",
                "--renderer",
                "simple",
                "--run-once",
            ]
        )
        assert args.ply == "scene.ply"
        assert args.pose_source == "static"
        assert args.static_position == [1.0, 2.0, 3.0]
        assert args.static_orientation == [0.0, 0.0, 0.0, 1.0]
        assert args.renderer == "simple"
        assert args.run_once is True

    def test_cli_sim2real_server_query_flags(self) -> None:
        """sim2real-server parser accepts query transport settings."""
        args = build_parser().parse_args(
            [
                "sim2real-server",
                "--ply",
                "scene.ply",
                "--pose-source",
                "query",
                "--query-transport",
                "zmq",
                "--query-endpoint",
                "tcp://127.0.0.1:6001",
            ]
        )
        assert args.pose_source == "query"
        assert args.query_transport == "zmq"
        assert args.query_endpoint == "tcp://127.0.0.1:6001"

    def test_cli_sim2real_server_auto_query_transport_flag(self) -> None:
        """sim2real-server parser accepts auto query transport selection."""
        args = build_parser().parse_args(
            [
                "sim2real-server",
                "--ply",
                "scene.ply",
                "--pose-source",
                "query",
                "--query-transport",
                "auto",
            ]
        )
        assert args.pose_source == "query"
        assert args.query_transport == "auto"

    def test_cli_export_splat_flags(self) -> None:
        """export parser accepts the antimatter15 .splat format flags."""
        args = build_parser().parse_args(
            [
                "export",
                "--model",
                "scene.ply",
                "--format",
                "splat",
                "--output",
                "scene.splat",
                "--max-points",
                "400000",
                "--splat-normalize-extent",
                "17.0",
                "--splat-min-opacity",
                "0.02",
                "--splat-max-scale",
                "2.0",
                "--splat-max-scale-percentile",
                "98",
            ]
        )
        assert args.format == "splat"
        assert args.splat_normalize_extent == 17.0
        assert args.splat_min_opacity == 0.02
        assert args.splat_max_scale == 2.0
        assert args.splat_max_scale_percentile == 98.0

    def test_cli_photos_to_splat_defaults(self) -> None:
        """photos-to-splat parser accepts the minimal form and has sane defaults."""
        args = build_parser().parse_args(["photos-to-splat", "--images", "photos/"])
        assert args.command == "photos-to-splat"
        assert args.preprocess == "dust3r"
        assert args.quality == "draft"
        assert args.num_frames == 20
        assert args.scene_graph == "complete"
        assert args.iterations == 3000
        assert args.splat_max_points == 400000
        assert args.splat_normalize_extent == 17.0
        assert args.splat_min_opacity == 0.02
        assert args.splat_max_scale == 2.0
        assert args.splat_max_scale_percentile is None

    def test_cli_photos_to_splat_clean_quality_resolves_heavier_defaults(self) -> None:
        """clean preset should spend more compute and filter browser-blurring gaussians."""
        args = build_parser().parse_args(["photos-to-splat", "--images", "photos/", "--quality", "clean"])
        resolved = _resolve_photos_to_splat_quality(args)
        assert resolved.num_frames == 64
        assert resolved.align_iters == 800
        assert resolved.iterations == 30000
        assert resolved.mast3r_subsample == 4
        assert resolved.splat_max_points == 600000
        assert resolved.splat_min_opacity == 0.035
        assert resolved.splat_max_scale == 1.0
        assert resolved.splat_max_scale_percentile == 98.0

    def test_cli_photos_to_splat_quality_preserves_explicit_overrides(self) -> None:
        """Quality presets should not clobber explicit user settings."""
        args = build_parser().parse_args(
            [
                "photos-to-splat",
                "--images",
                "photos/",
                "--quality",
                "hero",
                "--iterations",
                "7000",
                "--splat-max-scale-percentile",
                "99.5",
            ]
        )
        resolved = _resolve_photos_to_splat_quality(args)
        assert resolved.num_frames == 0
        assert resolved.iterations == 7000
        assert resolved.splat_max_scale_percentile == 99.5

    def test_cli_splat_filter_flags(self) -> None:
        """splat-filter parser accepts cleanup thresholds for existing browser splats."""
        args = build_parser().parse_args(
            [
                "splat-filter",
                "--input",
                "scene.splat",
                "--output",
                "scene.clean.splat",
                "--min-opacity",
                "0.08",
                "--max-scale",
                "1.0",
                "--max-scale-percentile",
                "98",
                "--max-points",
                "250000",
            ]
        )
        assert args.command == "splat-filter"
        assert args.input == "scene.splat"
        assert args.output == "scene.clean.splat"
        assert args.min_opacity == 0.08
        assert args.max_scale == 1.0
        assert args.max_scale_percentile == 98.0
        assert args.max_points == 250000

    def test_cli_splat_inspect_flags(self) -> None:
        """splat-inspect parser accepts reporting flags for existing browser splats."""
        args = build_parser().parse_args(
            [
                "splat-inspect",
                "--input",
                "scene.splat",
                "--low-opacity-threshold",
                "0.05",
                "--json",
            ]
        )
        assert args.command == "splat-inspect"
        assert args.input == "scene.splat"
        assert args.low_opacity_threshold == 0.05
        assert args.json is True

    def test_cli_photos_to_splat_mast3r(self) -> None:
        """photos-to-splat accepts mast3r as a pose-estimation backend."""
        args = build_parser().parse_args(
            [
                "photos-to-splat",
                "--images",
                "photos/",
                "--preprocess",
                "mast3r",
                "--mast3r-root",
                "/tmp/mast3r",
                "--mast3r-checkpoint",
                "/tmp/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                "--mast3r-subsample",
                "4",
            ]
        )
        assert args.preprocess == "mast3r"
        assert args.mast3r_root == "/tmp/mast3r"
        assert args.mast3r_checkpoint.endswith(".pth")
        assert args.mast3r_subsample == 4

    def test_cli_photos_to_splat_overrides(self) -> None:
        """photos-to-splat accepts explicit DUSt3R / training overrides."""
        args = build_parser().parse_args(
            [
                "photos-to-splat",
                "--images",
                "my_photos/",
                "--output",
                "outputs/my_splat",
                "--preprocess",
                "simple",
                "--num-frames",
                "12",
                "--scene-graph",
                "swin-5",
                "--align-iters",
                "500",
                "--iterations",
                "5000",
                "--splat-max-points",
                "200000",
                "--splat-normalize-extent",
                "30.0",
                "--splat-max-scale-percentile",
                "97.0",
                "--skip-data-check",
            ]
        )
        assert args.images == "my_photos/"
        assert args.output == "outputs/my_splat"
        assert args.preprocess == "simple"
        assert args.num_frames == 12
        assert args.scene_graph == "swin-5"
        assert args.align_iters == 500
        assert args.iterations == 5000
        assert args.splat_max_points == 200000
        assert args.splat_normalize_extent == 30.0
        assert args.splat_max_scale_percentile == 97.0
        assert args.skip_data_check is True

    def test_cli_export_scene_bundle_flags(self) -> None:
        """export parser accepts GitHub Pages scene bundle settings."""
        args = build_parser().parse_args(
            [
                "export",
                "--model",
                "scene.ply",
                "--format",
                "scene-bundle",
                "--output",
                "docs/assets/demo-room",
                "--bundle-asset-format",
                "binary",
                "--scene-id",
                "demo-room",
                "--label",
                "Demo Room",
                "--description",
                "GitHub Pages demo scene",
            ]
        )
        assert args.format == "scene-bundle"
        assert args.bundle_asset_format == "binary"
        assert args.scene_id == "demo-room"
        assert args.label == "Demo Room"
        assert args.description == "GitHub Pages demo scene"

    def test_cli_preprocess_waymo_optional_extraction_flags(self) -> None:
        """preprocess parser accepts Waymo depth and mask extraction toggles."""
        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                "data/waymo",
                "--method",
                "waymo",
                "--camera",
                "FRONT_LEFT",
                "--extract-lidar-depth",
                "--extract-dynamic-masks",
            ]
        )
        assert args.method == "waymo"
        assert args.camera == "FRONT_LEFT"
        assert args.extract_lidar_depth is True
        assert args.extract_dynamic_masks is True

    def test_cli_preprocess_accepts_colmap_path(self) -> None:
        """preprocess parser accepts a custom COLMAP executable path."""
        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                "data/images",
                "--method",
                "colmap",
                "--colmap-path",
                "/opt/colmap/bin/colmap",
            ]
        )
        assert args.method == "colmap"
        assert args.colmap_path == "/opt/colmap/bin/colmap"

    def test_cli_preprocess_mcd_optional_extraction_flags(self) -> None:
        """preprocess parser accepts MCD topic and extraction settings."""
        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                "data/mcd",
                "--method",
                "mcd",
                "--image-topic",
                "/d455t/color/image_raw",
                "--lidar-topic",
                "/os_cloud_node/points",
                "--imu-topic",
                "/vn200/imu",
                "--extract-lidar",
                "--extract-imu",
            ]
        )
        assert args.method == "mcd"
        assert args.image_topic == "/d455t/color/image_raw"
        assert args.lidar_topic == "/os_cloud_node/points"
        assert args.imu_topic == "/vn200/imu"
        assert args.extract_lidar is True
        assert args.extract_imu is True

    def test_cli_preprocess_mcd_lidar_seed_flags(self) -> None:
        """preprocess parser accepts MCD LiDAR seed flags."""
        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                "data/mcd",
                "--method",
                "mcd",
                "--mcd-lidar-frame",
                "os_sensor",
                "--mcd-skip-lidar-seed",
            ]
        )
        assert args.mcd_lidar_frame == "os_sensor"
        assert args.mcd_skip_lidar_seed is True

    def test_cli_preprocess_mcd_lidar_seed_flags_defaults(self) -> None:
        """preprocess parser defaults for MCD LiDAR seed flags are empty/false."""
        args = build_parser().parse_args(["preprocess", "--images", "data/mcd", "--method", "mcd"])
        assert args.mcd_lidar_frame == ""
        assert args.mcd_skip_lidar_seed is False

    def test_cli_mcd_static_calibration_flag_preprocess_run_demo(self) -> None:
        """preprocess / run / demo parsers accept MCD GNSS calibration flags."""
        p = build_parser()
        a = p.parse_args(
            [
                "preprocess",
                "--images",
                "x",
                "--output",
                "o",
                "--method",
                "mcd",
                "--mcd-static-calibration",
                "c.yaml",
                "--mcd-flatten-gnss-altitude",
                "--mcd-start-offset-sec",
                "35",
            ]
        )
        assert a.mcd_static_calibration == "c.yaml"
        assert a.mcd_flatten_gnss_altitude is True
        assert a.mcd_start_offset_sec == 35.0
        b = p.parse_args(
            [
                "run",
                "--images",
                "x",
                "--output",
                "o",
                "--mcd-static-calibration",
                "c.yaml",
                "--mcd-flatten-gnss-altitude",
                "--mcd-start-offset-sec",
                "35",
            ]
        )
        assert b.mcd_static_calibration == "c.yaml"
        assert b.mcd_flatten_gnss_altitude is True
        assert b.mcd_start_offset_sec == 35.0
        c = p.parse_args(
            [
                "demo",
                "--images",
                "x",
                "--output",
                "o",
                "--mcd-static-calibration",
                "c.yaml",
                "--mcd-flatten-gnss-altitude",
                "--mcd-start-offset-sec",
                "35",
            ]
        )
        assert c.mcd_static_calibration == "c.yaml"
        assert c.mcd_flatten_gnss_altitude is True
        assert c.mcd_start_offset_sec == 35.0

    def test_cli_preprocess_mcd_list_topics_flag(self) -> None:
        """preprocess parser accepts MCD topic listing mode."""
        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                "data/mcd",
                "--method",
                "mcd",
                "--list-topics",
            ]
        )
        assert args.method == "mcd"
        assert args.list_topics is True

    def test_cli_preprocess_lidar_slam_accepts_nmea_format(self) -> None:
        """preprocess parser accepts NMEA trajectory input for lidar-slam."""
        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                "data/images",
                "--method",
                "lidar-slam",
                "--trajectory",
                "data/gnss.nmea",
                "--trajectory-format",
                "nmea",
            ]
        )
        assert args.method == "lidar-slam"
        assert args.trajectory == "data/gnss.nmea"
        assert args.trajectory_format == "nmea"

    def test_cli_preprocess_external_slam_flags(self) -> None:
        """preprocess parser accepts external visual SLAM artifact settings."""
        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                "data/images",
                "--method",
                "external-slam",
                "--external-slam-system",
                "loger",
                "--external-slam-output",
                "outputs/loger",
                "--trajectory",
                "trajectory.txt",
                "--pointcloud",
                "points.ply",
                "--pinhole-calib",
                "camera.json",
                "--external-slam-dry-run",
                "--external-slam-manifest-format",
                "json",
                "--external-slam-fail-on-dry-run-gate",
                "--external-slam-min-aligned-frames",
                "3",
                "--external-slam-allow-dropped-images",
                "--external-slam-require-pointcloud",
                "--external-slam-min-point-count",
                "10",
            ]
        )
        assert args.method == "external-slam"
        assert args.external_slam_system == "loger"
        assert args.external_slam_output == "outputs/loger"
        assert args.trajectory == "trajectory.txt"
        assert args.pointcloud == "points.ply"
        assert args.pinhole_calib == "camera.json"
        assert args.external_slam_dry_run is True
        assert args.external_slam_manifest_format == "json"
        assert args.external_slam_fail_on_dry_run_gate is True
        assert args.external_slam_min_aligned_frames == 3
        assert args.external_slam_allow_dropped_images is True
        assert args.external_slam_require_pointcloud is True
        assert args.external_slam_min_point_count == 10

    def test_cli_run_lidar_slam_flags(self) -> None:
        """run parser accepts lidar-slam trajectory settings."""
        args = build_parser().parse_args(
            [
                "run",
                "--images",
                "data/images",
                "--preprocess-method",
                "lidar-slam",
                "--trajectory",
                "data/poses.nmea",
                "--trajectory-format",
                "nmea",
                "--pointcloud",
                "data/cloud.ply",
            ]
        )
        assert args.preprocess_method == "lidar-slam"
        assert args.trajectory == "data/poses.nmea"
        assert args.trajectory_format == "nmea"
        assert args.pointcloud == "data/cloud.ply"

    def test_cli_run_external_slam_flags(self) -> None:
        """run parser accepts external SLAM trajectory settings."""
        args = build_parser().parse_args(
            [
                "run",
                "--images",
                "data/images",
                "--preprocess-method",
                "external-slam",
                "--external-slam-system",
                "vggt-slam",
                "--external-slam-output",
                "outputs/vggt",
                "--trajectory",
                "poses.txt",
                "--pinhole-calib",
                "camera.json",
            ]
        )
        assert args.preprocess_method == "external-slam"
        assert args.external_slam_system == "vggt-slam"
        assert args.external_slam_output == "outputs/vggt"
        assert args.trajectory == "poses.txt"
        assert args.pinhole_calib == "camera.json"

    def test_cli_run_waymo_flags(self) -> None:
        """run parser accepts Waymo extraction settings."""
        args = build_parser().parse_args(
            [
                "run",
                "--images",
                "data/waymo",
                "--preprocess-method",
                "waymo",
                "--camera",
                "FRONT_LEFT",
                "--extract-lidar-depth",
                "--extract-dynamic-masks",
            ]
        )
        assert args.preprocess_method == "waymo"
        assert args.camera == "FRONT_LEFT"
        assert args.extract_lidar_depth is True
        assert args.extract_dynamic_masks is True

    def test_cli_run_accepts_colmap_path(self) -> None:
        """run parser accepts a custom COLMAP executable path."""
        args = build_parser().parse_args(
            [
                "run",
                "--images",
                "data/images",
                "--colmap-path",
                "/opt/colmap/bin/colmap",
            ]
        )
        assert args.colmap_path == "/opt/colmap/bin/colmap"

    def test_cli_run_accepts_no_gpu(self) -> None:
        """run parser accepts disabling GPU for COLMAP preprocessing."""
        args = build_parser().parse_args(
            [
                "run",
                "--images",
                "data/images",
                "--no-gpu",
            ]
        )
        assert args.no_gpu is True

    def test_cli_run_accepts_matching(self) -> None:
        """run parser accepts COLMAP matching strategy overrides."""
        args = build_parser().parse_args(
            [
                "run",
                "--images",
                "data/images",
                "--matching",
                "sequential",
            ]
        )
        assert args.matching == "sequential"

    def test_cli_run_accepts_config(self) -> None:
        """run parser accepts a training config override path."""
        args = build_parser().parse_args(
            [
                "run",
                "--images",
                "data/images",
                "--config",
                "configs/training_outdoor.yaml",
            ]
        )
        assert args.config == "configs/training_outdoor.yaml"

    def test_cli_run_mcd_flags(self) -> None:
        """run parser accepts MCD extraction settings."""
        args = build_parser().parse_args(
            [
                "run",
                "--images",
                "data/mcd",
                "--preprocess-method",
                "mcd",
                "--image-topic",
                "/d455t/color/image_raw",
                "--extract-lidar",
                "--extract-imu",
            ]
        )
        assert args.preprocess_method == "mcd"
        assert args.image_topic == "/d455t/color/image_raw"
        assert args.extract_lidar is True
        assert args.extract_imu is True

    def test_cli_demo_lidar_slam_flags(self) -> None:
        """demo parser accepts lidar-slam trajectory settings."""
        args = build_parser().parse_args(
            [
                "demo",
                "--images",
                "data/images",
                "--preprocess-method",
                "lidar-slam",
                "--trajectory",
                "data/poses.txt",
                "--trajectory-format",
                "tum",
            ]
        )
        assert args.preprocess_method == "lidar-slam"
        assert args.trajectory == "data/poses.txt"
        assert args.trajectory_format == "tum"

    def test_cli_demo_waymo_flags(self) -> None:
        """demo parser accepts Waymo extraction settings."""
        args = build_parser().parse_args(
            [
                "demo",
                "--images",
                "data/waymo",
                "--preprocess-method",
                "waymo",
                "--camera",
                "SIDE_LEFT",
                "--extract-lidar-depth",
            ]
        )
        assert args.preprocess_method == "waymo"
        assert args.camera == "SIDE_LEFT"
        assert args.extract_lidar_depth is True

    def test_cli_demo_accepts_colmap_path(self) -> None:
        """demo parser accepts a custom COLMAP executable path."""
        args = build_parser().parse_args(
            [
                "demo",
                "--images",
                "data/images",
                "--colmap-path",
                "/opt/colmap/bin/colmap",
            ]
        )
        assert args.colmap_path == "/opt/colmap/bin/colmap"

    def test_cli_demo_accepts_no_gpu(self) -> None:
        """demo parser accepts disabling GPU for COLMAP preprocessing."""
        args = build_parser().parse_args(
            [
                "demo",
                "--images",
                "data/images",
                "--no-gpu",
            ]
        )
        assert args.no_gpu is True

    def test_cli_demo_accepts_matching(self) -> None:
        """demo parser accepts COLMAP matching strategy overrides."""
        args = build_parser().parse_args(
            [
                "demo",
                "--images",
                "data/images",
                "--matching",
                "sequential",
            ]
        )
        assert args.matching == "sequential"

    def test_cli_demo_accepts_config(self) -> None:
        """demo parser accepts a training config override path."""
        args = build_parser().parse_args(
            [
                "demo",
                "--images",
                "data/images",
                "--config",
                "configs/training_outdoor.yaml",
            ]
        )
        assert args.config == "configs/training_outdoor.yaml"

    def test_cli_demo_mcd_flags(self) -> None:
        """demo parser accepts MCD extraction settings."""
        args = build_parser().parse_args(
            [
                "demo",
                "--images",
                "data/mcd",
                "--preprocess-method",
                "mcd",
                "--image-topic",
                "/d455b/color/image_raw",
                "--extract-lidar",
            ]
        )
        assert args.preprocess_method == "mcd"
        assert args.image_topic == "/d455b/color/image_raw"
        assert args.extract_lidar is True

    def test_cmd_preprocess_waymo_runs_optional_extractions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Waymo preprocess should call optional depth and mask extractors when requested."""
        from gs_sim2real import cli
        from gs_sim2real.datasets import waymo as waymo_module

        calls: list[tuple[str, str, str, int, int]] = []

        class FakeWaymoLoader:
            def __init__(self, data_dir: str):
                calls.append(("init", data_dir, "", 0, 0))

            def extract_frames(self, output_dir: str, camera: str, max_frames: int, every_n: int) -> str:
                calls.append(("frames", output_dir, camera, max_frames, every_n))
                return str(tmp_path / "out" / "images")

            def extract_lidar_depth(self, output_dir: str, camera: str, max_frames: int, every_n: int) -> str:
                calls.append(("depth", output_dir, camera, max_frames, every_n))
                return str(tmp_path / "out" / "depth")

            def extract_dynamic_masks(self, output_dir: str, camera: str, max_frames: int, every_n: int) -> str:
                calls.append(("masks", output_dir, camera, max_frames, every_n))
                return str(tmp_path / "out" / "masks")

        monkeypatch.setattr(waymo_module, "WaymoLoader", FakeWaymoLoader)

        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                str(tmp_path / "waymo"),
                "--output",
                str(tmp_path / "out"),
                "--method",
                "waymo",
                "--camera",
                "FRONT_RIGHT",
                "--max-frames",
                "12",
                "--every-n",
                "3",
                "--extract-lidar-depth",
                "--extract-dynamic-masks",
            ]
        )

        cli.cmd_preprocess(args)

        assert calls == [
            ("init", str(tmp_path / "waymo"), "", 0, 0),
            ("frames", str(tmp_path / "out"), "FRONT_RIGHT", 12, 3),
            ("depth", str(tmp_path / "out"), "FRONT_RIGHT", 12, 3),
            ("masks", str(tmp_path / "out"), "FRONT_RIGHT", 12, 3),
        ]

        out = capsys.readouterr().out
        assert "Waymo frames loaded from:" in out
        assert "Waymo LiDAR depth extracted to:" in out
        assert "Waymo dynamic masks extracted to:" in out

    def test_cmd_preprocess_mcd_runs_optional_extractions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """MCD preprocess should call optional LiDAR and IMU extractors when requested."""
        from gs_sim2real import cli
        from gs_sim2real.datasets import mcd as mcd_module

        calls: list[tuple[str, str, str, int, int | None]] = []

        class FakeMCDLoader:
            def __init__(self, data_dir: str):
                calls.append(("init", data_dir, "", 0, None))

            def extract_frames(
                self, output_dir: str, image_topic: str | None, max_frames: int, every_n: int, **kwargs
            ) -> str:
                del kwargs
                calls.append(("frames", output_dir, image_topic or "", max_frames, every_n))
                return str(tmp_path / "out" / "images")

            def extract_lidar(
                self, output_dir: str, lidar_topic: str | None, max_frames: int, every_n: int, **kwargs
            ) -> str:
                del kwargs
                calls.append(("lidar", output_dir, lidar_topic or "", max_frames, every_n))
                return str(tmp_path / "out" / "lidar")

            def extract_imu(self, output_dir: str, imu_topic: str | None) -> str:
                calls.append(("imu", output_dir, imu_topic or "", 0, None))
                return str(tmp_path / "out" / "imu.csv")

        monkeypatch.setattr(mcd_module, "MCDLoader", FakeMCDLoader)

        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                str(tmp_path / "mcd"),
                "--output",
                str(tmp_path / "out"),
                "--method",
                "mcd",
                "--image-topic",
                "/d455t/color/image_raw",
                "--lidar-topic",
                "/os_cloud_node/points",
                "--imu-topic",
                "/vn200/imu",
                "--max-frames",
                "8",
                "--every-n",
                "2",
                "--extract-lidar",
                "--extract-imu",
            ]
        )

        cli.cmd_preprocess(args)

        assert calls == [
            ("init", str(tmp_path / "mcd"), "", 0, None),
            ("frames", str(tmp_path / "out"), "/d455t/color/image_raw", 8, 2),
            ("lidar", str(tmp_path / "out"), "/os_cloud_node/points", 8, 2),
            ("imu", str(tmp_path / "out"), "/vn200/imu", 0, None),
        ]

        out = capsys.readouterr().out
        assert "MCD frames available at:" in out
        assert "MCD LiDAR extracted to:" in out
        assert "MCD IMU extracted to:" in out

    def test_cmd_preprocess_colmap_passes_custom_colmap_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """Plain COLMAP preprocess should forward --colmap-path to run_colmap."""
        from gs_sim2real import cli
        from gs_sim2real.preprocess import colmap as colmap_module

        calls: list[tuple[str, str, str, bool, str, bool]] = []

        def fake_run_colmap(
            image_dir,
            output_dir,
            matching="exhaustive",
            gpu_index=0,
            use_gpu=True,
            colmap_path="colmap",
            single_camera_per_folder=False,
        ):
            del gpu_index
            calls.append((str(image_dir), str(output_dir), matching, use_gpu, colmap_path, single_camera_per_folder))
            return tmp_path / "out" / "sparse" / "0"

        monkeypatch.setattr(colmap_module, "run_colmap", fake_run_colmap)

        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                str(tmp_path / "images"),
                "--output",
                str(tmp_path / "out"),
                "--method",
                "colmap",
                "--matching",
                "sequential",
                "--no-gpu",
                "--colmap-path",
                "/opt/colmap/bin/colmap",
            ]
        )

        cli.cmd_preprocess(args)

        assert calls == [
            (
                str(tmp_path / "images"),
                str(tmp_path / "out"),
                "sequential",
                False,
                "/opt/colmap/bin/colmap",
                False,
            )
        ]

    def test_cmd_preprocess_external_slam_delegates_import(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """external-slam preprocess should call the isolated artifact importer."""
        from gs_sim2real import cli
        from gs_sim2real.preprocess import external_slam as external_slam_module

        calls: list[dict[str, object]] = []

        def fake_import_external_slam(**kwargs):
            calls.append(kwargs)
            return tmp_path / "out" / "sparse" / "0"

        monkeypatch.setattr(external_slam_module, "import_external_slam", fake_import_external_slam)

        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                str(tmp_path / "images"),
                "--output",
                str(tmp_path / "out"),
                "--method",
                "external-slam",
                "--external-slam-system",
                "loger",
                "--external-slam-output",
                str(tmp_path / "loger"),
                "--trajectory",
                "trajectory.txt",
                "--pointcloud",
                "points.ply",
                "--pinhole-calib",
                "camera.json",
            ]
        )

        cli.cmd_preprocess(args)

        assert calls == [
            {
                "image_dir": tmp_path / "images",
                "output_dir": tmp_path / "out",
                "system": "loger",
                "artifact_dir": str(tmp_path / "loger"),
                "trajectory_path": "trajectory.txt",
                "trajectory_format": "tum",
                "pointcloud_path": "points.ply",
                "pinhole_calib_path": "camera.json",
                "nmea_time_offset_sec": 0.0,
            }
        ]
        assert "External SLAM import complete:" in capsys.readouterr().out

    def test_cmd_preprocess_external_slam_dry_run_prints_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """external-slam dry-run should resolve artifacts without importing them."""
        from gs_sim2real import cli
        from gs_sim2real.preprocess import external_slam as external_slam_module

        def forbidden_import_external_slam(**kwargs):
            raise AssertionError(f"dry-run should not import artifacts: {kwargs}")

        monkeypatch.setattr(external_slam_module, "import_external_slam", forbidden_import_external_slam)

        image_dir = tmp_path / "images"
        image_dir.mkdir()
        for idx in range(3):
            (image_dir / f"frame_{idx:06d}.jpg").write_bytes(b"jpg")
        artifact_dir = tmp_path / "mast3r_out"
        artifact_dir.mkdir()
        (artifact_dir / "poses.txt").write_text("0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n")
        (artifact_dir / "reconstruction.ply").write_text("ply\n")

        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                str(image_dir),
                "--output",
                str(tmp_path / "out"),
                "--method",
                "external-slam",
                "--external-slam-system",
                "mast3r-slam",
                "--external-slam-output",
                str(artifact_dir),
                "--external-slam-dry-run",
            ]
        )

        cli.cmd_preprocess(args)

        out = capsys.readouterr().out
        assert "External SLAM artifacts: MASt3R-SLAM (mast3r-slam)" in out
        assert "materialization=direct" in out
        assert "warning (2 aligned, 1 dropped images, 0 unused poses)" in out
        assert "External SLAM manifest gate: fail" in out
        assert "dropped_images" in out
        assert "External SLAM dry run complete." in out

        json_args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                str(image_dir),
                "--output",
                str(tmp_path / "out"),
                "--method",
                "external-slam",
                "--external-slam-system",
                "mast3r-slam",
                "--external-slam-output",
                str(artifact_dir),
                "--external-slam-dry-run",
                "--external-slam-manifest-format",
                "json",
            ]
        )

        cli.cmd_preprocess(json_args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["type"] == "external-slam-artifact-manifest"
        assert payload["system"] == "mast3r-slam"
        assert payload["images"]["imageCount"] == 3
        assert payload["alignment"]["status"] == "warning"
        assert payload["alignment"]["droppedImageCount"] == 1
        assert payload["gate"]["passed"] is False

        fail_args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                str(image_dir),
                "--output",
                str(tmp_path / "out"),
                "--method",
                "external-slam",
                "--external-slam-system",
                "mast3r-slam",
                "--external-slam-output",
                str(artifact_dir),
                "--external-slam-dry-run",
                "--external-slam-fail-on-dry-run-gate",
            ]
        )

        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_preprocess(fail_args)

        assert exc_info.value.code == 2

    def test_cmd_preprocess_external_slam_dry_run_json_reports_resolution_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """external-slam dry-run should emit machine-readable errors when artifacts are missing."""
        from gs_sim2real import cli
        from gs_sim2real.preprocess import external_slam as external_slam_module

        def forbidden_import_external_slam(**kwargs):
            raise AssertionError(f"dry-run should not import artifacts: {kwargs}")

        monkeypatch.setattr(external_slam_module, "import_external_slam", forbidden_import_external_slam)

        image_dir = tmp_path / "images"
        image_dir.mkdir()
        (image_dir / "frame_000000.jpg").write_bytes(b"jpg")
        artifact_dir = tmp_path / "pi3_out"
        artifact_dir.mkdir()

        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                str(image_dir),
                "--output",
                str(tmp_path / "out"),
                "--method",
                "external-slam",
                "--external-slam-system",
                "pi3",
                "--external-slam-output",
                str(artifact_dir),
                "--external-slam-dry-run",
                "--external-slam-manifest-format",
                "json",
                "--external-slam-fail-on-dry-run-gate",
            ]
        )

        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_preprocess(args)

        payload = json.loads(capsys.readouterr().out)
        assert exc_info.value.code == 2
        assert payload["type"] == "external-slam-artifact-manifest"
        assert payload["ready"] is False
        assert payload["system"] == "pi3"
        assert payload["error"]["type"] == "FileNotFoundError"
        assert payload["trajectory"] is None
        assert payload["images"]["imageCount"] == 1
        assert payload["resolution"]["trajectory"]["selectedPath"] is None
        assert "camera_poses.npz" in payload["resolution"]["trajectory"]["candidatePatterns"]
        assert payload["resolution"]["trajectory"]["reason"] == "no_candidate_match"
        assert len(payload["resolution"]["trajectory"]["trace"]) == len(
            payload["resolution"]["trajectory"]["candidatePatterns"]
        )

    def test_cmd_preprocess_mcd_list_topics_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """MCD topic listing mode should print topics and skip extraction."""
        from gs_sim2real import cli
        from gs_sim2real.datasets import mcd as mcd_module

        calls: list[str] = []

        class FakeMCDLoader:
            def __init__(self, data_dir: str):
                calls.append(f"init:{data_dir}")

            def list_topics(self):
                calls.append("list_topics")
                return [
                    {
                        "topic": "/d455t/color/image_raw",
                        "msgtype": "sensor_msgs/msg/Image",
                        "msgcount": 12,
                        "role": "image",
                        "is_preferred_default": True,
                    },
                    {
                        "topic": "/vn200/imu",
                        "msgtype": "sensor_msgs/msg/Imu",
                        "msgcount": 400,
                        "role": "imu",
                        "is_preferred_default": True,
                    },
                ]

            def extract_frames(self, *args, **kwargs):
                raise AssertionError("extract_frames should not run in list-topics mode")

        monkeypatch.setattr(mcd_module, "MCDLoader", FakeMCDLoader)

        args = build_parser().parse_args(
            [
                "preprocess",
                "--images",
                str(tmp_path / "mcd"),
                "--method",
                "mcd",
                "--list-topics",
            ]
        )

        cli.cmd_preprocess(args)

        assert calls == [f"init:{tmp_path / 'mcd'}", "list_topics"]
        out = capsys.readouterr().out
        assert "MCD rosbag topics:" in out
        assert "[image] /d455t/color/image_raw" in out
        assert "default" in out

    def test_cmd_run_mcd_extracts_frames_then_runs_colmap_and_training(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """run should support MCD bags by extracting frames before COLMAP and training."""
        from gs_sim2real import cli
        from gs_sim2real.datasets import mcd as mcd_module
        from gs_sim2real.preprocess import colmap as colmap_module
        from gs_sim2real.train import gsplat_trainer as gsplat_module

        calls: list[tuple[str, str, str]] = []

        class FakeMCDLoader:
            def __init__(self, data_dir: str):
                calls.append(("init", data_dir, ""))

            def extract_frames(
                self, output_dir: str, image_topic: str | None, max_frames: int, every_n: int, **kwargs
            ) -> str:
                del kwargs
                assert max_frames == 6
                assert every_n == 2
                calls.append(("frames", output_dir, image_topic or ""))
                return str(tmp_path / "out" / "colmap" / "images")

            def extract_lidar(
                self, output_dir: str, lidar_topic: str | None, max_frames: int, every_n: int, **kwargs
            ) -> str:
                del kwargs
                calls.append(("lidar", output_dir, lidar_topic or ""))
                return str(tmp_path / "out" / "colmap" / "lidar")

            def extract_imu(self, output_dir: str, imu_topic: str | None) -> str:
                calls.append(("imu", output_dir, imu_topic or ""))
                return str(tmp_path / "out" / "colmap" / "imu.csv")

        def fake_run_colmap(
            image_dir,
            output_dir,
            matching="exhaustive",
            gpu_index=0,
            use_gpu=True,
            colmap_path="colmap",
            single_camera_per_folder=False,
        ):
            del gpu_index
            calls.append(
                (
                    "colmap",
                    str(image_dir),
                    str(output_dir),
                    colmap_path,
                    str(use_gpu),
                    matching,
                    str(single_camera_per_folder),
                )
            )
            return tmp_path / "out" / "colmap" / "sparse" / "0"

        def fake_train_gsplat(data_dir, output_dir, config=None, num_iterations=30000):
            calls.append(("train", str(data_dir), str(output_dir)))
            assert num_iterations == 123
            assert config == {"appearance_embedding_dim": 32, "depth_loss_weight": 0.25}
            return tmp_path / "out" / "train" / "point_cloud.ply"

        monkeypatch.setattr(mcd_module, "MCDLoader", FakeMCDLoader)
        monkeypatch.setattr(colmap_module, "run_colmap", fake_run_colmap)
        monkeypatch.setattr(gsplat_module, "train_gsplat", fake_train_gsplat)
        config_path = tmp_path / "training.yaml"
        config_path.write_text("appearance_embedding_dim: 32\ndepth_loss_weight: 0.25\n")

        args = build_parser().parse_args(
            [
                "run",
                "--images",
                str(tmp_path / "mcd"),
                "--output",
                str(tmp_path / "out"),
                "--method",
                "gsplat",
                "--iterations",
                "123",
                "--config",
                str(config_path),
                "--preprocess-method",
                "mcd",
                "--colmap-path",
                "/opt/colmap/bin/colmap",
                "--matching",
                "sequential",
                "--no-gpu",
                "--image-topic",
                "/d455t/color/image_raw",
                "--lidar-topic",
                "/os_cloud_node/points",
                "--imu-topic",
                "/vn200/imu",
                "--extract-lidar",
                "--extract-imu",
                "--max-frames",
                "6",
                "--every-n",
                "2",
                "--no-viewer",
                "--skip-data-check",
            ]
        )

        cli.cmd_run(args)

        assert calls == [
            ("init", str(tmp_path / "mcd"), ""),
            ("frames", str(tmp_path / "out" / "colmap"), "/d455t/color/image_raw"),
            ("lidar", str(tmp_path / "out" / "colmap"), "/os_cloud_node/points"),
            ("imu", str(tmp_path / "out" / "colmap"), "/vn200/imu"),
            (
                "colmap",
                str(tmp_path / "out" / "colmap" / "images"),
                str(tmp_path / "out" / "colmap"),
                "/opt/colmap/bin/colmap",
                "False",
                "sequential",
                "False",
            ),
            ("train", str(tmp_path / "out" / "colmap"), str(tmp_path / "out" / "train")),
        ]

        out = capsys.readouterr().out
        assert "Step 1: Preprocessing (mcd)" in out
        assert "MCD frames available at:" in out
        assert "Step 2: Training" in out
        assert "Pipeline complete!" in out

    def test_cmd_run_mcd_multi_topic_uses_per_folder_colmap(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """Comma-separated MCD image topics should enable per-folder COLMAP intrinsics."""
        from gs_sim2real import cli
        from gs_sim2real.datasets import mcd as mcd_module
        from gs_sim2real.preprocess import colmap as colmap_module
        from gs_sim2real.train import gsplat_trainer as gsplat_module

        calls: list[tuple[str, object]] = []

        class FakeMCDLoader:
            def __init__(self, data_dir: str):
                calls.append(("init", data_dir))

            def extract_frames(self, output_dir, image_topic, max_frames, every_n, **kwargs):
                del max_frames, every_n, kwargs
                calls.append(("frames", image_topic))
                return str(tmp_path / "out" / "colmap" / "images")

        def fake_run_colmap(
            image_dir,
            output_dir,
            matching="exhaustive",
            gpu_index=0,
            use_gpu=True,
            colmap_path="colmap",
            single_camera_per_folder=False,
        ):
            del image_dir, output_dir, matching, gpu_index, use_gpu, colmap_path
            calls.append(("per_folder", single_camera_per_folder))
            return tmp_path / "out" / "colmap" / "sparse" / "0"

        monkeypatch.setattr(mcd_module, "MCDLoader", FakeMCDLoader)
        monkeypatch.setattr(colmap_module, "run_colmap", fake_run_colmap)
        monkeypatch.setattr(
            gsplat_module,
            "train_gsplat",
            lambda data_dir, output_dir, config=None, num_iterations=30000: (
                tmp_path / "out" / "train" / "point_cloud.ply"
            ),
        )

        args = build_parser().parse_args(
            [
                "run",
                "--images",
                str(tmp_path / "mcd"),
                "--output",
                str(tmp_path / "out"),
                "--preprocess-method",
                "mcd",
                "--image-topic",
                "/cam0,/cam1,/cam2",
                "--iterations",
                "1",
                "--no-viewer",
                "--skip-data-check",
            ]
        )

        cli.cmd_run(args)

        assert calls == [
            ("init", str(tmp_path / "mcd")),
            ("frames", ["/cam0", "/cam1", "/cam2"]),
            ("per_folder", True),
        ]

    def test_cmd_run_waymo_uses_loader_outputs_and_optional_extractions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """run should support Waymo tfrecords through the Waymo loader path."""
        from gs_sim2real import cli
        from gs_sim2real.datasets import waymo as waymo_module
        from gs_sim2real.train import gsplat_trainer as gsplat_module

        calls: list[tuple[str, str, str, int, int]] = []

        class FakeWaymoLoader:
            def __init__(self, data_dir: str):
                calls.append(("init", data_dir, "", 0, 0))

            def extract_frames(self, output_dir: str, camera: str, max_frames: int, every_n: int) -> str:
                calls.append(("frames", output_dir, camera, max_frames, every_n))
                params_path = Path(output_dir) / "camera_params.json"
                params_path.parent.mkdir(parents=True, exist_ok=True)
                params_path.write_text("{}")
                return str(Path(output_dir) / "images")

            def to_colmap_format(self, camera_params_path: str, output_dir: str) -> str:
                calls.append(("colmap", camera_params_path, output_dir, 0, 0))
                return str(Path(output_dir) / "sparse" / "0")

            def extract_lidar_depth(self, output_dir: str, camera: str, max_frames: int, every_n: int) -> str:
                calls.append(("depth", output_dir, camera, max_frames, every_n))
                return str(Path(output_dir) / "depth")

            def extract_dynamic_masks(self, output_dir: str, camera: str, max_frames: int, every_n: int) -> str:
                calls.append(("masks", output_dir, camera, max_frames, every_n))
                return str(Path(output_dir) / "masks")

        def fake_train_gsplat(data_dir, output_dir, config=None, num_iterations=30000):
            assert num_iterations == 55
            assert config is None
            assert str(data_dir) == str(tmp_path / "out" / "colmap")
            assert str(output_dir) == str(tmp_path / "out" / "train")
            return tmp_path / "out" / "train" / "point_cloud.ply"

        monkeypatch.setattr(waymo_module, "WaymoLoader", FakeWaymoLoader)
        monkeypatch.setattr(gsplat_module, "train_gsplat", fake_train_gsplat)

        args = build_parser().parse_args(
            [
                "run",
                "--images",
                str(tmp_path / "waymo"),
                "--output",
                str(tmp_path / "out"),
                "--method",
                "gsplat",
                "--iterations",
                "55",
                "--preprocess-method",
                "waymo",
                "--camera",
                "FRONT_RIGHT",
                "--max-frames",
                "9",
                "--every-n",
                "3",
                "--extract-lidar-depth",
                "--extract-dynamic-masks",
                "--no-viewer",
                "--skip-data-check",
            ]
        )

        cli.cmd_run(args)

        assert calls == [
            ("init", str(tmp_path / "waymo"), "", 0, 0),
            ("frames", str(tmp_path / "out" / "colmap"), "FRONT_RIGHT", 9, 3),
            (
                "colmap",
                str(tmp_path / "out" / "colmap" / "camera_params.json"),
                str(tmp_path / "out" / "colmap"),
                0,
                0,
            ),
            ("depth", str(tmp_path / "out" / "colmap"), "FRONT_RIGHT", 9, 3),
            ("masks", str(tmp_path / "out" / "colmap"), "FRONT_RIGHT", 9, 3),
        ]

        out = capsys.readouterr().out
        assert "Step 1: Preprocessing (waymo)" in out
        assert "Waymo frames extracted to:" in out
        assert "Waymo LiDAR depth extracted to:" in out
        assert "Waymo dynamic masks extracted to:" in out
        assert "Pipeline complete!" in out

    def test_cli_sim2real_server_websocket_query_flags(self) -> None:
        """sim2real-server parser accepts websocket query settings."""
        args = build_parser().parse_args(
            [
                "sim2real-server",
                "--ply",
                "scene.ply",
                "--query-transport",
                "ws",
                "--query-endpoint",
                "ws://127.0.0.1:8781/sim2real",
            ]
        )
        assert args.query_transport == "ws"
        assert args.query_endpoint == "ws://127.0.0.1:8781/sim2real"

    def test_cli_sim2real_query_flags(self) -> None:
        """sim2real-query parser accepts pose and output settings."""
        args = build_parser().parse_args(
            [
                "sim2real-query",
                "--endpoint",
                "tcp://127.0.0.1:6002",
                "--position",
                "1.0",
                "2.0",
                "3.0",
                "--yaw-degrees",
                "90.0",
                "--width",
                "320",
                "--height",
                "240",
                "--jpeg-out",
                "frame.jpg",
                "--depth-out",
                "depth.npy",
            ]
        )
        assert args.endpoint == "tcp://127.0.0.1:6002"
        assert args.position == [1.0, 2.0, 3.0]
        assert args.yaw_degrees == 90.0
        assert args.width == 320
        assert args.height == 240
        assert args.jpeg_out == "frame.jpg"
        assert args.depth_out == "depth.npy"

    def test_cli_sim2real_benchmark_images_flags(self) -> None:
        """sim2real-benchmark-images parser accepts run and metric settings."""
        args = build_parser().parse_args(
            [
                "sim2real-benchmark-images",
                "--endpoint",
                "ws://127.0.0.1:8781/sim2real",
                "--run",
                "run.json",
                "--alignment",
                "timestamp",
                "--metrics",
                "psnr",
                "ssim",
                "--lpips-net",
                "vgg",
                "--device",
                "auto",
                "--timeout-ms",
                "2500",
                "--max-frames",
                "12",
                "--output",
                "report.json",
            ]
        )
        assert args.endpoint == "ws://127.0.0.1:8781/sim2real"
        assert args.run == "run.json"
        assert args.alignment == "timestamp"
        assert args.metrics == ["psnr", "ssim"]
        assert args.lpips_net == "vgg"
        assert args.device == "auto"
        assert args.timeout_ms == 2500
        assert args.max_frames == 12
        assert args.output == "report.json"

    def test_cli_route_policy_benchmark_flags(self) -> None:
        """route-policy-benchmark parser accepts replay, goal, and threshold settings."""
        args = build_parser().parse_args(
            [
                "route-policy-benchmark",
                "--transitions-jsonl",
                "transitions.jsonl",
                "--scene-catalog",
                "scenes.json",
                "--scene-id",
                "unit-scene",
                "--benchmark-id",
                "unit-benchmark",
                "--episode-count",
                "4",
                "--seed-start",
                "10",
                "--max-steps",
                "8",
                "--goal",
                "1.0",
                "2.0",
                "3.0",
                "--neighbor-count",
                "3",
                "--action-keys",
                "payload.target.position.0",
                "payload.target.position.1",
                "payload.target.position.2",
                "--include-direct-baseline",
                "--min-success-rate",
                "0.9",
                "--output",
                "report.json",
                "--model-output",
                "model.json",
            ]
        )
        assert args.transitions_jsonl == "transitions.jsonl"
        assert args.scene_catalog == "scenes.json"
        assert args.scene_id == "unit-scene"
        assert args.benchmark_id == "unit-benchmark"
        assert args.episode_count == 4
        assert args.seed_start == 10
        assert args.max_steps == 8
        assert args.goal == [[1.0, 2.0, 3.0]]
        assert args.neighbor_count == 3
        assert args.action_keys == [
            "payload.target.position.0",
            "payload.target.position.1",
            "payload.target.position.2",
        ]
        assert args.include_direct_baseline is True
        assert args.min_success_rate == 0.9
        assert args.output == "report.json"
        assert args.model_output == "model.json"

    def test_cli_route_policy_benchmark_registry_flags(self) -> None:
        """route-policy-benchmark parser accepts registry and goal-suite artifacts."""
        args = build_parser().parse_args(
            [
                "route-policy-benchmark",
                "--policy-registry",
                "policies.json",
                "--goal-suite",
                "goals.json",
                "--scene-catalog",
                "scenes.json",
                "--benchmark-id",
                "registry-benchmark",
                "--episode-count",
                "8",
                "--output",
                "report.json",
            ]
        )
        assert args.policy_registry == "policies.json"
        assert args.goal_suite == "goals.json"
        assert args.scene_catalog == "scenes.json"
        assert args.benchmark_id == "registry-benchmark"
        assert args.episode_count == 8
        assert args.output == "report.json"

    def test_cli_route_policy_benchmark_history_flags(self) -> None:
        """route-policy-benchmark-history parser accepts history and regression gate settings."""
        args = build_parser().parse_args(
            [
                "route-policy-benchmark-history",
                "--report",
                "run-a.json",
                "--report",
                "run-b.json",
                "--baseline-report",
                "baseline.json",
                "--history-id",
                "unit-history",
                "--max-success-rate-drop",
                "0.05",
                "--max-collision-rate-increase",
                "0.01",
                "--max-truncation-rate-increase",
                "0.02",
                "--max-mean-reward-drop",
                "0.25",
                "--allow-missing-policies",
                "--allow-report-failures",
                "--fail-on-regression",
                "--output",
                "history.json",
                "--markdown-output",
                "history.md",
            ]
        )
        assert args.report == ["run-a.json", "run-b.json"]
        assert args.baseline_report == "baseline.json"
        assert args.history_id == "unit-history"
        assert args.max_success_rate_drop == 0.05
        assert args.max_collision_rate_increase == 0.01
        assert args.max_truncation_rate_increase == 0.02
        assert args.max_mean_reward_drop == 0.25
        assert args.allow_missing_policies is True
        assert args.allow_report_failures is True
        assert args.fail_on_regression is True
        assert args.output == "history.json"
        assert args.markdown_output == "history.md"

    def test_cli_route_policy_scenario_set_flags(self) -> None:
        """route-policy-scenario-set parser accepts scenario execution and history gate settings."""
        args = build_parser().parse_args(
            [
                "route-policy-scenario-set",
                "--scenario-set",
                "scenarios.json",
                "--policy-registry",
                "registry.json",
                "--report-dir",
                "reports",
                "--output",
                "scenario-run.json",
                "--markdown-output",
                "scenario-run.md",
                "--history-output",
                "history.json",
                "--history-markdown-output",
                "history.md",
                "--baseline-report",
                "baseline.json",
                "--max-success-rate-drop",
                "0.05",
                "--max-collision-rate-increase",
                "0.01",
                "--max-truncation-rate-increase",
                "0.02",
                "--max-mean-reward-drop",
                "0.25",
                "--allow-missing-policies",
                "--allow-report-failures",
                "--no-markdown",
                "--fail-on-regression",
            ]
        )
        assert args.scenario_set == "scenarios.json"
        assert args.policy_registry == "registry.json"
        assert args.report_dir == "reports"
        assert args.output == "scenario-run.json"
        assert args.markdown_output == "scenario-run.md"
        assert args.history_output == "history.json"
        assert args.history_markdown_output == "history.md"
        assert args.baseline_report == "baseline.json"
        assert args.max_success_rate_drop == 0.05
        assert args.max_collision_rate_increase == 0.01
        assert args.max_truncation_rate_increase == 0.02
        assert args.max_mean_reward_drop == 0.25
        assert args.allow_missing_policies is True
        assert args.allow_report_failures is True
        assert args.no_markdown is True
        assert args.fail_on_regression is True

    def test_cli_route_policy_scenario_matrix_flags(self) -> None:
        """route-policy-scenario-matrix parser accepts matrix expansion settings."""
        args = build_parser().parse_args(
            [
                "route-policy-scenario-matrix",
                "--matrix",
                "matrix.json",
                "--output-dir",
                "generated",
                "--index-output",
                "matrix-expansion.json",
                "--markdown-output",
                "matrix-expansion.md",
            ]
        )
        assert args.matrix == "matrix.json"
        assert args.output_dir == "generated"
        assert args.index_output == "matrix-expansion.json"
        assert args.markdown_output == "matrix-expansion.md"

    def test_cli_route_policy_scenario_shards_flags(self) -> None:
        """route-policy-scenario-shards parser accepts shard plan settings."""
        args = build_parser().parse_args(
            [
                "route-policy-scenario-shards",
                "--expansion",
                "matrix-expansion.json",
                "--max-scenarios-per-shard",
                "3",
                "--shard-plan-id",
                "unit-shards",
                "--output-dir",
                "shards",
                "--index-output",
                "shard-plan.json",
                "--markdown-output",
                "shard-plan.md",
            ]
        )
        assert args.expansion == "matrix-expansion.json"
        assert args.max_scenarios_per_shard == 3
        assert args.shard_plan_id == "unit-shards"
        assert args.output_dir == "shards"
        assert args.index_output == "shard-plan.json"
        assert args.markdown_output == "shard-plan.md"

    def test_cli_route_policy_scenario_shard_merge_flags(self) -> None:
        """route-policy-scenario-shard-merge parser accepts merge gate settings."""
        args = build_parser().parse_args(
            [
                "route-policy-scenario-shard-merge",
                "--run",
                "shard-a.json",
                "--run",
                "shard-b.json",
                "--merge-id",
                "unit-merge",
                "--baseline-report",
                "baseline.json",
                "--history-output",
                "history.json",
                "--history-markdown-output",
                "history.md",
                "--output",
                "merge.json",
                "--markdown-output",
                "merge.md",
                "--max-success-rate-drop",
                "0.05",
                "--max-collision-rate-increase",
                "0.01",
                "--max-truncation-rate-increase",
                "0.02",
                "--max-mean-reward-drop",
                "0.25",
                "--allow-missing-policies",
                "--allow-report-failures",
                "--fail-on-regression",
            ]
        )
        assert args.run == ["shard-a.json", "shard-b.json"]
        assert args.merge_id == "unit-merge"
        assert args.baseline_report == "baseline.json"
        assert args.history_output == "history.json"
        assert args.history_markdown_output == "history.md"
        assert args.output == "merge.json"
        assert args.markdown_output == "merge.md"
        assert args.max_success_rate_drop == 0.05
        assert args.max_collision_rate_increase == 0.01
        assert args.max_truncation_rate_increase == 0.02
        assert args.max_mean_reward_drop == 0.25
        assert args.allow_missing_policies is True
        assert args.allow_report_failures is True
        assert args.fail_on_regression is True

    def test_cli_route_policy_scenario_ci_manifest_flags(self) -> None:
        """route-policy-scenario-ci-manifest parser accepts matrix manifest settings."""
        args = build_parser().parse_args(
            [
                "route-policy-scenario-ci-manifest",
                "--shard-plan",
                "shard-plan.json",
                "--manifest-id",
                "unit-ci",
                "--report-dir",
                "reports",
                "--run-output-dir",
                "runs",
                "--history-output-dir",
                "histories",
                "--merge-id",
                "unit-merge",
                "--merge-output",
                "merge.json",
                "--merge-history-output",
                "history.json",
                "--merge-markdown-output",
                "merge.md",
                "--merge-history-markdown-output",
                "history.md",
                "--cache-key-prefix",
                "unit-cache",
                "--include-markdown",
                "--fail-on-regression",
                "--output",
                "manifest.json",
                "--markdown-output",
                "manifest.md",
            ]
        )
        assert args.shard_plan == "shard-plan.json"
        assert args.manifest_id == "unit-ci"
        assert args.report_dir == "reports"
        assert args.run_output_dir == "runs"
        assert args.history_output_dir == "histories"
        assert args.merge_id == "unit-merge"
        assert args.merge_output == "merge.json"
        assert args.merge_history_output == "history.json"
        assert args.merge_markdown_output == "merge.md"
        assert args.merge_history_markdown_output == "history.md"
        assert args.cache_key_prefix == "unit-cache"
        assert args.include_markdown is True
        assert args.fail_on_regression is True
        assert args.output == "manifest.json"
        assert args.markdown_output == "manifest.md"

    def test_cli_route_policy_scenario_ci_workflow_flags(self) -> None:
        """route-policy-scenario-ci-workflow parser accepts workflow materialization settings."""
        args = build_parser().parse_args(
            [
                "route-policy-scenario-ci-workflow",
                "--manifest",
                "manifest.json",
                "--workflow-id",
                "unit-workflow",
                "--workflow-name",
                "Unit Workflow",
                "--runs-on",
                "ubuntu-24.04",
                "--python-version",
                "3.12",
                "--install-command",
                "pip install -e .",
                "--artifact-root",
                "ci",
                "--artifact-retention-days",
                "5",
                "--no-workflow-dispatch",
                "--push-branch",
                "main",
                "--pull-request-branch",
                "main",
                "--fail-fast",
                "--workflow-output",
                "workflow.yml",
                "--index-output",
                "workflow.json",
                "--markdown-output",
                "workflow.md",
            ]
        )
        assert args.manifest == "manifest.json"
        assert args.workflow_id == "unit-workflow"
        assert args.workflow_name == "Unit Workflow"
        assert args.runs_on == "ubuntu-24.04"
        assert args.python_version == "3.12"
        assert args.install_command == "pip install -e ."
        assert args.artifact_root == "ci"
        assert args.artifact_retention_days == 5
        assert args.no_workflow_dispatch is True
        assert args.push_branch == ["main"]
        assert args.pull_request_branch == ["main"]
        assert args.fail_fast is True
        assert args.workflow_output == "workflow.yml"
        assert args.index_output == "workflow.json"
        assert args.markdown_output == "workflow.md"

    def test_cli_route_policy_scenario_ci_workflow_validate_flags(self) -> None:
        """route-policy-scenario-ci-workflow-validate parser accepts validation settings."""
        args = build_parser().parse_args(
            [
                "route-policy-scenario-ci-workflow-validate",
                "--manifest",
                "manifest.json",
                "--workflow-index",
                "workflow.json",
                "--workflow",
                "workflow.yml",
                "--validation-id",
                "unit-validation",
                "--output",
                "validation.json",
                "--markdown-output",
                "validation.md",
                "--fail-on-validation",
            ]
        )
        assert args.manifest == "manifest.json"
        assert args.workflow_index == "workflow.json"
        assert args.workflow == "workflow.yml"
        assert args.validation_id == "unit-validation"
        assert args.output == "validation.json"
        assert args.markdown_output == "validation.md"
        assert args.fail_on_validation is True

    def test_cli_route_policy_scenario_ci_workflow_activate_flags(self) -> None:
        """route-policy-scenario-ci-workflow-activate parser accepts activation settings."""
        args = build_parser().parse_args(
            [
                "route-policy-scenario-ci-workflow-activate",
                "--workflow-index",
                "workflow.json",
                "--validation-report",
                "validation.json",
                "--workflow",
                "workflow.generated.yml",
                "--active-workflow-output",
                ".github/workflows/unit.yml",
                "--activation-id",
                "unit-activation",
                "--output",
                "activation.json",
                "--markdown-output",
                "activation.md",
                "--overwrite",
                "--fail-on-activation",
            ]
        )
        assert args.workflow_index == "workflow.json"
        assert args.validation_report == "validation.json"
        assert args.workflow == "workflow.generated.yml"
        assert args.active_workflow_output == ".github/workflows/unit.yml"
        assert args.activation_id == "unit-activation"
        assert args.output == "activation.json"
        assert args.markdown_output == "activation.md"
        assert args.overwrite is True
        assert args.fail_on_activation is True

    def test_cli_route_policy_scenario_ci_review_flags(self) -> None:
        """route-policy-scenario-ci-review parser accepts review publishing settings."""
        args = build_parser().parse_args(
            [
                "route-policy-scenario-ci-review",
                "--shard-merge",
                "shard-merge.json",
                "--validation-report",
                "validation.json",
                "--activation-report",
                "activation.json",
                "--review-id",
                "unit-review",
                "--pages-base-url",
                "https://example.test/reviews/unit/",
                "--output",
                "review.json",
                "--markdown-output",
                "review.md",
                "--html-output",
                "index.html",
                "--bundle-dir",
                "pages/review",
                "--fail-on-review",
            ]
        )
        assert args.shard_merge == "shard-merge.json"
        assert args.validation_report == "validation.json"
        assert args.activation_report == "activation.json"
        assert args.review_id == "unit-review"
        assert args.pages_base_url == "https://example.test/reviews/unit/"
        assert args.output == "review.json"
        assert args.markdown_output == "review.md"
        assert args.html_output == "index.html"
        assert args.bundle_dir == "pages/review"
        assert args.fail_on_review is True

    def test_cli_route_policy_scenario_ci_review_event_aligned_flags(self) -> None:
        """route-policy-scenario-ci-review accepts --correlation-event-windows
        and the event-aligned stratification mode value."""
        args = build_parser().parse_args(
            [
                "route-policy-scenario-ci-review",
                "--shard-merge",
                "shard-merge.json",
                "--validation-report",
                "validation.json",
                "--activation-report",
                "activation.json",
                "--correlation-pair-distribution-strata",
                "3",
                "--correlation-pair-distribution-strata-mode",
                "event-aligned",
                "--correlation-event-windows",
                "events.json",
            ]
        )
        assert args.correlation_pair_distribution_strata == 3
        assert args.correlation_pair_distribution_strata_mode == "event-aligned"
        assert args.correlation_event_windows == "events.json"

    def test_cli_route_policy_scenario_ci_review_provenance_flags(self) -> None:
        """route-policy-scenario-ci-review accepts the new provenance flags."""
        args = build_parser().parse_args(
            [
                "route-policy-scenario-ci-review",
                "--shard-merge",
                "shard-merge.json",
                "--validation-report",
                "validation.json",
                "--activation-report",
                "activation.json",
                "--kind",
                "production",
                "--scene-id",
                "outdoor-demo",
                "--scenario-set-id",
                "outdoor-demo-shard-merge",
                "--matrix-hash",
                "sha256:abc123",
                "--policy-version",
                "v0.2.0",
                "--env-contract-version",
                "contract-v3",
                "--correlation-threshold-profile",
                "outdoor-demo-default",
                "--asset-source",
                "bag6",
                "--git-commit",
                "d866f1e",
                "--generated-at",
                "2026-05-15T08:30:00+00:00",
                "--provenance-extra",
                "runTrigger=nightly",
                "--provenance-extra",
                "ci=github",
            ]
        )
        assert args.kind == "production"
        assert args.scene_id == "outdoor-demo"
        assert args.scenario_set_id == "outdoor-demo-shard-merge"
        assert args.matrix_hash == "sha256:abc123"
        assert args.policy_version == "v0.2.0"
        assert args.env_contract_version == "contract-v3"
        assert args.correlation_threshold_profile == "outdoor-demo-default"
        assert args.asset_source == "bag6"
        assert args.git_commit == "d866f1e"
        assert args.generated_at == "2026-05-15T08:30:00+00:00"
        assert args.provenance_extra == ["runTrigger=nightly", "ci=github"]

    def test_cli_route_policy_scenario_ci_review_kind_required_when_provenance_set(self) -> None:
        """Setting --scene-id without --kind fails fast in the run helper."""
        import argparse

        from gs_sim2real.sim.policy_scenario_ci_review import (
            _build_review_provenance_from_cli,
        )

        ns = argparse.Namespace(
            kind=None,
            scene_id="outdoor-demo",
            scenario_set_id=None,
            matrix_hash=None,
            policy_version=None,
            env_contract_version=None,
            correlation_threshold_profile=None,
            asset_source=None,
            git_commit=None,
            generated_at=None,
            provenance_extra=None,
        )
        with pytest.raises(SystemExit) as excinfo:
            _build_review_provenance_from_cli(ns)
        assert excinfo.value.code == 2

    def test_cli_route_policy_scenario_ci_review_no_provenance_flags_keeps_v1_shape(self) -> None:
        """Omitting all provenance flags returns None so the v1 JSON shape is preserved."""
        import argparse

        from gs_sim2real.sim.policy_scenario_ci_review import (
            _build_review_provenance_from_cli,
        )

        ns = argparse.Namespace(
            kind=None,
            scene_id=None,
            scenario_set_id=None,
            matrix_hash=None,
            policy_version=None,
            env_contract_version=None,
            correlation_threshold_profile=None,
            asset_source=None,
            git_commit=None,
            generated_at=None,
            provenance_extra=None,
        )
        assert _build_review_provenance_from_cli(ns) is None

    def test_cli_route_policy_scenario_ci_workflow_promote_flags(self) -> None:
        """route-policy-scenario-ci-workflow-promote parser accepts promotion settings."""
        args = build_parser().parse_args(
            [
                "route-policy-scenario-ci-workflow-promote",
                "--review",
                "review.json",
                "--review-url",
                "https://example.test/reviews/unit/",
                "--promotion-id",
                "unit-promotion",
                "--trigger-mode",
                "push-and-pull-request",
                "--push-branch",
                "main",
                "--pull-request-branch",
                "main",
                "--output",
                "promotion.json",
                "--markdown-output",
                "promotion.md",
                "--fail-on-promotion",
            ]
        )
        assert args.review == "review.json"
        assert args.review_url == "https://example.test/reviews/unit/"
        assert args.promotion_id == "unit-promotion"
        assert args.trigger_mode == "push-and-pull-request"
        assert args.push_branch == ["main"]
        assert args.pull_request_branch == ["main"]
        assert args.output == "promotion.json"
        assert args.markdown_output == "promotion.md"
        assert args.fail_on_promotion is True

    def test_cli_experiment_localization_alignment_flags(self) -> None:
        """experiment-localization-alignment parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "localization-alignment",
                "--repetitions",
                "32",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "alignment-report.json",
            ]
        )
        assert args.repetitions == 32
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "alignment-report.json"

    def test_cli_experiment_render_backend_selection_flags(self) -> None:
        """experiment-render-backend-selection parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "render-backend-selection",
                "--repetitions",
                "16",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "render-backend-report.json",
            ]
        )
        assert args.repetitions == 16
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "render-backend-report.json"

    def test_cli_experiment_outdoor_training_features_flags(self) -> None:
        """experiment-outdoor-training-features parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "outdoor-training-features",
                "--repetitions",
                "12",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "outdoor-training-features-report.json",
            ]
        )
        assert args.repetitions == 12
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "outdoor-training-features-report.json"

    def test_cli_experiment_localization_import_flags(self) -> None:
        """experiment-localization-import parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "localization-import",
                "--repetitions",
                "24",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "localization-import-report.json",
            ]
        )
        assert args.repetitions == 24
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "localization-import-report.json"

    def test_cli_experiment_query_transport_selection_flags(self) -> None:
        """experiment-query-transport-selection parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "query-transport-selection",
                "--repetitions",
                "20",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-transport-report.json",
            ]
        )
        assert args.repetitions == 20
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-transport-report.json"

    def test_cli_experiment_query_request_import_flags(self) -> None:
        """experiment-query-request-import parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "query-request-import",
                "--repetitions",
                "12",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-request-import-report.json",
            ]
        )
        assert args.repetitions == 12
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-request-import-report.json"

    def test_cli_experiment_live_localization_stream_import_flags(self) -> None:
        """experiment-live-localization-stream-import parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "live-localization-stream-import",
                "--repetitions",
                "10",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "live-stream-import-report.json",
            ]
        )
        assert args.repetitions == 10
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "live-stream-import-report.json"

    def test_cli_experiment_route_capture_import_flags(self) -> None:
        """experiment-route-capture-import parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "route-capture-import",
                "--repetitions",
                "18",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "route-capture-import-report.json",
            ]
        )
        assert args.repetitions == 18
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "route-capture-import-report.json"

    def test_cli_experiment_sim2real_websocket_protocol_flags(self) -> None:
        """experiment-sim2real-websocket-protocol parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "sim2real-websocket-protocol",
                "--repetitions",
                "14",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "sim2real-websocket-protocol-report.json",
            ]
        )
        assert args.repetitions == 14
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "sim2real-websocket-protocol-report.json"

    def test_cli_experiment_localization_review_bundle_import_flags(self) -> None:
        """experiment-localization-review-bundle-import parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "localization-review-bundle-import",
                "--repetitions",
                "22",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "localization-review-bundle-import-report.json",
            ]
        )
        assert args.repetitions == 22
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "localization-review-bundle-import-report.json"

    def test_cli_experiment_query_response_build_flags(self) -> None:
        """experiment-query-response-build parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "query-response-build",
                "--repetitions",
                "24",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-response-build-report.json",
            ]
        )
        assert args.repetitions == 24
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-response-build-report.json"

    def test_cli_experiment_query_timeout_policy_flags(self) -> None:
        """experiment-query-timeout-policy parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "query-timeout-policy",
                "--repetitions",
                "26",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-timeout-policy-report.json",
            ]
        )
        assert args.repetitions == 26
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-timeout-policy-report.json"

    def test_cli_experiment_query_queue_policy_flags(self) -> None:
        """experiment-query-queue-policy parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "query-queue-policy",
                "--repetitions",
                "28",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-queue-policy-report.json",
            ]
        )
        assert args.repetitions == 28
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-queue-policy-report.json"

    def test_cli_experiment_query_cancellation_policy_flags(self) -> None:
        """experiment-query-cancellation-policy parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "query-cancellation-policy",
                "--repetitions",
                "30",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-cancellation-policy-report.json",
            ]
        )
        assert args.repetitions == 30
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-cancellation-policy-report.json"

    def test_cli_experiment_query_coalescing_policy_flags(self) -> None:
        """experiment-query-coalescing-policy parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment",
                "query-coalescing-policy",
                "--repetitions",
                "32",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-coalescing-policy-report.json",
            ]
        )
        assert args.repetitions == 32
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-coalescing-policy-report.json"

    def test_cli_no_command(self) -> None:
        """Running main with no arguments exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0
