"""Tests for the Isaac Sim (NuRec USDZ) export wrapper around 3dgrut."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from gs_sim2real.robotics.isaac_export import (
    THREEDGRUT_ENV_VAR,
    TRANSCODE_MODULE,
    build_transcode_command,
    export_usdz,
    find_threedgrut_root,
)


def _fake_threedgrut(tmp_path: Path) -> Path:
    root = tmp_path / "3dgrut"
    (root / "threedgrut" / "export").mkdir(parents=True)
    return root


class TestFindThreedgrutRoot:
    def test_explicit_root_wins(self, tmp_path, monkeypatch):
        monkeypatch.delenv(THREEDGRUT_ENV_VAR, raising=False)
        root = _fake_threedgrut(tmp_path)
        assert find_threedgrut_root(root) == root

    def test_explicit_root_must_be_a_checkout(self, tmp_path, monkeypatch):
        monkeypatch.delenv(THREEDGRUT_ENV_VAR, raising=False)
        with pytest.raises(FileNotFoundError):
            find_threedgrut_root(tmp_path / "nowhere")

    def test_env_var_fallback(self, tmp_path, monkeypatch):
        root = _fake_threedgrut(tmp_path)
        monkeypatch.setenv(THREEDGRUT_ENV_VAR, str(root))
        assert find_threedgrut_root(None) == root

    def test_returns_none_when_unavailable(self, monkeypatch):
        monkeypatch.delenv(THREEDGRUT_ENV_VAR, raising=False)
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        assert find_threedgrut_root(None) is None


class TestBuildTranscodeCommand:
    def test_default_command_shape(self):
        command = build_transcode_command(Path("in.ply"), Path("out.usdz"))
        assert command[0] == sys.executable
        assert command[1:3] == ["-m", TRANSCODE_MODULE]
        assert "in.ply" in command
        assert command[command.index("-o") + 1] == "out.usdz"
        assert command[command.index("--format") + 1] == "nurec"

    def test_rejects_unknown_format(self):
        with pytest.raises(ValueError):
            build_transcode_command(Path("in.ply"), Path("out.usdz"), export_format="splat")


class TestExportUsdz:
    def test_runs_transcode_with_pythonpath(self, tmp_path, monkeypatch):
        root = _fake_threedgrut(tmp_path)
        ply = tmp_path / "scene.ply"
        ply.write_bytes(b"ply\n")
        output = tmp_path / "nested" / "scene.usdz"
        calls = {}

        def fake_run(command, env=None, cwd=None, capture_output=True, text=True):
            calls["command"] = command
            calls["env"] = env
            calls["cwd"] = cwd
            output.write_bytes(b"usdz")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr("gs_sim2real.robotics.isaac_export.subprocess.run", fake_run)
        result = export_usdz(ply, output, threedgrut_root=root)
        assert result == output
        assert calls["cwd"] == str(root)
        assert calls["env"]["PYTHONPATH"].startswith(str(root))
        assert str(ply) in calls["command"]

    def test_missing_ply_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            export_usdz(tmp_path / "missing.ply", tmp_path / "out.usdz", threedgrut_root=_fake_threedgrut(tmp_path))

    def test_missing_threedgrut_explains_setup(self, tmp_path, monkeypatch):
        monkeypatch.delenv(THREEDGRUT_ENV_VAR, raising=False)
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        ply = tmp_path / "scene.ply"
        ply.write_bytes(b"ply\n")
        with pytest.raises(RuntimeError, match="git clone"):
            export_usdz(ply, tmp_path / "out.usdz")

    def test_transcode_failure_surfaces_stderr_tail(self, tmp_path, monkeypatch):
        root = _fake_threedgrut(tmp_path)
        ply = tmp_path / "scene.ply"
        ply.write_bytes(b"ply\n")

        def fake_run(command, env=None, cwd=None, capture_output=True, text=True):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom: no pxr")

        monkeypatch.setattr("gs_sim2real.robotics.isaac_export.subprocess.run", fake_run)
        with pytest.raises(RuntimeError, match="no pxr"):
            export_usdz(ply, tmp_path / "out.usdz", threedgrut_root=root)

    def test_silent_no_output_raises(self, tmp_path, monkeypatch):
        root = _fake_threedgrut(tmp_path)
        ply = tmp_path / "scene.ply"
        ply.write_bytes(b"ply\n")

        def fake_run(command, env=None, cwd=None, capture_output=True, text=True):
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr("gs_sim2real.robotics.isaac_export.subprocess.run", fake_run)
        with pytest.raises(RuntimeError, match="wrote no file"):
            export_usdz(ply, tmp_path / "out.usdz", threedgrut_root=root)


class TestCliWiring:
    def test_export_isaac_requires_exactly_one_input(self):
        from gs_sim2real.cli import build_parser, cmd_export_isaac

        args = build_parser().parse_args(["export-isaac"])
        with pytest.raises(SystemExit):
            cmd_export_isaac(args)

    def test_export_isaac_ply_path(self, tmp_path, monkeypatch, capsys):
        from gs_sim2real import cli

        ply = tmp_path / "scene.ply"
        ply.write_bytes(b"ply\n")
        seen = {}

        def fake_export(ply_path, output_path, *, threedgrut_root=None, export_format="nurec"):
            seen["args"] = (Path(ply_path), Path(output_path), threedgrut_root, export_format)
            Path(output_path).write_bytes(b"usdz")
            return Path(output_path)

        monkeypatch.setattr("gs_sim2real.robotics.isaac_export.export_usdz", fake_export)
        args = cli.build_parser().parse_args(["export-isaac", "--ply", str(ply)])
        cli.cmd_export_isaac(args)
        assert seen["args"][0] == ply
        assert seen["args"][1] == ply.with_suffix(".usdz")
        assert "Isaac Sim" in capsys.readouterr().out
