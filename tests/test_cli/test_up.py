"""Tests for headroom up/down CLI commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from headroom.cli.up import (
    PROXY_CONFIG_FILENAME,
    compute_port,
    find_project_root,
    is_proxy_alive,
    load_proxy_config,
    save_proxy_config,
)


def test_find_project_root_finds_headroom_dir(tmp_path: Path) -> None:
    project = tmp_path / "my-project"
    project.mkdir()
    (project / ".headroom").mkdir()
    assert find_project_root(project) == project


def test_find_project_root_finds_git_dir(tmp_path: Path) -> None:
    project = tmp_path / "my-project"
    project.mkdir()
    (project / ".git").mkdir()
    assert find_project_root(project) == project


def test_find_project_root_walks_up(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / ".headroom").mkdir()
    child = parent / "sub" / "deep"
    child.mkdir(parents=True)
    assert find_project_root(child) == parent


def test_find_project_root_falls_back_to_cwd(tmp_path: Path) -> None:
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_from_env_var(tmp_path: Path) -> None:
    with patch.dict(os.environ, {"HEADROOM_PROJECT_ROOT": str(tmp_path)}):
        assert find_project_root(Path("/tmp")) == tmp_path


def test_find_project_root_ignores_headroom_outside_git_boundary(tmp_path: Path) -> None:
    """A .headroom/ outside the git root must not be picked up."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    # .headroom in parent dir (simulates ~/.headroom)
    (tmp_path / ".headroom").mkdir()
    assert find_project_root(project) == project
    assert not (project / ".headroom").exists()


def test_compute_port_is_deterministic(tmp_path: Path) -> None:
    p1 = tmp_path / "project-a"
    p2 = tmp_path / "project-b"
    p1.mkdir()
    p2.mkdir()
    port_a = compute_port(p1)
    port_b = compute_port(p2)
    assert 8700 <= port_a <= 8799
    assert 8700 <= port_b <= 8799
    assert compute_port(p1) == port_a  # deterministic
    assert port_a != port_b  # different projects get different ports


def test_save_and_load_proxy_config(tmp_path: Path) -> None:
    save_proxy_config(tmp_path, {"port": 8765, "pid": 12345, "workspace_dir": str(tmp_path)})
    config_path = tmp_path / ".headroom" / PROXY_CONFIG_FILENAME
    assert config_path.is_file()
    loaded = load_proxy_config(tmp_path)
    assert loaded is not None
    assert loaded["port"] == 8765
    assert loaded["pid"] == 12345


def test_load_proxy_config_missing(tmp_path: Path) -> None:
    assert load_proxy_config(tmp_path) is None


def test_load_proxy_config_invalid_json(tmp_path: Path) -> None:
    headroom_dir = tmp_path / ".headroom"
    headroom_dir.mkdir()
    (headroom_dir / PROXY_CONFIG_FILENAME).write_text("not json")
    assert load_proxy_config(tmp_path) is None


def test_is_proxy_alive_no_pid(tmp_path: Path) -> None:
    assert not is_proxy_alive(999999999, 9999)
