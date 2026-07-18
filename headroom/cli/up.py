"""CLI commands for per-project Headroom proxy lifecycle.

``headroom up`` discovers or creates a ``.headroom/`` directory by walking up
from the current working directory (like git discovers ``.git/``), picks a
deterministic port, and starts a detached proxy process with
``HEADROOM_WORKSPACE_DIR`` set to the project's ``.headroom/`` directory.

``headroom down`` stops the running proxy for the current project.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import click

from .main import main

logger = logging.getLogger(__name__)

PROXY_CONFIG_FILENAME = "proxy.json"
HEADROOM_DIR_NAME = ".headroom"
PORT_MIN = 8700
PORT_RANGE = 100  # ports PORT_MIN to PORT_MIN+PORT_RANGE-1


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (or cwd) to find the project root.

    Resolution order:
    1. ``$HEADROOM_PROJECT_ROOT`` env var if set.
    2. Walk up from start looking for ``.headroom/``.
    3. Walk up from start looking for ``.git/``.
    4. ``start`` (cwd) as fallback.
    """
    env_root = os.environ.get("HEADROOM_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    cwd = (start or Path.cwd()).resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / HEADROOM_DIR_NAME).is_dir():
            return parent
    for parent in [cwd] + list(cwd.parents):
        if (parent / ".git").is_dir():
            return parent
    return cwd


def compute_port(project_root: Path) -> int:
    """Deterministic port within ``PORT_MIN``..``PORT_MIN+PORT_RANGE-1``."""
    raw = hash(str(project_root.resolve()))
    offset = raw % PORT_RANGE
    return PORT_MIN + offset


def load_proxy_config(project_root: Path) -> dict | None:
    """Load proxy config from ``.headroom/proxy.json``, or ``None``."""
    config_path = project_root / HEADROOM_DIR_NAME / PROXY_CONFIG_FILENAME
    if not config_path.is_file():
        return None
    try:
        with open(config_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_proxy_config(project_root: Path, config: dict) -> None:
    """Write proxy config to ``.headroom/proxy.json``."""
    headroom_dir = project_root / HEADROOM_DIR_NAME
    headroom_dir.mkdir(parents=True, exist_ok=True)
    config_path = headroom_dir / PROXY_CONFIG_FILENAME
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def is_proxy_alive(pid: int, port: int) -> bool:
    """Check if a process with ``pid`` is running and listening on ``port``."""
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except (OSError, TimeoutError):
        return False


@main.command()
@click.option(
    "--port",
    type=int,
    default=None,
    help="Force a specific port (default: deterministic from project path)",
)
@click.option(
    "--no-daemon",
    is_flag=True,
    help="Run proxy in foreground (for debugging)",
)
def up(port: int | None, no_daemon: bool = False) -> None:  # noqa: PLR0912 — CLI orchestration is inherently branchy
    """Start the Headroom proxy for the current project.

    Discovers the project root, creates ``.headroom/`` if needed, and
    starts a detached proxy with per-project metrics storage.
    """
    project_root = find_project_root()
    headroom_dir = project_root / HEADROOM_DIR_NAME
    headroom_dir.mkdir(parents=True, exist_ok=True)

    proxy_port = port or compute_port(project_root)

    # Check if already running
    existing = load_proxy_config(project_root)
    if existing is not None:
        pid = existing.get("pid")
        old_port = existing.get("port")
        if pid and old_port and is_proxy_alive(pid, old_port):
            click.echo(f"Headroom proxy already running on port {old_port} (pid {pid})")
            return

    # Check port availability
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", proxy_port))
        sock.close()
    except OSError:
        click.echo(
            f"Port {proxy_port} is in use by another process. "
            f"Use --port to specify a different port, or stop the other process.",
            err=True,
        )
        sys.exit(1)

    # Write base URL file for OpenCode `{file:.headroom/base_url}` config variables
    base_url_path = headroom_dir / "base_url"
    base_url_path.write_text(f"http://127.0.0.1:{proxy_port}/v1")

    # Load upstream routes
    routes_json = os.environ.get("HEADROOM_UPSTREAM_ROUTES")
    if not routes_json:
        routes_path = _find_routes_file(project_root)
        if routes_path:
            try:
                routes_json = routes_path.read_text().strip()
            except OSError:
                pass

    # Build env for child process
    env = os.environ.copy()
    env["HEADROOM_WORKSPACE_DIR"] = str(headroom_dir)
    if routes_json:
        env["HEADROOM_UPSTREAM_ROUTES"] = routes_json

    cmd = [sys.executable, "-m", "headroom", "proxy", "--port", str(proxy_port)]

    if no_daemon:
        click.echo(f"Starting Headroom proxy on port {proxy_port} (foreground)...")
        os.execve(
            sys.executable,
            [sys.executable, "-m", "headroom", "proxy", "--port", str(proxy_port)],
            env,
        )
    else:
        click.echo(f"Starting Headroom proxy on port {proxy_port}...")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        save_proxy_config(
            project_root,
            {
                "port": proxy_port,
                "pid": proc.pid,
                "workspace_dir": str(headroom_dir),
            },
        )
        click.echo(f"Headroom proxy started on port {proxy_port} (pid {proc.pid})")


@main.command()
def down() -> None:
    """Stop the Headroom proxy for the current project."""
    project_root = find_project_root()
    existing = load_proxy_config(project_root)
    if existing is None:
        click.echo("No Headroom proxy running for this project.")
        return

    pid = existing.get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            click.echo(f"Stopped Headroom proxy (pid {pid})")
        except ProcessLookupError:
            click.echo("Proxy process not found (already stopped).")
        except OSError as e:
            click.echo(f"Error stopping proxy: {e}", err=True)

    config_path = project_root / HEADROOM_DIR_NAME / PROXY_CONFIG_FILENAME
    if config_path.exists():
        config_path.unlink()


def _find_routes_file(project_root: Path) -> Path | None:
    """Look for upstream routes config file in standard locations."""
    candidates = [
        project_root / HEADROOM_DIR_NAME / "upstream_routes.json",
        Path.home() / HEADROOM_DIR_NAME / "upstream_routes.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None
