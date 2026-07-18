# UpstreamRouter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add model-prefix-based upstream routing to Headroom's OpenAI proxy handler so the proxy can forward `/v1/chat/completions` to different upstreams based on the model name prefix, eliminating manual `OPENAI_TARGET_API_URL` env var switching.

**Architecture:** A standalone `UpstreamRouter` component maps model-name prefixes to upstream URLs, read from `HEADROOM_UPSTREAM_ROUTES` env var JSON. Wired into `_resolve_openai_upstream()` with precedence: `x-headroom-base-url` header > model-prefix route > `OPENAI_API_URL` default. Per-project proxy instances via `headroom up` CLI command with auto-discovered `.headroom/` workspace.

**Tech Stack:** Python 3.10+, click (CLI), pytest (tests), fastapi/httpx (optional proxy deps).

## Global Constraints

- Zero behaviour change when no routes configured: existing tests must still pass unchanged.
- Header override (`x-headroom-base-url`) always wins over model routing.
- Invalid/malformed config must fail open to disabled (current default behaviour preserved).
- Model names are case-insensitive; matching uses `model.lower().startswith(prefix.lower())`.
- Longest matching prefix wins; tie on length → first configured wins.
- Python 3.10+ type hints; ruff lint/format, line length 100.
- No new dependencies.

---

### Task 1: UpstreamRouter component

**Files:**
- Create: `headroom/proxy/upstream_router.py`
- Test: `tests/test_proxy/test_upstream_router.py`

**Interfaces:**
- Consumes: nothing (standalone pure component)
- Produces: `UpstreamRoute` dataclass, `UpstreamRouter` class, `UpstreamRouterConfig` dataclass

- [ ] **Step 1: Write the failing test**

```python
"""Tests for upstream_router.UpstreamRouter."""
from __future__ import annotations

import pytest

from headroom.proxy.upstream_router import UpstreamRouter, UpstreamRouterConfig


def test_empty_router_returns_none() -> None:
    router = UpstreamRouter(None)
    assert router.resolve("deepseek-chat") is None


def test_no_routes_configured_returns_none() -> None:
    router = UpstreamRouter(UpstreamRouterConfig(routes=()))
    assert router.resolve("deepseek-chat") is None


def test_matches_prefix() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}, '
        '{"prefix": "gpt", "url": "https://api.openai.com/v1"}]'
    ))
    assert router.resolve("deepseek-chat") == "https://api.deepseek.com/v1"
    assert router.resolve("gpt-5") == "https://api.openai.com/v1"


def test_longest_prefix_wins() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}, '
        '{"prefix": "deepseek-reasoner", "url": "https://api.deepseek.com/reasoner"}]'
    ))
    assert router.resolve("deepseek-reasoner") == "https://api.deepseek.com/reasoner"
    assert router.resolve("deepseek-chat") == "https://api.deepseek.com/v1"


def test_case_insensitive_matching() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    assert router.resolve("DeepSeek-Chat") == "https://api.deepseek.com/v1"
    assert router.resolve("DEEPSEEK-REASONER") == "https://api.deepseek.com/v1"


def test_unknown_model_falls_back_to_none() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    assert router.resolve("claude-3") is None


def test_empty_model_returns_none() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    assert router.resolve("") is None


def test_malformed_json_fails_open() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env("not json"))
    assert router.resolve("deepseek-chat") is None


def test_missing_prefix_field_skips_route() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}, '
        '{"url": "https://api.openai.com/v1"}]'
    ))
    assert router.resolve("deepseek-chat") == "https://api.deepseek.com/v1"
    assert router.resolve("gpt-5") is None


def test_missing_url_field_skips_route() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}, '
        '{"prefix": "gpt"}]'
    ))
    assert router.resolve("deepseek-chat") == "https://api.deepseek.com/v1"
    assert router.resolve("gpt-5") is None


def test_not_a_list_fails_open() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env('{"prefix": "deepseek", "url": "https://example.com"}'))
    assert router.resolve("deepseek-chat") is None


def test_not_even_json_fails_open() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(""))
    assert router.resolve("deepseek-chat") is None
    router2 = UpstreamRouter(UpstreamRouterConfig.from_env(None))
    assert router2.resolve("deepseek-chat") is None


def test_non_dict_entry_skipped() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '["string", {"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    assert router.resolve("deepseek-chat") == "https://api.deepseek.com/v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kochan/workspace/headroom/code/headroom && uv run pytest tests/test_proxy/test_upstream_router.py -v`
Expected: ModuleNotFoundError / ImportError (no module yet)

- [ ] **Step 3: Write minimal implementation**

```python
"""Model-prefix-based upstream routing for the Headroom proxy.

Reads a JSON array of ``{prefix, url}`` rules from
``HEADROOM_UPSTREAM_ROUTES``. Resolves model names to upstream base URLs
by case-insensitive longest-prefix-first matching, so the proxy can
forward ``/v1/chat/completions`` to different upstreams (Deepseek,
OpenAI, Z.AI, …) without manual env var switching.

When no routes are configured or no prefix matches, returns ``None`` and
the caller's default upstream wins — zero behaviour change for existing
setups.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import NamedTuple

logger = logging.getLogger(__name__)


class UpstreamRoute(NamedTuple):
    """One model-prefix → upstream URL mapping."""

    prefix: str
    """Model name prefix to match (case-insensitive)."""

    url: str
    """Upstream base URL (e.g. ``https://api.deepseek.com/v1``)."""


@dataclass(frozen=True)
class UpstreamRouterConfig:
    """Configuration for :class:`UpstreamRouter`. Empty by default."""

    routes: tuple[UpstreamRoute, ...] = ()

    @classmethod
    def from_env(cls, raw: str | None) -> UpstreamRouterConfig:
        """Build config from a JSON string, failing open to empty.

        Expects a JSON array of ``{"prefix": "...", "url": "..."}`` objects.
        Malformed entries are skipped with a warning; a completely invalid
        or missing value yields an empty config so the proxy never crashes.
        """
        if not raw or not raw.strip():
            return cls()

        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError) as exc:
            logger.warning("invalid HEADROOM_UPSTREAM_ROUTES JSON; ignoring: %s", exc)
            return cls()

        if not isinstance(parsed, list):
            logger.warning("HEADROOM_UPSTREAM_ROUTES must be a JSON array; ignoring")
            return cls()

        routes: list[UpstreamRoute] = []
        for i, entry in enumerate(parsed):
            if not isinstance(entry, dict):
                logger.warning("upstream route #%d is not an object; skipping", i)
                continue
            prefix = entry.get("prefix")
            url = entry.get("url")
            if not isinstance(prefix, str) or not prefix:
                logger.warning("upstream route #%d missing string 'prefix'; skipping", i)
                continue
            if not isinstance(url, str) or not url:
                logger.warning("upstream route #%d missing string 'url'; skipping", i)
                continue
            routes.append(UpstreamRoute(prefix=prefix, url=url))
        return cls(routes=tuple(routes))


class UpstreamRouter:
    """Resolves model names to upstream base URLs by prefix matching."""

    def __init__(self, config: UpstreamRouterConfig | None) -> None:
        self._config = config or UpstreamRouterConfig()

    def resolve(self, model: str | None) -> str | None:
        """Return the matching upstream URL, or ``None`` for no match.

        Matching is case-insensitive with longest-prefix-first semantics.
        """
        if not model:
            return None

        model_lower = model.lower()
        best: UpstreamRoute | None = None
        for route in self._config.routes:
            if model_lower.startswith(route.prefix.lower()):
                if best is None or len(route.prefix) > len(best.prefix):
                    best = route
        return best.url if best is not None else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/kochan/workspace/headroom/code/headroom && uv run pytest tests/test_proxy/test_upstream_router.py -v`
Expected: all 14 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/kochan/workspace/headroom/code/headroom && git add headroom/proxy/upstream_router.py tests/test_proxy/test_upstream_router.py && git commit -m "feat: add UpstreamRouter for model-prefix upstream routing"
```

---

### Task 2: Register HEADROOM_UPSTREAM_ROUTES knob in runtime_env

**Files:**
- Modify: `headroom/proxy/runtime_env.py:56-73`

**Interfaces:**
- Consumes: UpstreamRouter and UpstreamRouterConfig (from Task 1)
- Produces: `HEADROOM_UPSTREAM_ROUTES` registered in RUNTIME_ENV_KNOBS

- [ ] **Step 1: Add the knob to RUNTIME_ENV_KNOBS**

After the existing knobs, add:
```python
    Knob(
        "HEADROOM_UPSTREAM_ROUTES", "str", "JSON array of model-prefix → upstream URL rules.",
    ),
```

- [ ] **Step 2: Verify it compiles**

Run: `cd /Users/kochan/workspace/headroom/code/headroom && uv run ruff check headroom/proxy/runtime_env.py`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
cd /Users/kochan/workspace/headroom/code/headroom && git add headroom/proxy/runtime_env.py && git commit -m "feat: register HEADROOM_UPSTREAM_ROUTES in runtime_env knobs"
```

---

### Task 3: Wire UpstreamRouter into ProxyConfig and HeadroomProxy

**Files:**
- Modify: `headroom/proxy/models.py:170` (add upstream_router field)
- Modify: `headroom/proxy/server.py:4799-4802` (pass config), `:724` (init router)
- Modify: `headroom/cli/proxy.py` (add --upstream-routes option, ~line 970)

**Interfaces:**
- Consumes: `UpstreamRouterConfig` (Task 1)
- Produces: `self.upstream_router` on `HeadroomProxy` instance

- [ ] **Step 1: Add upstream_router field to ProxyConfig**

```python
    # Model-prefix-based upstream routing. When configured, maps model name
    # prefixes to upstream base URLs. Falls back to OPENAI_API_URL when no
    # route matches. Disabled by default (empty config) so existing setups
    # are unchanged.
    upstream_router: UpstreamRouterConfig | None = None
```

Add import at top of models.py:
```python
from headroom.proxy.upstream_router import UpstreamRouterConfig
```

- [ ] **Step 2: Build config from env in _proxy_config_from_env**

In `headroom/proxy/server.py`, near line 4799, add:
```python
        upstream_router=UpstreamRouterConfig.from_env(
            os.environ.get("HEADROOM_UPSTREAM_ROUTES"),
        ),
```

Add import at top of server.py:
```python
from headroom.proxy.upstream_router import UpstreamRouter, UpstreamRouterConfig
```

- [ ] **Step 3: Initialize router in HeadroomProxy.__init__**

After line 724 (where `self.model_router` is set), add:
```python
        self.upstream_router = UpstreamRouter(config.upstream_router)
```

- [ ] **Step 4: Verify compilation**

Run: `cd /Users/kochan/workspace/headroom/code/headroom && uv run ruff check headroom/proxy/models.py headroom/proxy/server.py`
Expected: no errors

- [ ] **Step 5: Commit**

```bash
cd /Users/kochan/workspace/headroom/code/headroom && git add headroom/proxy/models.py headroom/proxy/server.py && git commit -m "feat: wire UpstreamRouter into ProxyConfig and HeadroomProxy"
```

---

### Task 4: Wire model-prefix routing into OpenAI handler

**Files:**
- Modify: `headroom/proxy/handlers/openai.py:1327-1336` (method signature + logic)
- Modify: `headroom/proxy/handlers/openai.py:2519` (first call site, pass model)
- Modify: `headroom/proxy/handlers/openai.py:2643` (second call site, pass model)
- Test: `tests/test_proxy/test_openai_upstream_header.py` (add model-based tests)

**Interfaces:**
- Consumes: `self.upstream_router` (Task 3), `model` from parsed body

- [ ] **Step 1: Update _resolve_openai_upstream signature and logic**

Change line 1327-1336:
```python
    def _resolve_openai_upstream(self, request: Request, model: str | None = None) -> str:
        """Return the OpenAI upstream base URL for ``request``.

        Precedence (first match wins)::

            1. ``x-headroom-base-url`` request header (existing behaviour).
            2. Model-prefix route (``HEADROOM_UPSTREAM_ROUTES``) when ``model``
               is provided and a matching prefix is configured.
            3. Configured ``OPENAI_API_URL`` (``OPENAI_TARGET_API_URL``).
        """
        # 1. Per-request header override
        header_url = _resolve_openai_upstream_base(request.headers)
        if header_url:
            return header_url
        # 2. Model prefix routing
        router_url = self.upstream_router.resolve(model) if hasattr(self, "upstream_router") else None
        if router_url:
            return router_url
        # 3. Fall back to configured default
        return self.OPENAI_API_URL
```

- [ ] **Step 2: Update first call site at line 2519**

Change:
```python
        upstream_base_url = self._resolve_openai_upstream(request)
```
To:
```python
        upstream_base_url = self._resolve_openai_upstream(request, model=model)
```

- [ ] **Step 3: Update second call site at line 2643**

This call site doesn't have `model` readily available. Add it:
```python
        upstream_base_url = _resolve_openai_upstream_base(request.headers)
```
→ This call site is in a re-resolution block. The `model` variable is already in scope (defined at line 2515). Change to:
```python
        upstream_base_url = self._resolve_openai_upstream(request, model=model)
```

- [ ] **Step 4: Write tests for model-based routing precedence**

Add to `tests/test_proxy/test_openai_upstream_header.py`:

```python
from headroom.proxy.upstream_router import UpstreamRouter, UpstreamRouterConfig


class _FakeRequest:
    ...


def _stub_proxy(fallback_url: str, upstream_router: UpstreamRouter | None = None) -> OpenAIHandlerMixin:
    """A bare mixin instance with OPENAI_API_URL and optional upstream_router."""
    return type(
        "_S",
        (OpenAIHandlerMixin,),
        {
            "OPENAI_API_URL": fallback_url,
            "upstream_router": upstream_router or UpstreamRouter(None),
        },
    )()


def test_model_route_matches() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    proxy = _stub_proxy("https://api.openai.test", router)
    request = _FakeRequest({})

    result = proxy._resolve_openai_upstream(request, model="deepseek-chat")
    assert result == "https://api.deepseek.com/v1"


def test_model_route_falls_back_to_default() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    proxy = _stub_proxy("https://api.openai.test", router)
    request = _FakeRequest({})

    result = proxy._resolve_openai_upstream(request, model="gpt-5")
    assert result == "https://api.openai.test"


def test_header_still_wins_over_model_route() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    proxy = _stub_proxy("https://api.openai.test", router)
    request = _FakeRequest({"x-headroom-base-url": "https://gateway.example"})

    result = proxy._resolve_openai_upstream(request, model="deepseek-chat")
    assert result == "https://gateway.example"


def test_no_router_no_header_uses_default() -> None:
    proxy = _stub_proxy("https://api.openai.test")
    request = _FakeRequest({})

    result = proxy._resolve_openai_upstream(request, model="deepseek-chat")
    assert result == "https://api.openai.test"


def test_longest_prefix_wins_in_handler() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}, '
        '{"prefix": "deepseek-reasoner", "url": "https://api.deepseek.com/reasoner"}]'
    ))
    proxy = _stub_proxy("https://api.openai.test", router)
    request = _FakeRequest({})

    assert proxy._resolve_openai_upstream(request, model="deepseek-reasoner") == "https://api.deepseek.com/reasoner"
    assert proxy._resolve_openai_upstream(request, model="deepseek-chat") == "https://api.deepseek.com/v1"


def test_model_routing_is_case_insensitive() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    proxy = _stub_proxy("https://api.openai.test", router)
    request = _FakeRequest({})

    assert proxy._resolve_openai_upstream(request, model="DEEPSEEK-CHAT") == "https://api.deepseek.com/v1"
```

- [ ] **Step 5: Run all upstream tests**

Run: `cd /Users/kochan/workspace/headroom/code/headroom && uv run pytest tests/test_proxy/test_openai_upstream_header.py tests/test_proxy/test_upstream_router.py -v`
Expected: all tests PASS (old + new)

- [ ] **Step 6: Commit**

```bash
cd /Users/kochan/workspace/headroom/code/headroom && git add headroom/proxy/handlers/openai.py tests/test_proxy/test_openai_upstream_header.py && git commit -m "feat: wire model-prefix routing into _resolve_openai_upstream"
```

---

### Task 5: CLI `headroom up` / `headroom down` commands

**Files:**
- Create: `headroom/cli/up.py`
- Modify: `headroom/cli/main.py:64-82` (register `up` and `down`)
- Test: `tests/test_cli/test_up.py`

**Interfaces:**
- Consumes: `headroom.proxy.server.create_app` (to verify a port has a running proxy)
- Produces: `.headroom/proxy.json` `{port, pid, workspace_dir}`

- [ ] **Step 1: Write tests for project root discovery**

```python
"""Tests for headroom up/down CLI commands."""
from __future__ import annotations

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from headroom.cli.main import main
from headroom.cli.up import find_project_root, PROXY_CONFIG_FILENAME


def test_find_project_root_finds_headroom_dir(tmp_path: Path) -> None:
    project = tmp_path / "my-project"
    project.mkdir()
    headroom_dir = project / ".headroom"
    headroom_dir.mkdir()
    assert find_project_root(project) == project


def test_find_project_root_finds_git_dir(tmp_path: Path) -> None:
    project = tmp_path / "my-project"
    project.mkdir()
    git_dir = project / ".git"
    git_dir.mkdir()
    assert find_project_root(project) == project


def test_find_project_root_walks_up(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    headroom_dir = parent / ".headroom"
    headroom_dir.mkdir()
    child = parent / "sub" / "deep"
    child.mkdir(parents=True)
    assert find_project_root(child) == parent


def test_find_project_root_falls_back_to_cwd(tmp_path: Path) -> None:
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_from_env_var(tmp_path: Path) -> None:
    with patch.dict(os.environ, {"HEADROOM_PROJECT_ROOT": str(tmp_path)}):
        assert find_project_root(tmp_path) == tmp_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kochan/workspace/headroom/code/headroom && uv run pytest tests/test_cli/test_up.py -v`
Expected: ModuleNotFoundError / ImportError

- [ ] **Step 3: Write find_project_root + proxy.json management**

Create `headroom/cli/up.py`:
```python
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
    import errno
    import socket

    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except (OSError, socket.timeout):
        return False


@main.command()
@click.option(
    "--port", type=int, default=None,
    help="Force a specific port (default: deterministic from project path)",
)
@click.option(
    "--no-daemon", is_flag=True,
    help="Run proxy in foreground (for debugging)",
)
def up(port: int | None, no_daemon: bool = False) -> None:
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

    # Warn about port conflicts
    import socket as _socket
    _sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        _sock.bind(("127.0.0.1", proxy_port))
        _sock.close()
    except OSError:
        click.echo(
            f"Port {proxy_port} is in use by another process. "
            f"Use --port to specify a different port, or stop the other process.",
            err=True,
        )
        sys.exit(1)

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
        os.execve(sys.executable, [sys.executable, "-m", "headroom", "proxy", "--port", str(proxy_port)], env)
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
        save_proxy_config(project_root, {
            "port": proxy_port,
            "pid": proc.pid,
            "workspace_dir": str(headroom_dir),
        })
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
    port = existing.get("port")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            click.echo(f"Stopped Headroom proxy (pid {pid})")
        except ProcessLookupError:
            click.echo("Proxy process not found (already stopped).")
        except OSError as e:
            click.echo(f"Error stopping proxy: {e}", err=True)

    # Clean up config
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
```

- [ ] **Step 4: Register CLI commands in main.py**

Add `up` to the import block:
```python
        up,  # noqa: F401
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/kochan/workspace/headroom/code/headroom && uv run pytest tests/test_cli/test_up.py -v`
Expected: all tests PASS

- [ ] **Step 6: Run existing tests to verify no regressions**

Run: `cd /Users/kochan/workspace/headroom/code/headroom && uv run pytest tests/test_proxy/test_openai_upstream_header.py tests/test_proxy/test_upstream_router.py -v`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
cd /Users/kochan/workspace/headroom/code/headroom && git add headroom/cli/up.py headroom/cli/main.py tests/test_cli/test_up.py && git commit -m "feat: add headroom up/down CLI commands for per-project proxy lifecycle"
```

---

### Task 6: Wire headroom up → routes file loading in settings.json

**Files:**
- Add: `headroom/proxy/settings_store.py` reference (if needed for routes)

The `headroom up` command already handles this by injecting `HEADROOM_UPSTREAM_ROUTES` into the child process env. The `settings_store.apply_to_environ()` call in `create_app` will then see it. No additional wiring needed — env var precedence is sufficient.

No-op task; verify by reading the settings store loading at `headroom/proxy/server.py:2253-2258`:

- [ ] **Step 1: Verify that env injection is sufficient**

The settings_store applies `os.environ.setdefault`, so explicit env (what `up` injects) always wins. Confirmed by reading `headroom/proxy/server.py:2253-2258`. No code change needed.

---

### Task 7: Final verification

- [ ] **Step 1: Run ruff check**

```bash
cd /Users/kochan/workspace/headroom/code/headroom && uv run ruff check headroom/proxy/upstream_router.py headroom/proxy/handlers/openai.py headroom/proxy/runtime_env.py headroom/proxy/models.py headroom/proxy/server.py headroom/cli/up.py
```
Expected: no errors

- [ ] **Step 2: Run ruff format**

```bash
cd /Users/kochan/workspace/headroom/code/headroom && uv run ruff format headroom/proxy/upstream_router.py headroom/proxy/handlers/openai.py headroom/proxy/runtime_env.py headroom/proxy/models.py headroom/proxy/server.py headroom/cli/up.py
```
Expected: no changes (or auto-format)

- [ ] **Step 3: Run full test suite for touched test files**

```bash
cd /Users/kochan/workspace/headroom/code/headroom && uv run pytest tests/test_proxy/test_openai_upstream_header.py tests/test_proxy/test_upstream_router.py tests/test_cli/test_up.py -v
```
Expected: all tests PASS

- [ ] **Step 4: Push branch**

```bash
cd /Users/kochan/workspace/headroom/code/headroom && git push origin feat/upstream-router
```
