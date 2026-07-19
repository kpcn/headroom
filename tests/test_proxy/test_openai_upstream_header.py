"""Tests for ``OpenAIHandlerMixin._resolve_openai_upstream``.

The dedicated OpenAI handlers (``/v1/chat/completions``,
``/v1/responses``) must honor the ``x-headroom-base-url`` request header
so OpenAI-compatible gateways (LiteLLM, CPA, self-hosted vLLM, Azure
OpenAI) route correctly — consistent with the generic passthrough route
that already honors it (see ``providers/proxy_routes.py``).

These tests pin the resolution contract:
- header present  → its value wins
- header absent   → configured ``OPENAI_API_URL`` fallback
- header empty or whitespace-only → fallback (no blanking)
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.datastructures import Headers  # noqa: E402

from headroom.proxy.handlers.openai import OpenAIHandlerMixin  # noqa: E402
from headroom.proxy.upstream_router import UpstreamRouter, UpstreamRouterConfig  # noqa: E402


class _FakeRequest:
    """Minimal stand-in exposing ``headers`` like a real Starlette request.

    Uses ``starlette.datastructures.Headers`` so header lookup is
    case-insensitive, matching the production ``request.headers`` — a
    plain ``dict`` would let case-folding regressions pass silently.
    """

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = Headers(headers=headers)


def _stub_proxy(fallback_url: str, upstream_router: UpstreamRouter | None = None) -> OpenAIHandlerMixin:
    """A bare mixin instance with ``OPENAI_API_URL`` and optional router."""
    return type(  # type: ignore[return-value]
        "_S",
        (OpenAIHandlerMixin,),
        {
            "OPENAI_API_URL": fallback_url,
            "upstream_router": upstream_router or UpstreamRouter(None),
        },
    )()


def test_header_overrides_configured_url() -> None:
    proxy = _stub_proxy("https://api.openai.test")
    # The transport sends the upstream origin (no /v1 path).
    request = _FakeRequest({"x-headroom-base-url": "https://gateway.example"})

    assert proxy._resolve_openai_upstream(request) == "https://gateway.example"


def test_missing_header_falls_back_to_configured_url() -> None:
    proxy = _stub_proxy("https://api.openai.test")
    request = _FakeRequest({})

    assert proxy._resolve_openai_upstream(request) == "https://api.openai.test"


def test_empty_header_falls_back_to_configured_url() -> None:
    """An explicitly empty or whitespace-only header must not blank the upstream."""
    proxy = _stub_proxy("https://api.openai.test")

    empty = _FakeRequest({"x-headroom-base-url": ""})
    assert proxy._resolve_openai_upstream(empty) == "https://api.openai.test"

    whitespace = _FakeRequest({"x-headroom-base-url": "   "})
    assert proxy._resolve_openai_upstream(whitespace) == "https://api.openai.test"


def test_header_lookup_is_case_insensitive() -> None:
    """Transports may send mixed-case header names; lookup must still resolve."""
    proxy = _stub_proxy("https://api.openai.test")
    # Real transports routinely send Title-Case header names.
    request = _FakeRequest({"X-Headroom-Base-Url": "https://gateway.example"})

    assert proxy._resolve_openai_upstream(request) == "https://gateway.example"


# ---------------------------------------------------------------------------
# Model-prefix routing tests (HEADROOM_UPSTREAM_ROUTES)
# ---------------------------------------------------------------------------


def test_model_route_matches() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    proxy = _stub_proxy("https://api.openai.test", router)
    request = _FakeRequest({})
    assert proxy._resolve_openai_upstream(request, model="deepseek-chat") == "https://api.deepseek.com/v1"


def test_model_route_falls_back_to_default() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    proxy = _stub_proxy("https://api.openai.test", router)
    request = _FakeRequest({})
    assert proxy._resolve_openai_upstream(request, model="gpt-5") == "https://api.openai.test"


def test_header_still_wins_over_model_route() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    proxy = _stub_proxy("https://api.openai.test", router)
    request = _FakeRequest({"x-headroom-base-url": "https://gateway.example"})
    assert proxy._resolve_openai_upstream(request, model="deepseek-chat") == "https://gateway.example"


def test_no_router_no_header_uses_default() -> None:
    proxy = _stub_proxy("https://api.openai.test")
    request = _FakeRequest({})
    assert proxy._resolve_openai_upstream(request, model="deepseek-chat") == "https://api.openai.test"


def test_longest_prefix_wins_in_handler() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}, '
        '{"prefix": "deepseek-reasoner", "url": "https://api.deepseek.com/reasoner"}]'
    ))
    proxy = _stub_proxy("https://api.openai.test", router)
    request = _FakeRequest({})
    assert proxy._resolve_openai_upstream(request, model="deepseek-reasoner") == "https://api.deepseek.com/reasoner"
    assert proxy._resolve_openai_upstream(request, model="deepseek-chat") == "https://api.deepseek.com/v1"


def test_model_routing_is_case_insensitive_in_handler() -> None:
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    proxy = _stub_proxy("https://api.openai.test", router)
    request = _FakeRequest({})
    assert proxy._resolve_openai_upstream(request, model="DEEPSEEK-CHAT") == "https://api.deepseek.com/v1"


def test_model_route_no_model_param_uses_default() -> None:
    """When no model is passed, model routing is skipped (backward compat)."""
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '[{"prefix": "deepseek", "url": "https://api.deepseek.com/v1"}]'
    ))
    proxy = _stub_proxy("https://api.openai.test", router)
    request = _FakeRequest({})
    assert proxy._resolve_openai_upstream(request) == "https://api.openai.test"


def test_header_with_subpath_preserves_path() -> None:
    """A custom upstream served from a sub-path (e.g. /api/v1) must keep the path,
    not be collapsed to the bare origin (#2047)."""
    proxy = _stub_proxy("https://api.openai.test")
    request = _FakeRequest({"x-headroom-base-url": "https://gateway.example/api/v1"})

    assert proxy._resolve_openai_upstream(request) == "https://gateway.example/api/v1"

    # Trailing slash is normalized away, not doubled.
    trailing = _FakeRequest({"x-headroom-base-url": "https://gateway.example/api/v1/"})
    assert proxy._resolve_openai_upstream(trailing) == "https://gateway.example/api/v1"
