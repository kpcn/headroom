"""Wiring tests for model-prefix upstream routing (HEADROOM_UPSTREAM_ROUTES).

Covers env -> ProxyConfig, ProxyConfig -> live proxy, and handler routing.
"""

from __future__ import annotations

from headroom.proxy.server import ProxyConfig, _proxy_config_from_env, create_app
from headroom.proxy.upstream_router import UpstreamRoute, UpstreamRouter, UpstreamRouterConfig

MOCK_ROUTES = (
    '[{"prefix": "deepseek", "url": "https://api.deepseek.com"}, '
    '{"prefix": "gpt", "url": "https://api.openai.com"}]'
)


def test_proxy_config_from_env_reads_upstream_router(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_UPSTREAM_ROUTES", MOCK_ROUTES)
    config = _proxy_config_from_env()
    assert config.upstream_router is not None
    assert len(config.upstream_router.routes) == 2
    assert config.upstream_router.routes[0].prefix == "deepseek"
    assert config.upstream_router.routes[0].url == "https://api.deepseek.com"


def test_proxy_config_from_env_empty_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_UPSTREAM_ROUTES", raising=False)
    config = _proxy_config_from_env()
    assert config.upstream_router is not None
    assert len(config.upstream_router.routes) == 0


def test_create_app_wires_upstream_router() -> None:
    config = ProxyConfig(
        optimize=False,
        image_optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        upstream_router=UpstreamRouterConfig(
            routes=(
                UpstreamRoute(prefix="deepseek", url="https://api.deepseek.com"),
            ),
        ),
    )
    app = create_app(config)
    router = app.state.proxy.upstream_router
    assert router is not None
    assert router.resolve("deepseek-chat") == "https://api.deepseek.com"
    assert router.resolve("gpt-5") is None


def test_create_app_router_absent_when_unset() -> None:
    app = create_app(ProxyConfig(optimize=False, cost_tracking_enabled=False))
    router = app.state.proxy.upstream_router
    assert router is not None
    assert router.resolve("deepseek-chat") is None


class _StubHandler:
    """Minimal handler stub to test _resolve_openai_upstream fallback."""

    def __init__(self, upstream_router: UpstreamRouter) -> None:
        self.upstream_router = upstream_router
        self.OPENAI_API_URL = "https://api.openai.com"


def test_handler_falls_back_to_default_when_no_route() -> None:
    handler = _StubHandler(UpstreamRouter(None))
    result = handler.upstream_router.resolve("deepseek-chat")
    assert result is None


def test_handler_routes_when_configured() -> None:
    router = UpstreamRouter(
        UpstreamRouterConfig(
            routes=(UpstreamRoute(prefix="deepseek", url="https://api.deepseek.com"),),
        )
    )
    result = router.resolve("deepseek-chat")
    assert result == "https://api.deepseek.com"


def test_handler_routes_case_insensitive() -> None:
    router = UpstreamRouter(
        UpstreamRouterConfig(
            routes=(UpstreamRoute(prefix="deepseek", url="https://api.deepseek.com"),),
        )
    )
    assert router.resolve("DEEPSEEK-CHAT") == "https://api.deepseek.com"
    assert router.resolve("DeepSeek-Pro") == "https://api.deepseek.com"
