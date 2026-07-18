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
    router = UpstreamRouter(UpstreamRouterConfig.from_env(
        '{"prefix": "deepseek", "url": "https://example.com"}'
    ))
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
