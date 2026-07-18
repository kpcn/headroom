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
