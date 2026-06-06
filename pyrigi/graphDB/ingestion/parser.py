"""
pyrigi.graphDB.ingestion.parser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`GraphParser` — decodes graph6 strings into networkx Graph objects.
"""

from __future__ import annotations

import logging

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "networkx is required for graph parsing. "
        "Install it with: pip install networkx"
    ) from exc

log = logging.getLogger(__name__)


class GraphParser:
    """Decodes graph6 strings into networkx Graph objects.

    Parameters
    ----------
    strict:
        If ``True``, raises on malformed strings.  If ``False`` (default),
        logs an error and skips.
    """

    def __init__(self, strict: bool = False) -> None:
        self._strict = strict

    def parse(self, g6: str) -> nx.Graph | None:
        """Return a :class:`networkx.Graph` or ``None`` on failure."""
        try:
            return nx.from_graph6_bytes(g6.encode("ascii"))
        except Exception as exc:
            if self._strict:
                raise
            log.error("Failed to parse graph6 string %r: %s", g6[:30], exc)
            return None
