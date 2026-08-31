"""MCP server entry point for Mnemo (supporting FastMCP and MCPServer).

Usage (stdio transport — for Claude Desktop / Antigravity / Cursor)::

    python -m mnemo.mcp.server

Or via the CLI::

    mnemo serve
"""

from __future__ import annotations

from typing import Any

from mnemo.engine.audn import AUDNClassifier
from mnemo.engine.retriever import HybridRetriever
from mnemo.engine.tier_manager import TierManager
from mnemo.storage.connection import Database
from mnemo.storage.fts_store import FTSStore
from mnemo.storage.graph_store import GraphStore
from mnemo.storage.vector_store import VectorStore, create_embedder


def _resolve_mcp_server_cls() -> Any:
    try:
        import mcp.server.mcpserver as _m2

        return _m2.MCPServer
    except Exception:
        pass
    try:
        import mcp.server.fastmcp as _m1

        fastmcp_cls = getattr(_m1, "FastMCP", None)
        if fastmcp_cls is not None:
            return fastmcp_cls
    except Exception:
        pass

    class _FallbackMCP:
        def __init__(self, name: str, **kwargs: Any) -> None:
            self.name = name

        def tool(self, *args: Any, **kwargs: Any) -> Any:
            def dec(fn: Any) -> Any:
                return fn

            return dec

        def resource(self, *args: Any, **kwargs: Any) -> Any:
            def dec(fn: Any) -> Any:
                return fn

            return dec

        def prompt(self, *args: Any, **kwargs: Any) -> Any:
            def dec(fn: Any) -> Any:
                return fn

            return dec

        def run(self, *args: Any, **kwargs: Any) -> None:
            pass

    return _FallbackMCP


FastMCP: Any = _resolve_mcp_server_cls()

# ---------------------------------------------------------------------------
# Shared singleton state  (initialised lazily on first use)
# ---------------------------------------------------------------------------

_db: Database | None = None
_vector_store: VectorStore | None = None
_fts_store: FTSStore | None = None
_graph_store: GraphStore | None = None
_audn: AUDNClassifier | None = None
_retriever: HybridRetriever | None = None
_tier_mgr: TierManager | None = None


def _get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(create_embedder())
    return _vector_store


def _get_fts_store() -> FTSStore:
    global _fts_store
    if _fts_store is None:
        _fts_store = FTSStore()
    return _fts_store


def _get_graph_store() -> GraphStore:
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore()
    return _graph_store


def _get_audn() -> AUDNClassifier:
    global _audn
    if _audn is None:
        _audn = AUDNClassifier(_get_vector_store())
    return _audn


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(
            _get_vector_store(),
            _get_fts_store(),
            _get_graph_store(),
        )
    return _retriever


def _get_tier_manager() -> TierManager:
    global _tier_mgr
    if _tier_mgr is None:
        _tier_mgr = TierManager()
    return _tier_mgr


# ---------------------------------------------------------------------------
# FastMCP application
# ---------------------------------------------------------------------------

mcp_app = FastMCP(
    "mnemo",
    version="0.1.0",
    description=(
        "Mnemo — cross-platform bitemporal long-term memory for AI agents. "
        "Provides hybrid 3-channel retrieval (Vector + FTS5 + Graph), "
        "Ebbinghaus salience decay, AUDN classification, and TOON-serialised output."
    ),
)

# Register tools, resources, and prompts from sub-modules.
# These modules call ``mcp_app.tool()`` etc. at import time.
from mnemo.mcp import prompts as _prompts  # noqa: F401, E402
from mnemo.mcp import resources as _resources  # noqa: F401, E402
from mnemo.mcp import tools as _tools  # noqa: F401, E402


def run_server() -> None:
    """Start the MCP server on stdio transport."""
    mcp_app.run(transport="stdio")


# Allow ``python -m mnemo.mcp.server``
if __name__ == "__main__":
    run_server()
