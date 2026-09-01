# Mnemo

Autonomous bitemporal memory engine for AI coding agents.

Mnemo provides persistent, cross-session architectural memory. It automatically maps repository structure, tracks context across edits, and exposes structured retrieval via Model Context Protocol (FastMCP) and clean CLI tools.

## Key Features

* **Autonomous AST Mapping:** Automatically parses modules, classes, and dependencies with incremental SHA-256 diff caching.
* **Bitemporal Knowledge Engine:** Tracks both event time and assertion time using SQLite with WAL mode and full-text search (FTS5).
* **Self-Healing Storage:** Dynamic schema migrations and automatic FTS index repair.
* **Dark Minimal Visualizer:** Interactive 3-column Sankey graph mapping categories, memory tiers, and architecture flows.

## Installation

```bash
pip install mnemo
```

## Quick Start

### 1. Initialize Memory Store

```bash
mnemo init
```

### 2. Autonomous Repository Scan

```bash
mnemo scan .
```

### 3. Interactive Knowledge Visualization

```bash
mnemo visualize
```

### 4. Health Diagnostics

```bash
mnemo doctor
```

## Model Context Protocol (FastMCP)

Mnemo integrates natively with Claude Desktop, Cursor, Antigravity, and any MCP client over stdio:

```json
{
  "mcpServers": {
    "mnemo": {
      "command": "mnemo",
      "args": ["serve"]
    }
  }
}
```

### Available MCP Tools

* `mnemo_remember`: Stores or updates facts through the AUDN classifier and auto-links mentioned code entities.
* `mnemo_search`: Hybrid 3-channel retrieval (Vector + FTS5 + Recursive CTE Graph) fused via Reciprocal Rank Fusion (RRF).
* `mnemo_scan_project`: Triggers an on-demand incremental AST scan of the repository.
* `mnemo_invalidate`: Soft-deletes records with bitemporal invalidation timestamps.
* `mnemo_tier_decay`: Recalculates Ebbinghaus salience decay and tier promotions/demotions.
* `mnemo_get_debt_ledger`: Detects decaying working-tier facts, stale core assumptions, and architectural debt.

## Architecture

* **Storage:** Local SQLite in WAL mode with FTS5 BM25, vector embeddings, and recursive CTE graph traversal.
* **Serialization:** Token-Optimized Object Notation (TOON) providing up to 56% token savings over standard JSON.
* **CLI:** Borderless Matrix interface powered by Typer and Rich.

## License

MIT License. Copyright (c) 2026 Mnemo Contributors.
