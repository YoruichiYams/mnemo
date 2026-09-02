# Mnemo

> Autonomous bitemporal memory engine and AST-based knowledge graph for AI agents via Model Context Protocol (MCP). Zero external infrastructure, single-file SQLite storage.

[![PyPI version](https://img.shields.io/pypi/v/mnemo-agentmcp.svg)](https://pypi.org/project/mnemo-mcp-agent/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Core Capabilities

* **Bitemporal Fact Store:** Separates validity timeline (`valid_time`) from system recording time (`ingest_time`) using SQLite WAL mode.
* **3-Channel Hybrid Retrieval:** Fuses dense vector embeddings, FTS5 full-text search, and recursive CTE graph traversal via Reciprocal Rank Fusion (RRF).
* **Ebbinghaus Memory Decay:** Four dynamic tiers (`Core`, `Working`, `Peripheral`, `Archived`) prevent context bloat through piecewise exponential decay.
* **AST Knowledge Graph:** Parses project syntax trees, tracks cross-file class/function relationships, and reconciles deleted modules automatically.
* **Zero Infrastructure:** Operates locally inside a single embedded database file without requiring Docker, external vector databases, or cloud accounts.

---

## Installation

Install the package directly from PyPI into your environment:

```bash
pip install mnemo-agentmcp
```

Or run commands directly via `uvx` without global installation:

```bash
uvx --from mnemo-agentmcp mnemo --help
```

---

## MCP Server Configuration

Connect Mnemo to your preferred AI environment (Cursor, Claude Desktop, Antigravity, Windsurf) by updating your client settings.

### Cursor / Antigravity (`.cursor/mcp.json` or `mcp_config.json`)

```json
{
  "mcpServers": {
    "mnemo": {
      "command": "python",
      "args": ["-m", "mnemo.mcp.server"]
    }
  }
}
```

### Zero-Setup via `uvx` (Cross-Platform)

```json
{
  "mcpServers": {
    "mnemo": {
      "command": "uvx",
      "args": ["--from", "mnemo-agentmcp", "mnemo", "serve"]
    }
  }
}
```

---

## Quick Start

Initialize the memory database in your project directory:

```bash
mnemo init
```

Analyze the repository topology and populate the initial knowledge graph:

```bash
mnemo scan .
```

Generate an interactive dark-themed Sankey visualization of your memory tiers and graph connections:

```bash
mnemo visualize
```

Verify your environment dependencies and database health:

```bash
mnemo doctor
```

---

## Available MCP Tools

Agents communicate with Mnemo over stdio JSON-RPC using these tools:

* `mnemo_remember`: Classify and persist facts using AUDN operations (Add, Update, Delete, Noop).
* `mnemo_search`: Retrieve facts using hybrid vector, lexical, and graph ranking.
* `mnemo_invalidate`: Soft-retire facts by ending their validity window without losing history.
* `mnemo_reinforce`: Boost fact salience upon confirmation or repeated access.
* `mnemo_inspect`: View full metadata, source trace, and bitemporal intervals for any fact.
* `mnemo_get_debt_ledger`: Audit active architectural tensions, deprecated items, and unresolved debt.
* `mnemo_scan_project`: Trigger project-wide AST dependency re-indexing directly from the agent.

---

## Architecture Overview

| Component | Technology | Responsibility |
|---|---|---|
| **Storage Engine** | SQLite (WAL mode) | ACID guarantees, bitemporal schema, and lightweight blob arrays |
| **Lexical Search** | SQLite FTS5 | Sub-millisecond keyword matching across architectural decisions |
| **Vector Search** | Cosine Similarity | Semantic retrieval and similarity clustering |
| **Graph Traversal** | Recursive CTEs | Fast multi-hop entity dependency traversal |
| **Agent Interface** | FastMCP (stdio) | Protocol-compliant JSON-RPC layer isolated on `stderr` |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
