# Mnemo

> **Production-Ready Cognitive Long-Term Memory & Code Knowledge Graph for AI Agents**  
> *Zero-Infrastructure, Local-First Model Context Protocol (MCP) Server and CLI.*

[![PyPI version](https://img.shields.io/pypi/v/mnemo-agentmcp.svg)](https://pypi.org/project/mnemo-mcp-agent/)
[![Python: >=3.11](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](pyproject.toml)
[![SQLite: WAL + FTS5](https://img.shields.io/badge/Storage-SQLite%20WAL%20%2B%20FTS5-blue.svg)](src/mnemo/storage/)
[![Protocol: Model Context Protocol](https://img.shields.io/badge/Protocol-MCP-purple.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version: 0.4.0](https://img.shields.io/badge/Release-v0.4.0-orange.svg)](pyproject.toml)

---

## Overview

**Mnemo** provides persistent, cross-session architectural memory and an autonomous code knowledge graph for AI coding agents. Unlike ephemeral prompt windows or naive vector-only RAG, Mnemo is an autonomous cognitive subsystem built directly on local SQLite.

It links natural-language rules and decisions directly to code entities, automatically detects architectural code drift, models memory retention through codebase activity ticks, executes true bitemporal point-in-time queries, and retrieves knowledge through a 3-channel Reciprocal Rank Fusion (RRF) search engine.

```
                              ┌──────────────────────────────────────────────┐
                              │             AI Coding Agent / IDE            │
                              └──────────────────────┬───────────────────────┘
                                                     │ MCP Protocol (JSON-RPC)
                                                     ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             MNEMO SUBSYSTEM                                            │
│                                                                                                        │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  ┌────────────────────────────────────┐  │
│  │     Zero-Trust Boundary │  │  Fact-Entity Linking & AST  │  │   Activity Spaced Repetition       │  │
│  │   Hardened MCP Server   │  │   Incremental SHA-256 Drift │  │   Activity-Tick Decay (No Time)    │  │
│  └────────────┬────────────┘  └──────────────┬──────────────┘  └─────────────────┬──────────────────┘  │
│               │                              │                                   │                     │
│               ▼                              ▼                                   ▼                     │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                 3-Channel Hybrid Retriever (RRF k=60)                            │  │
│  │                                                                                                  │  │
│  │       Dense Vector (ONNX)       Sparse FTS5 (Code BM25)      Graph CTE (fact_entity_links)       │  │
│  └───────────────────────────────────────────────────┬──────────────────────────────────────────────┘  │
│                                                      │                                                 │
│                                                      ▼                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                     Local SQLite Storage Engine (WAL Mode, BEGIN IMMEDIATE)                      │  │
│  │                                                                                                  │  │
│  │     facts (Bitemporal)  │  entities & relations  │  fact_entity_links  │  debt_ledger (Audit)    │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Architectural Pillars

### 1. Fact-Entity Linking & AST Drift Detection
* **Direct Association**: Memory facts are not isolated text strings; they are linked to AST entities (modules, classes, methods, functions) via the `fact_entity_links` table with SHA-256 node hashes (`entity_hash_at_link`).
* **Autonomous Drift Flagging**: When code is refactored, modified, or removed, subsequent incremental scans detect hash mismatches or missing nodes, instantly marking linked facts as `is_stale = 1` and queueing them in the `debt_ledger`.
* **Stale Penalization**: Stale facts suffer an accelerated decay rate ($\lambda_{\text{stale}} = 0.05$) and receive a $0.5\times$ penalty in graph ranking, warning agents against hallucinating outdated architectural patterns.

### 2. Activity-Based Spaced Repetition
* **Activity Ticks over Wall-Clock**: Real software development does not decay on a calendar clock. Memory salience $S$ is modeled against project activity ticks (`activity_ticks` incremented on memory ingestion, scanning, and edits) rather than wall-clock time:
  $$S = e^{-\frac{\Delta \text{ticks}}{R}}$$
  This eliminates artificial amnesia after weekends, holidays, or project pauses.
* **Multiplicative Stability Growth**: Repeated reinforcement increases retention stability exponentially:
  $$R = R_0 \cdot (1 + \gamma)^n$$
  (where $\gamma = 0.5$ and stability factor $n$ is clamped to prevent numerical overflows).
* **Search Without Reinforcement Bias**: Merely reading facts during search does not reinforce them. Reinforcement requires explicit agent acknowledgment (`mnemo_acknowledge_usage`), preventing "rich-get-richer" retrieval loops.

### 3. True Bitemporal Retrieval (`as_of`)
* **Dual Time Dimensions**: Strict separation of:
  * **Valid Time** (`valid_start`, `valid_end`): When the fact was true in the real world / project reality.
  * **System / Transaction Time** (`ingest_start`, `ingest_end`): When the fact was asserted, updated, or retracted in the database.
* **Point-in-Time Historical Reconstruction**: The `as_of` parameter allows agents and auditors to travel back in time:
  `mnemo_search("database driver", as_of="2026-01-15T00:00:00Z")`
* **Audit vs. Logical Erasure**: Updates and invalidations never overwrite history silently. Retractions record `valid_end = now` (logical expiration) or `ingest_end = now` (correction of error).

### 4. 3-Channel Hybrid RRF Search
Queries execute simultaneously across three complementary retrieval channels, combined using **Reciprocal Rank Fusion (RRF)**:
$$\text{RRF}(d) = \sum_{c \in C} \frac{w_c}{k + r_c(d)} \quad (k=60)$$

1. **Dense Vector Channel**: FastEmbed / ONNX Runtime local embeddings (`bge-small-en-v1.5`) capturing semantic concepts and intent.
2. **Sparse Lexical Channel (FTS5 BM25)**: SQLite FTS5 index tokenized with custom code sub-token extraction for `camelCase`, `snake_case`, and dotted qualified paths (e.g. `UserService.login_by_id`).
3. **Graph Dependency Channel (Recursive CTE)**: Recursive traversal starting from referenced symbols through AST relations (`CALLS`, `INHERITS`, `IMPORTS`, `DEPENDS_ON`), pulling downstream facts linked via `fact_entity_links`.

### 5. Enterprise Hardening & Zero-Trust
* **Protocol-Level Zero-Trust**: All facts written by agents over MCP are strictly constrained to `source_type="agent"` and `confidence=0.75`. Protected developer/git memory (`tier="core"`, `source_type="human_developer"`) cannot be overwritten or downgraded by agents.
* **Path Traversal & Symlink Defense**: AST scanner and MCP tools reject paths outside the repository root and block symlink escapes.
* **Deadlock-Free SQLite Concurrency**: Write transactions use `BEGIN IMMEDIATE` with exponential backoff retry; search operations execute in non-blocking WAL read mode (`session(readonly=True)`).
* **Administrative CLI-Only Purge**: Physical redaction (`mnemo_purge`) is excluded from MCP tool exposure and restricted to the terminal CLI (`mnemo purge <fact_id> --yes`) to prevent adversarial data destruction.

---

## Empirical Benchmarks & Evidence Base

A rigorous, reproducible benchmark suite (`benchmarks/run_benchmarks.py`) was evaluated over 54 multi-category benchmark queries across small (500 facts), medium (5,000 facts), and large (20,000 facts) synthetic code knowledge bases.

### 1. Channel Ablation Study (Retrieval Quality)

| Configuration | Retrieval Channels | Hit@1 | Hit@3 | Hit@5 | MRR | NDCG@5 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Config A (Dense Only)** | `vector` | 0.574 | 0.796 | 0.833 | 0.679 | 0.718 |
| **Config B (Sparse Only)** | `fts` | 0.259 | 0.259 | 0.259 | 0.259 | 0.259 |
| **Config C (Graph Only)** | `graph` | 0.222 | 0.518 | 0.593 | 0.353 | 0.413 |
| **Config D (Dense + Sparse)** | `vector + fts` | 0.574 | 0.796 | 0.833 | 0.679 | 0.718 |
| **Config E (Mnemo Full 3-Channel)** | **`vector + fts + graph`** | **0.667** | **0.815** | **0.833** | **0.736** | **0.761** |

#### Quality by Query Complexity (MRR / NDCG@5)

| Configuration | Semantic Intent (15) | Code Identifiers (15) | Graph Dependencies (16) | Bitemporal History (8) |
|---|:---:|:---:|:---:|:---:|
| **Config A (Dense Only)** | **0.439 / 0.490** | 1.000 / 1.000 | 0.554 / 0.627 | 1.000 / 1.000 |
| **Config B (Sparse Only)** | 0.000 / 0.000 | 0.867 / 0.867 | 0.000 / 0.000 | 0.250 / 0.250 |
| **Config C (Graph Only)** | 0.000 / 0.000 | 0.519 / 0.612 | 0.552 / 0.655 | 0.062 / 0.125 |
| **Config D (Dense + Sparse)** | **0.439 / 0.490** | 1.000 / 1.000 | 0.554 / 0.627 | 1.000 / 1.000 |
| **Config E (Mnemo Full 3-Channel)** | **0.439 / 0.490** | **1.000 / 1.000** | **0.708 / 0.771** | **1.000 / 1.000** |

> **Key Empirical Finding**: On Graph Dependency queries (where a top-level component relies on a downstream module containing the rule), standard Dense and Sparse channels miss the relationship (MRR = 0.554). Mnemo's 3-channel RRF boosts MRR to **0.708 (+27.8% relative gain)**, elevating global Hit@1 from **57.4% to 66.7%**.

### 2. Latency & Throughput Profile

Tested locally with SQLite WAL mode and local ONNX embeddings:

| Operation | Scale (Facts) | Median (p50) | 95th Percentile (p95) | Throughput (ops/sec) |
|---|:---:|:---:|:---:|:---:|
| **Ingestion (`mnemo_remember`)** | 500 | **1.15 ms** | 1.95 ms | ~870 ops/s |
| **Sparse Search (FTS5 BM25)** | 500 | **0.16 ms** | 0.35 ms | ~6,200 ops/s |
| **Dense Search (Cosine Vector)** | 500 | **0.38 ms** | 0.65 ms | ~2,600 ops/s |
| **Graph Traversal (Recursive CTE)** | 500 | **0.35 ms** | 0.82 ms | ~2,850 ops/s |
| **Full Hybrid Search (3-Channel RRF)** | 500 | **1.40 ms** | 2.10 ms | ~715 ops/s |
| **Full Hybrid Search (3-Channel RRF)** | 20,000 | **5.56 ms** | 7.90 ms | ~180 ops/s |
| **Incremental AST Scan (Unchanged)** | Repo | **1.41 ms** | 2.20 ms | ~710 ops/s |

---

## Model Context Protocol (MCP) Specification

Mnemo runs as a standard Model Context Protocol server over `stdio`, compatible with Claude Desktop, Cursor, Antigravity, and any MCP client.

### Active MCP Tools Exposed to Agents

| Tool Name | Parameters | Description |
|---|---|---|
| `mnemo_remember` | `text: str`<br>`category: str = "general"`<br>`target_fact_id: str = ""` *[opt]*<br>`force_op: str = ""` *[opt]*<br>`entity_identifiers: list[str] = []` *[opt]* | Ingests architectural decision or fact through AUDN classifier. Auto-links mentioned AST entities via SHA-256 hashes. Agent payload capped at 32k chars. |
| `mnemo_search` | `query: str`<br>`limit: int = 5`<br>`category: str = ""` *[opt]*<br>`tier: str = ""` *[opt]*<br>`valid_at: float = None` *[opt]*<br>`as_of: str \| float = None` *[opt]* | 3-Channel Hybrid Retrieval (Vector + FTS5 + Graph via RRF). Supports historical reconstruction via RFC3339 `as_of`. |
| `mnemo_acknowledge_usage`| `fact_ids: list[str]` | Reinforces salience and increments stability count ($n$) for facts that successfully aided the agent in completing a task. |
| `mnemo_invalidate` | `fact_id: str`<br>`reason: str = "outdated"` | Soft-deletes a fact by setting `valid_end = now`. Protected core/developer memories cannot be invalidated by agents. |
| `mnemo_inspect` | `fact_id: str` | Retrieves full bitemporal metadata, link hashes, tier, salience, and provenance audit for a specific fact. |
| `mnemo_get_debt_ledger` | *(none)* | Lists accumulated architectural memory debt (stale facts from AST drift, decayed working memories, orphan links). |
| `mnemo_scan_project` | `path: str = "."`<br>`force: bool = False` | Triggers incremental AST scan of repository. Protected against path traversal and symlink escapes. |

> [!CAUTION]
> **Administrative Security Boundary**: The physical erasure operation (`mnemo_purge`) is **intentionally excluded** from MCP exposure to prevent adversarial or compromised LLM subagents from permanently purging audit trails or leaked secrets. Physical purge must be performed by a human operator via the CLI:
> ```bash
> mnemo purge <fact_id> --yes
> ```

---

## Installation & Quick Start

### 1. Installation

Install directly via `pip` or `uv`:

```bash
pip install mnemo-agentmcp
```

Or install from source in development mode:

```bash
git clone https://github.com/YoruichiYams/mnemo.git
cd mnemo
pip install -e ".[dev]"
```

Or run directly without installation via `uvx`:

```bash
uvx --from mnemo-agentmcp mnemo --help
```

### 2. Configure MCP Client

Add Mnemo to your MCP client configuration:

#### Claude Desktop (`claude_desktop_config.json`)
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

#### Cursor / Antigravity (`.cursor/mcp.json` or `mcp_config.json`)
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

#### Zero-Setup via `uvx` (Cross-Platform)
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

### 3. CLI Management Commands

```bash
# Initialize database in current repository
mnemo init

# Scan codebase AST and extract architecture graph
mnemo scan .

# Search memories across channels
mnemo search "database transaction rules"

# Review system health, FTS sync, and schema status
mnemo doctor

# Inspect architectural memory debt
mnemo debt

# Launch dark-minimal interactive knowledge visualizer
mnemo visualize

# Administrative physical purge (human-only)
mnemo purge fact_abc123 --yes
```

### 4. Running Benchmarks

```bash
# Run complete benchmark suite (small scale)
python -m benchmarks.run_benchmarks --scale small

# Run across all scales (small, medium, large)
python -m benchmarks.run_benchmarks --all-scales
```

---

## Testing & Quality Assurance

Mnemo maintains a strict 100% automated test pass rate with full test isolation:

```bash
# Run complete test suite (141 unit, integration, and security tests)
pytest

# Run security and hardening test suite
pytest tests/test_audit_hardening.py

# Run benchmark smoke tests
pytest tests/test_benchmarks_smoke.py

# Static analysis and style verification
ruff check src tests benchmarks
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
