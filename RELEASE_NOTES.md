# Mnemo v0.4.0: Enterprise Hardening, Fact-Entity Linking & Bitemporal Zero-Trust

Mnemo v0.4.0 is a major milestone release transforming Mnemo into a production-ready, enterprise-hardened cognitive long-term memory engine and autonomous code knowledge graph for AI coding agents via the Model Context Protocol (MCP).

---

## 🌟 What's New in v0.4.0

### 1. Fact-Entity Linking & AST Drift Detection
* **AST Code-Memory Binding**: Facts are directly linked to AST entities (modules, classes, methods) via `fact_entity_links` storing SHA-256 node hashes.
* **Autonomous Drift Flagging**: Code modifications or deletions trigger incremental drift detection, instantly marking linked facts as `is_stale = 1` and recording them in the `debt_ledger`.
* **Stale Penalization**: Stale facts suffer an accelerated decay rate ($\lambda_{\text{stale}} = 0.05$) and receive a $0.5\times$ penalty in graph ranking.

### 2. Activity-Based Spaced Repetition
* **Activity-Tick Decay**: Replaced calendar wall-clock decay with project activity ticks (`activity_ticks`), eliminating amnesia across weekends, holidays, or project pauses ($S = e^{-\Delta \text{ticks} / R}$).
* **Multiplicative Stability Growth**: Spaced reinforcement scales stability exponentially ($R = R_0 \cdot 1.5^n$), capped at $n \le 100$.
* **Anti-Bias Retrieval**: Read queries do not reinforce facts. Reinforcement requires explicit agent acknowledgment (`mnemo_acknowledge_usage`), preventing "rich-get-richer" loops.

### 3. True Bitemporal Point-in-Time Search (`as_of`)
* **Dual Time Modeling**: Strict separation of Valid Time (`valid_start`, `valid_end`) and Transaction Time (`ingest_start`, `ingest_end`).
* **Point-in-Time Reconstruction**: Historical queries via RFC3339 timestamps (`as_of="2026-01-15T00:00:00Z"`).
* **Logical Invalidation vs Physical Corrections**: Differentiates between logical updates (`valid_end = now`) and physical corrections (`ingest_end = now`).

### 4. 3-Channel Hybrid Retrieval (RRF $k=60$)
* **Dense Vector**: Local FastEmbed / ONNX Runtime embeddings (`bge-small-en-v1.5`).
* **Sparse Lexical**: SQLite FTS5 BM25 with code identifier sub-tokenization (`camelCase`, `snake_case`, qualified paths).
* **Graph Traversal**: Recursive CTEs traversing AST dependencies via `fact_entity_links`.
* **Empirical Advantage**: +27.8% relative MRR increase on architectural dependency queries (0.708 vs 0.554) over dense-only search; global Hit@1 increases from 57.4% to 66.7%.

### 5. Enterprise Security Hardening & Zero-Trust
* **Protocol Zero-Trust**: Stripped `source_type` and `confidence` from agent MCP tool arguments; forced to `agent` / `0.75`. Protected developer/git memory (`tier="core"`, `source_type="human_developer"`) cannot be overwritten or downgraded by agents.
* **Path Traversal Defenses**: `ProjectScanner` and `mnemo_scan_project` reject paths outside project roots and block symlink escapes.
* **Deadlock Elimination**: SQLite write transactions use `BEGIN IMMEDIATE` with exponential backoff retry; reads run in non-blocking WAL mode.
* **Administrative CLI-Only Purge**: Physical redaction (`mnemo_purge`) excluded from MCP; restricted strictly to CLI (`mnemo purge <fact_id> --yes`).
* **Schema Migration v6**: Cascading foreign keys on `relations` (`ON DELETE CASCADE`).

---

## 📊 Benchmark Summary

| Configuration | Retrieval Channels | Hit@1 | Hit@3 | Hit@5 | MRR | NDCG@5 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Config A (Dense Only)** | `vector` | 0.574 | 0.796 | 0.833 | 0.679 | 0.718 |
| **Config B (Sparse Only)** | `fts` | 0.259 | 0.259 | 0.259 | 0.259 | 0.259 |
| **Config C (Graph Only)** | `graph` | 0.222 | 0.518 | 0.593 | 0.353 | 0.413 |
| **Config D (Dense + Sparse)** | `vector + fts` | 0.574 | 0.796 | 0.833 | 0.679 | 0.718 |
| **Config E (Mnemo Full 3-Channel)** | **`vector + fts + graph`** | **0.667** | **0.815** | **0.833** | **0.736** | **0.761** |

* **Full Hybrid Search Latency (p50)**: 1.40 ms (500 facts) — 5.56 ms (20,000 facts).
* **Incremental AST Scan (p50)**: 1.41 ms.

---

## 📦 Installation

```bash
pip install mnemo-agentmcp
```

Or run via `uvx`:

```bash
uvx --from mnemo-agentmcp mnemo --help
```
