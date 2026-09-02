# Mnemo

> Autonomous bitemporal memory engine and AST-based knowledge graph for AI agents via Model Context Protocol (MCP). Zero external infrastructure, single-file SQLite storage.

[![PyPI version](https://img.shields.io/pypi/v/mnemo-agentmcp.svg)](https://pypi.org/project/mnemo-agentmcp/)
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
