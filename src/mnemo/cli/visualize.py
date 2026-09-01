"""Interactive knowledge Sankey visualization module for Mnemo memory graph."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any


def export_knowledge_sankey_data(
    conn: sqlite3.Connection, *, now: float | None = None
) -> dict[str, Any]:
    """Extract bitemporally active facts, tiers, and entity relations for Sankey flows.

    Level 1: Fact Category -> Memory Tier (weighted by sum of fact salience).
    Level 2: Memory Tier -> Target Entities / Dependencies (weighted by relation weights and salience).

    Args:
        conn: Active SQLite connection.
        now: Reference timestamp (epoch seconds). Defaults to current time.

    Returns:
        Dictionary containing nodes, links, and summary metrics.
    """
    current_time = now if now is not None else time.time()

    # 1. Fetch active facts
    facts_rows = conn.execute(
        """
        SELECT category, tier, salience, text
        FROM facts
        WHERE ingest_end IS NULL
          AND (valid_end IS NULL OR valid_end > ?)
        """,
        (current_time,),
    ).fetchall()

    # 2. Fetch active entities and relations
    entities_rows = conn.execute(
        """
        SELECT id, name, entity_type, salience
        FROM entities
        WHERE ingest_end IS NULL
          AND (valid_end IS NULL OR valid_end > ?)
        """,
        (current_time,),
    ).fetchall()

    relations_rows = conn.execute(
        """
        SELECT r.source_id, r.target_id, r.relation_type, r.weight,
               e1.name AS source_name, e1.salience AS source_salience,
               e2.name AS target_name
        FROM relations r
        JOIN entities e1 ON e1.id = r.source_id
        JOIN entities e2 ON e2.id = r.target_id
        WHERE r.ingest_end IS NULL
          AND (r.valid_end IS NULL OR r.valid_end > ?)
          AND e1.ingest_end IS NULL AND (e1.valid_end IS NULL OR e1.valid_end > ?)
          AND e2.ingest_end IS NULL AND (e2.valid_end IS NULL OR e2.valid_end > ?)
        """,
        (current_time, current_time, current_time),
    ).fetchall()

    # Aggregate Level 1: Category -> Tier
    cat_tier_flows: dict[tuple[str, str], float] = {}
    total_salience = 0.0
    core_salience = 0.0
    tier_counts: dict[str, int] = {"core": 0, "working": 0, "peripheral": 0, "archived": 0}

    for row in facts_rows:
        cat = f"cat:{row['category']}"
        tier = str(row["tier"]).capitalize()
        tier_lower = str(row["tier"]).lower()
        sal = float(row["salience"])

        pair = (cat, tier)
        cat_tier_flows[pair] = cat_tier_flows.get(pair, 0.0) + sal

        total_salience += sal
        if tier_lower == "core":
            core_salience += sal
        if tier_lower in tier_counts:
            tier_counts[tier_lower] += 1

    # Aggregate Level 2: Tier -> Entity
    tier_entity_flows: dict[tuple[str, str], float] = {}

    for row in relations_rows:
        src_sal = float(row["source_salience"])
        tier_str = "Core" if src_sal >= 0.90 else ("Working" if src_sal >= 0.70 else "Peripheral")
        target_label = f"entity:{row['target_name']}"
        weight = float(row["weight"])

        pair = (tier_str, target_label)
        tier_entity_flows[pair] = tier_entity_flows.get(pair, 0.0) + (weight * src_sal)

    # If no explicit relations exist, map entities based on their own salience
    if not tier_entity_flows and entities_rows:
        for erow in entities_rows:
            esal = float(erow["salience"])
            etier = "Core" if esal >= 0.90 else ("Working" if esal >= 0.70 else "Peripheral")
            target_label = f"entity:{erow['name']}"
            pair = (etier, target_label)
            tier_entity_flows[pair] = tier_entity_flows.get(pair, 0.0) + max(0.5, esal)

    # If database is completely empty, provide fallback demo node structure
    if not cat_tier_flows and not tier_entity_flows:
        cat_tier_flows[("cat:general", "Working")] = 1.0
        tier_entity_flows[("Working", "entity:Mnemo")] = 1.0

    # Build unique nodes and index mapping
    all_nodes: list[str] = []
    node_indices: dict[str, int] = {}

    def get_or_add_node(name: str) -> int:
        if name not in node_indices:
            node_indices[name] = len(all_nodes)
            all_nodes.append(name)
        return node_indices[name]

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []

    # Add Level 1 links
    for (src, tgt), val in cat_tier_flows.items():
        sources.append(get_or_add_node(src))
        targets.append(get_or_add_node(tgt))
        values.append(round(val, 4))

    # Add Level 2 links
    for (src, tgt), val in tier_entity_flows.items():
        sources.append(get_or_add_node(src))
        targets.append(get_or_add_node(tgt))
        values.append(round(val, 4))

    # Compute strict 3-column coordinates: Col 0 (x=0.01), Col 1 (x=0.50), Col 2 (x=0.99)
    col_0_indices = [i for i, n in enumerate(all_nodes) if n.startswith("cat:")]
    col_1_indices = [
        i for i, n in enumerate(all_nodes) if n in {"Core", "Working", "Peripheral", "Archived"}
    ]
    col_2_indices = [
        i for i, n in enumerate(all_nodes) if i not in col_0_indices and i not in col_1_indices
    ]

    node_x: list[float] = [0.5] * len(all_nodes)
    node_y: list[float] = [0.5] * len(all_nodes)

    def _assign_column(indices: list[int], x_val: float) -> None:
        n = len(indices)
        if n == 0:
            return
        for k, idx in enumerate(indices):
            node_x[idx] = x_val
            if n == 1:
                node_y[idx] = 0.5
            else:
                node_y[idx] = round(0.08 + 0.84 * (k / (n - 1)), 4)

    _assign_column(col_0_indices, 0.01)
    _assign_column(col_1_indices, 0.50)
    _assign_column(col_2_indices, 0.99)

    # Node palette (Dark Minimal)
    node_colors: list[str] = []
    for node in all_nodes:
        if node.startswith("cat:"):
            node_colors.append("#58a6ff")  # Muted ice blue for categories
        elif node == "Core":
            node_colors.append("#3fb950")  # Muted forest green for core
        elif node == "Working":
            node_colors.append("#d29922")  # Muted amber for working
        elif node == "Peripheral":
            node_colors.append("#8b949e")  # Graphite grey for peripheral
        elif node == "Archived":
            node_colors.append("#484f58")  # Dark grey for archived
        elif node.startswith("entity:"):
            node_colors.append("#bc8cff")  # Muted lavender for entities
        else:
            node_colors.append("#7d8590")

    total_facts = len(facts_rows)
    core_pct = (core_salience / total_salience * 100.0) if total_salience > 0 else 0.0
    mean_salience = (total_salience / total_facts) if total_facts > 0 else 0.0

    return {
        "nodes": all_nodes,
        "node_colors": node_colors,
        "node_x": node_x,
        "node_y": node_y,
        "sources": sources,
        "targets": targets,
        "values": values,
        "metrics": {
            "total_facts": total_facts,
            "total_entities": len(entities_rows),
            "total_relations": len(relations_rows),
            "core_pct": round(core_pct, 1),
            "mean_salience": round(mean_salience, 3),
            "tier_counts": tier_counts,
        },
    }


def generate_mermaid_sankey(data: dict[str, Any]) -> str:
    """Generate Mermaid sankey-beta syntax diagram.

    Args:
        data: Exported sankey data dict.

    Returns:
        Mermaid sankey-beta diagram string.
    """
    nodes = data["nodes"]
    sources = data["sources"]
    targets = data["targets"]
    values = data["values"]

    lines = ["sankey-beta"]
    for src_idx, tgt_idx, val in zip(sources, targets, values, strict=True):
        src_name = nodes[src_idx].replace('"', "'")
        tgt_name = nodes[tgt_idx].replace('"', "'")
        lines.append(f'  "{src_name}", "{tgt_name}", {val:.2f}')

    return "\n".join(lines)


def generate_html_report(data: dict[str, Any], project_name: str = "Mnemo Memory") -> str:
    """Generate self-contained dark minimal HTML report with interactive Plotly Sankey.

    Args:
        data: Exported sankey data dict.
        project_name: Project header title.

    Returns:
        Full standalone HTML document string.
    """
    import html

    metrics = data.get("metrics", {})
    total_facts = metrics.get("total_facts", 0)
    total_entities = metrics.get("total_entities", 0)
    total_relations = metrics.get("total_relations", 0)
    core_pct = metrics.get("core_pct", 0.0)
    mean_salience = metrics.get("mean_salience", 0.0)

    def _safe_json_embed(obj: Any) -> str:
        # Prevent XSS via breaking out of <script> tags
        return json.dumps(obj).replace("</", "<\\/")

    safe_project_name = html.escape(project_name)
    safe_nodes = [html.escape(str(n)) for n in data.get("nodes", [])]

    nodes_json = _safe_json_embed(safe_nodes)
    node_colors_json = _safe_json_embed(data.get("node_colors", []))
    node_x_json = _safe_json_embed(data.get("node_x", []))
    node_y_json = _safe_json_embed(data.get("node_y", []))
    sources_json = _safe_json_embed(data.get("sources", []))
    targets_json = _safe_json_embed(data.get("targets", []))
    values_json = _safe_json_embed(data.get("values", []))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_project_name} - Knowledge Flow</title>
    <!-- Google Fonts: Google Sans Flex & JetBrains Mono -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Google+Sans+Flex:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <!-- Plotly.js for interactive zero-dependency dark Sankey -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {{
            --bg-primary: #050507;
            --bg-card: #0b0c0e;
            --border: #1a1c22;
            --text-primary: #f0f2f5;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --font-main: 'Google Sans Flex', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', ui-monospace, monospace;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            -webkit-font-smoothing: antialiased;
        }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: var(--font-main);
            padding: 32px 28px;
            min-height: 100vh;
        }}
        .header {{
            max-width: 1240px;
            margin: 0 auto 24px auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
        }}
        .title-group h1 {{
            font-size: 20px;
            font-weight: 600;
            letter-spacing: -0.3px;
            color: var(--text-primary);
        }}
        .title-group p {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 4px;
            font-weight: 400;
        }}
        .header-actions {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .zoom-controls {{
            display: flex;
            align-items: center;
            gap: 4px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 2px 4px;
        }}
        .zoom-btn {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 500;
            padding: 4px 8px;
            border-radius: 3px;
            cursor: pointer;
            transition: all 0.12s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
        }}
        .zoom-btn:hover {{
            color: var(--text-primary);
            background: #1a1c22;
        }}
        .reset-btn {{
            font-size: 10px;
            letter-spacing: 0.04em;
            border-left: 1px solid var(--border);
            border-radius: 0 3px 3px 0;
            padding-left: 8px;
        }}
        .badge {{
            font-size: 11px;
            font-family: var(--font-mono);
            background: #111318;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 4px 10px;
            border-radius: 4px;
            letter-spacing: 0.05em;
        }}
        .metrics-grid {{
            max-width: 1240px;
            margin: 0 auto 20px auto;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
        }}
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px 16px;
        }}
        .metric-card .label {{
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            font-weight: 500;
        }}
        .metric-card .value {{
            font-size: 20px;
            font-weight: 600;
            margin-top: 4px;
            color: var(--text-primary);
            font-family: var(--font-mono);
            letter-spacing: -0.02em;
        }}
        .chart-container {{
            max-width: 1240px;
            margin: 0 auto;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 20px;
            height: 600px;
            position: relative;
            overflow: hidden;
            cursor: grab;
        }}
        .chart-container:active {{
            cursor: grabbing;
        }}
        #sankey-plot {{
            width: 100%;
            height: 100%;
        }}
        /* Plotly Modebar Styling for Deep Black */
        .modebar-container {{
            top: 8px !important;
            right: 8px !important;
        }}
        .modebar {{
            background: transparent !important;
        }}
        .modebar-btn svg path {{
            fill: #70737d !important;
        }}
        .modebar-btn:hover svg path {{
            fill: #f0f2f5 !important;
        }}
        .modebar-btn.active svg path {{
            fill: #58a6ff !important;
        }}
        .footer {{
            max-width: 1240px;
            margin: 16px auto 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title-group">
            <h1>{safe_project_name}</h1>
            <p>Knowledge Flow &bull; Bitemporal Graph</p>
        </div>
        <div class="header-actions">
            <div class="zoom-controls">
                <button class="zoom-btn" onclick="zoomIn()" title="Zoom In">+</button>
                <button class="zoom-btn" onclick="zoomOut()" title="Zoom Out">&minus;</button>
                <button class="zoom-btn reset-btn" onclick="resetZoom()" title="Reset View">&#x27F2; Reset</button>
            </div>
            <div class="badge">SQLite WAL &bull; RRF k=60</div>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="label">Active Facts</div>
            <div class="value">{total_facts}</div>
        </div>
        <div class="metric-card">
            <div class="label">Entities / Nodes</div>
            <div class="value">{total_entities}</div>
        </div>
        <div class="metric-card">
            <div class="label">Graph Relations</div>
            <div class="value">{total_relations}</div>
        </div>
        <div class="metric-card">
            <div class="label">Core Tier Memory</div>
            <div class="value">{core_pct}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Mean Salience S(t)</div>
            <div class="value">{mean_salience:.3f}</div>
        </div>
    </div>

    <div class="chart-container" id="chartWrapper">
        <div id="sankey-plot"></div>
    </div>

    <div class="footer">
        <div>Mnemo Memory Engine</div>
        <div>Dark Minimal &bull; Zero-Dependency HTML</div>
    </div>

    <script>
        const nodes = {nodes_json};
        const nodeColors = {node_colors_json};
        const nodeX = {node_x_json};
        const nodeY = {node_y_json};
        const sources = {sources_json};
        const targets = {targets_json};
        const values = {values_json};

        // Format node display labels
        const cleanLabels = nodes.map(n => {{
            if (n.startsWith("cat:")) return n.replace("cat:", "📁 ");
            if (n.startsWith("entity:")) return n.replace("entity:", "⚡ ");
            return n;
        }});

        const plotData = [{{
            type: "sankey",
            orientation: "h",
            arrangement: "fixed",
            node: {{
                pad: 20,
                thickness: 18,
                line: {{ color: "#1a1c22", width: 1 }},
                label: cleanLabels,
                color: nodeColors,
                x: nodeX.length === nodes.length ? nodeX : undefined,
                y: nodeY.length === nodes.length ? nodeY : undefined,
                hoverlabel: {{
                    bgcolor: "#0b0c0e",
                    bordercolor: "#1a1c22",
                    font: {{ color: "#f0f2f5", family: "'JetBrains Mono', monospace", size: 12 }}
                }}
            }},
            link: {{
                source: sources,
                target: targets,
                value: values,
                color: "rgba(110, 118, 129, 0.18)",
                hovercolor: "rgba(88, 166, 255, 0.45)",
                hoverlabel: {{
                    bgcolor: "#0b0c0e",
                    bordercolor: "#1a1c22",
                    font: {{ color: "#f0f2f5", family: "'JetBrains Mono', monospace", size: 12 }}
                }}
            }}
        }}];

        const layout = {{
            font: {{
                family: "'Google Sans Flex', -apple-system, BlinkMacSystemFont, sans-serif",
                size: 13,
                color: "#c9d1d9"
            }},
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            margin: {{ l: 10, r: 10, t: 20, b: 20 }}
        }};

        const config = {{
            responsive: true,
            scrollZoom: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ["lasso2d", "select2d"],
            displaylogo: false
        }};

        Plotly.newPlot("sankey-plot", plotData, layout, config).then(() => {{
            setupZoomAndPan();
        }});

        let currentZoom = 1.0;
        let panX = 0;
        let panY = 0;
        let isPanning = false;
        let startX = 0;
        let startY = 0;

        function getTargetSvg() {{
            return document.querySelector("#sankey-plot .main-svg") || document.querySelector("#sankey-plot svg");
        }}

        function applyTransform() {{
            const svg = getTargetSvg();
            if (svg) {{
                svg.style.transformOrigin = "center center";
                svg.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{currentZoom}})`;
                svg.style.transition = isPanning ? "none" : "transform 0.12s ease-out";
            }}
        }}

        function zoomIn() {{
            currentZoom = Math.min(currentZoom * 1.25, 5.0);
            applyTransform();
        }}

        function zoomOut() {{
            currentZoom = Math.max(currentZoom / 1.25, 0.3);
            applyTransform();
        }}

        function resetZoom() {{
            currentZoom = 1.0;
            panX = 0;
            panY = 0;
            applyTransform();
        }}

        function setupZoomAndPan() {{
            const container = document.getElementById("chartWrapper");
            if (!container) return;

            container.addEventListener("wheel", (e) => {{
                e.preventDefault();
                const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
                currentZoom = Math.max(0.3, Math.min(5.0, currentZoom * zoomFactor));
                applyTransform();
            }}, {{ passive: false }});

            container.addEventListener("mousedown", (e) => {{
                if (e.button === 0) {{
                    isPanning = true;
                    startX = e.clientX - panX;
                    startY = e.clientY - panY;
                }}
            }});

            window.addEventListener("mousemove", (e) => {{
                if (!isPanning) return;
                panX = e.clientX - startX;
                panY = e.clientY - startY;
                applyTransform();
            }});

            window.addEventListener("mouseup", () => {{
                if (isPanning) {{
                    isPanning = false;
                    applyTransform();
                }}
            }});
        }}
    </script>
</body>
</html>
"""
