"""Mnemo CLI — Borderless Matrix interface.

Commands:
    init          Initialise the memory database.
    remember      Store a new fact (AUDN pipeline).
    search        Hybrid 3-channel search.
    invalidate    Soft-delete a fact.
    tier-decay    Recompute all salience scores and migrate tiers.
    debt          Detect and display knowledge debt.
    serve         Start the MCP stdio server.
    stats         Show memory statistics.
    visualize     Generate interactive knowledge Sankey diagram.
"""

from __future__ import annotations

import sys
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.table import Table

# Ensure UTF-8 output encoding across Windows legacy consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

app = typer.Typer(
    name="mnemo",
    help="Mnemo -- bitemporal long-term memory for AI agents.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console(highlight=False)


def _safe_str(val: Any) -> str:
    """Format string safely for display."""
    return str(val)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------
@app.command()
def init(
    db_path: str = typer.Option(
        "",
        "--db",
        help="Path to SQLite database. Leave empty for default (~/.mnemo/memory.db).",
    ),
) -> None:
    """Initialise (or re-initialise) the Mnemo memory database."""
    from mnemo.storage.connection import Database, default_db_path

    target = db_path if db_path else str(default_db_path())
    Database(db_path=target)
    console.print("→ [bold]mnemo init[/bold]")
    console.print(f"  [dim]database[/dim]  {_safe_str(target)}")


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------
@app.command()
def remember(
    text: str = typer.Argument(..., help="Fact text to store."),
    category: str = typer.Option("general", "-c", "--category", help="Fact category."),
    db_path: str = typer.Option("", "--db", help="Database path."),
) -> None:
    """Store a new fact through the AUDN pipeline."""
    from mnemo.core.models import AUDNOperation
    from mnemo.engine.audn import AUDNClassifier
    from mnemo.serialization.toon import encode_fact
    from mnemo.storage.connection import Database, default_db_path
    from mnemo.storage.vector_store import VectorStore, create_embedder

    target = db_path if db_path else str(default_db_path())
    db = Database(db_path=target)
    vs = VectorStore(create_embedder())
    audn = AUDNClassifier(vs)

    with db.session() as conn:
        op, existing_id = audn.classify(text, category, conn)

        if op == AUDNOperation.ADD:
            fact = audn.execute_add(text, category, conn)
            console.print("→ [bold]mnemo remember[/bold] [green]add[/green]")
            console.print(
                f"  [dim]fact[/dim]  {encode_fact(fact.model_dump(exclude={'embedding'}))}"
            )
        elif op == AUDNOperation.UPDATE and existing_id:
            fact = audn.execute_update(existing_id, text, category, conn)
            console.print("→ [bold]mnemo remember[/bold] [yellow]update[/yellow]")
            console.print(f"  [dim]old[/dim]   {existing_id[:8]}..")
            console.print(
                f"  [dim]new[/dim]   {encode_fact(fact.model_dump(exclude={'embedding'}))}"
            )
        elif op == AUDNOperation.NOOP and existing_id:
            audn.execute_noop(existing_id, conn)
            console.print("→ [bold]mnemo remember[/bold] [dim]noop[/dim]")
            console.print(f"  [dim]reinforced[/dim]  {existing_id[:8]}..")
        else:
            console.print(f"→ [bold]mnemo remember[/bold] [dim]{op.value}[/dim]")

    db.close()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(10, "-n", "--limit", help="Max results."),
    min_tier: str = typer.Option(
        "archived", "--min-tier", help="Minimum tier: core, working, peripheral, archived."
    ),
    db_path: str = typer.Option("", "--db", help="Database path."),
) -> None:
    """Hybrid 3-channel search (Vector + FTS5 + Graph + RRF)."""
    from mnemo.core.models import MemoryTier
    from mnemo.engine.retriever import HybridRetriever
    from mnemo.storage.connection import Database, default_db_path
    from mnemo.storage.fts_store import FTSStore
    from mnemo.storage.graph_store import GraphStore
    from mnemo.storage.vector_store import VectorStore, create_embedder

    target = db_path if db_path else str(default_db_path())
    db = Database(db_path=target)
    vs = VectorStore(create_embedder())
    retriever = HybridRetriever(vs, FTSStore(), GraphStore())
    tier = MemoryTier(min_tier.lower())

    with db.session() as conn:
        results = retriever.search(query, conn, limit=limit, min_tier=tier)

    if not results:
        console.print("[dim]No matching memories found.[/dim]")
        db.close()
        return

    table = Table(
        title="→ [bold]Search Results[/bold]",
        title_justify="left",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", justify="right", width=10)
    table.add_column("Tier", width=10)
    table.add_column("Category", width=12)
    table.add_column("Text", ratio=1)
    table.add_column("ID", style="dim", width=10)

    for i, r in enumerate(results, 1):
        tier_style = {
            "core": "bold green",
            "working": "bold yellow",
            "peripheral": "dim cyan",
            "archived": "dim",
        }.get(r.fact.tier.value, "")
        table.add_row(
            str(i),
            f"{r.score:.6f}",
            f"[{tier_style}]{r.fact.tier.value}[/{tier_style}]",
            r.fact.category,
            r.fact.text[:80],
            r.fact.id[:8] + "..",
        )

    console.print()
    console.print(table)
    console.print()
    db.close()


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------
@app.command()
def invalidate(
    fact_id: str = typer.Argument(..., help="UUID of the fact to invalidate."),
    db_path: str = typer.Option("", "--db", help="Database path."),
) -> None:
    """Soft-delete a fact (close its valid_end)."""
    from mnemo.engine.audn import AUDNClassifier
    from mnemo.storage.connection import Database, default_db_path
    from mnemo.storage.vector_store import VectorStore, create_embedder

    target = db_path if db_path else str(default_db_path())
    db = Database(db_path=target)
    vs = VectorStore(create_embedder())
    audn = AUDNClassifier(vs)

    with db.session() as conn:
        audn.execute_delete(fact_id, conn)

    console.print("→ [bold]mnemo invalidate[/bold]")
    console.print(f"  [dim]invalidated[/dim]  [red]{fact_id[:8]}..[/red]")
    db.close()


# ---------------------------------------------------------------------------
# tier-decay
# ---------------------------------------------------------------------------
@app.command("tier-decay")
def tier_decay(
    db_path: str = typer.Option("", "--db", help="Database path."),
) -> None:
    """Recompute all salience scores and migrate tiers."""
    from mnemo.engine.tier_manager import TierManager
    from mnemo.storage.connection import Database, default_db_path

    target = db_path if db_path else str(default_db_path())
    db = Database(db_path=target)
    mgr = TierManager()

    with db.session() as conn:
        result = mgr.decay_all(conn)

    console.print("→ [bold]mnemo tier-decay[/bold]")
    console.print(f"  [dim]processed[/dim]  [green]{result['processed']}[/green]")
    console.print(f"  [dim]migrated[/dim]   [yellow]{result['migrated']}[/yellow]")
    db.close()


# ---------------------------------------------------------------------------
# debt
# ---------------------------------------------------------------------------
@app.command()
def debt(
    db_path: str = typer.Option("", "--db", help="Database path."),
) -> None:
    """Detect and display knowledge / architectural debt."""
    from mnemo.engine.tier_manager import TierManager
    from mnemo.storage.connection import Database, default_db_path

    target = db_path if db_path else str(default_db_path())
    db = Database(db_path=target)
    mgr = TierManager()

    with db.session() as conn:
        mgr.decay_all(conn)
        items = mgr.detect_debt(conn)
        if items:
            mgr.persist_debt(items, conn)

    if not items:
        console.print("→ [bold]mnemo debt[/bold]")
        console.print("  [green]No debt detected.[/green]")
        db.close()
        return

    table = Table(
        title="→ [bold]Debt Ledger[/bold]",
        title_justify="left",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
    )
    table.add_column("Ceiling", width=10)
    table.add_column("Trigger", width=20)
    table.add_column("Context", ratio=1)

    for item in items:
        ceiling_style = {
            "critical": "bold red",
            "high": "red",
            "medium": "yellow",
            "low": "dim",
        }.get(item.ceiling, "")
        table.add_row(
            f"[{ceiling_style}]{item.ceiling}[/{ceiling_style}]",
            item.trigger,
            item.code_context,
        )

    console.print()
    console.print(table)
    console.print()
    db.close()


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
@app.command()
def serve() -> None:
    """Start the Mnemo MCP server (stdio transport)."""
    console.print("→ [bold]mnemo serve[/bold]")
    console.print("  [dim]transport[/dim]  stdio")
    console.print("  [dim]status[/dim]     running (press Ctrl+C to stop)")
    from mnemo.mcp.server import run_server

    run_server()


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
@app.command()
def stats(
    db_path: str = typer.Option("", "--db", help="Database path."),
) -> None:
    """Show memory store statistics."""
    import time

    from mnemo.storage.connection import Database, default_db_path

    target = db_path if db_path else str(default_db_path())
    db = Database(db_path=target)
    now = time.time()

    with db.session() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM facts WHERE ingest_end IS NULL").fetchone()
        active = conn.execute(
            "SELECT COUNT(*) AS c FROM facts WHERE ingest_end IS NULL "
            "AND (valid_end IS NULL OR valid_end > ?)",
            (now,),
        ).fetchone()
        tiers = conn.execute(
            "SELECT tier, COUNT(*) AS c FROM facts "
            "WHERE ingest_end IS NULL AND (valid_end IS NULL OR valid_end > ?) "
            "GROUP BY tier ORDER BY tier",
            (now,),
        ).fetchall()
        entities_count = conn.execute(
            "SELECT COUNT(*) AS c FROM entities WHERE ingest_end IS NULL "
            "AND (valid_end IS NULL OR valid_end > ?)",
            (now,),
        ).fetchone()
        relations_count = conn.execute(
            "SELECT COUNT(*) AS c FROM relations WHERE ingest_end IS NULL "
            "AND (valid_end IS NULL OR valid_end > ?)",
            (now,),
        ).fetchone()
        debt_count = conn.execute("SELECT COUNT(*) AS c FROM debt_ledger").fetchone()

    table = Table(
        title="→ [bold]Mnemo Memory Stats[/bold]",
        title_justify="left",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
    )
    table.add_column("Metric", style="bold", ratio=1)
    table.add_column("Value", justify="right", width=14)

    table.add_row("Total facts (all versions)", str(total["c"]))
    table.add_row("Active facts", str(active["c"]))
    for t in tiers:
        table.add_row(f"  · Tier: {t['tier']}", str(t["c"]))
    table.add_row("Active entities", str(entities_count["c"]))
    table.add_row("Active relations", str(relations_count["c"]))
    table.add_row("Debt ledger items", str(debt_count["c"]))
    table.add_row("Database", _safe_str(target))

    console.print()
    console.print(table)
    console.print()
    db.close()


# ---------------------------------------------------------------------------
# visualize
# ---------------------------------------------------------------------------
@app.command()
def visualize(
    output: str = typer.Option(
        "./mnemo_graph.html",
        "-o",
        "--output",
        help="Path to output HTML file.",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Automatically open generated HTML visualization in default browser.",
    ),
    format_type: str = typer.Option(
        "html",
        "--format",
        "-f",
        help="Visualization format: 'html' (interactive Sankey) or 'mermaid' (terminal output).",
    ),
    db_path: str = typer.Option("", "--db", help="Database path."),
) -> None:
    """Generate an interactive Dark Minimal Sankey flow diagram of the memory graph."""
    import webbrowser
    from pathlib import Path

    from mnemo.cli.visualize import (
        export_knowledge_sankey_data,
        generate_html_report,
        generate_mermaid_sankey,
    )
    from mnemo.storage.connection import Database, default_db_path

    target_db = db_path if db_path else str(default_db_path())
    db = Database(db_path=target_db)

    with db.session() as conn:
        data = export_knowledge_sankey_data(conn)

    db.close()

    if format_type.lower() == "mermaid":
        mermaid_code = generate_mermaid_sankey(data)
        console.print(mermaid_code)
        return

    html_content = generate_html_report(data)
    out_path = Path(output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")

    console.print("→ [bold]mnemo visualize[/bold]")
    console.print(f"  [dim]nodes[/dim]         {len(data['nodes'])}")
    console.print(f"  [dim]active flows[/dim]  {len(data['values'])}")
    console.print(f"  [dim]core memory[/dim]   {data['metrics']['core_pct']}%")
    console.print()
    console.print(f'the result is available by path "{_safe_str(out_path)}"')

    if open_browser:
        webbrowser.open(out_path.as_uri())


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
@app.command()
def doctor(
    db_path: str = typer.Option("", "--db", help="Database path."),
) -> None:
    """Diagnose database health, FTS5 index integrity, and graph statistics."""
    from mnemo.storage.connection import Database, default_db_path

    target = db_path if db_path else str(default_db_path())
    db = Database(db_path=target)
    health = db.check_integrity()

    status_color = "green" if health["integrity_ok"] else "red"
    fts_color = "green" if health["fts_in_sync"] else "yellow"

    console.print("→ [bold]mnemo doctor[/bold]")
    console.print(f"  [dim]database[/dim]        {_safe_str(target)}")
    console.print(
        f"  [dim]schema version[/dim]  v{health['schema_version']} (target v{health['target_version']})"
    )
    console.print(
        f"  [dim]sqlite integrity[/dim][{status_color}] {'ok' if health['integrity_ok'] else 'corrupted'}[/{status_color}]"
    )
    console.print(f"  [dim]journal mode[/dim]    {health['journal_mode']}")
    console.print(
        f"  [dim]fts5 sync[/dim]       [{fts_color}] {'synchronized' if health['fts_in_sync'] else 'repaired'}[/{fts_color}]"
    )

    counts = health.get("table_counts", {})
    console.print(f"  [dim]active facts[/dim]    {counts.get('facts', 0)}")
    console.print(f"  [dim]entities[/dim]        {counts.get('entities', 0)}")
    console.print(f"  [dim]relations[/dim]       {counts.get('relations', 0)}")
    console.print(f"  [dim]cached files[/dim]    {counts.get('file_scan_cache', 0)}")
    db.close()


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------
@app.command()
def scan(
    path: str = typer.Argument(".", help="Project directory path to scan."),
    db_path: str = typer.Option("", "--db", help="Database path."),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-scan all files."),
) -> None:
    """Perform an incremental AST scan to extract modules, classes, and dependencies."""
    from mnemo.engine.scanner import ProjectScanner
    from mnemo.storage.connection import Database, default_db_path

    target = db_path if db_path else str(default_db_path())
    db = Database(db_path=target)
    scanner = ProjectScanner(root_path=path)

    with db.session() as conn:
        res = scanner.scan(conn, force=force)

    console.print("→ [bold]mnemo scan[/bold]")
    console.print(f"  [dim]scanned files[/dim]    {res['scanned_files']}")
    console.print(f"  [dim]skipped files[/dim]    {res['skipped_files']}")
    console.print(f"  [dim]entities added[/dim]   {res['entities_added']}")
    console.print(f"  [dim]relations added[/dim]  {res['relations_added']}")
    db.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
