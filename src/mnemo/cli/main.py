"""Mnemo CLI — Typer + Rich interface.

Commands:
    init          Initialise the memory database.
    remember      Store a new fact (AUDN pipeline).
    search        Hybrid 3-channel search.
    invalidate    Soft-delete a fact.
    tier-decay    Recompute all salience scores and migrate tiers.
    debt          Detect and display knowledge debt.
    serve         Start the MCP stdio server.
    stats         Show memory statistics.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="mnemo",
    help="Mnemo -- bitemporal long-term memory for AI agents.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


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
    console.print(Panel(f"[green]Database initialised:[/green] {target}", title="mnemo init"))


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
    vs = VectorStore(create_embedder())  # type: ignore[arg-type]
    audn = AUDNClassifier(vs)

    with db.session() as conn:
        op, existing_id = audn.classify(text, category, conn)

        if op == AUDNOperation.ADD:
            fact = audn.execute_add(text, category, conn)
            console.print(
                Panel(
                    f"[green]ADD[/green] {encode_fact(fact.model_dump(exclude={'embedding'}))}",
                    title="mnemo remember",
                )
            )
        elif op == AUDNOperation.UPDATE and existing_id:
            fact = audn.execute_update(existing_id, text, category, conn)
            console.print(
                Panel(
                    f"[yellow]UPDATE[/yellow] old={existing_id[:8]}.. "
                    f"{encode_fact(fact.model_dump(exclude={'embedding'}))}",
                    title="mnemo remember",
                )
            )
        elif op == AUDNOperation.NOOP and existing_id:
            audn.execute_noop(existing_id, conn)
            console.print(
                Panel(
                    f"[dim]NOOP[/dim] reinforced existing fact {existing_id[:8]}..",
                    title="mnemo remember",
                )
            )
        else:
            console.print(f"[dim]{op.value}[/dim]")

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
    vs = VectorStore(create_embedder())  # type: ignore[arg-type]
    retriever = HybridRetriever(vs, FTSStore(), GraphStore())
    tier = MemoryTier(min_tier.lower())

    with db.session() as conn:
        results = retriever.search(query, conn, limit=limit, min_tier=tier)

    if not results:
        console.print("[dim]No matching memories found.[/dim]")
        db.close()
        return

    table = Table(title="Search Results", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", justify="right", width=10)
    table.add_column("Tier", width=8)
    table.add_column("Category", width=12)
    table.add_column("Text", ratio=1)
    table.add_column("ID", style="dim", width=10)

    for i, r in enumerate(results, 1):
        tier_style = {
            "core": "bold red",
            "working": "bold yellow",
            "peripheral": "cyan",
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

    console.print(table)
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
    vs = VectorStore(create_embedder())  # type: ignore[arg-type]
    audn = AUDNClassifier(vs)

    with db.session() as conn:
        audn.execute_delete(fact_id, conn)

    console.print(f"[red]Invalidated[/red] fact {fact_id[:8]}..")
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

    console.print(
        Panel(
            f"Processed: [green]{result['processed']}[/green] facts  "
            f"Migrated: [yellow]{result['migrated']}[/yellow]",
            title="Tier Decay",
        )
    )
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
        console.print("[green]No debt detected.[/green]")
        db.close()
        return

    table = Table(title="Debt Ledger", show_lines=True)
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

    console.print(table)
    db.close()


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
@app.command()
def serve() -> None:
    """Start the Mnemo MCP server (stdio transport)."""
    console.print(
        Panel(
            "[bold]Starting Mnemo MCP server...[/bold]\nTransport: stdio\nPress Ctrl+C to stop.",
            title="mnemo serve",
        )
    )
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

    table = Table(title="Mnemo Memory Stats", show_lines=True)
    table.add_column("Metric", style="bold", ratio=1)
    table.add_column("Value", justify="right", width=12)

    table.add_row("Total facts (all versions)", str(total["c"]))
    table.add_row("Active facts", str(active["c"]))
    for t in tiers:
        table.add_row(f"  Tier: {t['tier']}", str(t["c"]))
    table.add_row("Active entities", str(entities_count["c"]))
    table.add_row("Active relations", str(relations_count["c"]))
    table.add_row("Debt ledger items", str(debt_count["c"]))
    table.add_row("Database", target)

    console.print(table)
    db.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
