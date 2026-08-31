"""End-to-end unit tests for the Typer CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mnemo.cli.main import app


class TestCLICommands:
    """Test suite executing CLI commands against temporary SQLite databases."""

    def test_cli_help(self, cli_runner: CliRunner) -> None:
        """CLI --help returns 0 and lists all core commands."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Mnemo" in result.stdout
        assert "init" in result.stdout
        assert "remember" in result.stdout
        assert "search" in result.stdout
        assert "invalidate" in result.stdout
        assert "tier-decay" in result.stdout
        assert "debt" in result.stdout
        assert "stats" in result.stdout

    def test_cli_init_and_stats(self, cli_runner: CliRunner, temp_db_path: Path) -> None:
        """'mnemo init' initializes SQLite schema and 'mnemo stats' displays metrics."""
        db_str = str(temp_db_path)

        # 1. Init
        init_res = cli_runner.invoke(app, ["init", "--db", db_str])
        assert init_res.exit_code == 0
        assert "Database initialised" in init_res.stdout

        # 2. Stats
        stats_res = cli_runner.invoke(app, ["stats", "--db", db_str])
        assert stats_res.exit_code == 0
        assert "Total facts" in stats_res.stdout
        assert "Active facts" in stats_res.stdout

    def test_cli_full_memory_lifecycle(
        self,
        cli_runner: CliRunner,
        temp_db_path: Path,
    ) -> None:
        """Full CLI memory lifecycle: remember -> search -> invalidate -> decay -> debt -> stats."""
        db_str = str(temp_db_path)
        cli_runner.invoke(app, ["init", "--db", db_str])

        # 1. Remember fact 1
        rem_res = cli_runner.invoke(
            app,
            ["remember", "Antigravity orchestrates agentic workflows", "-c", "ai", "--db", db_str],
        )
        assert rem_res.exit_code == 0
        assert "ADD" in rem_res.stdout

        # 2. Remember fact 2
        rem_res2 = cli_runner.invoke(
            app,
            ["remember", "SQLite WAL mode improves concurrency", "-c", "database", "--db", db_str],
        )
        assert rem_res2.exit_code == 0
        assert "ADD" in rem_res2.stdout

        # 3. Search fact
        search_res = cli_runner.invoke(app, ["search", "Antigravity", "--db", db_str])
        assert search_res.exit_code == 0
        assert "Antigravity" in search_res.stdout

        # 4. Search with min_tier filter
        search_tier = cli_runner.invoke(
            app, ["search", "Antigravity", "--min-tier", "core", "--db", db_str]
        )
        assert search_tier.exit_code == 0

        # 5. Search with no results
        search_none = cli_runner.invoke(
            app, ["search", "NonexistentTokenQuery12345", "--db", db_str]
        )
        assert search_none.exit_code == 0

        # 6. Duplicate remember (NOOP)
        noop_res = cli_runner.invoke(
            app,
            ["remember", "Antigravity orchestrates agentic workflows", "-c", "ai", "--db", db_str],
        )
        assert noop_res.exit_code == 0
        assert "NOOP" in noop_res.stdout

        # 7. Invalidate
        import re

        match = re.search(r"i:([a-f0-9-]+)", rem_res.stdout)
        if match:
            fact_id = match.group(1)
            inval_res = cli_runner.invoke(app, ["invalidate", fact_id, "--db", db_str])
            assert inval_res.exit_code == 0
            assert "Invalidated" in inval_res.stdout

        # 8. Tier decay command
        decay_res = cli_runner.invoke(app, ["tier-decay", "--db", db_str])
        assert decay_res.exit_code == 0
        assert "Processed:" in decay_res.stdout

        # 9. Debt command
        debt_res = cli_runner.invoke(app, ["debt", "--db", db_str])
        assert debt_res.exit_code == 0

        # 10. Stats
        stats_res = cli_runner.invoke(app, ["stats", "--db", db_str])
        assert stats_res.exit_code == 0

    def test_cli_serve_mocked(self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        """'mnemo serve' runs server entrypoint."""
        ran = False

        def mock_run_server() -> None:
            nonlocal ran
            ran = True

        monkeypatch.setattr("mnemo.mcp.server.run_server", mock_run_server)
        res = cli_runner.invoke(app, ["serve"])
        assert res.exit_code == 0
        assert ran is True
