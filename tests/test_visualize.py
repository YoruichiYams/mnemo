"""Unit tests for the knowledge Sankey visualization module."""

from __future__ import annotations

from mnemo.cli.visualize import (
    export_knowledge_sankey_data,
    generate_html_report,
    generate_mermaid_sankey,
)
from mnemo.storage.connection import Database


class TestVisualizeModule:
    """Test suite for Sankey data export, HTML report generator, and Mermaid syntax."""

    def test_export_knowledge_sankey_data_populated(self, populated_db: Database) -> None:
        """Export sankey data from populated database contains 2-level flows."""
        with populated_db.session() as conn:
            data = export_knowledge_sankey_data(conn)

        assert "nodes" in data
        assert "node_x" in data
        assert "node_y" in data
        assert "sources" in data
        assert "targets" in data
        assert "values" in data
        assert "metrics" in data

        # Check nodes have categories, tiers, and entities
        node_names = data["nodes"]
        assert any(n.startswith("cat:") for n in node_names)
        assert any(n in ("Core", "Working", "Peripheral") for n in node_names)
        assert any(n.startswith("entity:") for n in node_names)

        # Check strict column coordinates
        for name, x in zip(data["nodes"], data["node_x"], strict=True):
            if name.startswith("cat:"):
                assert x == 0.01
            elif name in ("Core", "Working", "Peripheral", "Archived"):
                assert x == 0.50
            else:
                assert x == 0.99

        # Check metrics
        metrics = data["metrics"]
        assert metrics["total_facts"] > 0
        assert metrics["total_entities"] > 0
        assert metrics["core_pct"] > 0

    def test_export_knowledge_sankey_data_empty_db(self, in_memory_db: Database) -> None:
        """Empty database produces valid fallback node flows without crashing."""
        with in_memory_db.session() as conn:
            data = export_knowledge_sankey_data(conn)

        assert len(data["nodes"]) >= 2
        assert len(data["sources"]) >= 1
        assert len(data["values"]) >= 1

    def test_generate_mermaid_sankey(self, populated_db: Database) -> None:
        """Mermaid sankey generator outputs valid sankey-beta syntax."""
        with populated_db.session() as conn:
            data = export_knowledge_sankey_data(conn)

        mermaid_str = generate_mermaid_sankey(data)
        assert mermaid_str.startswith("sankey-beta")
        assert len(mermaid_str.splitlines()) > 2

    def test_generate_html_report(self, populated_db: Database) -> None:
        """HTML report generates complete Dark Minimal page with embedded JS and Plotly."""
        with populated_db.session() as conn:
            data = export_knowledge_sankey_data(conn)

        html = generate_html_report(data, project_name="Test Project")
        assert "<!DOCTYPE html>" in html
        assert "Test Project - Knowledge Flow" in html
        assert "#050507" in html  # Dark minimal deep black background
        assert "#0b0c0e" in html  # Card background
        assert "Google Sans Flex" in html
        assert "Plotly.newPlot" in html
        assert "sankey-plot" in html
        assert "fixed" in html
