"""Unit tests for FastMCP tools, resources, and prompt templates."""

from __future__ import annotations

import uuid

from mnemo.mcp.prompts import architectural_audit
from mnemo.mcp.resources import active_context_resource, core_identity_resource
from mnemo.mcp.tools import (
    mnemo_get_debt_ledger,
    mnemo_inspect,
    mnemo_invalidate,
    mnemo_reinforce,
    mnemo_remember,
    mnemo_search,
)


class TestMCPToolsAndResources:
    """Test suite for FastMCP server tool functions and resource endpoints."""

    def test_mcp_remember_search_and_inspect_tools(self) -> None:
        """'mnemo_remember' stores facts and 'mnemo_search' returns TOON-encoded hits."""
        unique_token = f"mcp_unique_{uuid.uuid4().hex[:8]}"
        fact_text = f"Validation fact with token {unique_token}"

        # 1. Remember
        rem_res = mnemo_remember(fact_text, category="testing", force_op="add")
        assert "[ADD]" in rem_res
        assert unique_token in rem_res

        # 2. Search
        search_res = mnemo_search(unique_token, limit=5)
        assert "sc:" in search_res
        assert "ch:rrf" in search_res
        assert unique_token in search_res

        # 3. Duplicate remember -> NOOP
        noop_res = mnemo_remember(fact_text, category="testing")
        assert "[NOOP]" in noop_res

        # 4. Search query format
        search_fmt = mnemo_search("query test")
        assert isinstance(search_fmt, str)

        # 5. Invalid force_op
        err_op = mnemo_remember("Some text", force_op="invalid_op")
        assert "[ERROR]" in err_op
        assert "invalid force_op" in err_op

        # 6. Invalid min_tier
        err_tier = mnemo_search("Some query", min_tier="invalid_tier")
        assert "[ERROR]" in err_tier
        assert "invalid min_tier" in err_tier

        # 7. Force update with non-existent text -> fallback to add
        fresh_token = f"fresh_{uuid.uuid4().hex[:8]}"
        fallback_res = mnemo_remember(f"Brand new fact {fresh_token}", force_op="update")
        assert "[ADD]" in fallback_res
        assert "fallback" in fallback_res

    def test_mcp_invalidate_and_reinforce_tools(self) -> None:
        """'mnemo_invalidate' and 'mnemo_reinforce' execute cleanly."""
        unique_token = f"inval_{uuid.uuid4().hex[:8]}"
        fact_text = f"Fact to reinforce and invalidate {unique_token}"

        rem_res = mnemo_remember(fact_text, category="temp", force_op="add")
        assert "[ADD]" in rem_res

        # Search to get id
        search_res = mnemo_search(unique_token)
        assert "f:(" in search_res

        # Parse ID from search result line (e.g. i:...)
        import re

        match = re.search(r"i:([a-f0-9-]+)", search_res)
        assert match is not None
        fact_id = match.group(1)

        # Inspect
        inspect_res = mnemo_inspect(fact_id)
        assert unique_token in inspect_res

        # Inspect non-existent
        inspect_none = mnemo_inspect(str(uuid.uuid4()))
        assert "[NOT_FOUND]" in inspect_none

        # Reinforce
        reinforce_res = mnemo_reinforce(fact_id, boost=0.15)
        assert "[REINFORCED]" in reinforce_res

        # Invalidate
        inval_res = mnemo_invalidate(fact_id)
        assert "[DEL]" in inval_res

    def test_mcp_debt_ledger_tool(self) -> None:
        """'mnemo_get_debt_ledger' tool returns ledger report."""
        res = mnemo_get_debt_ledger()
        assert isinstance(res, str)
        assert len(res) > 0

    def test_mcp_resources(self) -> None:
        """Memory resources return non-empty TOON responses."""
        core_res = core_identity_resource()
        assert isinstance(core_res, str)

        active_res = active_context_resource()
        assert isinstance(active_res, str)

    def test_mcp_prompts(self) -> None:
        """Prompt template 'architectural_audit' generates structured prompt."""
        prompt = architectural_audit()
        assert "architectural audit" in prompt
        assert "mnemo_get_debt_ledger" in prompt
        assert "Tier Distribution" in prompt
