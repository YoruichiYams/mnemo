"""FastMCP prompt templates for Mnemo."""

from __future__ import annotations

from mnemo.mcp.server import mcp_app


@mcp_app.prompt(
    name="architectural_audit",
    description=(
        "Generate a comprehensive architectural audit of the current memory store. "
        "Analyses tier distribution, debt items, stale facts, and coverage gaps."
    ),
)
def architectural_audit() -> str:
    """Return a structured prompt for the LLM to perform an architectural audit."""
    return (
        "You are performing an architectural audit of the Mnemo memory store.\n"
        "\n"
        "Steps:\n"
        "1. Call `mnemo_get_debt_ledger` to retrieve all current debt items.\n"
        "2. Call `mnemo_search` with query='architecture' and min_tier='core' "
        "to find all architectural rules.\n"
        "3. Read the `memory://core-identity` resource for permanent facts.\n"
        "4. Read the `memory://active-context` resource for current working context.\n"
        "\n"
        "Analysis required:\n"
        "- List all Core-tier facts and verify they are still accurate.\n"
        "- Identify any Working-tier facts at risk of demotion (salience < 0.75).\n"
        "- Flag contradictions between facts.\n"
        "- Identify orphaned entities with no relations.\n"
        "- Suggest facts that should be promoted to Core tier.\n"
        "- Suggest stale facts that should be invalidated.\n"
        "\n"
        "Format your response as a structured report with sections:\n"
        "## Tier Distribution\n"
        "## Debt Items\n"
        "## Contradictions\n"
        "## Recommendations\n"
    )
