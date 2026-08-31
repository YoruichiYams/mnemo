"""Token-Optimized Object Notation (TOON).

TOON is a super-compact text serialisation format designed to reduce LLM
context-window token consumption by 40-60 % compared to JSON.

Format rules:
  - Records are separated by newlines (``\\n``).
  - Fields within a record are separated by ``|``.
  - Each field is ``key:value``.
  - Short single/two-character key aliases are used for common fields.
  - Null / default values are omitted entirely.
  - Numbers use shortest representation (no trailing zeros).
  - Special characters (``|``, ``:``, quotes, newlines) are escaped and quoted.
  - Lists use comma-separated values inside ``[…]``.

Key alias table for Facts:
    i  -> id               x  -> text            c  -> category
    s  -> salience         a  -> access_count    T  -> tier
    vs -> valid_start      ve -> valid_end
    is -> ingest_start     ie -> ingest_end
    la -> last_accessed_at

Key alias table for SearchResults:
    f  -> fact (nested TOON)   sc -> score   ch -> channel

Key alias table for DebtLedgerItems:
    i  -> id   cl -> ceiling   tr -> trigger
    cx -> code_context         ca -> created_at
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Key alias mappings
# ---------------------------------------------------------------------------

_FACT_TO_SHORT: dict[str, str] = {
    "id": "i",
    "text": "x",
    "category": "c",
    "salience": "s",
    "access_count": "a",
    "tier": "T",
    "last_accessed_at": "la",
    "valid_start": "vs",
    "valid_end": "ve",
    "ingest_start": "is",
    "ingest_end": "ie",
}

_FACT_FROM_SHORT: dict[str, str] = {v: k for k, v in _FACT_TO_SHORT.items()}

_SEARCH_TO_SHORT: dict[str, str] = {
    "score": "sc",
    "channel": "ch",
}

_SEARCH_FROM_SHORT: dict[str, str] = {v: k for k, v in _SEARCH_TO_SHORT.items()}

_DEBT_TO_SHORT: dict[str, str] = {
    "id": "i",
    "ceiling": "cl",
    "trigger": "tr",
    "code_context": "cx",
    "created_at": "ca",
}

_DEBT_FROM_SHORT: dict[str, str] = {v: k for k, v in _DEBT_TO_SHORT.items()}

# Tier short codes
_TIER_SHORT: dict[str, str] = {
    "core": "C",
    "working": "W",
    "peripheral": "P",
    "archived": "A",
}
_TIER_LONG: dict[str, str] = {v: k for k, v in _TIER_SHORT.items()}


# ---------------------------------------------------------------------------
# Escaping & Value formatting
# ---------------------------------------------------------------------------


def _needs_escaping(s: str) -> bool:
    """Check if string contains characters requiring quoting/escaping."""
    if not s:
        return False
    if s.startswith('"') or s.endswith('"'):
        return True
    if s.startswith("[") or s.startswith("("):
        return True
    if s[0].isspace() or s[-1].isspace():
        return True
    return any(c in s for c in ("|", ":", '"', "\n", "\r", "\t", "\\"))


def _escape_str(s: str) -> str:
    """Escape special characters in string and wrap in quotes if needed."""
    if not _needs_escaping(s):
        return s

    escaped = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _unescape_str(s: str) -> str:
    """Unescape a quoted or unquoted TOON string value."""
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        inner = s[1:-1]
        res: list[str] = []
        i = 0
        n = len(inner)
        while i < n:
            if inner[i] == "\\" and i + 1 < n:
                nxt = inner[i + 1]
                if nxt == "n":
                    res.append("\n")
                elif nxt == "r":
                    res.append("\r")
                elif nxt == "t":
                    res.append("\t")
                elif nxt == '"':
                    res.append('"')
                elif nxt == "\\":
                    res.append("\\")
                elif nxt == "|":
                    res.append("|")
                elif nxt == ":":
                    res.append(":")
                else:
                    res.append(nxt)
                i += 2
            else:
                res.append(inner[i])
                i += 1
        return "".join(res)
    return s


def _fmt_val(value: Any) -> str:
    """Format a value as the shortest TOON string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            # Drop trailing zeros: 0.9000 -> 0.9 ; 1.0 -> 1
            s = f"{value:.6f}".rstrip("0").rstrip(".")
            return s
        return str(value)
    if isinstance(value, list):
        inner = ",".join(_fmt_val(v) for v in value)
        return f"[{inner}]"
    if isinstance(value, str):
        return _escape_str(value)
    return _escape_str(str(value))


# ---------------------------------------------------------------------------
# Tokenizer / Scanner
# ---------------------------------------------------------------------------


def _split_toon_fields(line: str) -> list[str]:
    """Split a TOON line into fields by '|', respecting quotes, escapes, and nested parens/brackets."""
    fields: list[str] = []
    cur: list[str] = []
    in_quotes = False
    escape = False
    paren_depth = 0
    bracket_depth = 0

    for char in line:
        if escape:
            cur.append(char)
            escape = False
            continue

        if char == "\\":
            cur.append(char)
            escape = True
            continue

        if char == '"':
            in_quotes = not in_quotes
            cur.append(char)
            continue

        if not in_quotes:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char == "|" and paren_depth == 0 and bracket_depth == 0:
                fields.append("".join(cur))
                cur = []
                continue

        cur.append(char)

    if cur or fields:
        fields.append("".join(cur))

    return fields


def _split_key_val(field: str) -> tuple[str, str]:
    """Split a single TOON field into key and value by the first unquoted, unnested ':'."""
    in_quotes = False
    escape = False
    paren_depth = 0
    bracket_depth = 0
    sep_idx = -1

    for idx, char in enumerate(field):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_quotes = not in_quotes
            continue
        if not in_quotes:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char == ":" and paren_depth == 0 and bracket_depth == 0:
                sep_idx = idx
                break

    if sep_idx == -1:
        return field.strip(), ""
    key = field[:sep_idx].strip()
    val = field[sep_idx + 1 :].strip()
    return key, val


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


def encode_fact(fact_dict: dict[str, Any]) -> str:
    """Encode a single Fact dictionary into a single-line TOON string.

    Args:
        fact_dict: Fact model dumped to dict (e.g. ``fact.model_dump()``).

    Returns:
        Single-line TOON string.
    """
    parts: list[str] = []
    for long_key, short_key in _FACT_TO_SHORT.items():
        val = fact_dict.get(long_key)
        if val is None:
            continue
        # Skip defaults to save tokens
        if long_key == "access_count" and val == 0:
            continue
        if long_key == "category" and val == "general":
            continue
        # Special: tier short code
        if long_key == "tier":
            val = _TIER_SHORT.get(str(val), str(val))
        formatted = _fmt_val(val)
        if not formatted:
            continue
        parts.append(f"{short_key}:{formatted}")
    return "|".join(parts)


def encode_search_result(result_dict: dict[str, Any]) -> str:
    """Encode a SearchResult dict into a single-line TOON string."""
    parts: list[str] = []
    for long_key, short_key in _SEARCH_TO_SHORT.items():
        val = result_dict.get(long_key)
        if val is not None:
            parts.append(f"{short_key}:{_fmt_val(val)}")

    fact_data = result_dict.get("fact")
    if fact_data is not None:
        fact_dict = fact_data if isinstance(fact_data, dict) else fact_data.model_dump()
        fact_toon = encode_fact(fact_dict)
        parts.append(f"f:({fact_toon})")

    return "|".join(parts)


def encode_debt(debt_dict: dict[str, Any]) -> str:
    """Encode a DebtLedgerItem dict into a single-line TOON string."""
    parts: list[str] = []
    for long_key, short_key in _DEBT_TO_SHORT.items():
        val = debt_dict.get(long_key)
        if val is None or (long_key == "code_context" and val == ""):
            continue
        formatted = _fmt_val(val)
        parts.append(f"{short_key}:{formatted}")
    return "|".join(parts)


def encode_facts(fact_dicts: list[dict[str, Any]]) -> str:
    """Encode a list of Fact dicts into multi-line TOON (one fact per line)."""
    return "\n".join(encode_fact(f) for f in fact_dicts)


def encode_search_results(results: list[dict[str, Any]]) -> str:
    """Encode a list of SearchResult dicts into multi-line TOON."""
    return "\n".join(encode_search_result(r) for r in results)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def _parse_value(raw: str) -> Any:
    """Parse a TOON value string back to a Python type."""
    if not raw:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1]
        if not inner:
            return []
        return [_parse_value(v.strip()) for v in inner.split(",")]

    unescaped = _unescape_str(raw)
    # If it was a quoted string, return as string
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return unescaped

    # Try numeric conversions
    try:
        if "." in unescaped:
            return float(unescaped)
        return int(unescaped)
    except ValueError:
        return unescaped


def decode_fact(line: str) -> dict[str, Any]:
    """Decode a single TOON line into a Fact-compatible dict.

    Handles escaped quotes, colons, pipes, and newlines safely.

    Args:
        line: A single TOON-encoded line.

    Returns:
        Dictionary with long key names, suitable for ``Fact(**d)``.
    """
    result: dict[str, Any] = {}
    fields = _split_toon_fields(line.strip())

    for field in fields:
        if not field:
            continue
        key, val_raw = _split_key_val(field)
        if not key:
            continue

        long_key = _FACT_FROM_SHORT.get(key, key)
        value = _parse_value(val_raw)

        # Expand tier short code
        if long_key == "tier" and isinstance(value, str):
            value = _TIER_LONG.get(value, value)

        result[long_key] = value

    return result


def decode_search_result(line: str) -> dict[str, Any]:
    """Decode a single TOON line into a SearchResult-compatible dict."""
    result: dict[str, Any] = {}
    fields = _split_toon_fields(line.strip())

    for field in fields:
        if not field:
            continue
        key, val_raw = _split_key_val(field)
        if not key:
            continue

        if key == "f":
            # Nested fact in parens: f:(...)
            inner = val_raw
            if inner.startswith("(") and inner.endswith(")"):
                inner = inner[1:-1]
            result["fact"] = decode_fact(inner)
        else:
            long_key = _SEARCH_FROM_SHORT.get(key, key)
            result[long_key] = _parse_value(val_raw)

    return result


def decode_facts(text: str) -> list[dict[str, Any]]:
    """Decode multi-line TOON into a list of Fact dicts."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return [decode_fact(ln) for ln in lines]
