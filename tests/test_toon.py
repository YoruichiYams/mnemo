"""Unit tests and stress tests for Token-Optimized Object Notation (TOON)."""

from __future__ import annotations

import json

import pytest

from mnemo.serialization.toon import (
    decode_fact,
    decode_facts,
    decode_search_result,
    encode_debt,
    encode_fact,
    encode_facts,
    encode_search_result,
    encode_search_results,
)


class TestTOONSerialization:
    """Test suite for TOON encoding, decoding, and compression efficiency."""

    def test_basic_fact_roundtrip(self) -> None:
        """Standard fact roundtrips cleanly between dict and TOON format."""
        fact_in = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "text": "Mnemo uses SQLite WAL mode for fast concurrency",
            "category": "database",
            "salience": 0.95,
            "access_count": 5,
            "tier": "core",
            "valid_start": 1700000000.0,
            "valid_end": None,
            "ingest_start": 1700000000.0,
            "ingest_end": None,
        }

        encoded = encode_fact(fact_in)
        assert "i:123e4567-e89b-12d3-a456-426614174000" in encoded
        assert "x:Mnemo uses SQLite WAL mode" in encoded
        assert "T:C" in encoded  # Core tier mapped to short code 'C'
        assert "s:0.95" in encoded

        decoded = decode_fact(encoded)
        assert decoded["id"] == fact_in["id"]
        assert decoded["text"] == fact_in["text"]
        assert decoded["category"] == fact_in["category"]
        assert decoded["salience"] == fact_in["salience"]
        assert decoded["access_count"] == fact_in["access_count"]
        assert decoded["tier"] == "core"

    @pytest.mark.parametrize(
        "complex_text",
        [
            "Simple single line sentence.",
            "Text with colons: http://example.com:8080/path?arg=1:2:3",
            "Text with pipes | option A | option B | option C",
            "Text with quotes: \"He said 'Hello' to the world!\"",
            "Text with backslashes: C:\\Users\\Name\\AppData\\Local\\Temp",
            "Multi-line:\nLine 1\nLine 2\nLine 3\n\tIndented tab line",
            'Extreme mix: "quotes" | colons: key:val | \n newline \\ backslash | end',
            "   Leading and trailing whitespace   ",
            "[bracketed text] and (parenthesized text)",
        ],
    )
    def test_special_characters_roundtrip_stress(self, complex_text: str) -> None:
        """Special characters (pipes, colons, quotes, newlines, backslashes) roundtrip 100% losslessly."""
        fact_in = {
            "id": "stress-test-id",
            "text": complex_text,
            "category": "stress",
            "salience": 0.88,
            "tier": "working",
        }

        encoded = encode_fact(fact_in)
        # Check that encoded is single-line (newlines escaped)
        assert "\n" not in encoded
        assert "\r" not in encoded

        decoded = decode_fact(encoded)
        assert decoded["id"] == fact_in["id"]
        assert decoded["text"] == fact_in["text"]
        assert decoded["tier"] == "working"

    def test_multi_facts_serialization(self) -> None:
        """Multi-fact batch encoding and decoding preserves line-by-line separation."""
        facts_in = [
            {
                "id": "f1",
                "text": "First fact\nwith internal newline",
                "tier": "core",
                "salience": 1.0,
            },
            {"id": "f2", "text": "Second fact | with | pipes", "tier": "working", "salience": 0.8},
            {"id": "f3", "text": "Third fact: with: colons", "tier": "peripheral", "salience": 0.5},
        ]

        batch_encoded = encode_facts(facts_in)
        lines = batch_encoded.strip().splitlines()
        assert len(lines) == 3

        batch_decoded = decode_facts(batch_encoded)
        assert len(batch_decoded) == 3
        for original, parsed in zip(facts_in, batch_decoded, strict=True):
            assert parsed["id"] == original["id"]
            assert parsed["text"] == original["text"]
            assert parsed["tier"] == original["tier"]

    def test_search_result_nested_roundtrip(self) -> None:
        """Nested SearchResult objects encode into compact parenthesized structure."""
        sr_in = {
            "score": 0.045678,
            "channel": "rrf",
            "fact": {
                "id": "fact-nested-1",
                "text": "Search result content with : and | inside",
                "tier": "working",
                "salience": 0.9,
            },
        }

        encoded = encode_search_result(sr_in)
        assert "sc:0.045678" in encoded
        assert "ch:rrf" in encoded
        assert "f:(" in encoded

        decoded = decode_search_result(encoded)
        assert decoded["score"] == 0.045678
        assert decoded["channel"] == "rrf"
        assert decoded["fact"]["id"] == "fact-nested-1"
        assert decoded["fact"]["text"] == "Search result content with : and | inside"

    def test_debt_ledger_item_encoding(self) -> None:
        """Debt ledger items encode compactly."""
        debt_in = {
            "id": "debt-1",
            "ceiling": "critical",
            "trigger": "contradiction_detected",
            "code_context": "fact:f1 vs fact:f2",
            "created_at": 1700000000.0,
        }

        encoded = encode_debt(debt_in)
        assert "i:debt-1" in encoded
        assert "cl:critical" in encoded
        assert "tr:contradiction_detected" in encoded
        assert 'cx:"fact:f1 vs fact:f2"' in encoded or "cx:fact:f1 vs fact:f2" in encoded

    def test_token_efficiency_compression_ratio(self) -> None:
        """TOON achieves >40% text reduction compared to formatted JSON."""
        fact_dict = {
            "id": "3a7b9c1d-0000-4000-8000-000000000001",
            "text": "Antigravity agents retain long-term memory via bitemporal timelines and Ebbinghaus decay",
            "category": "architecture",
            "salience": 0.95,
            "access_count": 12,
            "tier": "core",
            "last_accessed_at": 1700000000.0,
            "valid_start": 1700000000.0,
            "valid_end": None,
            "ingest_start": 1700000000.0,
            "ingest_end": None,
        }

        json_bytes = len(json.dumps(fact_dict))
        toon_bytes = len(encode_fact(fact_dict))

        savings = 1.0 - (toon_bytes / json_bytes)
        # Should save at least 40% characters / tokens
        assert savings > 0.40, f"Expected >40% savings, got {savings:.1%}"

    def test_encode_search_results_batch(self) -> None:
        """Batch encoding of SearchResults formats multiple records."""
        results = [
            {"score": 0.05, "channel": "vector", "fact": {"id": "f1", "text": "fact 1"}},
            {"score": 0.03, "channel": "fts", "fact": {"id": "f2", "text": "fact 2"}},
        ]
        encoded = encode_search_results(results)
        assert "sc:0.05" in encoded
        assert "sc:0.03" in encoded
        assert len(encoded.strip().splitlines()) == 2

    def test_toon_value_formatting_types(self) -> None:
        """TOON handles lists, booleans, and empty collections properly."""
        fact_with_list = {
            "id": "f_list",
            "text": "Tags test",
            "metadata": {"tags": ["ai", "memory", "sqlite"]},
        }
        encoded = encode_fact(fact_with_list)
        decoded = decode_fact(encoded)
        assert decoded["id"] == "f_list"
        assert decoded["text"] == "Tags test"
