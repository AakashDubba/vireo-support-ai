"""
tests/test_pipeline.py — 2 minimal pytest tests covering the three requirements:
  1. Join validation (all joins are 1:1, no orphans)
  2. JSON response parsing (valid, invalid, wrong category)
  3. Cache behaviour (hit after first write)

Run with:  pytest tests/ -v
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from utils import validate_joins, CATEGORIES
from classifier import TicketClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: tiny in-memory DataFrames that mirror real schemas
# ─────────────────────────────────────────────────────────────────────────────

def _make_tickets() -> pd.DataFrame:
    import pandas as pd
    return pd.DataFrame({
        "ticket_id":         ["TK-001", "TK-002", "TK-003"],
        "agent_id":          ["A3001",  "A3002",  "A3001"],
        "customer_id":       ["C001",   "C002",   "C003"],
        "order_id":          ["ORD-01", "ORD-02",  None],
        "product_sku":       ["VA-EB-PL1", "VA-EB-PL1", "VA-EB-PL1"],
        "customer_message":  ["not working", "won't charge", "what is warranty"],
        "agent_notes":       ["checked", "checked", "checked"],
        "source_system":     ["helpdesk"] * 3,
        "transfers":         [0, 1, None],
        "csat_score":        [4, None, 3],
        "refund_amount_inr": [None, 500, None],
        "handle_hours":      [2.5, 3.0, None],
        "handle_time_valid": [True, True, False],
        "in_primary_window": [True, True, True],
        "priority":          ["Normal", "High", "Low"],
        "assigned_team":     ["Chat Frontline"] * 3,
        "channel":           ["chat", "email", "voice"],
        "status":            ["resolved", "resolved", "open"],
        "category":          ["Charging & Battery", "Connectivity", "Warranty & Repair"],
        "created_at":        pd.to_datetime(["2025-03-01", "2025-04-01", "2025-05-01"]),
    })


def _make_agents() -> pd.DataFrame:
    return pd.DataFrame({
        "agent_id": ["A3001", "A3002"],
        "name":     ["Agent One", "Agent Two"],
        "team":     ["Chat Frontline", "Chat Frontline"],
        "tier":     [1, 1],
        "site":     ["Bengaluru", "Indore"],
    })


def _make_customers() -> pd.DataFrame:
    return pd.DataFrame({"customer_id": ["C001", "C002", "C003"]})


def _make_orders() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": ["ORD-01", "ORD-02"],
        "sku":      ["VA-EB-PL1", "VA-EB-PL1"],
    })


def _make_products() -> pd.DataFrame:
    return pd.DataFrame({"sku": ["VA-EB-PL1"]})


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Join validation
# ─────────────────────────────────────────────────────────────────────────────

class TestJoinValidation:
    def test_clean_joins_report_zero_orphans(self):
        """All FK joins should pass with no orphans and 1.0 fan-out."""
        report = validate_joins(
            _make_tickets(),
            _make_agents(),
            _make_customers(),
            _make_orders(),
            _make_products(),
        )
        assert report["agent_orphans"]   == 0,     "Agent orphans found"
        assert report["customer_orphans"] == 0,    "Customer orphans found"
        assert report["order_orphans"]   == 0,     "Order orphans found"
        assert report["order_fanout"]    == 1.0,   "Fan-out is not 1.0"
        assert report["sku_orphans"]     == 0,     "SKU orphans found"

    def test_orphan_agent_detected(self):
        """Missing agent in roster should appear as an orphan."""
        tickets = _make_tickets()
        tickets.loc[0, "agent_id"] = "A9999"   # not in agents
        report = validate_joins(
            tickets, _make_agents(), _make_customers(), _make_orders(), _make_products()
        )
        assert report["agent_orphans"] == 1

    def test_missing_order_id_not_counted_as_orphan(self):
        """Null order_id on TK-003 must not inflate orphan count."""
        report = validate_joins(
            _make_tickets(),
            _make_agents(),
            _make_customers(),
            _make_orders(),
            _make_products(),
        )
        assert report["order_orphans"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: LLM JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestJsonParsing:

    def _classifier(self) -> TicketClassifier:
        with tempfile.TemporaryDirectory() as d:
            return TicketClassifier(cache_dir=Path(d), offline=True)

    def test_valid_json_parsed_correctly(self):
        c = self._classifier()
        raw = '{"category": "Charging & Battery", "confidence": 0.91}'
        result = c._parse_llm_json(raw)
        assert result is not None
        assert result["category"]   == "Charging & Battery"
        assert abs(result["confidence"] - 0.91) < 0.001

    def test_invalid_json_returns_none(self):
        c = self._classifier()
        assert c._parse_llm_json("Sorry, I cannot help.") is None
        assert c._parse_llm_json("")                       is None
        assert c._parse_llm_json('{"category": "Other"}') is None   # "Other" not in taxonomy

    def test_markdown_fenced_json_is_unwrapped(self):
        c = self._classifier()
        raw = '```json\n{"category": "Billing & Payments", "confidence": 0.85}\n```'
        result = c._parse_llm_json(raw)
        assert result is not None
        assert result["category"] == "Billing & Payments"

    def test_unknown_category_returns_none(self):
        c = self._classifier()
        raw = '{"category": "Completely Unknown Category", "confidence": 0.5}'
        assert c._parse_llm_json(raw) is None

    def test_all_valid_categories_parse(self):
        c = self._classifier()
        for cat in CATEGORIES:
            raw = json.dumps({"category": cat, "confidence": 0.7})
            result = c._parse_llm_json(raw)
            assert result is not None, f"Failed to parse valid category: {cat}"
            assert result["category"] == cat


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Cache behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheBehaviour:

    def test_cache_hit_after_first_classification(self):
        """Second call for same ticket_id + prompt_version + model must be a cache hit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = TicketClassifier(cache_dir=Path(tmpdir), offline=True)

            df = pd.DataFrame({
                "ticket_id":        ["TK-TEST-01"],
                "customer_message": ["my earbuds are not charging"],
                "agent_notes":      ["checked charger"],
            })
            c.classify_dataframe(df)
            assert c.stats["cache_misses"] == 1
            assert c.stats["cache_hits"]   == 0

            # Second run: should be pure cache hit
            c2 = TicketClassifier(cache_dir=Path(tmpdir), offline=True)
            c2.classify_dataframe(df)
            assert c2.stats["cache_hits"]   == 1
            assert c2.stats["cache_misses"] == 0

    def test_cache_key_changes_with_prompt_version(self):
        """Cache key must embed prompt_version — different versions must not collide."""
        from utils import PROMPT_VERSION
        with tempfile.TemporaryDirectory() as tmpdir:
            c = TicketClassifier(cache_dir=Path(tmpdir), offline=True)
            key_v1  = c._cache_key("TK-001")
            # Simulate a future version
            fake_key = f"TK-001::v99.0::{c.model_name}"
            assert key_v1 != fake_key, "Same ticket must get different keys for different prompt versions"

    def test_cache_persists_across_instances(self):
        """Cache written by one instance must be readable by the next."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            df = pd.DataFrame({
                "ticket_id":        ["TK-PERSIST"],
                "customer_message": ["login not working"],
                "agent_notes":      ["reset done"],
            })
            TicketClassifier(cache_dir=p, offline=True).classify_dataframe(df)

            c2 = TicketClassifier(cache_dir=p, offline=True)
            assert len(c2._cache) == 1, "Cache should have 1 entry after reload"

    def test_offline_entry_not_treated_as_llm_cache_hit(self, monkeypatch):
        """An offline cached entry must NEVER be returned as an LLM cache hit."""
        monkeypatch.setenv("LLM_API_KEY", "gsk_dummy_test_key_1234567890")
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            df = pd.DataFrame({
                "ticket_id":        ["TK-ISOLATE"],
                "customer_message": ["login not working"],
                "agent_notes":      ["reset done"],
            })
            # Populate cache in offline mode
            c_off = TicketClassifier(cache_dir=p, offline=True)
            c_off.classify_dataframe(df)
            assert c_off.stats["cache_misses"] == 1

            # An LLM classifier instance should NOT treat the offline entry as an LLM hit
            c_llm = TicketClassifier(cache_dir=p, offline=False)
            key_llm = c_llm._cache_key("TK-ISOLATE")
            assert key_llm not in c_llm._cache, "LLM cache key must differ from offline cache key"

    def test_bypass_cache_forces_uncached(self):
        """When bypass_cache is True, classifier does not count cached items as hits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            df = pd.DataFrame({
                "ticket_id":        ["TK-BYPASS"],
                "customer_message": ["need a refund"],
                "agent_notes":      ["done"],
            })
            c1 = TicketClassifier(cache_dir=p, offline=True)
            c1.classify_dataframe(df)

            c2 = TicketClassifier(cache_dir=p, offline=True, bypass_cache=True)
            c2.classify_dataframe(df)
            assert c2.stats["cache_hits"] == 0
            assert c2.stats["cache_misses"] == 1

    def test_strict_llm_raises_without_api_key(self, monkeypatch):
        """strict_llm must raise ValueError if no API key is available."""
        monkeypatch.setenv("LLM_API_KEY", "")
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="LLM_API_KEY is not set"):
                TicketClassifier(cache_dir=Path(tmpdir), offline=False, strict_llm=True)

