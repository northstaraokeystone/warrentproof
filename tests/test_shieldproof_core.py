"""
ShieldProof v2.0 Core Module Tests

Tests for dual_hash, emit_receipt, merkle.
"""

import json
import pytest
import sys
from pathlib import Path

from src.shieldproof.core import (
    dual_hash,
    emit_receipt,
    merkle,
    load_ledger,
    query_receipts,
    clear_ledger,
    StopRule,
    TENANT_ID,
    RECEIPT_TYPES,
    MILESTONE_STATES,
    VERSION,
)


@pytest.fixture(autouse=True)
def clean_ledger():
    """Clear ledger before and after each test."""
    clear_ledger()
    yield
    clear_ledger()


class TestDualHash:
    """Tests for dual_hash function."""

    def test_dual_hash_string(self):
        """dual_hash should accept string input."""
        h = dual_hash("test")
        assert ":" in h
        assert len(h) == 129  # 64 + 1 + 64

    def test_dual_hash_bytes(self):
        """dual_hash should accept bytes input."""
        h = dual_hash(b"test")
        assert ":" in h
        assert len(h) == 129

    def test_dual_hash_deterministic(self):
        """dual_hash should be deterministic."""
        h1 = dual_hash("test")
        h2 = dual_hash("test")
        assert h1 == h2

    def test_dual_hash_different_inputs(self):
        """dual_hash should produce different outputs for different inputs."""
        h1 = dual_hash("test1")
        h2 = dual_hash("test2")
        assert h1 != h2

    def test_dual_hash_empty(self):
        """dual_hash should handle empty input."""
        h = dual_hash("")
        assert ":" in h


class TestEmitReceipt:
    """Tests for emit_receipt function."""

    def test_emit_receipt_structure(self):
        """emit_receipt should include required fields."""
        r = emit_receipt("test", {"key": "value"}, to_stdout=False)
        assert "receipt_type" in r
        assert "ts" in r
        assert "tenant_id" in r
        assert "payload_hash" in r

    def test_emit_receipt_type(self):
        """emit_receipt should set correct receipt_type."""
        r = emit_receipt("contract", {"data": "test"}, to_stdout=False)
        assert r["receipt_type"] == "contract"

    def test_emit_receipt_tenant_id(self):
        """emit_receipt should use correct tenant_id."""
        r = emit_receipt("test", {}, to_stdout=False)
        assert r["tenant_id"] == TENANT_ID

    def test_emit_receipt_payload_hash(self):
        """emit_receipt should include payload_hash."""
        r = emit_receipt("test", {"data": "test"}, to_stdout=False)
        assert ":" in r["payload_hash"]

    def test_emit_receipt_data_included(self):
        """emit_receipt should include data fields."""
        r = emit_receipt("test", {"custom_field": "custom_value"}, to_stdout=False)
        assert r["custom_field"] == "custom_value"

    def test_emit_receipt_to_ledger(self):
        """emit_receipt should append to ledger file."""
        emit_receipt("test", {"test": "ledger"}, to_stdout=False)
        receipts = load_ledger()
        assert len(receipts) >= 1
        assert receipts[-1]["test"] == "ledger"


class TestMerkle:
    """Tests for merkle function."""

    def test_merkle_empty(self):
        """merkle should handle empty list."""
        m = merkle([])
        assert ":" in m

    def test_merkle_single(self):
        """merkle should handle single item."""
        m = merkle([{"a": 1}])
        assert ":" in m

    def test_merkle_multiple(self):
        """merkle should handle multiple items."""
        m = merkle([{"a": 1}, {"b": 2}, {"c": 3}])
        assert ":" in m

    def test_merkle_deterministic(self):
        """merkle should be deterministic."""
        items = [{"a": 1}, {"b": 2}]
        m1 = merkle(items)
        m2 = merkle(items)
        assert m1 == m2

    def test_merkle_order_matters(self):
        """merkle should produce different results for different order."""
        m1 = merkle([{"a": 1}, {"b": 2}])
        m2 = merkle([{"b": 2}, {"a": 1}])
        assert m1 != m2


class TestQueryReceipts:
    """Tests for query_receipts function."""

    def test_query_receipts_empty(self):
        """query_receipts should return empty list for empty ledger."""
        receipts = query_receipts()
        assert receipts == []

    def test_query_receipts_filter_type(self):
        """query_receipts should filter by receipt_type."""
        emit_receipt("type_a", {"data": 1}, to_stdout=False)
        emit_receipt("type_b", {"data": 2}, to_stdout=False)
        emit_receipt("type_a", {"data": 3}, to_stdout=False)

        type_a = query_receipts("type_a")
        assert len(type_a) == 2
        assert all(r["receipt_type"] == "type_a" for r in type_a)

    def test_query_receipts_filter_field(self):
        """query_receipts should filter by field value."""
        emit_receipt("test", {"key": "value1"}, to_stdout=False)
        emit_receipt("test", {"key": "value2"}, to_stdout=False)

        filtered = query_receipts("test", key="value1")
        assert len(filtered) == 1
        assert filtered[0]["key"] == "value1"


class TestConstants:
    """Tests for module constants."""

    def test_tenant_id(self):
        """TENANT_ID should be shieldproof."""
        assert TENANT_ID == "shieldproof"

    def test_receipt_types(self):
        """RECEIPT_TYPES should have 3 types."""
        assert len(RECEIPT_TYPES) == 3
        assert "contract" in RECEIPT_TYPES
        assert "milestone" in RECEIPT_TYPES
        assert "payment" in RECEIPT_TYPES

    def test_milestone_states(self):
        """MILESTONE_STATES should have 5 states."""
        assert len(MILESTONE_STATES) == 5
        assert "PENDING" in MILESTONE_STATES
        assert "DELIVERED" in MILESTONE_STATES
        assert "VERIFIED" in MILESTONE_STATES
        assert "PAID" in MILESTONE_STATES
        assert "DISPUTED" in MILESTONE_STATES

    def test_version(self):
        """VERSION should be 2.0.0."""
        assert VERSION == "2.0.0"


class TestStopRule:
    """Tests for StopRule exception."""

    def test_stoprule_is_exception(self):
        """StopRule should be an Exception."""
        assert issubclass(StopRule, Exception)

    def test_stoprule_message(self):
        """StopRule should preserve message."""
        try:
            raise StopRule("test message")
        except StopRule as e:
            assert "test message" in str(e)
