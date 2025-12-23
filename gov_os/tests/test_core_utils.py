"""
Tests for Gov-OS Core Utils

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.utils import dual_hash, merkle_root, bytes_to_hex


class TestDualHash(unittest.TestCase):
    """Test dual hash function."""

    def test_dual_hash_deterministic(self):
        """Same input should produce same hash."""
        data = {"key": "value", "number": 42}
        hash1 = dual_hash(data)
        hash2 = dual_hash(data)
        self.assertEqual(hash1, hash2)

    def test_dual_hash_different_inputs(self):
        """Different inputs should produce different hashes."""
        hash1 = dual_hash({"a": 1})
        hash2 = dual_hash({"a": 2})
        self.assertNotEqual(hash1, hash2)

    def test_dual_hash_format(self):
        """Hash should be hex string of expected length."""
        result = dual_hash({"test": "data"})
        self.assertIsInstance(result, str)
        # SHA-256 produces 64 hex characters
        self.assertEqual(len(result), 64)

    def test_dual_hash_order_independent(self):
        """Dict key order should not affect hash."""
        hash1 = dual_hash({"a": 1, "b": 2})
        hash2 = dual_hash({"b": 2, "a": 1})
        self.assertEqual(hash1, hash2)


class TestMerkleRoot(unittest.TestCase):
    """Test Merkle root computation."""

    def test_merkle_empty(self):
        """Empty list should return empty string."""
        result = merkle_root([])
        self.assertEqual(result, "")

    def test_merkle_single(self):
        """Single item should return that item's hash."""
        result = merkle_root(["abc"])
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)

    def test_merkle_deterministic(self):
        """Same inputs should produce same root."""
        items = ["a", "b", "c", "d"]
        root1 = merkle_root(items)
        root2 = merkle_root(items)
        self.assertEqual(root1, root2)

    def test_merkle_order_matters(self):
        """Order of items should affect root."""
        root1 = merkle_root(["a", "b"])
        root2 = merkle_root(["b", "a"])
        self.assertNotEqual(root1, root2)

    def test_merkle_power_of_two(self):
        """Power of two items should work correctly."""
        result = merkle_root(["a", "b", "c", "d"])
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)

    def test_merkle_odd_count(self):
        """Odd number of items should work correctly."""
        result = merkle_root(["a", "b", "c"])
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)


class TestBytesToHex(unittest.TestCase):
    """Test bytes to hex conversion."""

    def test_bytes_to_hex(self):
        """Should convert bytes to hex string."""
        result = bytes_to_hex(b"\x00\xff")
        self.assertEqual(result, "00ff")

    def test_bytes_to_hex_empty(self):
        """Empty bytes should return empty string."""
        result = bytes_to_hex(b"")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
