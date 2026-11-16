"""Unit tests for the ToolManager module (zentromart_support_agent.agent.tools).

Covers:
1. ToolManager initialization, sys.path configuration, and empty state tracking.
2. Order lookup without verification value (verification_required returned, zero attempts consumed).
3. Failed verification attempt handling, attempt tracking, and retry budget.
4. Maximum attempts enforcement (max 1 retry / 2 total attempts).
5. Third attempt blocking without calling underlying tool.
6. Successful verification via phone last 4 digits.
7. Successful verification via delivery pincode.
8. In-memory caching and subsequent retrieval without re-verification.
9. Ticket creation validation (summary length >= 20 chars, error on short summary).
10. State reset functionality (clears attempts and cache).
11. Order ID normalization (numeric strings, integers, and prefixed codes).
12. Module-level convenience functions (get_tool_manager, lookup_order, create_ticket, reset_tools).
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from zentromart_support_agent.agent.tools import (
    ToolManager,
    create_ticket,
    get_tool_manager,
    lookup_order,
    reset_tools,
)


class TestToolManager(unittest.TestCase):
    """Test suite for the ToolManager customer-support tool wrapper."""

    # Sample test order from dataset:
    # Order ID: 69520867, Phone last 4: 0101, Pincode: 160017
    VALID_ORDER_ID = "69520867"
    VALID_PHONE_LAST4 = "0101"
    VALID_PINCODE = "160017"
    WRONG_PHONE = "9999"
    WRONG_PINCODE = "999999"

    def setUp(self) -> None:
        """Create a fresh ToolManager instance before each test."""
        self.mgr = ToolManager()
        self.mgr.reset()
        reset_tools()

    def tearDown(self) -> None:
        """Clean up tool state after each test."""
        self.mgr.reset()
        reset_tools()

    def test_init_state(self) -> None:
        """Test initial state has empty attempt and verified order mappings."""
        self.assertEqual(len(self.mgr.verification_attempts), 0)
        self.assertEqual(len(self.mgr.verified_orders), 0)
        self.assertTrue(self.mgr.can_retry_verification(self.VALID_ORDER_ID))
        self.assertFalse(self.mgr.is_verified(self.VALID_ORDER_ID))
        self.assertIsNone(self.mgr.get_verified_order(self.VALID_ORDER_ID))

    def test_lookup_order_without_verification_value(self) -> None:
        """First lookup without verification_value must return verification_required and consume 0 attempts."""
        res = self.mgr.lookup_order(self.VALID_ORDER_ID)
        self.assertTrue(res.get("verification_required"))
        self.assertFalse(res.get("verified"))
        self.assertIn("phone last 4 digits OR delivery pincode", res.get("message", ""))
        self.assertNotIn("items", res)
        self.assertNotIn("status", res)
        # Attempt count must still be 0 because no verification value was submitted
        self.assertEqual(self.mgr.verification_attempts.get(self.VALID_ORDER_ID, 0), 0)
        self.assertTrue(self.mgr.can_retry_verification(self.VALID_ORDER_ID))
        self.assertFalse(self.mgr.is_verified(self.VALID_ORDER_ID))

    def test_lookup_order_with_whitespace_verification_value(self) -> None:
        """Calling with whitespace verification_value should also require verification and consume 0 attempts."""
        res = self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value="   ")
        self.assertTrue(res.get("verification_required"))
        self.assertFalse(res.get("verified"))
        self.assertEqual(self.mgr.verification_attempts.get(self.VALID_ORDER_ID, 0), 0)

    def test_first_failed_verification_attempt(self) -> None:
        """One wrong verification value records 1 attempt and allows 1 retry."""
        res = self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value=self.WRONG_PHONE)
        self.assertFalse(res.get("verified"))
        self.assertTrue(res.get("verification_required"))
        self.assertEqual(res.get("attempts"), 1)
        self.assertTrue(res.get("can_retry"))
        self.assertNotIn("items", res)
        self.assertNotIn("status", res)
        # Verify manager state
        self.assertEqual(self.mgr.verification_attempts.get(self.VALID_ORDER_ID), 1)
        self.assertTrue(self.mgr.can_retry_verification(self.VALID_ORDER_ID))
        self.assertFalse(self.mgr.is_verified(self.VALID_ORDER_ID))

    def test_second_failed_verification_attempt_exhausts_retry_budget(self) -> None:
        """Two wrong verification values consume all attempts (max 1 retry / 2 total)."""
        # Attempt 1
        res1 = self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value=self.WRONG_PHONE)
        self.assertEqual(res1.get("attempts"), 1)
        self.assertTrue(self.mgr.can_retry_verification(self.VALID_ORDER_ID))

        # Attempt 2 (Retry 1)
        res2 = self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value=self.WRONG_PINCODE)
        self.assertFalse(res2.get("verified"))
        self.assertTrue(res2.get("verification_required"))
        self.assertEqual(res2.get("attempts"), 2)
        self.assertFalse(res2.get("can_retry"))
        self.assertTrue(res2.get("max_attempts_exceeded"))
        self.assertNotIn("items", res2)

        # Verification attempts is now 2
        self.assertEqual(self.mgr.verification_attempts.get(self.VALID_ORDER_ID), 2)
        self.assertFalse(self.mgr.can_retry_verification(self.VALID_ORDER_ID))
        self.assertFalse(self.mgr.is_verified(self.VALID_ORDER_ID))

    def test_third_attempt_is_blocked_without_invoking_mock(self) -> None:
        """Third attempt is blocked immediately without calling the underlying tool."""
        # Exhaust 2 attempts
        self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value=self.WRONG_PHONE)
        self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value=self.WRONG_PINCODE)
        self.assertFalse(self.mgr.can_retry_verification(self.VALID_ORDER_ID))

        # Spy or patch mock function
        with patch.object(self.mgr, "_lookup_order_fn") as mock_lookup:
            res3 = self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value=self.VALID_PHONE_LAST4)
            # Must NOT call mock_lookup because limit is exceeded
            mock_lookup.assert_not_called()
            self.assertFalse(res3.get("verified"))
            self.assertTrue(res3.get("max_attempts_exceeded"))
            self.assertIn("Maximum verification attempts exceeded", res3.get("message", ""))

    def test_inquiry_after_exhaustion_still_blocked(self) -> None:
        """Calling lookup_order without verification_value after exhaustion remains blocked."""
        self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value=self.WRONG_PHONE)
        self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value=self.WRONG_PINCODE)
        res = self.mgr.lookup_order(self.VALID_ORDER_ID)
        self.assertFalse(res.get("verified"))
        self.assertTrue(res.get("max_attempts_exceeded"))

    def test_successful_verification_via_phone_last4(self) -> None:
        """Verification succeeds when providing registered phone last 4 digits."""
        res = self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value=self.VALID_PHONE_LAST4)
        self.assertTrue(res.get("found"))
        self.assertTrue(res.get("verified"))
        self.assertEqual(res.get("order_id"), self.VALID_ORDER_ID)
        self.assertIn("items", res)
        self.assertIn("customer_first_name", res)
        self.assertEqual(res.get("customer_first_name"), "Naveen")
        self.assertNotIn("phone_last4", res)  # Must never leak expected phone field

        # Check manager state
        self.assertTrue(self.mgr.is_verified(self.VALID_ORDER_ID))
        cached = self.mgr.get_verified_order(self.VALID_ORDER_ID)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.get("order_id"), self.VALID_ORDER_ID)

    def test_successful_verification_via_delivery_pincode(self) -> None:
        """Verification succeeds when providing 6-digit delivery pincode."""
        res = self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value=self.VALID_PINCODE)
        self.assertTrue(res.get("found"))
        self.assertTrue(res.get("verified"))
        self.assertTrue(self.mgr.is_verified(self.VALID_ORDER_ID))

    def test_verified_order_cached_on_subsequent_lookup(self) -> None:
        """Subsequent lookup of a verified order returns cached data without requiring verification."""
        # First verify
        self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value=self.VALID_PHONE_LAST4)
        self.assertTrue(self.mgr.is_verified(self.VALID_ORDER_ID))

        # Subsequent lookup without verification_value returns the cached order
        subsequent = self.mgr.lookup_order(self.VALID_ORDER_ID)
        self.assertTrue(subsequent.get("verified"))
        self.assertTrue(subsequent.get("found"))
        self.assertIn("items", subsequent)
        self.assertEqual(subsequent.get("order_id"), self.VALID_ORDER_ID)

    def test_normalization_of_order_id(self) -> None:
        """Normalization handles ints, formatted strings like ORD-69520867, and spaces."""
        self.assertEqual(self.mgr._normalize_order_id(69520867), "69520867")
        self.assertEqual(self.mgr._normalize_order_id("ORD-69520867"), "69520867")
        self.assertEqual(self.mgr._normalize_order_id(" 69520867 "), "69520867")
        self.assertEqual(self.mgr._normalize_order_id("UNKNOWN"), "UNKNOWN")
        self.assertEqual(self.mgr._normalize_order_id(None), "")

        # Verify cross-format cache lookup
        self.mgr.lookup_order("ORD-69520867", verification_value=self.VALID_PHONE_LAST4)
        self.assertTrue(self.mgr.is_verified("69520867"))
        self.assertTrue(self.mgr.is_verified(69520867))
        self.assertTrue(self.mgr.is_verified("ORD-69520867"))

    def test_reset_clears_all_state(self) -> None:
        """reset() clears attempts and cached orders completely."""
        self.mgr.lookup_order(self.VALID_ORDER_ID, verification_value=self.VALID_PHONE_LAST4)
        self.assertTrue(self.mgr.is_verified(self.VALID_ORDER_ID))
        self.assertEqual(len(self.mgr.verification_attempts), 1)
        self.assertEqual(len(self.mgr.verified_orders), 1)

        self.mgr.reset()
        self.assertFalse(self.mgr.is_verified(self.VALID_ORDER_ID))
        self.assertIsNone(self.mgr.get_verified_order(self.VALID_ORDER_ID))
        self.assertEqual(len(self.mgr.verification_attempts), 0)
        self.assertEqual(len(self.mgr.verified_orders), 0)
        self.assertTrue(self.mgr.can_retry_verification(self.VALID_ORDER_ID))

    def test_create_ticket_short_summary_fails(self) -> None:
        """Ticket creation must fail if summary is less than 20 characters."""
        res = self.mgr.create_ticket(summary="Too short", order_id=self.VALID_ORDER_ID)
        self.assertFalse(res.get("created"))
        self.assertIn("summary too short", res.get("error", ""))

    def test_create_ticket_valid_summary_succeeds(self) -> None:
        """Ticket creation succeeds with descriptive summary (>= 20 characters)."""
        summary = "Customer requested return after damaged package received. Verified delivery."
        res = self.mgr.create_ticket(
            summary=summary,
            order_id=self.VALID_ORDER_ID,
            reason="damaged_goods",
            intent="return",
        )
        self.assertTrue(res.get("created"))
        self.assertIn("ticket_id", res)
        self.assertTrue(res.get("ticket_id", "").startswith("TKT"))

    def test_module_level_convenience_functions(self) -> None:
        """Module-level lookup_order and create_ticket functions delegate properly."""
        reset_tools()
        # Initial inquiry
        res1 = lookup_order(self.VALID_ORDER_ID)
        self.assertTrue(res1.get("verification_required"))

        # Verify
        res2 = lookup_order(self.VALID_ORDER_ID, verification_value=self.VALID_PHONE_LAST4)
        self.assertTrue(res2.get("verified"))

        # Ticket creation
        tkt = create_ticket(
            summary="Module level test for escalation logging with sufficient detail.",
            order_id=self.VALID_ORDER_ID,
            reason="test_escalation",
        )
        self.assertTrue(tkt.get("created"))
        self.assertIn("ticket_id", tkt)

        # Reset
        reset_tools()
        tm = get_tool_manager()
        self.assertFalse(tm.is_verified(self.VALID_ORDER_ID))


if __name__ == "__main__":
    unittest.main()
