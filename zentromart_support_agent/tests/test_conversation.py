"""Unit tests for ConversationManager and ConversationState.

Verifies:
- Dataclass fields and initialization
- Order ID extraction (continuous, spaced, hyphenated, non-8 digit rejection)
- Verification credential extraction (4-digit phone suffix, 6-digit pincode)
- Intent classification across all 9 canonical intents
- [ESC-1] Real-time escalation triggers (damage > 10000, unauthorized activity,
  legal mention, repeat contact >= 2, late refund, human request, safety risk)
- [IDV-1] Verification requirements gating
- Multi-turn state updates and context summaries
"""

import sys
from pathlib import Path
import unittest

# Ensure parent directory is in sys.path
_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _TESTS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from agent.conversation import (
    ConversationState,
    ConversationManager,
    INTENT_ORDER_STATUS,
    INTENT_RETURN,
    INTENT_REFUND,
    INTENT_CANCELLATION,
    INTENT_DELIVERY,
    INTENT_DAMAGE_REPORT,
    INTENT_WARRANTY,
    INTENT_GENERAL_QUERY,
    INTENT_ESCALATION_REQUEST,
    PENDING_AWAITING_VERIFICATION,
    PENDING_AWAITING_ORDER_ID,
)


class TestConversationState(unittest.TestCase):
    """Tests for ConversationState dataclass."""

    def test_default_initialization(self):
        state = ConversationState()
        self.assertIsNotNone(state.conversation_id)
        self.assertGreater(len(state.conversation_id), 10)
        self.assertEqual(state.turns, [])
        self.assertEqual(state.detected_language, "en")
        self.assertIsNone(state.current_order_id)
        self.assertFalse(state.verified)
        self.assertIsNone(state.customer_name)
        self.assertFalse(state.is_plus)
        self.assertEqual(state.intent_history, [])
        self.assertEqual(state.cited_sections, [])
        self.assertIsNone(state.pending_action)
        self.assertFalse(state.ticket_created)
        self.assertIsNone(state.ticket_id)

    def test_to_dict_and_from_dict(self):
        state = ConversationState(
            detected_language="hi-en",
            current_order_id="89782837",
            verified=True,
            customer_name="Pooja",
            is_plus=True,
            intent_history=[INTENT_ORDER_STATUS, INTENT_RETURN],
            cited_sections=["RET-2", "IDV-2"],
            pending_action=None,
            ticket_created=True,
            ticket_id="TKT12345",
        )
        data = state.to_dict()
        self.assertEqual(data["current_order_id"], "89782837")
        self.assertEqual(data["customer_name"], "Pooja")
        self.assertTrue(data["is_plus"])

        reconstructed = ConversationState.from_dict(data)
        self.assertEqual(reconstructed.conversation_id, state.conversation_id)
        self.assertEqual(reconstructed.current_order_id, "89782837")
        self.assertTrue(reconstructed.verified)
        self.assertEqual(reconstructed.customer_name, "Pooja")
        self.assertEqual(reconstructed.intent_history, [INTENT_ORDER_STATUS, INTENT_RETURN])
        self.assertEqual(reconstructed.cited_sections, ["RET-2", "IDV-2"])


class TestExtractOrderId(unittest.TestCase):
    """Tests for 8-digit order ID extraction."""

    def test_continuous_8_digits(self):
        self.assertEqual(ConversationManager.extract_order_id("My order is 89782837"), "89782837")

    def test_spaced_4_4_digits(self):
        self.assertEqual(ConversationManager.extract_order_id("Order number hai 8978 2837."), "89782837")
        self.assertEqual(ConversationManager.extract_order_id("The order number is 6649 2475."), "66492475")

    def test_hyphenated_order_id(self):
        self.assertEqual(ConversationManager.extract_order_id("Order # 8978-2837"), "89782837")

    def test_spaced_2_2_2_2_digits(self):
        self.assertEqual(ConversationManager.extract_order_id("Order: 89 78 28 37"), "89782837")

    def test_prefixed_order_id(self):
        self.assertEqual(ConversationManager.extract_order_id("OD89782837"), "89782837")
        self.assertEqual(ConversationManager.extract_order_id("ORD-89782837"), "89782837")

    def test_does_not_match_non_8_digits(self):
        # 10-digit phone
        self.assertIsNone(ConversationManager.extract_order_id("Call me on 9876543210"))
        # 6-digit pincode
        self.assertIsNone(ConversationManager.extract_order_id("Delivery pincode 560001"))
        # 4-digit last4
        self.assertIsNone(ConversationManager.extract_order_id("Last 4 digits 4533"))
        # empty / None
        self.assertIsNone(ConversationManager.extract_order_id(""))
        self.assertIsNone(ConversationManager.extract_order_id(None))


class TestExtractVerificationValue(unittest.TestCase):
    """Tests for 4-digit phone suffix and 6-digit pincode extraction."""

    def test_4_digit_phone_suffix(self):
        self.assertEqual(ConversationManager.extract_verification_value("Last four digits 4533 hain."), "4533")
        self.assertEqual(ConversationManager.extract_verification_value("The last four digits are 6073."), "6073")
        self.assertEqual(ConversationManager.extract_verification_value("आख़िरी चार अंक 2949 हैं।"), "2949")

    def test_4_digit_leading_zero(self):
        # Preserves leading zero as string
        self.assertEqual(ConversationManager.extract_verification_value("Last 4 digits are 0783."), "0783")

    def test_6_digit_pincode(self):
        self.assertEqual(ConversationManager.extract_verification_value("Delivery pincode is 560001."), "560001")
        self.assertEqual(ConversationManager.extract_verification_value("560 001"), "560001")
        self.assertEqual(ConversationManager.extract_verification_value("पिनकोड 110001 है"), "110001")

    def test_excludes_order_id_from_verification_value(self):
        # When order ID is present with last 4 digits
        text = "Order 8978 2837, last 4 digits are 4533"
        self.assertEqual(ConversationManager.extract_verification_value(text), "4533")

        # When ONLY order ID is present
        text_only_order = "Order number hai 8978 2837."
        self.assertIsNone(ConversationManager.extract_verification_value(text_only_order))

    def test_no_digits_returns_none(self):
        self.assertIsNone(ConversationManager.extract_verification_value("Hello, I need help."))
        self.assertIsNone(ConversationManager.extract_verification_value(""))
        self.assertIsNone(ConversationManager.extract_verification_value(None))


class TestDetectIntent(unittest.TestCase):
    """Tests for keyword-based intent classification across all 9 intents."""

    def test_order_status(self):
        self.assertEqual(ConversationManager.detect_intent("Where is my order?"), INTENT_ORDER_STATUS)
        self.assertEqual(ConversationManager.detect_intent("Mera package kahan hai?"), INTENT_ORDER_STATUS)
        self.assertEqual(ConversationManager.detect_intent("Please track my order"), INTENT_ORDER_STATUS)
        self.assertEqual(ConversationManager.detect_intent("Has it shipped yet?"), INTENT_ORDER_STATUS)
        self.assertEqual(ConversationManager.detect_intent("Is my parcel delivered?"), INTENT_ORDER_STATUS)
        self.assertEqual(ConversationManager.detect_intent("स्टेटस चेक करना है"), INTENT_ORDER_STATUS)

    def test_return(self):
        self.assertEqual(ConversationManager.detect_intent("Can I return this jeans?"), INTENT_RETURN)
        self.assertEqual(ConversationManager.detect_intent("Mujhe item wapas karna hai"), INTENT_RETURN)
        self.assertEqual(ConversationManager.detect_intent("I want to exchange this item"), INTENT_RETURN)
        self.assertEqual(ConversationManager.detect_intent("How to send back the product?"), INTENT_RETURN)
        self.assertEqual(ConversationManager.detect_intent("सामान वापस करना है"), INTENT_RETURN)

    def test_refund(self):
        self.assertEqual(ConversationManager.detect_intent("Where is my refund?"), INTENT_REFUND)
        self.assertEqual(ConversationManager.detect_intent("Mera paisa kab aayega?"), INTENT_REFUND)
        self.assertEqual(ConversationManager.detect_intent("I have not received my money back"), INTENT_REFUND)
        self.assertEqual(ConversationManager.detect_intent("रिफ़ंड कब मिलेगा?"), INTENT_REFUND)

    def test_cancellation(self):
        self.assertEqual(ConversationManager.detect_intent("Please cancel my order"), INTENT_CANCELLATION)
        self.assertEqual(ConversationManager.detect_intent("Order band kar do"), INTENT_CANCELLATION)
        self.assertEqual(ConversationManager.detect_intent("ऑर्डर कैंसिल करना है"), INTENT_CANCELLATION)

    def test_delivery(self):
        self.assertEqual(ConversationManager.detect_intent("When will you deliver?"), INTENT_DELIVERY)
        self.assertEqual(ConversationManager.detect_intent("Why is shipping late?"), INTENT_DELIVERY)
        self.assertEqual(ConversationManager.detect_intent("There is a delivery delay"), INTENT_DELIVERY)
        self.assertEqual(ConversationManager.detect_intent("डिलीवरी में देरी क्यों है?"), INTENT_DELIVERY)

    def test_damage_report(self):
        self.assertEqual(ConversationManager.detect_intent("Item arrived damaged"), INTENT_DAMAGE_REPORT)
        self.assertEqual(ConversationManager.detect_intent("The screen is broken"), INTENT_DAMAGE_REPORT)
        self.assertEqual(ConversationManager.detect_intent("You sent the wrong product"), INTENT_DAMAGE_REPORT)
        self.assertEqual(ConversationManager.detect_intent("Charger is missing from box"), INTENT_DAMAGE_REPORT)
        self.assertEqual(ConversationManager.detect_intent("The unit is defective"), INTENT_DAMAGE_REPORT)
        self.assertEqual(ConversationManager.detect_intent("सामान टूटा हुआ मिला"), INTENT_DAMAGE_REPORT)

    def test_warranty(self):
        self.assertEqual(ConversationManager.detect_intent("Is this covered under warranty?"), INTENT_WARRANTY)
        self.assertEqual(ConversationManager.detect_intent("What is the guarantee period?"), INTENT_WARRANTY)
        self.assertEqual(ConversationManager.detect_intent("Can I send this for repair?"), INTENT_WARRANTY)
        self.assertEqual(ConversationManager.detect_intent("वारंटी क्लेम कैसे करें?"), INTENT_WARRANTY)

    def test_escalation_request(self):
        self.assertEqual(ConversationManager.detect_intent("I want to speak with a human agent"), INTENT_ESCALATION_REQUEST)
        self.assertEqual(ConversationManager.detect_intent("Connect me to a manager"), INTENT_ESCALATION_REQUEST)
        self.assertEqual(ConversationManager.detect_intent("I want to file a complaint"), INTENT_ESCALATION_REQUEST)
        self.assertEqual(ConversationManager.detect_intent("I will take legal action"), INTENT_ESCALATION_REQUEST)
        self.assertEqual(ConversationManager.detect_intent("मुझे किसी इंसान से बात करनी है"), INTENT_ESCALATION_REQUEST)

    def test_general_query(self):
        self.assertEqual(ConversationManager.detect_intent("Hello, good morning"), INTENT_GENERAL_QUERY)
        self.assertEqual(ConversationManager.detect_intent("What are the membership perks?"), INTENT_GENERAL_QUERY)
        self.assertEqual(ConversationManager.detect_intent(""), INTENT_GENERAL_QUERY)
        self.assertEqual(ConversationManager.detect_intent(None), INTENT_GENERAL_QUERY)


class TestShouldEscalate(unittest.TestCase):
    """Tests for [ESC-1] escalation triggers."""

    def setUp(self):
        self.mgr = ConversationManager()

    def test_trigger_1_high_value_damage(self):
        self.mgr.add_user_message("I received a damaged TV.")
        order_data = {
            "order_id": "89782837",
            "damage_sku": "ELE-0021",
            "items": [
                {"sku": "ELE-0021", "title": "4K TV", "unit_price_inr": 31649, "qty": 1}
            ],
            "total_inr": 31649,
        }
        escalate, reason = self.mgr.should_escalate(order_data)
        self.assertTrue(escalate)
        self.assertTrue("10,000" in reason or "ELE-0021" in reason)

    def test_trigger_1_damage_low_value_no_escalation(self):
        self.mgr.add_user_message("My tote bag is slightly damaged.")
        order_data = {
            "order_id": "89782837",
            "damage_sku": "FSH-0038",
            "items": [
                {"sku": "FSH-0038", "title": "Tote Bag", "unit_price_inr": 979, "qty": 1}
            ],
            "total_inr": 979,
        }
        escalate, reason = self.mgr.should_escalate(order_data)
        self.assertFalse(escalate)
        self.assertEqual(reason, "")

    def test_trigger_2_unauthorized_activity(self):
        self.mgr.add_user_message("I did not make this order, someone hacked my account!")
        escalate, reason = self.mgr.should_escalate()
        self.assertTrue(escalate)
        self.assertIn("unauthorized", reason.lower())

        self.mgr.reset()
        self.mgr.add_user_message("maine order nahi kiya ye fraud hai")
        escalate, reason = self.mgr.should_escalate()
        self.assertTrue(escalate)

    def test_trigger_3_legal_mention(self):
        self.mgr.add_user_message("If you don't resolve this I will go to consumer court.")
        escalate, reason = self.mgr.should_escalate()
        self.assertTrue(escalate)
        self.assertTrue("consumer court" in reason.lower() or "legal" in reason.lower())

        self.mgr.reset()
        self.mgr.add_user_message("I am filing an FIR with the police")
        escalate, reason = self.mgr.should_escalate()
        self.assertTrue(escalate)
        self.assertIn("police", reason.lower())

    def test_trigger_4_repeat_contacts(self):
        # Order lookup returns contacts_last_7_days >= 2
        order_data = {"order_id": "45011817", "contacts_last_7_days": 2}
        escalate, reason = self.mgr.should_escalate(order_data)
        self.assertTrue(escalate)
        self.assertTrue("contacts_last_7_days" in reason or "Third or later contact" in reason)

    def test_trigger_4_repeat_contact_text(self):
        self.mgr.add_user_message("Main is hafte teesri baar call kar rahi hoon.")
        escalate, reason = self.mgr.should_escalate()
        self.assertTrue(escalate)
        self.assertIn("third or later", reason.lower())

    def test_trigger_5_late_refund(self):
        # UPI refund initiated on 2026-07-20 (REFERENCE_DATE is 2026-10-20)
        # UPI max is 3 business days; grace is 3 business days. Overdue!
        order_data = {
            "order_id": "16640894",
            "return_refund_initiated_on": "2026-07-20",
            "return_refund_method": "upi",
            "return_refund_completed_on": None,
        }
        escalate, reason = self.mgr.should_escalate(order_data)
        self.assertTrue(escalate)
        self.assertTrue("business days past initiation" in reason or "Refund" in reason)

    def test_trigger_6_human_request(self):
        self.mgr.add_user_message("Actually, can I just speak to a person?")
        escalate, reason = self.mgr.should_escalate()
        self.assertTrue(escalate)
        self.assertIn("human agent", reason.lower())

        self.mgr.reset()
        self.mgr.add_user_message("क्या मैं किसी इंसान से बात कर सकती हूँ?")
        escalate, reason = self.mgr.should_escalate()
        self.assertTrue(escalate)

    def test_trigger_7_product_safety_risk(self):
        self.mgr.add_user_message("The phone battery is swelling and sparks came out!")
        escalate, reason = self.mgr.should_escalate()
        self.assertTrue(escalate)
        self.assertIn("safety risk", reason.lower())

        self.mgr.reset()
        self.mgr.add_user_message("There is smoke and burning smell coming from the device")
        escalate, reason = self.mgr.should_escalate()
        self.assertTrue(escalate)

    def test_no_escalation_for_normal_query(self):
        self.mgr.add_user_message("Hi, where is my order 8978 2837?")
        order_data = {
            "order_id": "89782837",
            "contacts_last_7_days": 0,
            "total_inr": 1500,
        }
        escalate, reason = self.mgr.should_escalate(order_data)
        self.assertFalse(escalate)
        self.assertEqual(reason, "")


class TestNeedsVerificationAndConversationFlow(unittest.TestCase):
    """Tests for identity verification gating and conversation flow."""

    def test_needs_verification_gating(self):
        mgr = ConversationManager()

        # Step 1: Greeting without order ID -> no verification needed yet
        mgr.add_user_message("Hello, mujhe order ke baare mein help chahiye.")
        self.assertFalse(mgr.needs_verification())
        self.assertEqual(mgr.state.pending_action, PENDING_AWAITING_ORDER_ID)

        # Step 2: Customer provides order ID -> needs verification now!
        mgr.add_user_message("Order number hai 8978 2837.")
        self.assertEqual(mgr.state.current_order_id, "89782837")
        self.assertTrue(mgr.needs_verification())
        self.assertEqual(mgr.state.pending_action, PENDING_AWAITING_VERIFICATION)

        # Step 3: Identity verification succeeds
        mgr.set_verified(verified=True, customer_name="Pooja", is_plus=True)
        self.assertFalse(mgr.needs_verification())
        self.assertTrue(mgr.state.verified)
        self.assertEqual(mgr.state.customer_name, "Pooja")
        self.assertTrue(mgr.state.is_plus)
        self.assertIsNone(mgr.state.pending_action)

    def test_agent_message_and_citations(self):
        mgr = ConversationManager()
        mgr.add_user_message("Can I return this item?")
        mgr.add_agent_message(
            "You can return it within 15 calendar days under [RET-2].",
            cited_sections=["RET-2"],
        )

        self.assertEqual(len(mgr.state.turns), 2)
        self.assertIn("RET-2", mgr.state.cited_sections)
        agent_turn = mgr.state.turns[1]
        self.assertEqual(agent_turn["role"], "agent")
        self.assertIn("RET-2", agent_turn["cited_sections"])

    def test_context_summary(self):
        mgr = ConversationManager()
        mgr.add_user_message("Hello, order 8978 2837 track karna hai.")
        mgr.add_agent_message("Please provide your delivery pincode [IDV-2].")

        summary = mgr.get_context_summary()
        self.assertIn("Conversation ID:", summary)
        self.assertIn("Current Order ID: 89782837", summary)
        self.assertIn("Identity Verified: No", summary)
        self.assertIn("IDV-2", summary)
        self.assertIn("Recent Turns", summary)

    def test_reset(self):
        mgr = ConversationManager()
        mgr.add_user_message("Order 89782837")
        old_id = mgr.state.conversation_id

        new_state = mgr.reset()
        self.assertNotEqual(new_state.conversation_id, old_id)
        self.assertEqual(new_state.turns, [])
        self.assertIsNone(new_state.current_order_id)
        self.assertFalse(new_state.verified)


if __name__ == "__main__":
    unittest.main()
