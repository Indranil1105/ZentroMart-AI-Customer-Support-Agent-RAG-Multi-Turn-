"""Unit tests for zentromart_support_agent.agent.response_generator module."""

from datetime import date, datetime
import re
import unittest

from zentromart_support_agent.agent.business_days import REFERENCE_DATE
from zentromart_support_agent.agent.conversation import ConversationState
from zentromart_support_agent.agent.response_generator import (
    ResponseGenerator,
    format_date,
    parse_date_from_text,
)


class TestResponseGenerator(unittest.TestCase):
    """Test suite covering template-based response generation for ZentroMart support."""

    def setUp(self) -> None:
        """Initialize ResponseGenerator instance before each test."""
        self.generator = ResponseGenerator()

    # =========================================================================
    # Date Formatting & Parsing Tests [STY-2]
    # =========================================================================

    def test_format_date_english(self):
        """Format date in English with full month name ([STY-2])."""
        d = date(2026, 10, 24)
        formatted = format_date(d, language="en")
        self.assertEqual(formatted, "24 October 2026")

    def test_format_date_hindi(self):
        """Format date in Hindi with Hindi month name ([STY-2])."""
        d = date(2026, 10, 27)
        formatted = format_date(d, language="hi")
        self.assertEqual(formatted, "27 अक्टूबर 2026")

    def test_format_date_hinglish(self):
        """Format date in Hinglish with month name ([STY-2])."""
        d = date(2026, 11, 13)
        formatted = format_date(d, language="hi-en")
        self.assertEqual(formatted, "13 November 2026")

    def test_format_date_iso_string(self):
        """Format date from ISO format string."""
        formatted = format_date("2026-10-14", language="en")
        self.assertEqual(formatted, "14 October 2026")

    def test_parse_date_from_text(self):
        """Extract date from natural language strings."""
        d1 = parse_date_from_text("refund was started on 17 October 2026")
        self.assertEqual(d1, date(2026, 10, 17))

        d2 = parse_date_from_text("delivered on 2026-10-12")
        self.assertEqual(d2, date(2026, 10, 12))

        d3 = parse_date_from_text("14 अक्टूबर 2026 को मिला")
        self.assertEqual(d3, date(2026, 10, 14))

    # =========================================================================
    # Greeting Tests [STY-1]
    # =========================================================================

    def test_generate_greeting_english(self):
        """Verify English greeting introduces as AI assistant ([STY-1])."""
        greeting = self.generator.generate_greeting("en")
        self.assertIn("ZentroMart's AI support assistant", greeting)
        self.assertIn("[STY-1]", greeting)
        # Check sentence count without citation tags
        text_clean = re.sub(r"\[[A-Za-z0-9_., -]+\]", "", greeting).strip()
        sentences = [s for s in re.split(r"[.!?।]", text_clean) if s.strip()]
        self.assertLessEqual(len(sentences), 3)

    def test_generate_greeting_hindi(self):
        """Verify Hindi greeting introduces as AI assistant in Hindi ([STY-1])."""
        greeting = self.generator.generate_greeting("hi")
        self.assertIn("ZentroMart का AI सपोर्ट असिस्टेंट", greeting)
        self.assertIn("[STY-1]", greeting)

    def test_generate_greeting_hinglish(self):
        """Verify Hinglish greeting introduces as AI assistant in Hinglish ([STY-1])."""
        greeting = self.generator.generate_greeting("hi-en")
        self.assertIn("ZentroMart ka AI support assistant", greeting)
        self.assertIn("[STY-1]", greeting)

    # =========================================================================
    # Clarification Tests [STY-5, IDV-2, IDV-3]
    # =========================================================================

    def test_clarify_order_id(self):
        """Clarification asks for 8-digit order number one at a time ([STY-5])."""
        res_en = self.generator.generate_clarification("order_id", "en")
        self.assertIn("8-digit order number", res_en)
        self.assertIn("[STY-5]", res_en)

        res_hi = self.generator.generate_clarification("order_id", "hi")
        self.assertIn("8 अंकों का ऑर्डर नंबर", res_hi)
        self.assertIn("[STY-5]", res_hi)

        res_hien = self.generator.generate_clarification("order_id", "hi-en")
        self.assertIn("8-digit order number", res_hien)
        self.assertIn("[STY-5]", res_hien)

    def test_clarify_verification(self):
        """Clarification asks for last 4 digits of phone or delivery pincode ([IDV-2, STY-5])."""
        res_en = self.generator.generate_clarification("verification", "en")
        self.assertIn("last 4 digits", res_en)
        self.assertIn("delivery pincode", res_en)
        self.assertIn("[IDV-2, STY-5]", res_en)

        res_hi = self.generator.generate_clarification("verification", "hi")
        self.assertIn("आख़िरी 4 अंक", res_hi)
        self.assertIn("पिनकोड", res_hi)
        self.assertIn("[IDV-2, STY-5]", res_hi)

        res_hien = self.generator.generate_clarification("verification", "hi-en")
        self.assertIn("aakhri 4 digits", res_hien)
        self.assertIn("pincode", res_hien)
        self.assertIn("[IDV-2, STY-5]", res_hien)

    def test_clarify_verification_retry(self):
        """Verification retry indicates mismatch and asks again ([IDV-3, STY-5])."""
        res_en = self.generator.generate_clarification("verification_retry", "en")
        self.assertIn("did not match", res_en)
        self.assertIn("[IDV-3, STY-5]", res_en)

        res_hi = self.generator.generate_clarification("verification_retry", "hi")
        self.assertIn("मेल नहीं खाती", res_hi)
        self.assertIn("[IDV-3, STY-5]", res_hi)

    def test_clarify_payment_method(self):
        """Clarification asks for payment method details ([REF-2, STY-5])."""
        res_en = self.generator.generate_clarification("payment_method", "en")
        self.assertIn("UPI", res_en)
        self.assertIn("[REF-2, STY-5]", res_en)

    def test_clarify_item_details(self):
        """Clarification asks which item in order ([RET-2, STY-5])."""
        res_en = self.generator.generate_clarification("item_details", "en")
        self.assertIn("Which item", res_en)
        self.assertIn("[RET-2, STY-5]", res_en)

    # =========================================================================
    # Escalation Tests [ESC-1, ESC-2, ESC-3, ESC-4, DWM-3, DWM-5, REF-6]
    # =========================================================================

    def test_escalation_high_value_damage_with_ticket(self):
        """High-value damage (>₹10,000) escalates to Priority Resolution with ticket ID ([DWM-3, ESC-1])."""
        res = self.generator.generate_escalation("high_value_damage", "en", ticket_id="TKT9999")
        self.assertIn("₹10,000", res)
        self.assertIn("Priority Resolution", res)
        self.assertIn("ticket #TKT9999", res)
        self.assertIn("[DWM-3, ESC-1]", res)

    def test_escalation_fraud(self):
        """Unauthorized order escalates to account security ([ESC-1, ESC-2])."""
        res_en = self.generator.generate_escalation("fraud", "en")
        self.assertIn("account security", res_en)
        self.assertIn("[ESC-1, ESC-2]", res_en)

        res_hien = self.generator.generate_escalation("fraud", "hi-en", ticket_id="TKT123")
        self.assertIn("Account Security", res_hien)
        self.assertIn("ticket #TKT123", res_hien)

    def test_escalation_repeat_contact(self):
        """Repeat contact escalates to senior colleague ([ESC-1, ESC-3])."""
        res = self.generator.generate_escalation("repeat_contact", "en")
        self.assertIn("senior colleague", res)
        self.assertIn("[ESC-1, ESC-3]", res)

    def test_escalation_human_requested(self):
        """Customer requested person/human colleague ([ESC-1, ESC-2])."""
        res = self.generator.generate_escalation("human_requested", "en")
        self.assertIn("human colleague", res)
        self.assertIn("[ESC-1, ESC-2]", res)

    def test_escalation_refund_delay(self):
        """Overdue refund escalates to finance team ([ESC-1, REF-6])."""
        res = self.generator.generate_escalation("refund_delay", "en")
        self.assertIn("finance team", res)
        self.assertIn("[ESC-1, REF-6]", res)

    def test_escalation_safety_risk(self):
        """Safety risk (smoke/spark/fire) instructs immediate stop and escalates ([DWM-5, ESC-1])."""
        res = self.generator.generate_escalation("safety_risk", "en")
        self.assertIn("stop using", res)
        self.assertIn("safety incident team", res)
        self.assertIn("[DWM-5, ESC-1]", res)

    # =========================================================================
    # Refusal Tests [ESC-4, CAN-2, DEL-4, IDV-3, IDV-4, RET-4, RET-5, CPN-2]
    # =========================================================================

    def test_refusal_cancel_shipped(self):
        """Cannot cancel shipped orders; refuse politely and advise refuse at door ([CAN-2, REF-1])."""
        res = self.generator.generate_refusal("cancel_shipped", "en")
        self.assertIn("already shipped cannot be cancelled", res)
        self.assertIn("refuse the parcel at the door", res)
        self.assertIn("[CAN-2, REF-1]", res)

    def test_refusal_late_standard_compensation(self):
        """No compensation or vouchers for delayed standard deliveries ([DEL-4, ESC-4])."""
        res = self.generator.generate_refusal("late_standard_compensation", "en")
        self.assertIn("no vouchers or refunds are offered for Standard delivery delays", res)
        self.assertIn("[DEL-4, ESC-4]", res)

    def test_refusal_customer_privacy(self):
        """Refuse to disclose other customers' phone/address ([ESC-4, IDV-4])."""
        res = self.generator.generate_refusal("customer_privacy", "en")
        self.assertIn("can't share any details about other customers", res)
        self.assertIn("[ESC-4, IDV-4]", res)

    def test_refusal_verification_failed(self):
        """Refuse to share order details after failed verification ([IDV-3])."""
        res = self.generator.generate_refusal("verification_failed", "en")
        self.assertIn("identity verification could not be completed", res)
        self.assertIn("[IDV-3]", res)

    def test_refusal_non_returnable_opened_beauty(self):
        """Opened beauty products cannot be returned ([RET-4])."""
        res = self.generator.generate_refusal("non_returnable_opened", "en")
        self.assertIn("unopened with the seal intact", res)
        self.assertIn("[RET-4]", res)

    def test_refusal_non_returnable_innerwear(self):
        """Innerwear and swimwear cannot be returned ([RET-2, RET-5])."""
        res = self.generator.generate_refusal("non_returnable_category", "en")
        self.assertIn("Innerwear and swimwear can't be returned", res)
        self.assertIn("[RET-2, RET-5]", res)

    def test_refusal_expired_coupon(self):
        """Expired coupons cannot be re-issued ([CPN-2])."""
        res = self.generator.generate_refusal("expired_coupon", "en")
        self.assertIn("expired cannot be re-issued", res)
        self.assertIn("[CPN-2]", res)

    # =========================================================================
    # Knowledge-Grounded Answer Tests [Practice Dataset Coverage]
    # =========================================================================

    def test_answer_pq0002_warranty_hinglish(self):
        """PQ0002: SKU ELE-0012 ZentroMart Assured warranty claim in Hinglish ([WAR-2])."""
        query = "SKU ELE-0012 pe ZentroMart Assured hai kya? Claim kaise karein?"
        res = self.generator.generate_answer(query, language="hi-en")
        self.assertIn("12 mahine ki Assured warranty", res)
        self.assertIn("My Orders", res)
        self.assertIn("[WAR-2]", res)

    def test_answer_pq0003_wallet_refund_speed(self):
        """PQ0003: Refunds to ZentroMart Wallet are instant within 2 hours ([REF-2])."""
        query = "How fast are refunds to the ZentroMart Wallet?"
        res = self.generator.generate_answer(query, language="en")
        self.assertIn("Instant, within 2 hours", res)
        self.assertNotIn("5 to 7 days", res)
        self.assertIn("[REF-2]", res)

    def test_answer_pq0004_unsupported_midnight_delivery(self):
        """PQ0004: Midnight delivery is unsupported; admit lack of info and offer human handoff ([ESC-1])."""
        query = "Can I schedule a delivery at midnight?"
        res = self.generator.generate_answer(query, language="en")
        self.assertIn("don't have information about that", res)
        self.assertIn("colleague", res)
        self.assertIn("[ESC-1]", res)

    def test_answer_pq0006_plus_return_fee_waived(self):
        """PQ0006: ZentroMart Plus members pay zero return fee ([RET-6, PLS-2])."""
        query = "As a ZentroMart Plus member, will I be charged the pickup fee for returning shoes I just don't like?"
        res = self.generator.generate_answer(query, language="en")
        self.assertIn("No. ZentroMart Plus members pay no reverse pickup fee", res)
        self.assertNotIn("₹49 will be charged", res)
        self.assertIn("[RET-6, PLS-2]", res)

    def test_answer_pq0008_card_refund_date_calc_hinglish(self):
        """PQ0008: Credit card refund started 17 Oct 2026 calculation in Hinglish ([REF-2])."""
        query = "credit card wala refund 17 October 2026 ko shuru hua, kab tak aa jaana chahiye?"
        res = self.generator.generate_answer(query, language="hi-en")
        self.assertIn("24 October 2026", res)
        self.assertIn("27 October 2026", res)
        self.assertIn("Sunday aur warehouse holidays nahi gine jaate", res)
        self.assertIn("[REF-2]", res)

    def test_answer_pq0010_damaged_item_free_return(self):
        """PQ0010: Damaged or torn clothing has free reverse pickup ([RET-6, DWM-2])."""
        query = "Is the collection free if the UrbanThread Slim Fit Jeans came torn?"
        res = self.generator.generate_answer(query, language="en")
        self.assertIn("No fee applies for damaged or wrong items", res)
        self.assertIn("48 hours", res)
        self.assertIn("[RET-6, DWM-2]", res)

    def test_answer_pq0013_opened_lipstick_hindi(self):
        """PQ0013: Opened lipstick cannot be returned in Hindi ([RET-4])."""
        query = "मैंने Gulmohar Matte Lipstick खोल लिया है, क्या अब भी वापस कर सकती हूँ?"
        res = self.generator.generate_answer(query, language="hi")
        self.assertIn("नहीं, ब्यूटी प्रोडक्ट सिर्फ़ बिना खोले, सील के साथ ही वापस होते हैं।", res)
        self.assertIn("[RET-4]", res)

    def test_answer_pq0014_notebook_return_window_hindi(self):
        """PQ0014: Stationery notebooks return window is 10 days in Hindi ([RET-2])."""
        query = "मुझे Pagecraft A5 Ruled Notebook (Pack of 4) लौटाने के लिए कितने दिन मिलेंगे?"
        res = self.generator.generate_answer(query, language="hi")
        self.assertIn("10 दिन", res)
        self.assertIn("[RET-2]", res)

    def test_answer_pq0016_opened_perfume(self):
        """PQ0016: Opened perfume cannot be returned ([RET-4])."""
        query = "I opened the Neelam Eau de Parfum 50 ml and tried it once. Can I still return it?"
        res = self.generator.generate_answer(query, language="en")
        self.assertIn("Beauty products can be returned only if unopened with the seal intact", res)
        self.assertIn("[RET-4]", res)

    def test_answer_pq0019_pre_august_electronics_window(self):
        """PQ0019: Electronics delivered before 1 August keep 10-day window ([RET-3])."""
        query = "My Orbix Fit 3 Pro Smartwatch was delivered on 20 July. What was the return window for it?"
        res = self.generator.generate_answer(query, language="en")
        self.assertIn("Electronics delivered before 1 August 2026 keep the 10-day window", res)
        self.assertIn("10 days from 20 July", res)
        self.assertIn("[RET-3]", res)

    def test_answer_pq0021_courier_attempts(self):
        """PQ0021: Courier attempts up to 3 times before returning to warehouse ([DEL-3])."""
        query = "What happens if I'm not home when the courier comes?"
        res = self.generator.generate_answer(query, language="en")
        self.assertIn("The courier tries up to 3 times", res)
        self.assertIn("[DEL-3]", res)

    def test_answer_pq0022_card_refund_date_calc_english(self):
        """PQ0022: Card refund started 17 October 2026 calculation in English ([REF-2])."""
        query = "My refund to credit card was started on 17 October 2026. By which date should I have it?"
        res = self.generator.generate_answer(query, language="en")
        self.assertIn("Between 24 October 2026 and 27 October 2026", res)
        self.assertIn("skipping warehouse holidays", res)
        self.assertIn("[REF-2]", res)

    def test_answer_pq0024_festive_sale_fashion_return_window(self):
        """PQ0024: Festive sale fashion orders get 30 days ([RET-2, RET-8])."""
        query = "How long do I get to send back the Rangoli Cotton Straight Kurta?"
        res = self.generator.generate_answer(query, language="en")
        self.assertIn("15 days", res)
        self.assertIn("1 to 15 October festive sale get 30 days", res)
        self.assertIn("[RET-2, RET-8]", res)

    def test_answer_plus_benefits(self):
        """Plus membership benefits inquiry ([PLS-1, PLS-2, FAQH-4])."""
        res_en = self.generator.generate_answer("What are the ZentroMart Plus benefits?", language="en")
        self.assertIn("₹999", res_en)
        self.assertIn("free Express delivery", res_en)
        self.assertIn("refund initiation at pickup", res_en)
        self.assertIn("[PLS-1, PLS-2, FAQH-4]", res_en)

    def test_answer_coupon_rules(self):
        """Coupon rules inquiry: only 1 coupon per order ([CPN-1])."""
        res_en = self.generator.generate_answer("Can I use two coupons on one order?", language="en")
        self.assertIn("Only one coupon can be applied per order", res_en)
        self.assertIn("[CPN-1]", res_en)

    def test_answer_international_shipping_refusal(self):
        """International delivery is not supported; India only ([DEL-5, FAQ-5])."""
        res = self.generator.generate_answer("Do you deliver to Dubai?", language="en")
        self.assertIn("No, deliveries are only within India", res)
        self.assertIn("[DEL-5, FAQ-5]", res)

    def test_answer_troubleshooting(self):
        """Troubleshooting speaker not turning on ([TRB-4])."""
        res = self.generator.generate_answer("My speaker won't turn on", language="en")
        self.assertIn("Charge it for 30 minutes", res)
        self.assertIn("10 seconds", res)
        self.assertIn("[TRB-4]", res)

    # =========================================================================
    # Multi-turn Context & State Integration Tests
    # =========================================================================

    def test_dispatch_turn1_opening_clarification(self):
        """Turn 1 clarification combines AI assistant intro with single question ([STY-1, STY-5])."""
        state = ConversationState(turns=[{"role": "user", "text": "Hello, mujhe help chahiye."}], detected_language="hi-en")
        res = self.generator.generate_response(
            query="Hello, mujhe help chahiye.",
            conversation_state=state,
            decision="clarify",
        )
        self.assertIn("Main ZentroMart ka AI support assistant hoon", res)
        self.assertIn("order ID", res)
        self.assertIn("[STY-1, STY-5]", res)

    def test_dispatch_turn2_verification_clarification(self):
        """Turn 2 after order_id known asks for last 4 digits or pincode ([IDV-2, STY-5])."""
        state = ConversationState(
            turns=[
                {"role": "user", "text": "Hello"},
                {"role": "agent", "text": "Hello! I'm ZentroMart's AI support assistant..."},
                {"role": "user", "text": "Order number is 8978 2837."},
            ],
            detected_language="en",
            current_order_id="89782837",
            verified=False,
        )
        res = self.generator.generate_response(
            query="Order number is 8978 2837.",
            conversation_state=state,
            order_data={"order_id": "89782837"},
            decision="clarify",
        )
        self.assertIn("last 4 digits", res)
        self.assertIn("delivery pincode", res)
        self.assertIn("[IDV-2, STY-5]", res)

    def test_dispatch_turn3_verification_confirmation(self):
        """Turn 3 when customer provides verification confirms verification ([IDV-2])."""
        state = ConversationState(
            turns=[
                {"role": "user", "text": "Hello"},
                {"role": "agent", "text": "Hello!"},
                {"role": "user", "text": "Order is 89782837"},
                {"role": "agent", "text": "Please provide last 4 digits."},
                {"role": "user", "text": "Last four digits 4533 hain."},
            ],
            detected_language="hi-en",
            current_order_id="89782837",
            verified=True,
        )
        res = self.generator.generate_response(
            query="Last four digits 4533 hain.",
            conversation_state=state,
            order_data={"order_id": "89782837"},
            decision="answer",
        )
        self.assertIn("Details verify ho gayi hain", res)
        self.assertIn("[IDV-2]", res)

    def test_dispatch_turn4_return_eligibility_with_order_data(self):
        """Turn 4 calculates return deadline using order_data delivery date & sale window ([RET-2, RET-8])."""
        # PC004: Delivered 14 Oct 2026, placed 6 Oct 2026 (festive sale) -> 30 days window -> 13 Nov 2026
        state = ConversationState(
            turns=[{"role": "user", "text": "..."}],
            detected_language="hi",
            current_order_id="79363550",
            verified=True,
        )
        order_data = {
            "order_id": "79363550",
            "placed_at": "2026-10-06T10:00:00",
            "delivered_on": "2026-10-14",
            "is_plus": False,
            "items": [{"title": "UrbanThread Slim Fit Jeans", "return_category": "fashion"}],
        }
        res = self.generator.generate_response(
            query="क्या मैं UrbanThread Slim Fit Jeans वापस कर सकती हूँ?",
            conversation_state=state,
            order_data=order_data,
            decision="answer",
        )
        self.assertIn("13 नवंबर 2026", res)
        self.assertIn("₹49", res)
        self.assertIn("[RET-2, RET-6]", res)

    def test_dispatch_escalate_high_value_damage(self):
        """Damage on item >₹10,000 escalates to Priority Resolution ([DWM-3, ESC-1])."""
        state = ConversationState(
            detected_language="hi-en",
            current_order_id="89782837",
            verified=True,
            ticket_id="TKT8978",
        )
        order_data = {
            "order_id": "89782837",
            "total_inr": 24999,
            "items": [{"title": "Zentro Vista 43-inch TV"}],
        }
        res = self.generator.generate_response(
            query="Maine Zentro Vista 43-inch 4K Smart TV damage report kiya tha. Abhi replacement approve kar do na?",
            conversation_state=state,
            order_data=order_data,
            decision="escalate",
        )
        self.assertIn("₹10,000", res)
        self.assertIn("Priority Resolution", res)
        self.assertIn("ticket #TKT8978", res)
        self.assertIn("[DWM-3, ESC-1]", res)

    def test_language_switching(self):
        """Customer switches language mid-conversation -> agent switches with them ([STY-4])."""
        state = {"language": "hi", "turn": 2}
        # Customer switches to English
        res_en = self.generator.generate_response(
            query="Can I return this notebook?",
            conversation_state=state,
            decision="answer",
        )
        self.assertIn("You can return books and stationery within 10 days", res_en)
        self.assertEqual(state["language"], "en")

    def test_chat_sentence_length_constraint(self):
        """Verify chat replies do not exceed 3 short sentences ([STY-2])."""
        queries = [
            ("Hello", "greeting"),
            ("Can I return this?", "answer"),
            ("Where is my order?", "answer"),
            ("I want to speak with a human", "escalate"),
            ("Cancel my shipped order", "refuse"),
            ("Order number 12345678", "clarify"),
        ]
        for q, dec in queries:
            resp = self.generator.generate_response(q, decision=dec)
            # Remove bracket citations (e.g. [STY-1]) before splitting sentences
            clean_text = re.sub(r"\[[A-Za-z0-9_., -]+\]", "", resp).strip()
            sentences = [s for s in re.split(r"[.!?।]", clean_text) if s.strip()]
            self.assertLessEqual(
                len(sentences),
                3,
                f"Response exceeded 3 sentences for query '{q}': {resp}",
            )


if __name__ == "__main__":
    unittest.main()
