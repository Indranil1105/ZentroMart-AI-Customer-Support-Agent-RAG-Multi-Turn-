"""Conversation state manager for ZentroMart AI Customer Support Agent.

This module provides conversation state tracking, multi-turn context management,
intent detection, customer identity verification gating, and human escalation
evaluation adhering to ZentroMart support policies:
    - [STY-1] to [STY-5]: Conversation style, language switching, and step-by-step guidance.
    - [IDV-1] to [IDV-4]: Identity verification before disclosing specific order details.
    - [ESC-1] to [ESC-4]: Escalation criteria, ticket handover summaries, and policy safeguards.
    - [REF-2], [RET-2]: Business day calendars, return windows, and refund timelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from uuid import uuid4

# Ensure parent directory is accessible in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _CURRENT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# Import language detector and business day utilities
try:
    from agent.language_detector import detect_language
    from agent.business_days import REFERENCE_DATE, count_business_days
except ImportError:
    from zentromart_support_agent.agent.language_detector import detect_language
    from zentromart_support_agent.agent.business_days import (
        REFERENCE_DATE,
        count_business_days,
    )

logger = logging.getLogger(__name__)

# ============================================================================
# Intent Constants
# ============================================================================

INTENT_ORDER_STATUS: str = "order_status"
INTENT_RETURN: str = "return"
INTENT_REFUND: str = "refund"
INTENT_CANCELLATION: str = "cancellation"
INTENT_DELIVERY: str = "delivery"
INTENT_DAMAGE_REPORT: str = "damage_report"
INTENT_WARRANTY: str = "warranty"
INTENT_GENERAL_QUERY: str = "general_query"
INTENT_ESCALATION_REQUEST: str = "escalation_request"

ALL_INTENTS: Tuple[str, ...] = (
    INTENT_ORDER_STATUS,
    INTENT_RETURN,
    INTENT_REFUND,
    INTENT_CANCELLATION,
    INTENT_DELIVERY,
    INTENT_DAMAGE_REPORT,
    INTENT_WARRANTY,
    INTENT_GENERAL_QUERY,
    INTENT_ESCALATION_REQUEST,
)

# Pending Action States
PENDING_AWAITING_VERIFICATION: str = "awaiting_verification"
PENDING_AWAITING_ORDER_ID: str = "awaiting_order_id"

# Section ID extraction pattern (e.g. [RET-2], [ESC-1], [IDV-2])
SECTION_ID_PATTERN: re.Pattern = re.compile(r"\[([A-Za-z0-9_.-]+)\]")

# Maximum business days allowed per refund payment method per [REF-2]
REFUND_MAX_BUSINESS_DAYS: Dict[str, int] = {
    "upi": 3,
    "netbanking": 5,
    "card": 7,
    "credit_card": 7,
    "debit_card": 7,
    "wallet": 0,
    "cod_wallet": 0,
    "cod_bank": 7,
    "bank_transfer": 7,
}


# ============================================================================
# Conversation State Dataclass
# ============================================================================

@dataclass
class ConversationState:
    """Represents the complete state of a customer support conversation session.

    Attributes:
        conversation_id: Unique identifier for the conversation session (UUID string).
        turns: Ordered history of message turns, each a dict with role, text, timestamp,
            and optional metadata such as cited_sections.
        detected_language: ISO language code of the customer ('en', 'hi', 'hi-en', etc.).
        current_order_id: 8-digit order identifier actively discussed, or None.
        verified: True if customer identity was verified for current_order_id per [IDV-2].
        customer_name: Customer first name or full name when identified, or None.
        is_plus: True if customer has active ZentroMart Plus membership.
        intent_history: List of detected intents across user turns.
        cited_sections: Policy section identifiers cited during the interaction (e.g. [RET-2]).
        pending_action: Current workflow state: None, 'awaiting_verification', or 'awaiting_order_id'.
        ticket_created: True if an escalation ticket was created via create_ticket tool.
        ticket_id: Identifier of created escalation ticket, or None.
    """

    conversation_id: str = field(default_factory=lambda: str(uuid4()))
    turns: List[Dict[str, Any]] = field(default_factory=list)
    detected_language: str = "en"
    current_order_id: Optional[str] = None
    verified: bool = False
    customer_name: Optional[str] = None
    is_plus: bool = False
    intent_history: List[str] = field(default_factory=list)
    cited_sections: List[str] = field(default_factory=list)
    pending_action: Optional[str] = None  # None | 'awaiting_verification' | 'awaiting_order_id'
    ticket_created: bool = False
    ticket_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a plain dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """Instantiate ConversationState from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# Conversation Manager Class
# ============================================================================

class ConversationManager:
    """Manages multi-turn conversation flow, identity gating, and escalation detection.

    Adheres strictly to ZentroMart Customer Support rules:
    - [STY-4] Language policy: updates detected language as customer communicates.
    - [IDV-1], [IDV-2]: Identity verification required before sharing specific order data.
    - [ESC-1]: Real-time escalation triggers (damage > 10,000, unauthorized activity,
      legal mention, repeat contacts >= 2, late refunds, human agent request, safety risks).
    """

    def __init__(self, state: Optional[ConversationState] = None) -> None:
        """Initialize ConversationManager with a fresh state or adopt an existing state.

        Args:
            state: Optional existing ConversationState. If None, a new instance is created.
        """
        self.state: ConversationState = state if state is not None else ConversationState()

    def reset(self) -> ConversationState:
        """Reset the conversation session to a fresh state with a new UUID.

        Returns:
            ConversationState: The newly initialized conversation state.
        """
        self.state = ConversationState()
        return self.state

    # ------------------------------------------------------------------------
    # Message Handling Methods
    # ------------------------------------------------------------------------

    def add_user_message(self, text: str) -> Dict[str, Any]:
        """Process an incoming user message, updating conversation state.

        Performs:
            1. Appends user message to turns with current timestamp.
            2. Detects message language via language_detector and updates detected_language [STY-4].
            3. Extracts 8-digit order ID if present and updates current_order_id.
            4. Detects intent and appends to intent_history.
            5. Evaluates and manages pending_action state ('awaiting_order_id' / 'awaiting_verification').

        Args:
            text: Raw message text sent by the customer.

        Returns:
            Dict[str, Any]: The newly created turn dictionary.
        """
        if text is None:
            text = ""

        # 1. Create turn record
        turn: Dict[str, Any] = {
            "role": "user",
            "text": text,
            "timestamp": datetime.now().isoformat(),
        }
        self.state.turns.append(turn)

        # 2. Detect language [STY-4]
        detected_lang = detect_language(text)
        self.state.detected_language = detected_lang

        # 3. Extract order ID
        extracted_oid = self.extract_order_id(text)
        if extracted_oid:
            if self.state.current_order_id != extracted_oid:
                self.state.current_order_id = extracted_oid
                # Switching or setting order ID resets verification for safety [IDV-1]
                self.state.verified = False
            if self.state.pending_action == PENDING_AWAITING_ORDER_ID:
                self.state.pending_action = PENDING_AWAITING_VERIFICATION

        # 4. Detect intent
        intent = self.detect_intent(text)
        self.state.intent_history.append(intent)

        # 5. Manage pending_action state
        order_intents = {
            INTENT_ORDER_STATUS,
            INTENT_RETURN,
            INTENT_REFUND,
            INTENT_CANCELLATION,
            INTENT_DELIVERY,
            INTENT_DAMAGE_REPORT,
        }
        text_lower = text.lower()
        has_order_cue = (intent in order_intents) or any(
            k in text_lower for k in ["order", "ऑर्डर"]
        )

        if not self.state.verified:
            if self.state.current_order_id is None:
                if has_order_cue:
                    self.state.pending_action = PENDING_AWAITING_ORDER_ID
            else:
                self.state.pending_action = PENDING_AWAITING_VERIFICATION
        else:
            if self.state.pending_action in (
                PENDING_AWAITING_ORDER_ID,
                PENDING_AWAITING_VERIFICATION,
            ):
                self.state.pending_action = None

        return turn

    def add_agent_message(
        self, text: str, cited_sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Record an agent reply, extracting and tracking policy citations.

        Performs:
            1. Appends agent message to turns.
            2. Ingests caller-provided cited_sections.
            3. Extracts inline section IDs like [RET-2], [ESC-1] from message text.
            4. Updates pending_action based on agent prompt cues if awaiting user input.

        Args:
            text: Text message sent by the agent.
            cited_sections: Optional explicit list of cited policy section IDs.

        Returns:
            Dict[str, Any]: The newly created turn dictionary.
        """
        if text is None:
            text = ""

        all_cited: List[str] = []
        if cited_sections:
            for s in cited_sections:
                sec_clean = str(s).strip()
                if sec_clean and sec_clean not in all_cited:
                    all_cited.append(sec_clean)

        # Extract inline section IDs [ABC-1] from text
        inline_matches = SECTION_ID_PATTERN.findall(text)
        for sec in inline_matches:
            if sec not in all_cited:
                all_cited.append(sec)

        # Append to global cited_sections
        for sec in all_cited:
            if sec not in self.state.cited_sections:
                self.state.cited_sections.append(sec)

        turn: Dict[str, Any] = {
            "role": "agent",
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "cited_sections": all_cited,
        }
        self.state.turns.append(turn)

        # Update pending action cues from agent prompt if unverified
        if not self.state.verified:
            lower = text.lower()
            if self.state.current_order_id is None:
                if any(k in lower for k in ["order id", "order number", "ऑर्डर नंबर", "order no"]):
                    self.state.pending_action = PENDING_AWAITING_ORDER_ID
            else:
                if any(
                    k in lower
                    for k in [
                        "last 4",
                        "last four",
                        "pincode",
                        "pin code",
                        "phone number",
                        "aakhri",
                        "पिनकोड",
                    ]
                ):
                    self.state.pending_action = PENDING_AWAITING_VERIFICATION

        return turn

    # ------------------------------------------------------------------------
    # Extraction Utilities
    # ------------------------------------------------------------------------

    @staticmethod
    def extract_order_id(text: Optional[str]) -> Optional[str]:
        """Extract an 8-digit ZentroMart order ID from text.

        Allows formats such as:
            - Continuous: "89782837"
            - Spaced: "8978 2837"
            - Hyphenated: "8978-2837"
            - 2-2-2-2 spaced: "89 78 28 37"
            - Prefixed: "OD89782837", "Order # 8978 2837"

        Does NOT match 10-digit phone numbers, 6-digit pincodes, or 4-digit codes.

        Args:
            text: Customer text containing potential order ID.

        Returns:
            Optional[str]: Cleaned 8-digit order ID string, or None if not found.
        """
        if not text or not isinstance(text, str):
            return None

        # Pattern 1: 4 digits + optional space/hyphen + 4 digits
        # Lookbehind/lookahead ensures no adjacent digits (e.g. not a 10-digit phone number)
        match_4_4 = re.search(r"(?<!\d)(\d{4})[\s\-]?(\d{4})(?!\d)", text)
        if match_4_4:
            return f"{match_4_4.group(1)}{match_4_4.group(2)}"

        # Pattern 2: 2-2-2-2 spaced digits (e.g. "89 78 28 37")
        match_2_2 = re.search(
            r"(?<!\d)(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})(?!\d)", text
        )
        if match_2_2:
            return (
                f"{match_2_2.group(1)}{match_2_2.group(2)}"
                f"{match_2_2.group(3)}{match_2_2.group(4)}"
            )

        return None

    @staticmethod
    def extract_verification_value(text: Optional[str]) -> Optional[str]:
        """Extract verification credential (4-digit phone suffix or 6-digit delivery pincode).

        To avoid misidentifying parts of an 8-digit order ID as verification digits,
        any 8-digit order ID in the text is stripped first.

        Supports:
            - 6-digit pincodes: "560001", "110001", "560 001"
            - 4-digit phone suffixes: "4533", "0783", "2108"
            - Keyword-assisted disambiguation (e.g. "pincode is...", "last 4 digits...")

        Args:
            text: Customer input containing phone suffix or delivery pincode.

        Returns:
            Optional[str]: 4-digit or 6-digit credential string, or None if not found.
        """
        if not text or not isinstance(text, str):
            return None

        # Step 1: Strip 8-digit order IDs so their components aren't mistaken for verification values
        cleaned = re.sub(r"(?<!\d)\d{4}[\s\-]?\d{4}(?!\d)", " ", text)
        cleaned = re.sub(r"(?<!\d)(?:\d{2}\s+){3}\d{2}(?!\d)", " ", cleaned)

        lower = cleaned.lower()

        # Step 2: Search for 6-digit delivery pincode (continuous or 3+3 spaced)
        pincode_match = re.search(r"(?<!\d)(\d{3})\s*(\d{3})(?!\d)", cleaned)
        pincode_val = (
            f"{pincode_match.group(1)}{pincode_match.group(2)}"
            if pincode_match
            else None
        )

        # Step 3: Search for 4-digit phone suffix (preserves leading zero like '0783')
        phone_matches = list(re.finditer(r"(?<!\d)(\d{4})(?!\d)", cleaned))

        # Check keyword intent cues
        wants_phone = any(
            k in lower
            for k in [
                "last 4",
                "last four",
                "phone",
                "mobile",
                "aakhri",
                "char",
                "digit",
                "अंक",
                "चार",
                "फ़ोन",
                "फोन",
            ]
        )
        wants_pin = any(
            k in lower
            for k in [
                "pin",
                "pincode",
                "pin code",
                "delivery pin",
                "पिन",
                "पिनकोड",
            ]
        )

        if wants_phone and phone_matches:
            return phone_matches[-1].group(1)

        if wants_pin and pincode_val:
            return pincode_val

        # If no explicit keyword preference, default: pincode (6-digit) then phone (4-digit)
        if pincode_val:
            return pincode_val

        if phone_matches:
            return phone_matches[0].group(1)

        return None

    # ------------------------------------------------------------------------
    # Intent Detection
    # ------------------------------------------------------------------------

    @staticmethod
    def detect_intent(text: Optional[str]) -> str:
        """Classify customer query intent using keyword and phrase heuristics.

        Recognized Intents:
            - escalation_request: human, agent, manager, complaint, legal
            - damage_report: damaged, broken, wrong, missing, defective
            - warranty: warranty, guarantee, repair
            - cancellation: cancel, band
            - refund: refund, money, paisa
            - return: return, wapas, exchange, send back
            - order_status: status, where, track, kahan, shipped, delivered
            - delivery: deliver, shipping, late, delay
            - general_query: everything else (fallback)

        Args:
            text: Message text to classify.

        Returns:
            str: One of the canonical intent constants.
        """
        if not text or not isinstance(text, str):
            return INTENT_GENERAL_QUERY

        lower = text.lower()

        # 1. Escalation Request (Highest priority: human intervention / legal)
        escalation_pattern = re.compile(
            r"(?:\bhuman\b|\bagent\b|\bmanager\b|\bcomplaint\b|\blegal\b|"
            r"\bperson\b|\brepresentative\b|\bsupervisor\b|\bexecutive\b|"
            r"\bcourt\b|\bpolice\b|\blawyer\b|\badvocate\b|"
            r"talk to (?:a person|someone|a human|an agent)|"
            r"speak to (?:a person|someone|a human|an agent)|"
            r"connect me|"
            r"इंसान|एजेंट|मैनेजर|शिकायत|लीगल|कस्टमर केयर|अधिकारी|"
            r"insan se baat|agent se baat|human se baat)",
            re.IGNORECASE,
        )
        if escalation_pattern.search(lower):
            return INTENT_ESCALATION_REQUEST

        # 2. Damage Report
        damage_pattern = re.compile(
            r"(?:\bdamaged?\b|\bbroken\b|\bwrong\b|\bmissing\b|\bdefective\b|"
            r"\bfaulty\b|\bcrack(?:ed)?\b|\bscratch(?:ed)?\b|"
            r"डैमेज|खराब|टूटा|गलत|गायब|नुकसान)",
            re.IGNORECASE,
        )
        if damage_pattern.search(lower):
            return INTENT_DAMAGE_REPORT

        # 3. Warranty
        warranty_pattern = re.compile(
            r"(?:\bwarranty\b|\bguarantee\b|\brepair\b|\bwarrenty\b|"
            r"वारंटी|गारंटी|मरम्मत|रिपेयर)",
            re.IGNORECASE,
        )
        if warranty_pattern.search(lower):
            return INTENT_WARRANTY

        # 4. Cancellation
        cancel_pattern = re.compile(
            r"(?:\bcancel\b|\bcancelling\b|\bcancellation\b|\bcancelled\b|\bband\b|"
            r"कैंसिल|रद्द|बंद)",
            re.IGNORECASE,
        )
        if cancel_pattern.search(lower):
            return INTENT_CANCELLATION

        # 5. Refund
        refund_pattern = re.compile(
            r"(?:\brefund\b|\brefunds\b|\brefunding\b|\bmoney\b|\bpaisa\b|\bpaise\b|"
            r"रिफ़ंड|रिफंड|पैसा|पैसे|रुपया|रुपये)",
            re.IGNORECASE,
        )
        if refund_pattern.search(lower):
            return INTENT_REFUND

        # 6. Return
        return_pattern = re.compile(
            r"(?:\breturn\b|\breturning\b|\breturns\b|\bwapas\b|\bvaapas\b|"
            r"\bwaapas\b|\bvapas\b|\bexchange\b|\bsend back\b|"
            r"वापस|रिटर्न|एक्सचेंज)",
            re.IGNORECASE,
        )
        if return_pattern.search(lower):
            return INTENT_RETURN

        # 7. Order Status (specific status inquiries, tracking, location, or delivered milestone)
        order_status_pattern = re.compile(
            r"(?:\bstatus\b|\bwhere\b|\btrack\b|\btracking\b|\bkahan\b|\bkidhar\b|"
            r"\bshipped\b|\bdelivered\b|"
            r"कहाँ|किधर|ट्रैक|स्टेटस|कब आएगा|पहुँचा|कहा)",
            re.IGNORECASE,
        )
        if order_status_pattern.search(lower):
            return INTENT_ORDER_STATUS

        # 8. Delivery (general shipping inquiries, speed, delay, partners)
        delivery_pattern = re.compile(
            r"(?:\bdeliver\b|\bdelivery\b|\bshipping\b|\blate\b|\bdelay\b|\bdelayed\b|"
            r"\bcourier\b|\bdispatch\b|"
            r"डिलीवरी|शिपिंग|देरी|लेट|डिले)",
            re.IGNORECASE,
        )
        if delivery_pattern.search(lower):
            return INTENT_DELIVERY

        # 9. General Query (default fallback)
        return INTENT_GENERAL_QUERY

    # ------------------------------------------------------------------------
    # Escalation & Verification Gating
    # ------------------------------------------------------------------------

    def should_escalate(
        self, order_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """Check [ESC-1] triggers to determine if conversation must hand over to a human.

        Evaluates the following 7 triggers defined in agent_escalation_handoff.md [ESC-1]:
            1. Damaged, defective or wrong item worth more than ₹10,000.
            2. Customer reports an order, payment, or account activity they did not make.
            3. Customer mentions legal action, a consumer court, or a police complaint.
            4. Third or later contact about the same issue within 7 days (contacts_last_7_days >= 2).
            5. Refund more than 3 business days past its maximum timeline.
            6. Customer asks for a human agent.
            7. Product safety risk such as a swelling battery, burning smell, sparks, or injury.

        Args:
            order_data: Optional dictionary returned by lookup_order tool with order details.

        Returns:
            Tuple[bool, str]: (should_escalate, escalation_reason).
                If escalation is required, returns (True, reason_string).
                Otherwise returns (False, "").
        """
        latest_user_text = self.get_last_user_message() or ""
        all_user_text = " ".join(
            t.get("text", "") for t in self.state.turns if t.get("role") == "user"
        )
        recent_combined = f"{latest_user_text} {all_user_text}".strip()

        # --------------------------------------------------------------------
        # Trigger 7: Product Safety Risk [ESC-1.7]
        # --------------------------------------------------------------------
        safety_pattern = re.compile(
            r"(?:swelling|swollen battery|battery swelling|bulging battery|battery bulge|"
            r"burning smell|burning|smoke|smoking|\bfire\b|\bspark\b|\bsparks\b|sparking|"
            r"electric shock|\bshock\b|\bcurrent\b|short circuit|"
            r"injury|injured|\bhurt\b|bleeding|wound|"
            r"exploded|explosion|\bblast\b|"
            r"धुआँ|धुआं|आग|जलने की बदबू|करंट|चोट|ब्लास्ट|विस्फोट|बैटरी फूल)",
            re.IGNORECASE,
        )
        if safety_pattern.search(recent_combined):
            return (
                True,
                "Product safety risk reported (battery swelling, sparks, fire, smoke, injury) [ESC-1]",
            )

        if order_data and isinstance(order_data, dict):
            damage_issue = str(order_data.get("damage_issue") or "")
            if safety_pattern.search(damage_issue):
                return (
                    True,
                    f"Product safety risk in order record: {damage_issue} [ESC-1]",
                )

        # --------------------------------------------------------------------
        # Trigger 2: Unauthorized Activity [ESC-1.2]
        # --------------------------------------------------------------------
        unauthorized_pattern = re.compile(
            r"(?:did(?:n['’]t| not) (?:make|place|order)|"
            r"not my order|never ordered|haven['’]t ordered|"
            r"unauthori[sz]ed|fraud|fraudulent|hacked|stolen|scam|"
            r"maine order nahi kiya|maine nahi (?:kiya|mangwaya)|kisi aur ne (?:order|kiya)|"
            r"dhokhadhadi|धोखाधड़ी|अनधिकृत|मैंने ऑर्डर नहीं किया|मैंने नहीं मंगाया|गलत ट्रांजेक्शन)",
            re.IGNORECASE,
        )
        if unauthorized_pattern.search(recent_combined):
            return (
                True,
                "Customer reports unauthorized order, payment, or account activity [ESC-1]",
            )

        # --------------------------------------------------------------------
        # Trigger 3: Legal, Consumer Court, Police Complaint [ESC-1.3]
        # --------------------------------------------------------------------
        legal_pattern = re.compile(
            r"(?:legal (?:action|notice)|consumer (?:court|forum|complaint)|"
            r"\bpolice\b|police complaint|\bfir\b|\bsue\b|\blawyer\b|\badvocate\b|"
            r"उपभोक्ता अदालत|उपभोक्ता फोरम|कोर्ट|अदालत|पुलिस|एफआईआर|वकील|कानूनी कार्रवाई|"
            r"kanuni|kanooni)",
            re.IGNORECASE,
        )
        if legal_pattern.search(recent_combined):
            return (
                True,
                "Customer mentioned legal action, consumer court, or police complaint [ESC-1]",
            )

        # --------------------------------------------------------------------
        # Trigger 6: Customer Asks for Human Agent [ESC-1.6]
        # --------------------------------------------------------------------
        human_pattern = re.compile(
            r"(?:human(?: agent)?|\bperson\b|real person|representative|executive|"
            r"supervisor|manager|speak to (?:a person|someone|a human)|"
            r"talk to (?:a person|someone|a human|an agent)|connect (?:me )?to (?:a human|an agent)|"
            r"transfer (?:me )?to (?:a human|an agent)|"
            r"इंसान|इंसान से बात|एजेंट से बात|मैनेजर|कस्टमर केयर|अधिकारी|"
            r"insan se baat|agent se baat|human se baat|kisi bande se baat)",
            re.IGNORECASE,
        )
        if human_pattern.search(latest_user_text):
            return True, "Customer requested handover to a human agent [ESC-1]"

        # --------------------------------------------------------------------
        # Trigger 4: Repeat Contact (3rd+ in 7 days via contacts_last_7_days >= 2) [ESC-1.4]
        # --------------------------------------------------------------------
        if order_data and isinstance(order_data, dict):
            contacts = order_data.get("contacts_last_7_days")
            if contacts is not None:
                try:
                    if int(contacts) >= 2:
                        return (
                            True,
                            f"Third or later contact within 7 days (contacts_last_7_days={contacts} >= 2) [ESC-1]",
                        )
                except (ValueError, TypeError):
                    pass

        repeat_pattern = re.compile(
            r"(?:third (?:time|contact)|3rd time|teesri baar|तीसरी बार|"
            r"fourth (?:time|contact)|4th time|calling (?:again|for the third time)|"
            r"multiple times this week)",
            re.IGNORECASE,
        )
        if repeat_pattern.search(latest_user_text):
            return (
                True,
                "Customer indicated third or later contact within 7 days [ESC-1]",
            )

        # --------------------------------------------------------------------
        # Trigger 1: Damaged, Defective or Wrong Item > ₹10,000 [ESC-1.1]
        # --------------------------------------------------------------------
        if order_data and isinstance(order_data, dict):
            damage_sku = order_data.get("damage_sku")
            damage_status = order_data.get("damage_status")
            has_damage_claim = bool(
                damage_sku or damage_status or order_data.get("damage_issue")
            )

            # Check if current user message reports damage
            if not has_damage_claim:
                if self.detect_intent(latest_user_text) == INTENT_DAMAGE_REPORT:
                    has_damage_claim = True

            if has_damage_claim:
                items = order_data.get("items", [])
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            it_sku = it.get("sku")
                            unit_price = it.get("unit_price_inr") or 0
                            qty = it.get("qty") or 1
                            item_total = unit_price * qty

                            # If a specific damage_sku is recorded, match it
                            if damage_sku and it_sku == damage_sku:
                                if item_total > 10000 or unit_price > 10000:
                                    return (
                                        True,
                                        f"Damaged item {damage_sku} value (₹{item_total:,}) exceeds ₹10,000 [ESC-1]",
                                    )
                            elif not damage_sku and (
                                item_total > 10000 or unit_price > 10000
                            ):
                                return (
                                    True,
                                    f"Damaged item {it_sku} value (₹{item_total:,}) exceeds ₹10,000 [ESC-1]",
                                )

                # Fallback to total_inr / subtotal_inr when damage is reported on high-value order
                order_total = (
                    order_data.get("total_inr")
                    or order_data.get("subtotal_inr")
                    or 0
                )
                if order_total > 10000 and (damage_sku or damage_status):
                    return (
                        True,
                        f"Damage claim for order totaling ₹{order_total:,} exceeds ₹10,000 [ESC-1]",
                    )

        # Check user text for damage intent accompanied by amount > ₹10,000
        if self.detect_intent(latest_user_text) == INTENT_DAMAGE_REPORT:
            amounts = re.findall(
                r"(?:₹|rs\.?|inr)?\s*(\d{5,7})\b", latest_user_text, re.IGNORECASE
            )
            for amt_str in amounts:
                if int(amt_str) > 10000:
                    return (
                        True,
                        f"Customer reported damage for item/order worth ₹{int(amt_str):,} exceeding ₹10,000 [ESC-1]",
                    )

        # --------------------------------------------------------------------
        # Trigger 5: Late Refund (More than 3 Business Days past Max Timeline) [ESC-1.5]
        # --------------------------------------------------------------------
        if order_data and isinstance(order_data, dict):
            refund_prefixes = ["return_refund", "cancellation_refund", "rto_refund"]
            for prefix in refund_prefixes:
                init_date_val = order_data.get(f"{prefix}_initiated_on")
                completed_date_val = order_data.get(f"{prefix}_completed_on")
                method_val = (
                    str(order_data.get(f"{prefix}_method") or "").lower().strip()
                )

                if init_date_val and not completed_date_val:
                    max_bdays = REFUND_MAX_BUSINESS_DAYS.get(method_val, 7)
                    grace_bdays = 3
                    threshold_bdays = max_bdays + grace_bdays

                    try:
                        elapsed_bdays = count_business_days(
                            init_date_val, REFERENCE_DATE
                        )
                        if elapsed_bdays > threshold_bdays:
                            return (
                                True,
                                (
                                    f"Refund ({method_val or 'standard'}) initiated on {init_date_val} is "
                                    f"{elapsed_bdays} business days past initiation, exceeding maximum "
                                    f"timeline ({max_bdays} days) + {grace_bdays} business days grace [ESC-1]"
                                ),
                            )
                    except Exception as err:
                        logger.warning(
                            f"Error calculating refund business days for escalation: {err}"
                        )

        # No escalation trigger met
        return False, ""

    def needs_verification(self) -> bool:
        """Check if current conversation state requires customer identity verification.

        Per [IDV-1]:
            Before sharing anything about a specific order (status, items, address,
            amounts, refund details), verify the customer.
            General policy questions need no verification.

        Returns:
            bool: True if order details are referenced and customer is not verified.
        """
        # If already verified, no further verification needed
        if self.state.verified:
            return False

        # If awaiting verification, identity check is active
        if self.state.pending_action == PENDING_AWAITING_VERIFICATION:
            return True

        # If an order ID has been captured but not verified
        if self.state.current_order_id is not None:
            return True

        return False

    # ------------------------------------------------------------------------
    # State Helpers & Context Summary
    # ------------------------------------------------------------------------

    def set_verified(
        self,
        verified: bool = True,
        customer_name: Optional[str] = None,
        is_plus: Optional[bool] = None,
    ) -> None:
        """Mark identity verification as completed and store customer attributes.

        Args:
            verified: True if verification succeeded per [IDV-2].
            customer_name: Optional customer name retrieved from lookup_order.
            is_plus: Optional boolean indicating ZentroMart Plus membership.
        """
        self.state.verified = verified
        if customer_name is not None:
            self.state.customer_name = customer_name
        if is_plus is not None:
            self.state.is_plus = is_plus
        if verified and self.state.pending_action == PENDING_AWAITING_VERIFICATION:
            self.state.pending_action = None

    def set_ticket(self, ticket_id: str) -> None:
        """Record created escalation ticket in state per [ESC-2].

        Args:
            ticket_id: Ticket ID generated by create_ticket tool.
        """
        self.state.ticket_created = True
        self.state.ticket_id = ticket_id

    def set_pending_action(self, pending_action: Optional[str]) -> None:
        """Manually update pending action state.

        Args:
            pending_action: One of None, 'awaiting_verification', 'awaiting_order_id'.
        """
        self.state.pending_action = pending_action

    def get_last_user_message(self) -> Optional[str]:
        """Return text of the most recent user message, or None."""
        for turn in reversed(self.state.turns):
            if turn.get("role") == "user":
                return turn.get("text")
        return None

    def get_last_agent_message(self) -> Optional[str]:
        """Return text of the most recent agent message, or None."""
        for turn in reversed(self.state.turns):
            if turn.get("role") == "agent":
                return turn.get("text")
        return None

    def get_context_summary(self) -> str:
        """Generate a structured summary of the conversation state.

        Suitable for LLM context prompts, ticket escalation summaries [ESC-3],
        and session inspection.

        Returns:
            str: Multi-line formatted context summary.
        """
        latest_intent = (
            self.state.intent_history[-1] if self.state.intent_history else "None"
        )
        all_intents_str = (
            ", ".join(self.state.intent_history)
            if self.state.intent_history
            else "None"
        )
        cited_str = (
            ", ".join(self.state.cited_sections)
            if self.state.cited_sections
            else "None"
        )

        customer_desc = self.state.customer_name or "Unidentified Customer"
        plus_desc = "Plus Member" if self.state.is_plus else "Standard Member"
        order_desc = self.state.current_order_id or "None"
        verified_desc = "Yes" if self.state.verified else "No"
        ticket_desc = (
            f"Created ({self.state.ticket_id})"
            if self.state.ticket_created
            else "None"
        )

        lines: List[str] = [
            "=== Conversation Context Summary ===",
            f"Conversation ID: {self.state.conversation_id}",
            f"Detected Language: {self.state.detected_language}",
            f"Customer: {customer_desc} ({plus_desc})",
            f"Current Order ID: {order_desc} (Verified: {verified_desc})",
            f"Identity Verified: {verified_desc}",
            f"Pending Action: {self.state.pending_action or 'None'}",
            f"Escalation Ticket: {ticket_desc}",
            f"Current Intent: {latest_intent}",
            f"Intent History: {all_intents_str}",
            f"Cited Sections: {cited_str}",
            f"Total Turns: {len(self.state.turns)}",
        ]

        if self.state.turns:
            lines.append("--- Recent Turns ---")
            for turn in self.state.turns[-6:]:
                role = turn.get("role", "unknown")
                text = turn.get("text", "")
                lines.append(f"  [{role}]: {text}")

        return "\n".join(lines)
