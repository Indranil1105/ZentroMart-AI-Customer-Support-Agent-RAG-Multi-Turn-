"""Main agent orchestrator for ZentroMart AI Customer Support Agent.

This is the central brain that ties together all components:
- Conversation state management
- Identity verification flow
- RAG-powered knowledge retrieval
- Policy-grounded response generation
- Escalation handling with ticket creation
- Multi-turn context tracking
- Multilingual support (EN/HI/Hinglish)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from agent.conversation import (
    ConversationManager,
    ConversationState,
    PENDING_AWAITING_VERIFICATION,
    PENDING_AWAITING_ORDER_ID,
)
from agent.tools import ToolManager
from agent.rag import retrieve, retrieve_with_context, get_cited_section_ids, initialize_rag
from agent.response_generator import ResponseGenerator
from agent.business_days import (
    count_business_days,
    add_business_days,
    days_since,
    business_days_since,
    REFERENCE_DATE,
)
from agent.language_detector import detect_language

logger = logging.getLogger(__name__)


class ZentroMartAgent:
    """End-to-end customer support agent for ZentroMart.

    Orchestrates the full conversation lifecycle:
    1. Receives user messages
    2. Detects language and intent
    3. Manages identity verification flow
    4. Retrieves relevant KB context via RAG
    5. Generates policy-grounded responses
    6. Handles escalation triggers
    7. Tracks multi-turn conversation state

    Attributes:
        conversation: ConversationManager instance for state tracking.
        tools: ToolManager instance for order lookup and ticket creation.
        response_gen: ResponseGenerator instance for template-based responses.
        rag_initialized: Whether the RAG pipeline has been initialized.
    """

    def __init__(self) -> None:
        """Initialize the ZentroMart support agent with all subsystems."""
        self.conversation = ConversationManager()
        self.tools = ToolManager()
        self.response_gen = ResponseGenerator()
        self.rag_initialized = False

    def initialize(self) -> None:
        """Initialize the RAG pipeline (one-time setup, downloads model on first run)."""
        if not self.rag_initialized:
            try:
                initialize_rag()
                self.rag_initialized = True
                logger.info("RAG pipeline initialized successfully.")
            except Exception as e:
                logger.error("Failed to initialize RAG pipeline: %s", e)
                self.rag_initialized = True  # Mark as done to avoid retry loops

    def process_message(self, user_message: str) -> Dict[str, Any]:
        """Process an incoming user message and return a structured response.

        This is the main entry point for the agent. It handles the full
        conversation flow including greeting, verification, retrieval, and response.

        Args:
            user_message: The raw text message from the customer.

        Returns:
            Dict with keys:
                - response (str): The agent's reply text
                - decision (str): answer/clarify/escalate/refuse
                - intent (str): Detected intent category
                - cited_sections (list[str]): KB section IDs used
                - tool_calls (list[str]): Tool calls made this turn
                - language (str): Detected language code
                - needs_input (str|None): What we're waiting for, if anything
        """
        if not user_message:
            user_message = ""

        # Step 1: Add user message to conversation (updates language, order_id, intent)
        self.conversation.add_user_message(user_message)
        language = self.conversation.state.detected_language
        intent = self.conversation.state.intent_history[-1] if self.conversation.state.intent_history else "general_query"

        tool_calls: List[str] = []
        order_data: Optional[Dict[str, Any]] = None
        cited_sections: List[str] = []
        decision = "answer"
        response_parts: List[str] = []

        # Step 2: First message greeting [STY-1] & Turn 1 inquiry check
        is_first_turn = len(self.conversation.state.turns) <= 1
        just_verified = False

        # Step 3: Handle ongoing verification flow
        if self.conversation.state.pending_action == PENDING_AWAITING_VERIFICATION:
            result = self._handle_verification(user_message, language)
            result["intent"] = intent
            if result.get("decision") in ("clarify", "refuse"):
                return result
            # Verification succeeded — continue to answer
            order_data = result.get("_order_data")
            tool_calls = result.get("tool_calls", [])
            just_verified = True

        # Step 4: Check for new order ID
        order_id = self.conversation.extract_order_id(user_message)
        if order_id:
            self.conversation.state.current_order_id = order_id

        # Step 5: Check if this is a general/vague opening requesting order help without specific details
        # e.g., "Hello, mujhe order ke baare mein help chahiye." / "Hi, I need some help with an order."
        current_oid = self.conversation.state.current_order_id
        if is_first_turn and not current_oid:
            user_msg_clean = user_message.strip().lower()
            vague_openers = [
                "help chahiye", "help with an order", "help with my order", 
                "मदद चाहिए", "order ke baare mein help", "hasn't arrived yet. where is it"
            ]
            # If the user message is just a general request for order help (under 60 chars) without asking a specific factual policy question
            is_vague_help = any(v in user_msg_clean for v in vague_openers) and len(user_msg_clean) < 70
            if is_vague_help:
                decision = "clarify"
                self.conversation.state.pending_action = PENDING_AWAITING_ORDER_ID
                greeting = self.response_gen.generate_greeting(language)
                clarification = self.response_gen.generate_clarification("order_id", language)
                response = greeting + " " + clarification
                self.conversation.add_agent_message(response)
                return {
                    "response": response,
                    "decision": decision,
                    "intent": intent,
                    "cited_sections": [],
                    "tool_calls": tool_calls,
                    "language": language,
                    "needs_input": PENDING_AWAITING_ORDER_ID,
                }

        # Step 6: If we have an unverified order ID, start verification
        if current_oid and not self.tools.is_verified(current_oid) and not order_data:
            if order_id:  # Only trigger verification if order ID was mentioned in this turn
                result = self.tools.lookup_order(current_oid)
                tool_calls.append(f"lookup_order(order_id={current_oid})")
                self.conversation.state.pending_action = PENDING_AWAITING_VERIFICATION

                decision = "clarify"
                greeting = ""
                if is_first_turn:
                    greeting = self.response_gen.generate_greeting(language) + " "

                clarification = self.response_gen.generate_clarification("verification", language)
                response = greeting + clarification

                self.conversation.add_agent_message(response)
                return {
                    "response": response,
                    "decision": decision,
                    "intent": intent,
                    "cited_sections": [],
                    "tool_calls": tool_calls,
                    "language": language,
                    "needs_input": PENDING_AWAITING_VERIFICATION,
                }

        # Step 7: Get verified order data if available
        if current_oid and self.tools.is_verified(current_oid) and not order_data:
            order_data = self.tools.get_verified_order(current_oid)

        # Step 8: Check escalation triggers [ESC-1]
        # Notice: When user JUST submitted verification digits, they haven't stated their problem/question yet!
        # Do NOT escalate immediately on the verification turn itself.
        should_esc = False
        esc_reason = ""
        if not just_verified:
            should_esc, esc_reason = self.conversation.should_escalate(order_data)
            if intent == "escalation_request":
                should_esc = True
                esc_reason = esc_reason or "Customer requested handover to a human agent [ESC-1]"

        if should_esc:
            return self._handle_escalation(
                intent, language, esc_reason, order_data, tool_calls, is_first_turn
            )

        # Step 9: If just verified, reply with verification confirmation and order status
        if just_verified and order_data:
            decision = "answer"
            status = order_data.get("status", "")
            promised_date = order_data.get("promised_date", "")
            first_name = order_data.get("customer_first_name", "")

            # Format status response with promised date if shipped
            formatted_date = ""
            if promised_date:
                from agent.response_generator import format_date
                formatted_date = format_date(promised_date, language)

            if language == "hi":
                if status == "shipped" and formatted_date:
                    resp = f"सत्यापन सफल रहा। आपका ऑर्डर भेजा जा चुका है और {formatted_date} तक डिलीवर होने की उम्मीद है। मैं आज आपकी क्या सहायता कर सकता हूँ? [TRK-1]"
                else:
                    resp = f"सत्यापन सफल रहा। आपका ऑर्डर मिल गया है। मैं आज आपकी क्या मदद कर सकता हूँ? [IDV-2]"
            elif language == "hi-en":
                if status == "shipped" and formatted_date:
                    resp = f"Verification successful. Aapka order ship ho chuka hai aur {formatted_date} tak deliver hone ki umeed hai. Main aapki kya madad kar sakta hoon? [TRK-1]"
                else:
                    resp = f"Verification successful. Main aapki kya madad kar sakta hoon? [IDV-2]"
            else:
                if status == "shipped" and formatted_date:
                    resp = f"Thank you for verifying. Your order has shipped and is promised by {formatted_date}. How can I assist you with this order? [TRK-1]"
                else:
                    resp = f"Thank you for verifying. How can I assist you with your order today? [IDV-2]"

            self.conversation.add_agent_message(resp)
            return {
                "response": resp,
                "decision": decision,
                "intent": intent,
                "cited_sections": ["TRK-1"] if (status == "shipped" and formatted_date) else ["IDV-2"],
                "tool_calls": tool_calls,
                "language": language,
                "needs_input": None,
            }

        # Step 8: RAG retrieval
        if not self.rag_initialized:
            self.initialize()

        try:
            retrieved = retrieve(user_message, top_k=5)
            cited_sections = get_cited_section_ids(retrieved)
            context = retrieve_with_context(user_message, order_data)
        except Exception as e:
            logger.error("RAG retrieval failed: %s", e)
            retrieved = []
            context = ""
            cited_sections = []

        # Step 9: Generate response
        answer = self.response_gen.generate_response(
            query=user_message,
            context=context,
            conversation_state=self.conversation.state,
            order_data=order_data,
            decision="answer",
        )

        # Add greeting for first turn if not already included
        if is_first_turn and not any(
            "ZentroMart" in t.get("text", "") 
            for t in self.conversation.state.turns 
            if t.get("role") == "agent"
        ):
            greeting = self.response_gen.generate_greeting(language)
            response = greeting + " " + answer
        else:
            response = answer

        # Step 10: Record agent message
        self.conversation.add_agent_message(response, cited_sections)
        self.conversation.state.cited_sections = cited_sections

        return {
            "response": response,
            "decision": decision,
            "intent": intent,
            "cited_sections": cited_sections,
            "tool_calls": tool_calls,
            "language": language,
            "needs_input": None,
        }

    def _handle_verification(
        self, user_message: str, language: str
    ) -> Dict[str, Any]:
        """Handle the verification sub-flow when awaiting verification value.

        Args:
            user_message: The user's latest message.
            language: Detected language code.

        Returns:
            Dict with response details. If verification succeeds, includes '_order_data'.
        """
        tool_calls: List[str] = []
        oid = self.conversation.state.current_order_id

        if not oid:
            return {
                "response": self.response_gen.generate_clarification("order_id", language),
                "decision": "clarify",
                "intent": "",
                "cited_sections": [],
                "tool_calls": [],
                "language": language,
                "needs_input": PENDING_AWAITING_ORDER_ID,
            }

        ver_val = self.conversation.extract_verification_value(user_message)

        if ver_val:
            result = self.tools.lookup_order(oid, ver_val)
            tool_calls.append(f"lookup_order(order_id={oid}, verification_value={ver_val})")

            if result.get("verified"):
                # Verification succeeded
                self.conversation.state.verified = True
                self.conversation.state.pending_action = None
                self.conversation.state.customer_name = result.get("customer_first_name")
                self.conversation.state.is_plus = result.get("is_plus", False)
                return {
                    "decision": "answer",
                    "tool_calls": tool_calls,
                    "_order_data": result,
                    "language": language,
                }
            else:
                # Verification failed
                if not self.tools.can_retry_verification(oid):
                    # Max attempts exceeded
                    self.conversation.state.pending_action = None
                    response = self.response_gen.generate_refusal("verification_failed", language)
                    self.conversation.add_agent_message(response)
                    return {
                        "response": response,
                        "decision": "refuse",
                        "intent": "",
                        "cited_sections": [],
                        "tool_calls": tool_calls,
                        "language": language,
                        "needs_input": None,
                    }
                else:
                    # Allow retry
                    response = self.response_gen.generate_clarification("verification_retry", language)
                    self.conversation.add_agent_message(response)
                    return {
                        "response": response,
                        "decision": "clarify",
                        "intent": "",
                        "cited_sections": [],
                        "tool_calls": tool_calls,
                        "language": language,
                        "needs_input": PENDING_AWAITING_VERIFICATION,
                    }
        else:
            # No verification value found in message
            response = self.response_gen.generate_clarification("verification", language)
            self.conversation.add_agent_message(response)
            return {
                "response": response,
                "decision": "clarify",
                "intent": "",
                "cited_sections": [],
                "tool_calls": tool_calls,
                "language": language,
                "needs_input": PENDING_AWAITING_VERIFICATION,
            }

    def _handle_escalation(
        self,
        intent: str,
        language: str,
        esc_reason: str,
        order_data: Optional[Dict[str, Any]],
        tool_calls: List[str],
        is_first_turn: bool,
    ) -> Dict[str, Any]:
        """Handle escalation by creating a ticket and generating an escalation response.

        Args:
            intent: Detected intent.
            language: Detected language.
            esc_reason: Reason for escalation.
            order_data: Optional verified order data.
            tool_calls: List of tool calls made so far.
            is_first_turn: Whether this is the first turn.

        Returns:
            Dict with escalation response details.
        """
        oid = self.conversation.state.current_order_id

        # Build a detailed summary for the ticket [ESC-3]
        summary_parts = [
            f"Issue: {intent}",
            f"Order: {oid or 'N/A'}",
            f"Language: {language}",
            f"Reason: {esc_reason}",
        ]
        if self.conversation.state.customer_name:
            summary_parts.append(f"Customer: {self.conversation.state.customer_name}")

        # Add conversation context
        user_messages = [
            t.get("text", "")[:100]
            for t in self.conversation.state.turns
            if t.get("role") == "user"
        ]
        if user_messages:
            summary_parts.append(f"Customer said: {' | '.join(user_messages[-3:])}")

        summary = ". ".join(summary_parts)
        if len(summary) < 20:
            summary = summary + " — Escalation required per ZentroMart policy."

        ticket_result = self.tools.create_ticket(
            summary=summary,
            order_id=oid,
            reason=esc_reason,
            intent=intent,
        )
        tool_calls.append("create_ticket()")

        ticket_id = ticket_result.get("ticket_id") if ticket_result.get("created") else None
        self.conversation.state.ticket_created = ticket_result.get("created", False)
        self.conversation.state.ticket_id = ticket_id

        # Determine escalation category for response template
        esc_category = self._categorize_escalation(esc_reason)

        response = self.response_gen.generate_escalation(esc_category, language, ticket_id)

        if is_first_turn:
            greeting = self.response_gen.generate_greeting(language)
            response = greeting + " " + response

        self.conversation.add_agent_message(response)
        self.conversation.state.pending_action = None

        return {
            "response": response,
            "decision": "escalate",
            "intent": intent,
            "cited_sections": [],
            "tool_calls": tool_calls,
            "language": language,
            "needs_input": None,
        }

    @staticmethod
    def _categorize_escalation(reason: str) -> str:
        """Map an escalation reason string to a response template category.

        Args:
            reason: The raw escalation reason string.

        Returns:
            str: Category key for response_generator.generate_escalation().
        """
        reason_lower = reason.lower()
        if "safety" in reason_lower or "fire" in reason_lower or "battery" in reason_lower:
            return "safety_risk"
        if "unauthorized" in reason_lower or "fraud" in reason_lower:
            return "fraud"
        if "legal" in reason_lower or "court" in reason_lower or "police" in reason_lower:
            return "legal"
        if "contact" in reason_lower or "repeat" in reason_lower or "third" in reason_lower:
            return "repeat_contact"
        if "refund" in reason_lower and ("delay" in reason_lower or "past" in reason_lower or "exceed" in reason_lower):
            return "refund_delay"
        if "10,000" in reason_lower or "10000" in reason_lower or "damage" in reason_lower:
            return "high_value_damage"
        if "human" in reason_lower or "agent" in reason_lower or "request" in reason_lower:
            return "human_requested"
        return "generic"

    def reset(self) -> None:
        """Reset the agent for a new conversation."""
        self.conversation.reset()
        self.tools.reset()
        logger.info("Agent reset for new conversation.")

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a summary of the current conversation state for the UI.

        Returns:
            Dict with conversation state details.
        """
        state = self.conversation.state
        return {
            "conversation_id": state.conversation_id,
            "turn_count": len(state.turns),
            "language": state.detected_language,
            "current_order_id": state.current_order_id,
            "verified": state.verified,
            "customer_name": state.customer_name,
            "is_plus": state.is_plus,
            "intent_history": state.intent_history,
            "cited_sections": state.cited_sections,
            "pending_action": state.pending_action,
            "ticket_created": state.ticket_created,
            "ticket_id": state.ticket_id,
        }


__all__ = ["ZentroMartAgent"]
