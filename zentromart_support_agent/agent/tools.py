"""Tool manager wrapping synthetic customer-support tools for the ZentroMart capstone.

This module provides the ToolManager class which manages interactions with the underlying
simulated tools (lookup_order and create_ticket). It implements:
1. Identity verification gating: callers must provide registered phone last 4 digits OR delivery pincode.
2. Attempt tracking: enforces a strict limit of max 1 retry (2 total attempts per order_id).
3. Result caching: successfully verified orders are cached in memory.
4. Information protection: never discloses expected verification values or order details prior to verification.
5. Ticket escalation: wraps create_ticket for logging human handover requests with issue details.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Optional, Union

# Set up logger
logger = logging.getLogger(__name__)

# Fallback dataset directory path if config import is unavailable
DEFAULT_DATASET_DIR: str = r"c:\Users\Dell\Downloads\Capstone_Project_Dataset_FTTT_2026\Capstone_Project_Dataset_FTTT_2026"

try:
    from zentromart_support_agent.config import DATASET_DIR
    _DATASET_PATH: str = str(DATASET_DIR)
except ImportError:
    _DATASET_PATH = DEFAULT_DATASET_DIR


class ToolManager:
    """Manager for customer support tools with identity verification gating, caching, and ticket creation.

    Wraps the synthetic Capstone tools (lookup_order, create_ticket), providing:
    - Order identity verification tracking (maximum 1 retry, i.e., 2 total attempts per order_id).
    - In-memory caching of verified order details.
    - Safe ticket creation logging for human handovers.
    - Zero information leakage for unverified inquiries.

    Attributes:
        verification_attempts: Mapping of order_id to number of verification attempts made.
        verified_orders: Mapping of order_id to cached verified order details.
        MAX_VERIFICATION_ATTEMPTS: Maximum allowable verification attempts per order (2).
    """

    MAX_VERIFICATION_ATTEMPTS: int = 2
    VERIFICATION_MESSAGE: str = (
        "Ask for the registered phone last 4 digits OR delivery pincode."
    )

    def __init__(self, dataset_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize ToolManager, ensuring the dataset tools are on sys.path and imported.

        Args:
            dataset_dir: Optional directory path to the Capstone dataset containing the tools/ package.
                         Defaults to config.DATASET_DIR or the standard Capstone path.
        """
        target_dataset_dir = str(dataset_dir) if dataset_dir else _DATASET_PATH
        if target_dataset_dir not in sys.path:
            sys.path.insert(0, target_dataset_dir)

        # Import mock tools from tools.tools
        try:
            from tools.tools import create_ticket as mock_create_ticket
            from tools.tools import lookup_order as mock_lookup_order

            self._lookup_order_fn: Callable[..., Dict[str, Any]] = mock_lookup_order
            self._create_ticket_fn: Callable[..., Dict[str, Any]] = mock_create_ticket
        except ImportError as exc:
            logger.error("Failed to import tools.tools from %s: %s", target_dataset_dir, exc)
            raise ImportError(
                f"Could not import tools.tools from '{target_dataset_dir}'. "
                "Ensure Capstone_Project_Dataset_FTTT_2026 is accessible."
            ) from exc

        # State tracking
        self.verification_attempts: Dict[str, int] = {}
        self.verified_orders: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _normalize_order_id(order_id: Any) -> str:
        """Normalize order_id to a numeric digit string or stripped string.

        Args:
            order_id: Order identifier as string, integer, or raw token.

        Returns:
            str: Digit string if digits are present, otherwise stripped string.
        """
        if order_id is None:
            return ""
        raw = str(order_id).strip()
        digits = "".join(ch for ch in raw if "0" <= ch <= "9")
        return digits if digits else raw

    def is_verified(self, order_id: Any) -> bool:
        """Check whether an order has been successfully verified.

        Args:
            order_id: Order identifier.

        Returns:
            bool: True if the order has been verified and cached, False otherwise.
        """
        oid = self._normalize_order_id(order_id)
        if not oid:
            return False
        return oid in self.verified_orders

    def get_verified_order(self, order_id: Any) -> Optional[Dict[str, Any]]:
        """Retrieve verified order details if already verified, otherwise None.

        Args:
            order_id: Order identifier.

        Returns:
            dict or None: Cached order details dictionary if verified, else None.
        """
        oid = self._normalize_order_id(order_id)
        if not oid:
            return None
        cached = self.verified_orders.get(oid)
        if cached is None:
            return None
        return dict(cached)

    def can_retry_verification(self, order_id: Any) -> bool:
        """Return True if verification attempts for order_id are less than 2.

        Enforces the policy rule: Maximum 1 retry (2 total attempts).

        Args:
            order_id: Order identifier.

        Returns:
            bool: True if attempts < 2, False otherwise.
        """
        oid = self._normalize_order_id(order_id)
        if not oid:
            return True
        return self.verification_attempts.get(oid, 0) < self.MAX_VERIFICATION_ATTEMPTS

    def lookup_order(
        self,
        order_id: Any,
        verification_value: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Wrap mock lookup_order tool with identity verification gating and caching.

        Rules enforced:
        - If the order has already been verified, returns the cached verified order data immediately.
        - First lookup without verification_value returns verification_required asking for
          EITHER phone last 4 digits OR delivery pincode.
        - Does not increment attempts when no verification_value is provided.
        - Increments attempt count upon each verification value submission.
        - Max 1 retry allowed (2 total attempts per order_id).
        - If attempts >= 2, blocks further attempts and requires escalation.
        - Never reveals expected values or order details prior to successful verification.
        - Caches successfully verified order details in verified_orders.

        Args:
            order_id: Order identifier (e.g. '69520867', 'ORD-69520867', or 69520867).
            verification_value: Customer-supplied verification value (phone last 4 digits OR pincode).

        Returns:
            Dict[str, Any]: Order details if verified, or verification challenge/status dictionary.
        """
        oid = self._normalize_order_id(order_id)
        if not oid:
            return {
                "found": False,
                "verified": False,
                "verification_required": True,
                "error": "Invalid or missing order_id",
                "message": "A valid order ID must be provided.",
            }

        # 1. Return cached verified data if already verified
        if self.is_verified(oid):
            cached = self.get_verified_order(oid)
            if cached is not None:
                return cached

        # Normalize verification_value
        val_str = str(verification_value).strip() if verification_value is not None else ""

        # 2. Case: No verification value provided (initial inquiry or clarification)
        if not val_str:
            # Check if user has already exhausted verification attempts previously
            if not self.can_retry_verification(oid):
                return {
                    "found": False,
                    "verified": False,
                    "verification_required": True,
                    "max_attempts_exceeded": True,
                    "order_id": oid,
                    "attempts": self.verification_attempts.get(oid, 0),
                    "can_retry": False,
                    "message": "Maximum verification attempts exceeded (2/2). Escalation to human agent required.",
                }

            # Call mock lookup without verification value (does NOT increment attempts)
            try:
                res = self._lookup_order_fn(oid, verification_value=None)
                out = dict(res)
                out["verified"] = False
                out["verification_required"] = True
                out["can_retry"] = True
                out["attempts"] = self.verification_attempts.get(oid, 0)
                return out
            except Exception as exc:
                logger.error("lookup_order mock call failed for %s: %s", oid, exc)
                return {
                    "verified": False,
                    "verification_required": True,
                    "order_id": oid,
                    "attempts": self.verification_attempts.get(oid, 0),
                    "can_retry": True,
                    "message": self.VERIFICATION_MESSAGE,
                }

        # 3. Case: Verification value provided -> Check attempt budget
        if not self.can_retry_verification(oid):
            return {
                "found": False,
                "verified": False,
                "verification_required": True,
                "max_attempts_exceeded": True,
                "order_id": oid,
                "attempts": self.verification_attempts.get(oid, 0),
                "can_retry": False,
                "message": "Maximum verification attempts exceeded (2/2). Escalation to human agent required.",
            }

        # Increment verification attempt count
        current_attempts = self.verification_attempts.get(oid, 0) + 1
        self.verification_attempts[oid] = current_attempts

        # Call underlying mock tool
        try:
            raw_result = self._lookup_order_fn(oid, verification_value=val_str)
        except Exception as exc:
            logger.error("Error invoking mock lookup_order for %s: %s", oid, exc)
            raw_result = {
                "verified": False,
                "verification_required": True,
                "message": "Verification failed or order not found.",
            }

        # Check verification outcome
        if raw_result.get("verified") is True:
            # Verification successful -> Cache and return
            self.verified_orders[oid] = dict(raw_result)
            return dict(raw_result)

        # Verification failed
        can_retry = self.can_retry_verification(oid)
        out = dict(raw_result)
        out["verified"] = False
        out["verification_required"] = True
        out["order_id"] = oid
        out["attempts"] = current_attempts
        out["can_retry"] = can_retry

        if not can_retry:
            out["max_attempts_exceeded"] = True
            out["message"] = (
                "Verification failed or order not found. Maximum verification attempts reached (2/2)."
            )

        return out

    def create_ticket(
        self,
        summary: str,
        order_id: Optional[Union[str, int]] = None,
        reason: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Wrap mock create_ticket tool to log a human-handover support ticket.

        Args:
            summary: Escalation summary detailing customer issue, checks made, and escalation reason
                     (minimum 20 characters).
            order_id: Optional order ID associated with the escalation ticket.
            reason: Optional escalation reason category or description.
            intent: Optional customer intent classification (e.g. 'cancellation', 'return', 'delivery_inquiry').

        Returns:
            Dict[str, Any]: Result with 'created': True and 'ticket_id' on success,
                            or 'created': False and 'error' description on failure.
        """
        oid_str = self._normalize_order_id(order_id) if order_id is not None else None
        try:
            return self._create_ticket_fn(
                summary=summary,
                order_id=oid_str or (str(order_id) if order_id is not None else None),
                reason=reason,
                intent=intent,
            )
        except Exception as exc:
            logger.error("Ticket creation failed with exception: %s", exc)
            return {"created": False, "error": str(exc)}

    def reset(self) -> None:
        """Clear all session verification attempts and cached verified orders."""
        self.verification_attempts.clear()
        self.verified_orders.clear()


# Module-level default singleton instance
_DEFAULT_TOOL_MANAGER: Optional[ToolManager] = None


def get_tool_manager(dataset_dir: Optional[Union[str, Path]] = None) -> ToolManager:
    """Get or create the default global ToolManager instance.

    Args:
        dataset_dir: Optional directory path to the dataset.

    Returns:
        ToolManager: Shared ToolManager instance.
    """
    global _DEFAULT_TOOL_MANAGER
    if _DEFAULT_TOOL_MANAGER is None:
        _DEFAULT_TOOL_MANAGER = ToolManager(dataset_dir=dataset_dir)
    return _DEFAULT_TOOL_MANAGER


def lookup_order(
    order_id: Any,
    verification_value: Optional[Any] = None,
) -> Dict[str, Any]:
    """Look up an order using the default global ToolManager instance."""
    return get_tool_manager().lookup_order(order_id, verification_value=verification_value)


def create_ticket(
    summary: str,
    order_id: Optional[Union[str, int]] = None,
    reason: Optional[str] = None,
    intent: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an escalation ticket using the default global ToolManager instance."""
    return get_tool_manager().create_ticket(
        summary=summary,
        order_id=order_id,
        reason=reason,
        intent=intent,
    )


def reset_tools() -> None:
    """Reset the default global ToolManager instance."""
    global _DEFAULT_TOOL_MANAGER
    if _DEFAULT_TOOL_MANAGER is not None:
        _DEFAULT_TOOL_MANAGER.reset()


__all__ = [
    "ToolManager",
    "get_tool_manager",
    "lookup_order",
    "create_ticket",
    "reset_tools",
]
