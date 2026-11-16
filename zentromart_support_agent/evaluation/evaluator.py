"""Evaluation runner for ZentroMart support agent.

Runs the agent against practice questions (52 single-turn) and
practice conversations (23 multi-turn, 96 total customer turns),
collecting per-question and per-turn metrics.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from config import QUESTIONS_CSV, CONVERSATIONS_CSV
from evaluation.metrics import (
    retrieval_precision,
    retrieval_recall,
    decision_accuracy,
    must_include_score,
    must_not_include_score,
    overall_score,
)

logger = logging.getLogger(__name__)


def _parse_gold_sections(gold_evidence: str) -> List[str]:
    """Parse gold_evidence field into a list of section IDs.

    Examples:
        "RET-2 ; DEL-3" -> ["RET-2", "DEL-3"]
        "RET-2 ; products.csv:ELE-0012" -> ["RET-2", "products.csv:ELE-0012"]
    """
    if not gold_evidence or pd.isna(gold_evidence):
        return []
    return [s.strip() for s in gold_evidence.split(" ; ") if s.strip()]


class SingleQuestionEvaluator:
    """Evaluates the agent on 52 single-turn practice questions.

    Each question is processed independently with a fresh agent state.
    Results include decision accuracy, retrieval metrics, and content scores.
    """

    def __init__(self, agent: Any, progress_callback: Optional[Callable] = None) -> None:
        """Initialize with a ZentroMartAgent instance.

        Args:
            agent: A ZentroMartAgent instance.
            progress_callback: Optional callback(current, total) for progress tracking.
        """
        self.agent = agent
        self.progress_callback = progress_callback

    def evaluate_question(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single practice question.

        Args:
            row: A dict with keys from questions.csv (question, expected_decision, etc.)

        Returns:
            Dict with evaluation results for this question.
        """
        self.agent.reset()

        question = str(row.get("question", ""))
        expected_decision = str(row.get("expected_decision", ""))
        acceptable_decisions = str(row.get("acceptable_decisions", ""))
        gold_evidence_str = str(row.get("gold_evidence", ""))
        must_inc_str = str(row.get("must_include", ""))
        must_not_str = str(row.get("must_not_include", ""))
        question_id = str(row.get("id", ""))

        # Run agent
        try:
            result = self.agent.process_message(question)
        except Exception as e:
            logger.error("Agent error on question %s: %s", question_id, e)
            result = {
                "response": f"Error: {e}",
                "decision": "error",
                "cited_sections": [],
                "intent": "error",
                "language": "en",
            }

        response = result.get("response", "")
        pred_decision = result.get("decision", "")
        pred_sections = result.get("cited_sections", [])

        # Parse gold evidence
        gold_sections = _parse_gold_sections(gold_evidence_str)

        # Calculate metrics
        ret_p = retrieval_precision(pred_sections, gold_sections)
        ret_r = retrieval_recall(pred_sections, gold_sections)
        dec_ok = decision_accuracy(pred_decision, expected_decision, acceptable_decisions)
        mi_score = must_include_score(response, must_inc_str) if must_inc_str and not pd.isna(must_inc_str) else 1.0
        mni_ok = must_not_include_score(response, must_not_str) if must_not_str and not pd.isna(must_not_str) else True
        total = overall_score(ret_p, ret_r, dec_ok, mi_score, mni_ok)

        return {
            "id": question_id,
            "question": question[:100],
            "language": str(row.get("language", "")),
            "intent": str(row.get("intent", "")),
            "expected_decision": expected_decision,
            "predicted_decision": pred_decision,
            "decision_correct": dec_ok,
            "retrieval_precision": round(ret_p, 3),
            "retrieval_recall": round(ret_r, 3),
            "must_include_score": round(mi_score, 3),
            "must_not_include_ok": mni_ok,
            "overall_score": round(total, 3),
            "response": response[:200],
            "predicted_sections": pred_sections,
            "gold_sections": gold_sections,
        }

    def evaluate_all(self, questions_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Run evaluation on all practice questions.

        Args:
            questions_df: Optional DataFrame of questions. Loads from CSV if None.

        Returns:
            DataFrame with one row per question and all metrics.
        """
        if questions_df is None:
            questions_df = pd.read_csv(str(QUESTIONS_CSV), dtype=str, keep_default_na=False)

        results = []
        total = len(questions_df)

        for idx, row in questions_df.iterrows():
            row_dict = row.to_dict()
            try:
                result = self.evaluate_question(row_dict)
                results.append(result)
            except Exception as e:
                logger.error("Error evaluating question %s: %s", row_dict.get("id"), e)
                results.append({
                    "id": row_dict.get("id", ""),
                    "question": str(row_dict.get("question", ""))[:100],
                    "error": str(e),
                    "overall_score": 0.0,
                })

            if self.progress_callback:
                self.progress_callback(idx + 1, total)

        return pd.DataFrame(results)


class ConversationEvaluator:
    """Evaluates the agent on 23 multi-turn practice conversations.

    Each conversation is processed sequentially, maintaining state between turns.
    The agent is reset between conversations.
    """

    def __init__(self, agent: Any, progress_callback: Optional[Callable] = None) -> None:
        """Initialize with a ZentroMartAgent instance.

        Args:
            agent: A ZentroMartAgent instance.
            progress_callback: Optional callback(current, total) for progress tracking.
        """
        self.agent = agent
        self.progress_callback = progress_callback

    def evaluate_conversation(self, conv_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Evaluate a single multi-turn conversation.

        Args:
            conv_df: DataFrame rows for one conversation, sorted by turn.

        Returns:
            List of result dicts, one per turn.
        """
        self.agent.reset()
        results = []

        for _, row in conv_df.iterrows():
            row_dict = row.to_dict()
            customer_message = str(row_dict.get("customer_message", ""))
            expected_decision = str(row_dict.get("expected_decision", ""))
            must_inc_str = str(row_dict.get("must_include", ""))
            must_not_str = str(row_dict.get("must_not_include", ""))
            gold_evidence_str = str(row_dict.get("gold_evidence", ""))
            conv_id = str(row_dict.get("conversation_id", ""))
            turn = str(row_dict.get("turn", ""))

            try:
                result = self.agent.process_message(customer_message)
            except Exception as e:
                logger.error("Agent error on conv %s turn %s: %s", conv_id, turn, e)
                result = {
                    "response": f"Error: {e}",
                    "decision": "error",
                    "cited_sections": [],
                    "tool_calls": [],
                }

            response = result.get("response", "")
            pred_decision = result.get("decision", "")
            pred_sections = result.get("cited_sections", [])
            gold_sections = _parse_gold_sections(gold_evidence_str)

            ret_p = retrieval_precision(pred_sections, gold_sections)
            ret_r = retrieval_recall(pred_sections, gold_sections)
            dec_ok = decision_accuracy(pred_decision, expected_decision)
            mi_score = must_include_score(response, must_inc_str) if must_inc_str else 1.0
            mni_ok = must_not_include_score(response, must_not_str) if must_not_str else True
            total = overall_score(ret_p, ret_r, dec_ok, mi_score, mni_ok)

            results.append({
                "conversation_id": conv_id,
                "turn": turn,
                "scenario": str(row_dict.get("scenario", "")),
                "customer_message": customer_message[:100],
                "expected_decision": expected_decision,
                "predicted_decision": pred_decision,
                "decision_correct": dec_ok,
                "retrieval_precision": round(ret_p, 3),
                "retrieval_recall": round(ret_r, 3),
                "must_include_score": round(mi_score, 3),
                "must_not_include_ok": mni_ok,
                "overall_score": round(total, 3),
                "response": response[:200],
                "tool_calls": result.get("tool_calls", []),
                "expected_tool_calls": str(row_dict.get("expected_tool_calls", "")),
            })

        return results

    def evaluate_all(self, conversations_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Run evaluation on all practice conversations.

        Args:
            conversations_df: Optional DataFrame. Loads from CSV if None.

        Returns:
            DataFrame with one row per turn across all conversations.
        """
        if conversations_df is None:
            conversations_df = pd.read_csv(str(CONVERSATIONS_CSV), dtype=str, keep_default_na=False)

        all_results = []
        conv_ids = conversations_df["conversation_id"].unique()
        total_convs = len(conv_ids)

        for idx, conv_id in enumerate(conv_ids):
            conv_df = conversations_df[conversations_df["conversation_id"] == conv_id].sort_values("turn")
            try:
                results = self.evaluate_conversation(conv_df)
                all_results.extend(results)
            except Exception as e:
                logger.error("Error evaluating conversation %s: %s", conv_id, e)

            if self.progress_callback:
                self.progress_callback(idx + 1, total_convs)

        return pd.DataFrame(all_results)
