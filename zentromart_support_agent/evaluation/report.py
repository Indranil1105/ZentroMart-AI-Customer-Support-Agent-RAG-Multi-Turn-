"""Evaluation report generator with plotly visualizations.

Generates summary statistics, detailed markdown reports, and interactive
plotly charts for the ZentroMart support agent evaluation dashboard.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Color palette
COLORS = {
    "primary": "#6C3CE9",
    "accent": "#FF6B35",
    "success": "#00C851",
    "danger": "#FF4444",
    "warning": "#FFBB33",
    "info": "#33B5E5",
    "dark": "#1a1a2e",
}


class EvaluationReport:
    """Generate evaluation summaries, reports, and plotly visualizations."""

    def generate_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute overall summary statistics from evaluation results.

        Args:
            results_df: DataFrame with per-question or per-turn results.

        Returns:
            Dict with summary metrics.
        """
        if results_df.empty:
            return {
                "total_questions": 0,
                "decision_accuracy": 0.0,
                "avg_retrieval_precision": 0.0,
                "avg_retrieval_recall": 0.0,
                "avg_must_include": 0.0,
                "must_not_compliance": 0.0,
                "avg_overall_score": 0.0,
            }

        total = len(results_df)
        dec_col = "decision_correct"
        if dec_col not in results_df.columns:
            dec_accuracy = 0.0
        else:
            dec_accuracy = results_df[dec_col].astype(bool).mean()

        return {
            "total_questions": total,
            "decision_accuracy": round(dec_accuracy * 100, 1),
            "avg_retrieval_precision": round(results_df.get("retrieval_precision", pd.Series([0])).mean() * 100, 1),
            "avg_retrieval_recall": round(results_df.get("retrieval_recall", pd.Series([0])).mean() * 100, 1),
            "avg_must_include": round(results_df.get("must_include_score", pd.Series([0])).mean() * 100, 1),
            "must_not_compliance": round(results_df.get("must_not_include_ok", pd.Series([True])).astype(bool).mean() * 100, 1),
            "avg_overall_score": round(results_df.get("overall_score", pd.Series([0])).mean() * 100, 1),
        }

    def generate_detailed_report(self, results_df: pd.DataFrame) -> str:
        """Generate a detailed markdown evaluation report.

        Args:
            results_df: DataFrame with evaluation results.

        Returns:
            str: Markdown-formatted report.
        """
        summary = self.generate_summary(results_df)

        lines = [
            "# ZentroMart Agent Evaluation Report",
            "",
            "## Summary Metrics",
            "",
            f"| Metric | Score |",
            f"|--------|-------|",
            f"| Total Questions | {summary['total_questions']} |",
            f"| Decision Accuracy | {summary['decision_accuracy']}% |",
            f"| Retrieval Precision | {summary['avg_retrieval_precision']}% |",
            f"| Retrieval Recall | {summary['avg_retrieval_recall']}% |",
            f"| Must-Include Coverage | {summary['avg_must_include']}% |",
            f"| Must-Not Compliance | {summary['must_not_compliance']}% |",
            f"| **Overall Score** | **{summary['avg_overall_score']}%** |",
            "",
        ]

        # Per-intent breakdown
        if "intent" in results_df.columns:
            lines.append("## Performance by Intent")
            lines.append("")
            lines.append("| Intent | Count | Decision Acc. | Overall Score |")
            lines.append("|--------|-------|---------------|---------------|")

            for intent, group in results_df.groupby("intent"):
                count = len(group)
                dec_acc = group.get("decision_correct", pd.Series()).astype(bool).mean() * 100 if "decision_correct" in group else 0
                avg_score = group.get("overall_score", pd.Series([0])).mean() * 100
                lines.append(f"| {intent} | {count} | {dec_acc:.0f}% | {avg_score:.0f}% |")
            lines.append("")

        # Error analysis
        if "decision_correct" in results_df.columns:
            failures = results_df[~results_df["decision_correct"].astype(bool)]
            if not failures.empty:
                lines.append("## Error Analysis")
                lines.append("")
                lines.append(f"**{len(failures)} questions with incorrect decisions:**")
                lines.append("")
                for _, row in failures.head(10).iterrows():
                    q_id = row.get("id", row.get("conversation_id", "?"))
                    q_text = str(row.get("question", row.get("customer_message", "")))[:80]
                    exp = row.get("expected_decision", "?")
                    pred = row.get("predicted_decision", "?")
                    lines.append(f"- **{q_id}**: Expected `{exp}`, got `{pred}` — \"{q_text}...\"")
                lines.append("")

        return "\n".join(lines)

    def get_plotly_figures(self, results_df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Generate plotly figures for the evaluation dashboard.

        Args:
            results_df: DataFrame with evaluation results.

        Returns:
            Dict mapping chart names to plotly Figure objects.
        """
        figures = {}

        if results_df.empty:
            return figures

        # 1. Decision Accuracy by Intent
        if "intent" in results_df.columns and "decision_correct" in results_df.columns:
            intent_stats = results_df.groupby("intent").agg(
                correct=("decision_correct", lambda x: x.astype(bool).sum()),
                total=("decision_correct", "count"),
            ).reset_index()
            intent_stats["accuracy"] = (intent_stats["correct"] / intent_stats["total"] * 100).round(1)
            intent_stats["incorrect"] = intent_stats["total"] - intent_stats["correct"]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=intent_stats["intent"], y=intent_stats["correct"],
                name="Correct", marker_color=COLORS["success"]
            ))
            fig.add_trace(go.Bar(
                x=intent_stats["intent"], y=intent_stats["incorrect"],
                name="Incorrect", marker_color=COLORS["danger"]
            ))
            fig.update_layout(
                title="Decision Accuracy by Intent",
                barmode="stack",
                xaxis_title="Intent",
                yaxis_title="Questions",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            figures["decision_by_intent"] = fig

        # 2. Retrieval Metrics Distribution
        if "retrieval_precision" in results_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=results_df["retrieval_precision"],
                name="Precision", marker_color=COLORS["primary"],
                opacity=0.7, nbinsx=10,
            ))
            if "retrieval_recall" in results_df.columns:
                fig.add_trace(go.Histogram(
                    x=results_df["retrieval_recall"],
                    name="Recall", marker_color=COLORS["accent"],
                    opacity=0.7, nbinsx=10,
                ))
            fig.update_layout(
                title="Retrieval Precision & Recall Distribution",
                barmode="overlay",
                xaxis_title="Score",
                yaxis_title="Count",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            figures["retrieval_distribution"] = fig

        # 3. Must-Include Coverage
        if "must_include_score" in results_df.columns:
            bins = [0, 0.25, 0.5, 0.75, 1.01]
            labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
            results_df = results_df.copy()
            results_df["mi_bin"] = pd.cut(
                results_df["must_include_score"], bins=bins, labels=labels, include_lowest=True
            )
            mi_counts = results_df["mi_bin"].value_counts().sort_index()
            fig = px.bar(
                x=mi_counts.index.astype(str), y=mi_counts.values,
                labels={"x": "Coverage Range", "y": "Questions"},
                title="Must-Include Fact Coverage",
                color=mi_counts.values,
                color_continuous_scale=["#FF4444", "#FFBB33", "#33B5E5", "#00C851"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            figures["must_include_coverage"] = fig

        # 4. Language Breakdown
        if "language" in results_df.columns:
            lang_counts = results_df["language"].value_counts()
            fig = px.pie(
                values=lang_counts.values,
                names=lang_counts.index,
                title="Questions by Language",
                color_discrete_sequence=[COLORS["primary"], COLORS["accent"], COLORS["info"], COLORS["success"]],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            figures["language_breakdown"] = fig

        # 5. Overall Score Distribution
        if "overall_score" in results_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=results_df["overall_score"],
                marker_color=COLORS["primary"],
                nbinsx=15,
            ))
            fig.add_vline(
                x=results_df["overall_score"].mean(),
                line_dash="dash", line_color=COLORS["accent"],
                annotation_text=f"Mean: {results_df['overall_score'].mean():.2f}",
            )
            fig.update_layout(
                title="Overall Score Distribution",
                xaxis_title="Score",
                yaxis_title="Count",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            figures["overall_score_dist"] = fig

        return figures
