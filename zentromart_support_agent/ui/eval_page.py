"""Evaluation dashboard page for ZentroMart agent assessment."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from evaluation.evaluator import SingleQuestionEvaluator, ConversationEvaluator
from evaluation.report import EvaluationReport


def render_eval_page(agent) -> None:
    """Render the evaluation dashboard page.

    Args:
        agent: ZentroMartAgent instance.
    """
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #6C3CE9;">✅ Agent Evaluation Dashboard</h2>
        <p style="color: #888;">Test against 52 questions & 23 multi-turn conversations</p>
    </div>
    """, unsafe_allow_html=True)

    report_gen = EvaluationReport()

    # Tabs for different evaluations
    tab1, tab2 = st.tabs(["📝 Single Questions (52)", "💬 Conversations (23)"])

    with tab1:
        _render_questions_tab(agent, report_gen)

    with tab2:
        _render_conversations_tab(agent, report_gen)


def _render_questions_tab(agent, report_gen: EvaluationReport) -> None:
    """Render the single-question evaluation tab."""

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Run Question Evaluation", type="primary", use_container_width=True):
            st.session_state.run_q_eval = True

    if st.session_state.get("run_q_eval"):
        progress_bar = st.progress(0, text="Evaluating questions...")

        def update_progress(current, total):
            progress_bar.progress(current / total, text=f"Question {current}/{total}")

        evaluator = SingleQuestionEvaluator(agent, progress_callback=update_progress)

        with st.spinner("Running evaluation on 52 practice questions..."):
            results_df = evaluator.evaluate_all()
            st.session_state.q_results = results_df
            st.session_state.run_q_eval = False

        progress_bar.progress(1.0, text="Evaluation complete! ✅")

    # Display results
    if "q_results" in st.session_state:
        results_df = st.session_state.q_results
        summary = report_gen.generate_summary(results_df)

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("🎯 Decision Accuracy", f"{summary['decision_accuracy']}%")
        with m2:
            st.metric("🔍 Retrieval Precision", f"{summary['avg_retrieval_precision']}%")
        with m3:
            st.metric("📊 Retrieval Recall", f"{summary['avg_retrieval_recall']}%")
        with m4:
            st.metric("⭐ Overall Score", f"{summary['avg_overall_score']}%")

        st.divider()

        # Charts
        figures = report_gen.get_plotly_figures(results_df)

        c1, c2 = st.columns(2)
        with c1:
            if "decision_by_intent" in figures:
                st.plotly_chart(figures["decision_by_intent"], use_container_width=True)
        with c2:
            if "retrieval_distribution" in figures:
                st.plotly_chart(figures["retrieval_distribution"], use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if "must_include_coverage" in figures:
                st.plotly_chart(figures["must_include_coverage"], use_container_width=True)
        with c4:
            if "overall_score_dist" in figures:
                st.plotly_chart(figures["overall_score_dist"], use_container_width=True)

        if "language_breakdown" in figures:
            st.plotly_chart(figures["language_breakdown"], use_container_width=True)

        # Results table
        st.markdown("### 📋 Per-Question Results")
        display_cols = [
            col for col in ["id", "intent", "language", "expected_decision",
                            "predicted_decision", "decision_correct",
                            "retrieval_precision", "retrieval_recall",
                            "must_include_score", "overall_score"]
            if col in results_df.columns
        ]
        if display_cols:
            styled = results_df[display_cols].copy()
            st.dataframe(
                styled,
                use_container_width=True,
                height=400,
                column_config={
                    "decision_correct": st.column_config.CheckboxColumn("✅ Correct"),
                    "overall_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
                    "retrieval_precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=1),
                    "retrieval_recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1),
                    "must_include_score": st.column_config.ProgressColumn("Must-Include", min_value=0, max_value=1),
                },
            )

        # Detailed report download
        report_md = report_gen.generate_detailed_report(results_df)
        st.download_button(
            "📥 Download Detailed Report",
            report_md,
            file_name="evaluation_report.md",
            mime="text/markdown",
        )
    else:
        st.info("Click **Run Question Evaluation** to start testing the agent against 52 practice questions.")


def _render_conversations_tab(agent, report_gen: EvaluationReport) -> None:
    """Render the multi-turn conversation evaluation tab."""

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Run Conversation Evaluation", type="primary", use_container_width=True):
            st.session_state.run_c_eval = True

    if st.session_state.get("run_c_eval"):
        progress_bar = st.progress(0, text="Evaluating conversations...")

        def update_progress(current, total):
            progress_bar.progress(current / total, text=f"Conversation {current}/{total}")

        evaluator = ConversationEvaluator(agent, progress_callback=update_progress)

        with st.spinner("Running evaluation on 23 multi-turn conversations..."):
            results_df = evaluator.evaluate_all()
            st.session_state.c_results = results_df
            st.session_state.run_c_eval = False

        progress_bar.progress(1.0, text="Evaluation complete! ✅")

    if "c_results" in st.session_state:
        results_df = st.session_state.c_results
        summary = report_gen.generate_summary(results_df)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("🎯 Decision Accuracy", f"{summary['decision_accuracy']}%")
        with m2:
            n_convs = results_df["conversation_id"].nunique() if "conversation_id" in results_df.columns else 0
            st.metric("💬 Conversations", f"{n_convs}")
        with m3:
            st.metric("🔄 Total Turns", f"{len(results_df)}")
        with m4:
            st.metric("⭐ Overall Score", f"{summary['avg_overall_score']}%")

        st.divider()

        # Per-conversation breakdown
        if "conversation_id" in results_df.columns:
            st.markdown("### 📋 Per-Conversation Results")
            conv_summary = results_df.groupby("conversation_id").agg(
                scenario=("scenario", "first"),
                turns=("turn", "count"),
                decision_acc=("decision_correct", lambda x: x.astype(bool).mean()),
                avg_score=("overall_score", "mean"),
            ).reset_index()
            conv_summary["decision_acc"] = (conv_summary["decision_acc"] * 100).round(0)
            conv_summary["avg_score"] = (conv_summary["avg_score"] * 100).round(0)

            st.dataframe(conv_summary, use_container_width=True, height=350)

        # Turn-level details
        st.markdown("### 🔍 Turn-Level Details")
        display_cols = [
            col for col in ["conversation_id", "turn", "scenario",
                            "expected_decision", "predicted_decision",
                            "decision_correct", "overall_score",
                            "customer_message", "response"]
            if col in results_df.columns
        ]
        if display_cols:
            st.dataframe(results_df[display_cols], use_container_width=True, height=400)
    else:
        st.info("Click **Run Conversation Evaluation** to test the agent on 23 multi-turn conversations.")
