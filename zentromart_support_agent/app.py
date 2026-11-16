"""ZentroMart AI Customer Support Agent — Streamlit Application.

A production-grade, RAG-powered customer support chatbot with:
- Semantic search over 21 policy/help documents
- Identity-verified order lookup with tool calling
- Multi-turn conversation management
- Multilingual support (English, Hindi, Hinglish)
- Interactive analytics dashboard
- Comprehensive evaluation suite

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root and its parent are importable
_PROJECT_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _PROJECT_DIR.parent
for p in (str(_PARENT_DIR), str(_PROJECT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

# ── Page Configuration ──
st.set_page_config(
    page_title="ZentroMart AI Support Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background-color: #0E1117;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #6C3CE9 0%, #3b1f8c 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        margin: 0;
    }
    .main-header p {
        color: #d4c5f9;
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #6C3CE9;
        font-size: 1.8rem;
    }
    [data-testid="stMetricLabel"] {
        color: #FAFAFA;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #1a1a2e;
        border-radius: 12px;
        border: 1px solid #2a2a4a;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background-color: #6C3CE9;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #5a2fd4;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 8px 16px;
    }

    /* Dividers */
    hr {
        border-color: #2a2a4a;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 1rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Agent Initialization ──
@st.cache_resource(show_spinner="🚀 Initializing ZentroMart AI Agent...")
def get_agent():
    """Initialize and cache the ZentroMart support agent."""
    from agent.core import ZentroMartAgent
    agent = ZentroMartAgent()
    agent.initialize()
    return agent


def main():
    """Main application entry point."""

    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #6C3CE9;">🛒 ZentroMart</h2>
            <p style="color: #888; font-size: 0.9rem;">AI Support Agent</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        page = st.radio(
            "Navigation",
            options=["💬 Chat", "📊 Analytics", "✅ Evaluation"],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        # Agent status
        st.markdown("### ⚙️ System Status")
        try:
            agent = get_agent()
            st.success("Agent: Online ✅")
            st.caption(f"RAG: {'Initialized' if agent.rag_initialized else 'Pending'}")
        except Exception as e:
            st.error(f"Agent Error: {e}")
            agent = None

        st.divider()
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.75rem;">
            <p>Built with Sentence-Transformers,<br>ChromaDB & Streamlit</p>
            <p>Capstone Project FTTT 2026</p>
        </div>
        """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛒 ZentroMart AI Customer Support Agent</h1>
        <p>RAG-Powered · Multilingual · Policy-Grounded · Identity-Verified</p>
    </div>
    """, unsafe_allow_html=True)

    if agent is None:
        st.error("⚠️ Agent failed to initialize. Please check logs.")
        return

    # Page routing
    if page == "💬 Chat":
        from ui.chat_page import render_chat_page
        render_chat_page(agent)
    elif page == "📊 Analytics":
        from ui.analytics_page import render_analytics_page
        render_analytics_page()
    elif page == "✅ Evaluation":
        from ui.eval_page import render_eval_page
        render_eval_page(agent)

    # Footer
    st.markdown("""
    <div class="footer">
        ZentroMart AI Support Agent · Capstone Project FTTT 2026 · All data is synthetic
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
