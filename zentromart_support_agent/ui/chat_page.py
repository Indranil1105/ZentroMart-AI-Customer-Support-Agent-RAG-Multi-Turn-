"""Chat page for ZentroMart AI Support Agent Streamlit app."""

import streamlit as st
from typing import Any, Dict


def render_chat_page(agent: Any) -> None:
    """Render the chat interface page.

    Args:
        agent: ZentroMartAgent instance from session state.
    """
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = {}

    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #6C3CE9;">💬 Customer Support Chat</h2>
        <p style="color: #888;">Ask about orders, returns, refunds, policies & more</p>
    </div>
    """, unsafe_allow_html=True)

    # Layout: chat area + sidebar
    chat_col, info_col = st.columns([3, 1])

    with info_col:
        st.markdown("### 📋 Session Info")

        # New conversation button
        if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
            agent.reset()
            st.session_state.messages = []
            st.session_state.agent_state = {}
            st.rerun()

        # Show conversation state
        state = agent.get_state_summary()

        # Language badge
        lang_map = {"en": "🇬🇧 English", "hi": "🇮🇳 Hindi", "hi-en": "🇮🇳 Hinglish"}
        lang = lang_map.get(state.get("language", "en"), f"🌐 {state.get('language', 'en')}")
        st.markdown(f"**Language:** {lang}")

        # Order info
        if state.get("current_order_id"):
            verified_icon = "✅" if state.get("verified") else "⏳"
            st.markdown(f"**Order:** `{state['current_order_id']}` {verified_icon}")
            if state.get("customer_name"):
                st.markdown(f"**Customer:** {state['customer_name']}")
            if state.get("is_plus"):
                st.markdown("**Membership:** ⭐ Plus Member")

        # Intent history
        if state.get("intent_history"):
            latest_intent = state["intent_history"][-1]
            intent_icons = {
                "order_status": "📦", "return": "↩️", "refund": "💰",
                "cancellation": "❌", "delivery": "🚚", "damage_report": "🔨",
                "warranty": "🛡️", "general_query": "❓", "escalation_request": "🆘",
            }
            icon = intent_icons.get(latest_intent, "💭")
            st.markdown(f"**Intent:** {icon} `{latest_intent}`")

        # Cited sections
        if state.get("cited_sections"):
            st.markdown("**📚 KB Sections Used:**")
            for sec in state["cited_sections"][-5:]:
                st.markdown(f"  `[{sec}]`")

        # Ticket info
        if state.get("ticket_created"):
            st.markdown(f"**🎫 Ticket:** `{state.get('ticket_id', 'N/A')}`")

        # Pending action
        if state.get("pending_action"):
            action_map = {
                "awaiting_verification": "🔐 Awaiting verification",
                "awaiting_order_id": "🔍 Awaiting order ID",
            }
            st.markdown(f"**Status:** {action_map.get(state['pending_action'], state['pending_action'])}")

        st.divider()
        st.markdown("### 💡 Try asking:")
        examples = [
            "What is the return window for electronics?",
            "My order 8978 2837 hasn't arrived yet",
            "Mera refund kab aayega?",
            "Is reverse pickup free for Plus members?",
            "I received a damaged TV worth ₹25,000",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
                st.session_state.example_query = ex
                st.rerun()

    with chat_col:
        # Chat container
        chat_container = st.container(height=500)

        with chat_container:
            # Display chat history
            for msg in st.session_state.messages:
                role = msg["role"]
                avatar = "👤" if role == "user" else "🤖"
                with st.chat_message(role, avatar=avatar):
                    st.markdown(msg["content"])

                    # Show metadata for agent messages
                    if role == "assistant" and msg.get("metadata"):
                        meta = msg["metadata"]
                        badges = []
                        if meta.get("decision"):
                            badges.append(f"`{meta['decision']}`")
                        if meta.get("intent"):
                            badges.append(f"`{meta['intent']}`")
                        if meta.get("cited_sections"):
                            secs = ", ".join(f"`[{s}]`" for s in meta["cited_sections"][:3])
                            badges.append(secs)
                        if badges:
                            st.caption(" · ".join(badges))

        # Chat input
        example_query = st.session_state.pop("example_query", None)
        user_input = st.chat_input("Type your message here...", key="chat_input")

        if example_query:
            user_input = example_query

        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Process with agent
            with st.spinner("Thinking..."):
                result = agent.process_message(user_input)

            # Add agent response
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "metadata": {
                    "decision": result.get("decision"),
                    "intent": result.get("intent"),
                    "cited_sections": result.get("cited_sections", []),
                    "tool_calls": result.get("tool_calls", []),
                    "language": result.get("language"),
                },
            })

            st.session_state.agent_state = agent.get_state_summary()
            st.rerun()
