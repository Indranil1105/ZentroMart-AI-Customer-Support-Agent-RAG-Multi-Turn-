"""Analytics dashboard page for ZentroMart data insights."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from data_loader.loader import (
    get_orders, get_customers, get_products,
    get_order_items, get_pincodes, get_support_logs,
)

COLORS = ["#6C3CE9", "#FF6B35", "#00C851", "#33B5E5", "#FFBB33", "#FF4444", "#AA66CC", "#99CC00"]


@st.cache_data(ttl=600)
def _load_analytics_data():
    """Load and preprocess data for analytics."""
    orders = get_orders()
    customers = get_customers()
    products = get_products()
    order_items = get_order_items()
    support_logs = get_support_logs()
    pincodes = get_pincodes()

    # Convert numeric columns
    for col in ["total_inr", "subtotal_inr", "delivery_fee_inr"]:
        if col in orders.columns:
            orders[col] = pd.to_numeric(orders[col], errors="coerce").fillna(0)

    if "price_inr" in products.columns:
        products["price_inr"] = pd.to_numeric(products["price_inr"], errors="coerce").fillna(0)

    if "unit_price_inr" in order_items.columns:
        order_items["unit_price_inr"] = pd.to_numeric(order_items["unit_price_inr"], errors="coerce").fillna(0)
    if "qty" in order_items.columns:
        order_items["qty"] = pd.to_numeric(order_items["qty"], errors="coerce").fillna(1).astype(int)

    if "csat" in support_logs.columns:
        support_logs["csat"] = pd.to_numeric(support_logs["csat"], errors="coerce")

    return orders, customers, products, order_items, support_logs, pincodes


def render_analytics_page() -> None:
    """Render the analytics dashboard page."""
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #6C3CE9;">📊 ZentroMart Data Analytics</h2>
        <p style="color: #888;">Insights from 1,000 orders, 300 customers, 70 products</p>
    </div>
    """, unsafe_allow_html=True)

    orders, customers, products, order_items, support_logs, pincodes = _load_analytics_data()

    # ── KPI Cards ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Total Orders", f"{len(orders):,}")
    with col2:
        total_rev = orders["total_inr"].sum()
        st.metric("💰 Total Revenue", f"₹{total_rev:,.0f}")
    with col3:
        st.metric("👥 Customers", f"{len(customers):,}")
    with col4:
        plus_pct = (customers["is_plus"].str.lower() == "true").mean() * 100
        st.metric("⭐ Plus Members", f"{plus_pct:.0f}%")

    st.divider()

    # ── Row 1: Order Status + Payment Method ──
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        status_counts = orders["status"].value_counts()
        fig = px.pie(
            values=status_counts.values, names=status_counts.index,
            title="Order Status Distribution",
            color_discrete_sequence=COLORS,
            hole=0.4,
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        payment_rev = orders.groupby("payment_method")["total_inr"].sum().sort_values(ascending=True)
        fig = px.bar(
            x=payment_rev.values, y=payment_rev.index,
            title="Revenue by Payment Method",
            orientation="h",
            color=payment_rev.index,
            color_discrete_sequence=COLORS,
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, xaxis_title="Revenue (₹)", yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Top Products + Plus vs Regular ──
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        item_counts = order_items.groupby("title")["qty"].sum().nlargest(10).sort_values()
        fig = px.bar(
            x=item_counts.values, y=item_counts.index,
            title="Top 10 Products by Units Sold",
            orientation="h",
            color_discrete_sequence=[COLORS[0]],
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Units", yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        plus_counts = customers["is_plus"].str.lower().map({"true": "Plus ⭐", "false": "Regular"}).value_counts()
        fig = px.pie(
            values=plus_counts.values, names=plus_counts.index,
            title="Plus vs Regular Customers",
            color_discrete_sequence=[COLORS[0], COLORS[1]],
            hole=0.5,
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Orders Timeline + Delivery Performance ──
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        orders_copy = orders.copy()
        orders_copy["placed_date"] = pd.to_datetime(orders_copy["placed_at"], errors="coerce").dt.date
        daily = orders_copy.groupby("placed_date").size().reset_index(name="count")
        fig = px.line(
            daily, x="placed_date", y="count",
            title="Orders per Day",
            color_discrete_sequence=[COLORS[0]],
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Date", yaxis_title="Orders",
        )
        st.plotly_chart(fig, use_container_width=True)

    with r3c2:
        delivered = orders[orders["delivered_on"].str.strip() != ""].copy()
        if not delivered.empty:
            delivered["promised"] = pd.to_datetime(delivered["promised_date"], errors="coerce")
            delivered["actual"] = pd.to_datetime(delivered["delivered_on"], errors="coerce")
            delivered["on_time"] = delivered["actual"] <= delivered["promised"]
            on_time_pct = delivered["on_time"].mean() * 100

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=on_time_pct,
                title={"text": "On-Time Delivery Rate"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLORS[0]},
                    "steps": [
                        {"range": [0, 50], "color": "#FF4444"},
                        {"range": [50, 75], "color": "#FFBB33"},
                        {"range": [75, 100], "color": "#00C851"},
                    ],
                },
                number={"suffix": "%"},
            ))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: State Distribution + Return Rate ──
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        state_counts = customers["state"].value_counts().head(10).sort_values()
        fig = px.bar(
            x=state_counts.values, y=state_counts.index,
            title="Top 10 States by Customers",
            orientation="h",
            color_discrete_sequence=[COLORS[2]],
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Customers", yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    with r4c2:
        returns = orders[orders["return_status"].str.strip() != ""]
        if not returns.empty:
            merged = returns.merge(order_items[["order_id", "return_category"]].drop_duplicates(), on="order_id", how="left")
            cat_col = "return_category"
            if cat_col in merged.columns:
                return_rates = merged[cat_col].value_counts()
                fig = px.bar(
                    x=return_rates.index, y=return_rates.values,
                    title="Returns by Product Category",
                    color=return_rates.index,
                    color_discrete_sequence=COLORS,
                )
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False, xaxis_title="Category", yaxis_title="Returns",
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Row 5: Support Analysis ──
    st.divider()
    st.markdown("### 🎧 Support Log Analysis")

    s1, s2, s3 = st.columns(3)

    with s1:
        if "intent" in support_logs.columns:
            intent_counts = support_logs["intent"].value_counts().head(8)
            fig = px.bar(
                x=intent_counts.values, y=intent_counts.index,
                title="Top Support Intents",
                orientation="h",
                color_discrete_sequence=[COLORS[4]],
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Count", yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)

    with s2:
        if "csat" in support_logs.columns:
            csat_data = support_logs["csat"].dropna()
            if not csat_data.empty:
                fig = px.histogram(
                    csat_data, nbins=5,
                    title=f"CSAT Distribution (Avg: {csat_data.mean():.1f})",
                    color_discrete_sequence=[COLORS[0]],
                )
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="CSAT Score", yaxis_title="Count",
                )
                st.plotly_chart(fig, use_container_width=True)

    with s3:
        if "escalated" in support_logs.columns:
            esc_counts = support_logs["escalated"].str.lower().map(
                {"true": "Escalated", "false": "Resolved"}
            ).value_counts()
            fig = px.pie(
                values=esc_counts.values, names=esc_counts.index,
                title="Escalation Rate",
                color_discrete_sequence=[COLORS[5], COLORS[2]],
                hole=0.4,
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
