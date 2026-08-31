import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Streamlit Page Config
st.set_page_config(
    page_title="LLMOps Evaluation Telemetry",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark-mode aesthetic
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3440;
    }
    .badge-passed {
        background-color: #0e3a2f;
        color: #27c93f;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #1ba94c;
    }
    .badge-failed {
        background-color: #3b1419;
        color: #ff5f56;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #ff3b30;
    }
</style>
""", unsafe_allow_html=True)


def load_history_data(history_path: str = "runs/history.jsonl") -> pd.DataFrame:
    """Parses runs/history.jsonl into a flat pandas DataFrame."""
    if not os.path.exists(history_path):
        return pd.DataFrame()

    records = []
    try:
        with open(history_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                agg = row.get("aggregate_metrics", {})

                # Normalize latency (convert float seconds to ms if < 10)
                lat = agg.get("latency", 0.0)
                if lat < 50.0:  # stored in seconds
                    lat_ms = lat * 1000.0
                else:
                    lat_ms = lat

                records.append({
                    "run_id": row.get("run_id", "")[:8],
                    "full_run_id": row.get("run_id", ""),
                    "timestamp": row.get("timestamp", ""),
                    "git_commit": row.get("git_commit", "n/a"),
                    "config_path": row.get("config_path", ""),
                    "dataset_path": row.get("dataset_path", ""),
                    "sut_provider": row.get("sut_provider", "Unknown"),
                    "judge_provider": row.get("judge_provider", "Unknown"),
                    "pass_rate": agg.get("pass_rate", 0.0),
                    "latency_ms": lat_ms,
                    "cost_usd": agg.get("cost", 0.0),
                    "faithfulness": agg.get("faithfulness", 0.0),
                    "relevance": agg.get("relevance", 0.0),
                    "correctness": agg.get("correctness", 0.0),
                    "context_precision": agg.get("context_precision"),
                    "context_recall": agg.get("context_recall"),
                    "sla_status": row.get("sla_status", "UNKNOWN")
                })
    except Exception as e:
        st.error(f"Error reading run history log: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt", ascending=True).reset_index(drop=True)
    return df


def main():
    st.title("⚡ LLMOps CI/CD Evaluation Telemetry")
    st.caption("Strictly Read-Only Presentation Layer — Parsed from `runs/history.jsonl`")

    df = load_history_data()

    if df.empty:
        st.warning("⚠️ No evaluation run records found in `runs/history.jsonl`. Run an evaluation first using `python3 run_eval.py`.")
        return

    # Sidebar Controls & Filters
    st.sidebar.header("🔍 Evaluation Filters")

    configs = ["All"] + sorted(list(df["config_path"].unique()))
    selected_config = st.sidebar.selectbox("Filter by Config", configs)

    suts = ["All"] + sorted(list(df["sut_provider"].unique()))
    selected_sut = st.sidebar.selectbox("Filter by SUT Provider", suts)

    statuses = ["All", "PASSED", "FAILED"]
    selected_status = st.sidebar.selectbox("Filter by SLA Status", statuses)

    # Apply Filters
    filtered_df = df.copy()
    if selected_config != "All":
        filtered_df = filtered_df[filtered_df["config_path"] == selected_config]
    if selected_sut != "All":
        filtered_df = filtered_df[filtered_df["sut_provider"] == selected_sut]
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["sla_status"] == selected_status]

    if filtered_df.empty:
        st.info("No runs match the selected filters.")
        return

    # Latest Run Telemetry
    latest_run = filtered_df.iloc[-1]
    prev_run = filtered_df.iloc[-2] if len(filtered_df) > 1 else None

    # Calculate Deltas
    pass_delta = f"{latest_run['pass_rate'] - prev_run['pass_rate']:.1f}%" if prev_run is not None else None
    lat_delta = f"{latest_run['latency_ms'] - prev_run['latency_ms']:.1f} ms" if prev_run is not None else None
    cost_delta = f"${latest_run['cost_usd'] - prev_run['cost_usd']:.6f}" if prev_run is not None else None

    # 2. KPI Metrics Cards
    st.subheader("📌 Latest Run Telemetry")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Pass Rate", f"{latest_run['pass_rate']:.1f}%", delta=pass_delta)

    with col2:
        st.metric("Average Latency", f"{latest_run['latency_ms']:.2f} ms", delta=lat_delta, delta_color="inverse")

    with col3:
        st.metric("Total Evaluation Cost", f"${latest_run['cost_usd']:.6f}", delta=cost_delta, delta_color="inverse")

    with col4:
        status_badge = "✅ PASSED" if latest_run['sla_status'] == "PASSED" else "❌ FAILED"
        st.metric("Latest SLA Status", status_badge)

    st.markdown("---")

    # 3. Interactive Telemetry Charts
    st.subheader("📈 Performance & Telemetry Trends")

    tab1, tab2 = st.tabs(["Quality Scores Trend", "Latency & Expenditure Profiling"])

    with tab1:
        # Prepare quality metrics plot dataframe
        trend_cols = ["faithfulness", "relevance", "correctness"]
        if filtered_df["context_precision"].notnull().any():
            trend_cols.append("context_precision")
        if filtered_df["context_recall"].notnull().any():
            trend_cols.append("context_recall")

        plot_df = filtered_df[["timestamp_dt", "run_id", "git_commit"] + trend_cols].copy()
        plot_df["run_label"] = plot_df["run_id"] + " (" + plot_df["git_commit"] + ")"

        fig = go.Figure()
        colors = {
            "faithfulness": "#00d2ff",
            "relevance": "#3a7bd5",
            "correctness": "#00f2fe",
            "context_precision": "#f7b731",
            "context_recall": "#fa8231"
        }

        for col in trend_cols:
            valid_rows = plot_df.dropna(subset=[col])
            fig.add_trace(go.Scatter(
                x=valid_rows["run_label"],
                y=valid_rows[col],
                mode="lines+markers",
                name=col.replace("_", " ").title(),
                line=dict(width=3, color=colors.get(col, "#ffffff")),
                marker=dict(size=8)
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            yaxis=dict(range=[0.0, 1.05], title="Score (0.0 - 1.0)"),
            xaxis=dict(title="Run ID (Git SHA)"),
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig_lat = px.line(
            filtered_df,
            x="run_id",
            y=["latency_ms", "cost_usd"],
            markers=True,
            title="Latency (ms) and Cost ($) over Run History",
            template="plotly_dark"
        )
        fig_lat.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    st.markdown("---")

    # 4. Raw Run History Ledger
    st.subheader("📜 Run History Ledger")
    display_cols = [
        "run_id", "timestamp", "git_commit", "sut_provider", "judge_provider",
        "pass_rate", "latency_ms", "cost_usd", "faithfulness", "relevance",
        "correctness", "context_precision", "context_recall", "sla_status"
    ]
    st.dataframe(
        filtered_df[display_cols].sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True
    )


if __name__ == "__main__":
    main()
