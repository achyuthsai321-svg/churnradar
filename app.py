"""
src/dashboard/app.py
ChurnRadar — Streamlit retention intelligence dashboard.
Run: streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnRadar",
    page_icon="🎯",
    layout="wide",
)

# ── Load model artifacts ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("models/xgboost_churn.pkl")
    scaler   = joblib.load("models/scaler.pkl")
    features = joblib.load("models/feature_names.pkl")
    return model, scaler, features

@st.cache_data
def load_scored_data():
    """Load processed dataset and run batch scoring for the dashboard."""
    df = pd.read_csv("data/processed/telco_clean.csv")
    model, scaler, features = load_artifacts()

    X = df[features]
    X_sc = scaler.transform(X)
    probs = model.predict_proba(X_sc)[:, 1]

    df["churn_probability"] = probs
    df["risk_level"] = pd.cut(
        probs,
        bins=[-0.01, 0.40, 0.65, 1.01],
        labels=["LOW", "MEDIUM", "HIGH"]
    )
    return df

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("## 🎯 ChurnRadar — Retention Intelligence Dashboard")
st.caption("Predicts customer churn 30–60 days before it happens.")
st.divider()

df = load_scored_data()

# ── KPI Cards ──────────────────────────────────────────────────────────────────
total       = len(df)
high_risk   = int((df["risk_level"] == "HIGH").sum())
medium_risk = int((df["risk_level"] == "MEDIUM").sum())
low_risk    = int((df["risk_level"] == "LOW").sum())
actual_churn = int(df["Churn"].sum()) if "Churn" in df.columns else "—"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers",  f"{total:,}")
col2.metric("🔴 High Risk",     f"{high_risk:,}",   delta=f"{high_risk/total*100:.1f}%")
col3.metric("🟡 Medium Risk",   f"{medium_risk:,}", delta=f"{medium_risk/total*100:.1f}%")
col4.metric("🟢 Low Risk",      f"{low_risk:,}")
col5.metric("Actual Churn",     f"{actual_churn:,}" if isinstance(actual_churn, int) else actual_churn)

st.divider()

# ── Charts row ─────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Churn Probability Distribution")
    fig = px.histogram(
        df, x="churn_probability", nbins=40,
        color_discrete_sequence=["#E24B4A"],
        labels={"churn_probability": "Churn Probability"},
    )
    fig.update_layout(bargap=0.05, height=300)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Risk Segment Breakdown")
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    colors = {"HIGH": "#E24B4A", "MEDIUM": "#F5A623", "LOW": "#1D9E75"}
    fig2 = px.pie(
        risk_counts, names="Risk Level", values="Count",
        color="Risk Level", color_discrete_map=colors,
        hole=0.45,
    )
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

# ── Churn by contract type ──────────────────────────────────────────────────────
if "Contract" in df.columns:
    st.subheader("Avg Churn Probability by Contract Type")
    contract_churn = df.groupby("Contract")["churn_probability"].mean().reset_index()
    fig3 = px.bar(
        contract_churn, x="Contract", y="churn_probability",
        color_discrete_sequence=["#3B5BDB"],
        labels={"churn_probability": "Avg Churn Probability"},
    )
    fig3.update_layout(height=280)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── At-risk customer table ──────────────────────────────────────────────────────
st.subheader("🔴 High-Risk Customers — Action Required")

risk_filter = st.selectbox("Filter by risk level", ["HIGH", "MEDIUM", "LOW"], index=0)
filtered    = df[df["risk_level"] == risk_filter].sort_values(
    "churn_probability", ascending=False
).head(50)

display_cols = ["churn_probability", "risk_level", "tenure",
                "MonthlyCharges", "Contract"]
display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(
    filtered[display_cols].style.background_gradient(
        subset=["churn_probability"], cmap="RdYlGn_r"
    ).format({"churn_probability": "{:.2%}"}),
    use_container_width=True,
    height=350,
)

# ── Single customer scorer ──────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Score a Single Customer")

with st.expander("Enter customer details"):
    t1, t2, t3 = st.columns(3)
    tenure    = t1.number_input("Tenure (months)", 0, 72, 12)
    monthly   = t2.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total     = t3.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)

    contract  = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet  = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

    if st.button("Predict Churn Risk", type="primary"):
        model, scaler, features = load_artifacts()

        # Build minimal feature row
        row = {f: 0 for f in features}
        row["tenure"]          = tenure
        row["MonthlyCharges"]  = monthly
        row["TotalCharges"]    = total
        row["AvgMonthlyCharge"]= total / (tenure + 1)
        row["NewCustomer"]     = int(tenure <= 3)
        row["HighValue"]       = int(monthly >= 80)

        X_in  = pd.DataFrame([row])[features]
        X_sc  = scaler.transform(X_in)
        prob  = model.predict_proba(X_sc)[0][1]
        risk  = "HIGH" if prob >= 0.65 else "MEDIUM" if prob >= 0.40 else "LOW"

        risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        st.metric("Churn Probability", f"{prob:.1%}", delta=risk)
        st.info(f"{risk_color[risk]} Risk Level: **{risk}**")
