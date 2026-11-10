import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="US Stock Analysis Dashboard", layout="wide")
st.markdown("""<style>.stApp{background:#fff;color:#10141A}.kicker{letter-spacing:.08em;font-size:.78rem;color:#64748b}.warning{padding:1rem;border-left:4px solid #b45309;background:#fff7ed}</style>""", unsafe_allow_html=True)

picks = pd.DataFrame([
    {"Ticker":"MU","Peer group":"Semiconductor","Rank":1,"Entry price":"$690.00","Composite score":77.2},
    {"Ticker":"APP","Peer group":"AI software","Rank":2,"Entry price":"$391.22","Composite score":76.4},
    {"Ticker":"INTU","Peer group":"AI software","Rank":2,"Entry price":"$341.50","Composite score":75.4},
    {"Ticker":"LRCX","Peer group":"Semiconductor","Rank":3,"Entry price":"$200.44","Composite score":75.7},
])

st.title("US Stock Analysis Dashboard")
st.caption("Read-only public snapshot • data date: 2026-11-07 • research project, not investment advice")
page = st.sidebar.radio("Page", ["Daily Picks", "Stock Deep Dive", "Strategy Performance"])

if page == "Daily Picks":
    st.subheader("Daily picks")
    a,b,c = st.columns(3)
    a.metric("Universe", "60 listed companies")
    b.metric("Snapshot date", "2026-11-07")
    c.metric("Top score", "77.2")
    st.dataframe(picks, use_container_width=True, hide_index=True)
    st.caption("Scores and entry levels are preserved from the static snapshot; this demo does not fetch live market data.")

elif page == "Stock Deep Dive":
    st.subheader("Stock deep dive — LRCX")
    a,b,c = st.columns(3)
    a.metric("Composite score", "75.7")
    b.metric("Snapshot price", "$200.44")
    c.metric("Peer group", "Semiconductor")
    st.write("Lam Research is shown here as a static company-profile example from the 2026-11-07 snapshot.")
    days = pd.date_range("2026-10-01", periods=28)
    prices = 184 + np.cumsum(np.array([1.2,-.4,1.8,.3,-.6,1.1,.8,-.2,1.6,.4,-.7,1.3,.9,-.5,1.0,.2,.8,-.3,1.4,.5,-.4,1.1,.6,-.2,.9,.4,1.0,.3]))
    fig = go.Figure(go.Scatter(x=days, y=prices, mode="lines", line=dict(color="#15385C", width=3), name="LRCX snapshot"))
    fig.update_layout(height=330, margin=dict(l=0,r=0,t=25,b=0), yaxis_title="Price (illustrative path)", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("Strategy performance")
    a,b,c = st.columns(3)
    a.metric("Strategy cumulative return", "+34.30%")
    b.metric("Excess return vs SPY", "+27.07%")
    c.metric("SPY buy-and-hold", "+7.22%")
    dates = pd.date_range("2026-04-21", periods=12, freq="15D")
    strategy = [0,3,6,11,10,14,17,21,24,27,31,34.3]
    spy = [0,.4,1.3,2.1,1.7,3.5,4.2,4.9,5.3,6.0,6.6,7.22]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=strategy,name="Strategy",line=dict(color="#15385C",width=3)))
    fig.add_trace(go.Scatter(x=dates,y=spy,name="SPY buy-and-hold",line=dict(color="#64748B",dash="dash")))
    fig.update_layout(height=330, margin=dict(l=0,r=0,t=25,b=0), yaxis_title="Cumulative return (%)", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div class='warning'><strong>Concentration warning.</strong> Performance is concentrated in one name: MU contributes 108.9%. The sample is too narrow to treat the stock-picking result as consistent.</div>", unsafe_allow_html=True)

st.divider()
st.caption("Static demo snapshot only. No live data, credentials, or background collection are included.")
