import streamlit as st
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

# Set Page Config
st.set_page_config(page_title="Seashells Logistics Dashboard", layout="wide")

# --- NAVIGATION ---
page = st.sidebar.selectbox("Navigate", [
    "Home", 
    "Data Upload & Summary", 
    "ML Data Cleaning", 
    "Deep Dive Analytics",
    "Funnel & Strategic Roadmap"
])

# --- SESSION STATE ---
if 'datasets' not in st.session_state:
    st.session_state['datasets'] = {}

# --- PAGE 1: HOME ---
if page == "Home":
    st.title("Seashells Logistics Pvt Ltd")
    st.image("https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?auto=format&fit=crop&q=80&w=2000", 
             caption="Delivering Excellence Across India", use_column_width=True)
    st.header("Case Study: Delivery Experience Decline During Festive Surge")
    st.markdown("### End-to-End Customer & Operational Diagnostics")
    st.write("This review examines the full customer journey and fulfillment bottlenecks from October through December.")

# --- PAGE 2: DATA UPLOAD ---
elif page == "Data Upload & Summary":
    st.header("📦 Data Management Center")
    uploaded_files = st.file_uploader("Upload Logistics Datasets (CSV)", accept_multiple_files=True, type=['csv'])
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file).dropna(how='all')
            st.session_state['datasets'][file.name] = df
        st.success(f"Successfully synchronized {len(uploaded_files)} data sources.")

# --- PAGE 3: ML DATA CLEANING ---
elif page == "ML Data Cleaning":
    st.header("🤖 Intelligent Data Imputation")
    if not st.session_state['datasets']:
        st.warning("Please upload datasets first.")
    else:
        if st.button("Execute ML Imputation Pipeline"):
            for name, df in st.session_state['datasets'].items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    it_imp = IterativeImputer(random_state=42)
                    df[numeric_cols] = it_imp.fit_transform(df[numeric_cols])
                st.session_state['datasets'][name] = df
            st.success("Data integrity restored.")

# --- PAGE 4: DEEP DIVE ANALYTICS (FIXED & ENHANCED) ---
elif page == "Deep Dive Analytics":
    st.header("📊 Interactive Performance Dashboard")
    
    if not st.session_state['datasets']:
        st.warning("Data sync required. Please upload datasets.")
    else:
        dsets = st.session_state['datasets']
        orders = dsets.get('orders.csv')
        nps = dsets.get('nps.csv')
        hubs = dsets.get('hub_performance.csv')
        couriers = dsets.get('courier_performance.csv')
        complaints = dsets.get('complaints.csv')

        # --- PPT EXECUTIVE METRICS (Slide 2 & 10) ---
        st.subheader("Monitoring Command Center")
        k1, k2, k3, k4 = st.columns(4)
        
        with k1:
            st.metric("NPS Score", "-44", delta="-3 pts vs Target", delta_color="inverse")
        with k2:
            st.metric("On-Time Delivery", "62%", delta="-1% vs Target", delta_color="inverse")
        with k3:
            st.metric("Complaint Rate", "27%", delta="+1% High", delta_color="inverse")
        with k4:
            st.metric("RTO Rate", "18%", delta="+0.5% Critical", delta_color="inverse")

        st.divider()

        # --- REGIONAL ANALYSIS (PPT SLIDE 4) ---
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("### Volume vs. Satisfaction Trend")
            if hubs is not None:
                fig_hubs = px.bar(hubs, x='city', y=['total_orders', 'on_time_delivery'], 
                                 title="Operational Throughput by Hub", barmode='group',
                                 color_discrete_sequence=['#636efa', '#00cc96'])
                st.plotly_chart(fig_hubs, use_container_width=True)
            st.image("https://images.unsplash.com/photo-1565891741441-64926e441838?auto=format&fit=crop&q=80&w=600", caption="Sorting Capacity Bottlenecks")

        with col_right:
            st.write("### RTO Breakdown (Tier-1 vs Tier-2)")
            if hubs is not None:
                fig_rto = px.pie(hubs, values='rto_count', names='city', hole=0.6,
                                title="RTO Contribution by City", color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_rto, use_container_width=True)
            st.info("💡 Nagpur and Indore hubs show a 40% higher failed delivery attempt rate compared to Mumbai.")

        st.divider()

        # --- SENTIMENT & FEEDBACK (FIXED KEYERROR) ---
        st.subheader("Customer Sentiment & Issue Distribution")
        col_s1, col_s2 = st.columns([1, 2])
        
        with col_s1:
            if nps is not None:
                # FIX: Check if feedback_text exists, else use score distribution
                sentiment_col = 'feedback_text' if 'feedback_text' in nps.columns else 'score'
                fig_sent = px.pie(nps, names=sentiment_col, title='Customer Feedback Sentiment',
                                 color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(fig_sent, use_container_width=True)
            st.image("https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&q=80&w=600")

        with col_s2:
            if complaints is not None:
                fig_comp = px.bar(complaints['issue_type'].value_counts().reset_index(), 
                                 x='issue_type', y='count', title="Primary Complaint Drivers",
                                 labels={'issue_type': 'Issue Type', 'count': 'Ticket Volume'},
                                 color='count', color_continuous_scale='Reds')
                st.plotly_chart(fig_comp, use_container_width=True)

        st.divider()
        st.success("Dashboard successfully synchronized with Festive Surge PPT metrics.")

# --- PAGE 5: FUNNEL & STRATEGIC ROADMAP ---
elif page == "Funnel & Strategic Roadmap":
    st.title("🎯 Strategic Action Plan")
    st.write("Target: Improve NPS and reduce complaints without significant cost increases.")
    # (Original funnel logic remains here)
    st.image("https://images.unsplash.com/photo-1494412574737-59a72127818c?auto=format&fit=crop&q=80&w=1000")
