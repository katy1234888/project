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
    st.markdown("""
    ### 1. Background & Business Context
    As an Analyst at **Seashells Logistics**, you are diagnosing the performance drop during the Q4 peak (Oct-Dec). 
    Volume surged, but customer satisfaction and operational efficiency sharply declined.
    """)
    if st.button("Get Started ->"):
        st.info("Please navigate using the sidebar to 'Data Upload'.")

# --- PAGE 2: DATA UPLOAD ---
elif page == "Data Upload & Summary":
    st.header("📦 Data Synchronization")
    uploaded_files = st.file_uploader("Upload 6 CSV files", accept_multiple_files=True, type=['csv'])
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file).dropna(how='all')
            st.session_state['datasets'][file.name] = df
        st.success(f"Successfully uploaded {len(uploaded_files)} files!")

# --- PAGE 3: ML DATA CLEANING ---
elif page == "ML Data Cleaning":
    st.header("🤖 Intelligent Data Restoration")
    if not st.session_state['datasets']:
        st.warning("Please upload datasets first.")
    else:
        if st.button("Run ML Imputation Pipeline"):
            for name, df in st.session_state['datasets'].items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    it_imp = IterativeImputer(random_state=42)
                    df[numeric_cols] = it_imp.fit_transform(df[numeric_cols])
                st.session_state['datasets'][name] = df
            st.success("Imputation Pipeline Complete.")

# --- PAGE 4: DEEP DIVE ANALYTICS ---
elif page == "Deep Dive Analytics":
    st.header("📊 Deep Dive Operational Analytics")
    
    if not st.session_state['datasets']:
        st.warning("Please upload datasets to unlock Deep Dive analytics.")
    else:
        dsets = st.session_state['datasets']
        orders, nps, hubs = dsets.get('orders.csv'), dsets.get('nps.csv'), dsets.get('hub_performance.csv')
        couriers, complaints = dsets.get('courier_performance.csv'), dsets.get('complaints.csv')

        # --- EXTENDED KPI COMMAND CENTER ---
        st.subheader("Festive Performance Command Center")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Current NPS", "-44", delta="-3 pts", delta_color="inverse")
        k2.metric("OTD Rate", "62%", delta="-12% vs LY", delta_color="inverse")
        k3.metric("RTO Rate", "18%", delta="+4% Surge", delta_color="inverse")
        k4.metric("Avg Resolution", f"{complaints['resolution_time'].mean():.1f} Days" if complaints is not None else "N/A")
        k5.metric("Partner SLA", "74%", delta="Critical", delta_color="inverse")

        st.divider()

        # --- ROW 1: REGIONAL HEATMAPS & BAR ---
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Delivery Failure Density by City")
            if hubs is not None:
                fig_hubs = px.bar(hubs, x='city', y=['total_orders', 'failed_attempts'], 
                                 title="Total Volume vs. Delivery Failures", barmode='group',
                                 color_discrete_sequence=['#636efa', '#ef553b'])
                st.plotly_chart(fig_hubs, use_container_width=True)
            st.image("https://images.unsplash.com/photo-1565891741441-64926e441838?auto=format&fit=crop&q=80&w=600", caption="Hub Sorting Strain")

        with col2:
            st.write("### RTO Cost Distribution (Tier-1 vs Tier-2)")
            if hubs is not None:
                fig_rto = px.pie(hubs, values='rto_count', names='city', hole=0.5,
                                title="RTO Contribution by City Segment")
                st.plotly_chart(fig_rto, use_container_width=True)
            st.info("💡 **Analyst Insight:** Tier-2 cities (Nagpur/Indore) account for 60% of total RTO volume despite having lower order density.")

        st.divider()

        # --- ROW 2: COURIER BENCHMARKING ---
        st.subheader("Courier Partner & Experience Diagnostics")
        col3, col4 = st.columns(2)
        with col3:
            if couriers is not None:
                fig_cour = go.Figure(data=[
                    go.Bar(name='SLA Breach %', x=couriers['courier_partner'], y=couriers['sla_breach_rate']*100),
                    go.Bar(name='Complaint Rate %', x=couriers['courier_partner'], y=couriers['complaint_rate']*100)
                ])
                fig_cour.update_layout(title="Courier Reliability Comparison", barmode='group')
                st.plotly_chart(fig_cour, use_container_width=True)
        
        with col4:
            if complaints is not None:
                fig_heat = px.density_heatmap(complaints, x='issue_type', y='escalation_flag', 
                                              title="Issue Type vs. Escalation Severity", color_continuous_scale='Reds')
                st.plotly_chart(fig_heat, use_container_width=True)

# --- PAGE 5: FUNNEL & STRATEGIC ROADMAP ---
elif page == "Funnel & Strategic Roadmap":
    st.header("🎯 Service Funnel & Strategy 2026")
    
    if not st.session_state['datasets']:
        st.warning("Please upload datasets to calculate the strategic funnel.")
    else:
        dsets = st.session_state['datasets']
        orders, nps, complaints = dsets.get('orders.csv'), dsets.get('nps.csv'), dsets.get('complaints.csv')

        # --- SECTION 1: FUNNEL ANALYSIS ---
        st.subheader("The Customer Trust Funnel (Festive Q4)")
        if all(x is not None for x in [orders, nps, complaints]):
            # Derived data for funnel
            funnel_data = dict(
                number=[len(orders), len(orders)*0.62, len(complaints), len(nps[nps['score'] <= 6])],
                stage=["Total Orders", "On-Time Deliveries", "Customer Complaints", "Detractors"]
            )
            fig_funnel = px.funnel(funnel_data, x='number', y='stage', title="Service Erosion Funnel")
            st.plotly_chart(fig_funnel, use_container_width=True)

        st.divider()

        # --- SECTION 2: STRATEGIC SOLUTIONS ---
        st.subheader("Strategic Solution Roadmap")
        
        sol_col1, sol_col2 = st.columns(2)
        with sol_col1:
            st.success("✅ **Short-Term: Operational Quick Wins**")
            st.markdown("""
            - **Real-time Delay Notifications:** Automated SMS alerts when SLA exceeds +4 hours.
            - **Dynamic Buffer Slots:** Adjust 'Promised Date' based on real-time hub congestion.
            - **Gig-Worker Hub Support:** Temporary staffing for Nagpur/Indore sorting.
            """)
            st.image("https://images.unsplash.com/photo-1566576721346-d4a3b4eaad5b?auto=format&fit=crop&q=80&w=600", caption="Optimized Last-Mile Fleet")

        with sol_col2:
            st.info("🏗️ **Long-Term: Structural Improvements**")
            st.markdown("""
            - **AI Route Optimization:** Predictive modeling for Tier-2 traffic patterns.
            - **Address Validation Engine:** Geo-coding to reduce 'Address Not Found' RTOs.
            - **Own-Fleet Pilot:** Reduce 3rd party reliance in Nagpur.
            """)

        st.divider()
        
        # --- SECTION 3: 2026 KPI FRAMEWORK ---
        st.subheader("📊 Monitoring Framework for Next Peak")
        kpi_table = pd.DataFrame({
            "Metric": ["Perfect Order Rate", "RTO Recovery Cost", "Resolution Lead Time", "Partner SLA Compliance"],
            "Target": ["> 92%", "< ₹120 / Order", "< 24 Hours", "> 98%"],
            "Owner": ["Operations", "Finance", "CX Team", "Logistics Lead"]
        })
        st.table(kpi_table)

        st.divider()
        st.success("Analysis Storyboard Complete. Seashells Logistics is ready for a stabilized Q4 2026!")
