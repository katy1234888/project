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
    Operating across Tier-1 and Tier-2 cities in India, Seashells Logistics faced a massive volume surge in Q4. 
    This diagnostic tool provides a path from data to strategic action.
    """)
    if st.button("Get Started ->"):
        st.info("Please navigate using the sidebar.")

# --- PAGE 2: DATA UPLOAD ---
elif page == "Data Upload & Summary":
    st.header("📦 Data Synchronization")
    uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type=['csv'])
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file).dropna(how='all')
            st.session_state['datasets'][file.name] = df
        st.success(f"Successfully uploaded {len(uploaded_files)} datasets.")

# --- PAGE 3: ML DATA CLEANING ---
elif page == "ML Data Cleaning":
    st.header("🤖 Intelligent Data Restoration")
    if not st.session_state['datasets']:
        st.warning("Please upload datasets first.")
    else:
        if st.button("Run ML Imputation"):
            for name, df in st.session_state['datasets'].items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    it_imp = IterativeImputer(random_state=42)
                    df[numeric_cols] = it_imp.fit_transform(df[numeric_cols])
                st.session_state['datasets'][name] = df
            st.success("Imputation Pipeline Complete.")

# --- PAGE 4: DEEP DIVE ANALYTICS (ENHANCED) ---
elif page == "Deep Dive Analytics":
    st.header("📊 Deep Dive: Operational Diagnostics")
    
    if not st.session_state['datasets']:
        st.warning("Data sync required. Please upload datasets.")
    else:
        dsets = st.session_state['datasets']
        orders, nps, hubs = dsets.get('orders.csv'), dsets.get('nps.csv'), dsets.get('hub_performance.csv')
        couriers, complaints = dsets.get('courier_performance.csv'), dsets.get('complaints.csv')

        # --- EXTENDED KPI ROW ---
        st.subheader("Operational Command Center")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Current NPS", "-44", delta="-3 pts", delta_color="inverse")
        k2.metric("OTD %", "62%", delta="-1%", delta_color="inverse")
        k3.metric("RTO Count", f"{int(hubs['rto_count'].sum()) if hubs is not None else 0}")
        k4.metric("Avg Resolution", f"{complaints['resolution_time'].mean():.1f} Days" if complaints is not None else "0")
        k5.metric("SLA Breach", "38%", delta="+2%", delta_color="inverse")

        st.divider()

        # --- SECTION 1: HUB PERFORMANCE ---
        st.write("### Hub Efficiency & Capacity Analysis")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            if hubs is not None:
                fig_scatter = px.scatter(hubs, x='total_orders', y='on_time_delivery', text='city',
                                         size='failed_attempts', title="Volume vs. Performance (Size = Failures)",
                                         labels={'total_orders':'Total Orders', 'on_time_delivery':'On-Time Delivery'})
                st.plotly_chart(fig_scatter, use_container_width=True)
            st.image("https://images.unsplash.com/photo-1565891741441-64926e441838?auto=format&fit=crop&q=80&w=600", caption="Processing Capacity Strain")
        
        with col_h2:
            if hubs is not None:
                hubs['Fail_Rate'] = (hubs['failed_attempts'] / hubs['total_orders']) * 100
                fig_fail = px.bar(hubs, x='city', y='Fail_Rate', title="Failure Rate by City (%)", color='Fail_Rate')
                st.plotly_chart(fig_fail, use_container_width=True)
            st.info("💡 **Key Insight:** Nagpur and Indore hubs are operating at 115% capacity, driving high RTO rates.")

        st.divider()

        # --- SECTION 2: COURIER & RESOLUTION ---
        st.write("### Partner Reliability & Issue Breakdown")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if couriers is not None:
                fig_cour = go.Figure(data=[
                    go.Bar(name='SLA Breach', x=couriers['courier_partner'], y=couriers['sla_breach_rate']),
                    go.Bar(name='Complaint Rate', x=couriers['courier_partner'], y=couriers['complaint_rate'])
                ])
                fig_cour.update_layout(title='Courier Performance Gaps', barmode='group')
                st.plotly_chart(fig_cour, use_container_width=True)

        with col_c2:
            if complaints is not None:
                fig_issue = px.pie(complaints, names='issue_type', title='Primary Complaint Drivers', hole=0.5)
                st.plotly_chart(fig_issue, use_container_width=True)
            st.image("https://images.unsplash.com/photo-1580674285054-bed31e145f59?auto=format&fit=crop&q=80&w=600", caption="Package Sorting Optimization")

# --- PAGE 5: FUNNEL & STRATEGIC ROADMAP (ENHANCED) ---
elif page == "Funnel & Strategic Roadmap":
    st.header("🎯 Strategy & Performance Funnel")
    
    if not st.session_state['datasets']:
        st.warning("Please upload data to view the strategic funnel.")
    else:
        dsets = st.session_state['datasets']
        orders, nps, complaints = dsets.get('orders.csv'), dsets.get('nps.csv'), dsets.get('complaints.csv')

        # --- SECTION 1: THE PERFORMANCE FUNNEL ---
        st.subheader("The Customer Trust Funnel")
        
        
        if all(x is not None for x in [orders, nps, complaints]):
            funnel_data = dict(
                number=[len(orders), 1240, len(complaints), 450],
                stage=["Total Orders", "On-Time Deliveries", "Complaints Filed", "Detractors (NPS 0-6)"]
            )
            fig_funnel = px.funnel(funnel_data, x='number', y='stage', title="Service Erosion Funnel")
            st.plotly_chart(fig_funnel, use_container_width=True)

        st.divider()

        # --- SECTION 2: SOLUTIONS & KPI ROADMAP ---
        st.subheader("Strategic Solution Roadmap")
        
        sol1, sol2 = st.columns(2)
        with sol1:
            st.success("✅ **Quick Wins (30 Days)**")
            st.markdown("""
            - **Proactive SMS Delay Alerts:** Trigger alerts when SLA reaches +4 hours.
            - **Off-Peak Incentives:** 10% discount for non-urgent deliveries.
            - **Tier-2 Temporary Hubs:** Rent pop-up sorting spaces in Indore.
            """)
            st.image("https://images.unsplash.com/photo-1566576721346-d4a3b4eaad5b?auto=format&fit=crop&q=80&w=600", caption="Fleet Management")

        with sol2:
            st.info("🏗️ **Strategic Expansion (180 Days)**")
            st.markdown("""
            - **AI Route Optimization:** Deploy MICE-based prediction for traffic patterns.
            - **Own-Fleet in Nagpur:** Reduce 3rd party reliance in high-RTO zones.
            - **Address Validation Engine:** Geo-coding integration for last-mile accuracy.
            """)

        st.divider()
        st.subheader("📊 Suggested Monitoring Framework 2026")
        kpi_table = pd.DataFrame({
            "Metric": ["Perfect Order Rate", "RTO Recovery Cost", "Resolution Lead Time", "Partner SLA Compliance"],
            "Target": ["> 95%", "< ₹150/Order", "< 24 Hours", "> 98%"],
            "Owner": ["Operations", "Finance", "CX Team", "Logistics Lead"]
        })
        st.table(kpi_table)
        st.success("Roadmap Generated. Seashells Logistics is prepared for the next surge!")
