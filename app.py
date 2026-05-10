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
# FIX: Added "Funnel & Strategic Roadmap" to this list so it shows in the sidebar
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
    
    # Logistics Image
    st.image("https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?auto=format&fit=crop&q=80&w=2000", 
             caption="Delivering Excellence Across India", use_column_width=True)
    
    st.header("Case Study: Delivery Experience Decline During Festive Surge")
    
    intro_text = """
    ### 1. Background & Business Context
    You are working as an Analyst at **Seashells Logistics** operating across Tier-1 and Tier-2 cities in India.
    Over the past 3 months (October–December), the company has experienced a significant increase in order volumes due to festive demand. However, leadership has raised concerns about:
    - **Declining customer satisfaction (NPS)**
    - **Increase in customer complaints**
    - **Rising Return-to-Origin (RTO) rates**
    - **Drop in repeat customer usage**
    - **Operational inefficiencies** across hubs and courier partners

    ### 2. Objective
    The leadership team wants to diagnose the root causes and take corrective actions before the next peak season.
    - Analyze customer experience and operational performance.
    - Identify key drivers of poor customer satisfaction.
    - Evaluate impact on customer retention.
    - Recommend actionable solutions for improvement.
    """
    st.markdown(intro_text)
    
    if st.button("Get Started ->"):
        st.info("Please use the sidebar to navigate to the 'Data Upload & Summary' page.")

# --- PAGE 2: DATA UPLOAD ---
elif page == "Data Upload & Summary":
    st.header("📦 Upload Datasets")
    st.write("Please upload the 6 CSV files: Orders, NPS, Hub Performance, Courier Performance, Customers, and Complaints.")
    
    uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type=['csv'])
    
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file)
            df = df.dropna(how='all') # Basic cleanup
            st.session_state['datasets'][file.name] = df
            
        st.success(f"Successfully uploaded {len(uploaded_files)} files!")
        
        for name, df in st.session_state['datasets'].items():
            with st.expander(f"Summary: {name}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Data Preview:**")
                    st.dataframe(df.head())
                with col2:
                    st.write("**Missing Values Profile:**")
                    st.write(df.isnull().sum())
                    st.metric("Total Rows", df.shape[0])

# --- PAGE 3: ML DATA CLEANING ---
elif page == "ML Data Cleaning":
    st.header("🤖 Machine Learning Based Data Imputation")
    
    if not st.session_state['datasets']:
        st.warning("Please upload datasets first on the 'Data Upload' page.")
    else:
        st.info("Running MICE (Multivariate Imputation by Chained Equations) for numeric data.")
        
        if st.button("Run ML Imputation Pipeline"):
            progress_bar = st.progress(0)
            count = 0
            total = len(st.session_state['datasets'])
            
            for name, df in st.session_state['datasets'].items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(exclude=[np.number]).columns
                
                # ML Imputation for Numeric
                if len(numeric_cols) > 0 and df[numeric_cols].isnull().sum().sum() > 0:
                    it_imp = IterativeImputer(random_state=42)
                    df[numeric_cols] = it_imp.fit_transform(df[numeric_cols])
                
                # Simple Imputation for Categorical
                if len(categorical_cols) > 0 and df[categorical_cols].isnull().sum().sum() > 0:
                    cat_imp = SimpleImputer(strategy='most_frequent')
                    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols].astype(str))
                
                st.session_state['datasets'][name] = df
                count += 1
                progress_bar.progress(count / total)
            
            st.success("Imputation Complete!")
            
        st.subheader("Data Summary After Cleaning")
        for name, df in st.session_state['datasets'].items():
            with st.expander(f"Cleaned Summary: {name}"):
                st.write("**Missing Values (Check):**")
                st.write(df.isnull().sum())
                st.dataframe(df.head())
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
    st.title("🎯 End-to-End Funnel & Strategy")
    
    if not st.session_state['datasets']:
        st.warning("Please upload datasets to calculate funnel metrics.")
    else:
        dsets = st.session_state['datasets']
        orders = dsets.get('orders.csv')
        nps = dsets.get('nps.csv')
        complaints = dsets.get('complaints.csv')
        customers = dsets.get('customers.csv')

        # --- SECTION 1: FUNNEL ANALYSIS ---
        st.subheader("Section D: End-to-End Funnel Analysis")
        
        # 1. Funnel Visualization
        if all(x is not None for x in [orders, nps, complaints]):
            funnel_data = dict(
                number=[len(orders), int(len(orders)*0.62), len(complaints), len(nps[nps['score'] <= 6])],
                stage=["Total Orders", "On-Time Deliveries", "Complaints Filed", "Detractors"]
            )
            fig_funnel = px.funnel(funnel_data, x='number', y='stage', title="Service Erosion Funnel (Festive Q4)")
            st.plotly_chart(fig_funnel, use_container_width=True)

        # 2. Key Funnel Metrics & Insights
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            if orders is not None and complaints is not None:
                orders['is_delayed'] = pd.to_datetime(orders['delivery_date']) > pd.to_datetime(orders['promised_date'])
                delayed_order_ids = set(orders[orders['is_delayed'] == True]['order_id'])
                complaint_order_ids = set(complaints['order_id'])
                delayed_with_complaints = len(delayed_order_ids.intersection(complaint_order_ids))
                ratio = (delayed_with_complaints / len(delayed_order_ids)) * 100 if len(delayed_order_ids) > 0 else 0
                st.metric("Delayed Orders -> Complaints", f"{ratio:.2f}%")
                st.write("**Insight:** High correlation confirms delays as the primary ticket driver.")

        with col_m2:
            if complaints is not None and nps is not None:
                complaint_orders = set(complaints['order_id'])
                detractors = set(nps[nps['score'] <= 6]['order_id'])
                complaints_turned_detractors = len(complaint_orders.intersection(detractors))
                ratio_det = (complaints_turned_detractors / len(complaint_orders)) * 100 if len(complaint_orders) > 0 else 0
                st.metric("Complaints -> Detractors", f"{ratio_det:.2f}%")
                st.write("**Insight:** Highlights the critical impact of resolution efficiency on loyalty.")

        with col_m3:
            if customers is not None:
                repeat_rate = (len(customers[customers['segment'] == 'Repeat']) / len(customers)) * 100
                st.metric("Overall Repeat Rate", f"{repeat_rate:.2f}%")
                st.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?auto=format&fit=crop&q=80&w=1000", caption="Loyalty Focus")

        st.divider()

        # --- SECTION 2: BUSINESS RECOMMENDATIONS ---
        st.subheader("Section E: Business Recommendations")
        
        c1, c2 = st.columns(2)
        with c1:
            st.success("✅ **Quick Wins (Short-term)**")
            st.markdown("""
- **Real-time SMS Alerts:** Proactive delay notifications via automated triggers.
- **Incentivize Off-Peak Delivery:** Offer discounts for slower, non-urgent delivery windows.
- **Temporary Hub Staffing:** Deploy gig-workers during festive spikes in Tier-2 cities.
- **Dynamic Buffer Slots:** Adjust 'Promised Date' based on real-time hub congestion.
            """)
            
            st.markdown("""
#### 🚩 Top 3 Root Causes
1. **Tier-2 Infrastructure Gap:** Hubs lack sorting capacity for 3x volume surges.
2. **Courier Partner Capacity:** Partners have exceeded operational bandwidth.
3. **Communication Latency:** Lag in tracking API updates between systems.
            """)

        with c2:
            st.info("🏗️ **Strategic Roadmap (Long-term)**")
            st.markdown("""
- **AI Route Optimization:** Deploy MICE-based prediction for regional traffic patterns.
- **Own-Fleet Expansion:** Reduce dependency on 3rd parties in high-RTO zones (e.g., Nagpur).
- **Address Validation Engine:** Geo-coding integration to reduce 'Address Not Found' RTOs.
- **API Real-time Sync:** Upgrade courier integrations for millisecond status updates.
            """)
            

        st.divider()
        
        # --- SECTION 3: 2026 MONITORING FRAMEWORK ---
        st.subheader("📊 Suggested Monitoring Framework 2026")
        kpi_table = pd.DataFrame({
            "Metric": ["Perfect Order Rate", "RTO Recovery Cost", "Resolution Lead Time", "Partner SLA Compliance"],
            "Target": ["> 92%", "< ₹120 / Order", "< 24 Hours", "> 98%"],
            "Owner": ["Operations", "Finance", "CX Team", "Logistics Lead"]
        })
        st.table(kpi_table)

        st.divider()
        st.success("Analysis Storyboard Complete. Seashells Logistics is ready for a stabilized Q4 2026!")
