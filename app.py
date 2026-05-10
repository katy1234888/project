import streamlit as st
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import plotly.express as px

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

# --- PAGE 4: DEEP DIVE ANALYTICS ---
elif page == "Deep Dive Analytics":
    st.header("🔍 Deep Dive: Festive Surge Analysis")
    
    if not st.session_state['datasets']:
        st.warning("Please upload datasets first to unlock analytics.")
    else:
        dsets = st.session_state['datasets']
        orders = dsets.get('orders.csv')
        nps = dsets.get('nps.csv')
        complaints = dsets.get('complaints.csv')
        hubs = dsets.get('hub_performance.csv')
        couriers = dsets.get('courier_performance.csv')

        st.subheader("1. The Voice of the Customer (NPS)")
        if nps is not None:
            col1, col2 = st.columns([2, 1])
            with col1:
                def get_cat(s):
                    if s >= 9: return "Promoter"
                    if s <= 6: return "Detractor"
                    return "Passive"
                nps['Category'] = nps['score'].apply(get_cat)
                fig_nps = px.pie(nps, names='Category', title='NPS Breakdown', hole=0.5,
                                color_discrete_map={'Detractor':'#ef553b', 'Promoter':'#00cc96', 'Passive':'#ab63fa'})
                st.plotly_chart(fig_nps, use_container_width=True)
            with col2:
                st.write("### Insights")
                st.write("The high volume of **Detractors** suggests that the festive surge overwhelmed the logistics network.")
                st.image("https://images.unsplash.com/photo-1556742044-3c52d6e88c62?auto=format&fit=crop&q=80&w=600")

        st.divider()

        st.subheader("2. Operational Bottlenecks by City")
        if hubs is not None:
            hubs['SLA_Breach_Rate'] = ((hubs['total_orders'] - hubs['on_time_delivery']) / hubs['total_orders']) * 100
            fig_hubs = px.bar(hubs, x='city', y='SLA_Breach_Rate', color='city', title="SLA Breach Percentage by City")
            st.plotly_chart(fig_hubs, use_container_width=True)

        st.divider()

        st.subheader("3. Why are customers complaining?")
        if complaints is not None:
            c_dist = complaints['issue_type'].value_counts().reset_index()
            fig_issues = px.bar(c_dist, x='count', y='issue_type', orientation='h', title="Complaint Categories", color='issue_type')
            st.plotly_chart(fig_issues, use_container_width=True)

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

        st.subheader("Section D: End-to-End Funnel Analysis")
        
        # 1. Delayed Orders -> Complaints %
        if orders is not None and complaints is not None:
            orders['is_delayed'] = pd.to_datetime(orders['delivery_date']) > pd.to_datetime(orders['promised_date'])
            delayed_order_ids = set(orders[orders['is_delayed'] == True]['order_id'])
            complaint_order_ids = set(complaints['order_id'])
            
            delayed_with_complaints = len(delayed_order_ids.intersection(complaint_order_ids))
            ratio = (delayed_with_complaints / len(delayed_order_ids)) * 100 if len(delayed_order_ids) > 0 else 0
            
            st.metric("Delayed Orders resulting in Complaints", f"{ratio:.2f}%")
            st.write("**Insight:** High correlation indicates that late delivery is the primary trigger for support tickets.")

        # 2. Complaints -> Detractors %
        if complaints is not None and nps is not None:
            complaint_orders = set(complaints['order_id'])
            detractors = set(nps[nps['score'] <= 6]['order_id'])
            
            complaints_turned_detractors = len(complaint_orders.intersection(detractors))
            ratio_det = (complaints_turned_detractors / len(complaint_orders)) * 100 if len(complaint_orders) > 0 else 0
            
            st.metric("Complaints turning into Detractors", f"{ratio_det:.2f}%")
            st.write("**Insight:** This shows the impact of poor issue resolution on brand loyalty.")

        # 3. Impact on Repeat Usage
        if customers is not None:
            repeat_rate = (len(customers[customers['segment'] == 'Repeat']) / len(customers)) * 100
            st.metric("Overall Repeat Customer Rate", f"{repeat_rate:.2f}%")
            st.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?auto=format&fit=crop&q=80&w=1000", caption="Customer Loyalty Focus")

        st.divider()

        st.subheader("Section E: Business Recommendations")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            #### 🚩 Top 3 Root Causes
            1. **Tier-2 Infrastructure Gap:** Hubs lack sorting capacity for 3x volume surges.
            2. **Courier Partner Capacity:** Partners have exceeded their operational bandwidth.
            3. **Communication Latency:** Lag in tracking API updates between systems.
            """)
            
            st.markdown("""
            #### ⚡ Quick Wins (Short-term)
            - **Real-time SMS Alerts:** Proactive delay notifications.
            - **Incentivize Off-Peak Delivery:** Offer discounts for slower delivery windows.
            - **Temporary Hub Staffing:** Deploy gig-workers during spikes.
            """)

        with c2:
            st.markdown("""
            #### 🏗️ Long-term Improvements
            - **Route Optimization AI:** Predict festive traffic and adjust dates dynamically.
            - **Own-Fleet Expansion:** Reduce dependency on 3rd parties in high-RTO zones.
            - **Address Validation Engine:** Reduce RTOs caused by faulty addresses.
            """)
            
            st.markdown("""
            #### 📊 Suggested KPIs for 2026
            - **Perfect Order Rate**
            - **RTO Recovery Cost**
            - **First-Response-Time (FRT)**
            """)

        st.image("https://images.unsplash.com/photo-1494412574737-59a72127818c?auto=format&fit=crop&q=80&w=1000")
        st.success("Analysis Complete. Seashells Logistics is ready for the next peak season!")
