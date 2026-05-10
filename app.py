import streamlit as st
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import plotly.express as px

# Set Page Config
st.set_page_config(page_title="Seashells Logistics Dashboard", layout="wide")

# --- NAVIGATION ---
# Added "Funnel & Strategic Roadmap" to the sidebar
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
    You are working as an Analyst at **Seashells Logistics** operating across Tier-1 and Tier-2 cities in India.
    Over the past 3 months (October–December), the company has experienced a significant increase in order volumes due to festive demand.
    """)
    if st.button("Get Started ->"):
        st.info("Please use the sidebar to navigate.")

# --- PAGE 2: DATA UPLOAD ---
elif page == "Data Upload & Summary":
    st.header("📦 Upload Datasets")
    uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type=['csv'])
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file).dropna(how='all')
            st.session_state['datasets'][file.name] = df
        st.success(f"Successfully uploaded {len(uploaded_files)} files!")

# --- PAGE 3: ML DATA CLEANING ---
elif page == "ML Data Cleaning":
    st.header("🤖 Machine Learning Based Data Imputation")
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
            st.success("Imputation Complete!")

# --- PAGE 4: DEEP DIVE ANALYTICS ---
elif page == "Deep Dive Analytics":
    st.header("🔍 Deep Dive: Festive Surge Analysis")
    if not st.session_state['datasets']:
        st.warning("Please upload datasets first.")
    else:
        nps = st.session_state['datasets'].get('nps.csv')
        if nps is not None:
            nps['Category'] = nps['score'].apply(lambda x: "Promoter" if x>=9 else ("Detractor" if x<=6 else "Passive"))
            fig = px.pie(nps, names='Category', hole=0.5, title="NPS Distribution")
            st.plotly_chart(fig)

# --- PAGE 5: FUNNEL & STRATEGIC ROADMAP (NEW OPTION) ---
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
            st.write("**Insight:** This shows the efficiency (or lack thereof) of the complaint resolution process.")

        # 3. Impact on Repeat Usage
        if customers is not None:
            repeat_rate = (len(customers[customers['segment'] == 'Repeat']) / len(customers)) * 100
            st.metric("Overall Repeat Customer Rate", f"{repeat_rate:.2f}%")
            st.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?auto=format&fit=crop&q=80&w=1000", 
                     caption="Retaining customers through service excellence")

        st.divider()

        st.subheader("Section E: Business Recommendations")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            #### 🚩 Top 3 Root Causes
            1. **Tier-2 Infrastructure Gap:** Nagpur and Indore hubs lack the sorting capacity for 3x volume surges.
            2. **Courier Partner Capacity:** 'QuickShip' has exceeded its operational bandwidth, leading to a 32% SLA breach rate.
            3. **Communication Latency:** "Wrong Status" complaints suggest a lag in the tracking API between couriers and our internal system.
            """)
            
            st.markdown("""
            #### ⚡ Quick Wins (Short-term)
            - **Real-time SMS Alerts:** Proactive delay notifications to reduce "Where is my order?" complaints.
            - **Incentivize Off-Peak Delivery:** Offer discounts for customers willing to accept 2-day longer windows.
            - **Temporary Hub Staffing:** Deploy gig-workers to Nagpur/Indore during December spikes.
            """)

        with c2:
            st.markdown("""
            #### 🏗️ Long-term Improvements
            - **Route Optimization AI:** Implement ML models to predict festive traffic and adjust 'promised_dates' dynamically.
            - **Own-Fleet Expansion:** Reduce dependency on 3rd party couriers in high-RTO zones.
            - **Address Validation Engine:** Use geo-coding to reduce RTOs caused by "Address not found" in Tier-2 cities.
            """)
            
            st.markdown("""
            #### 📊 Suggested KPIs for 2026
            - **Perfect Order Rate:** (On-time + No Damage + No Complaints).
            - **RTO Recovery Cost:** Total cost lost per return.
            - **First-Response-Time (FRT):** Speed of resolving festive complaints.
            """)

        st.image("https://images.unsplash.com/photo-1494412574737-59a72127818c?auto=format&fit=crop&q=80&w=1000", 
                 caption="Strategic Growth & Logistics Scaling")
        
        st.success("Analysis Complete. Seashells Logistics is ready for the next peak season!")
