import streamlit as st
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import plotly.express as px

# Set Page Config
st.set_page_config(page_title="Seashells Logistics Dashboard", layout="wide")

# --- NAVIGATION ---
# Added "Strategic Business Theory" to the sidebar
page = st.sidebar.selectbox("Navigate", [
    "Home", 
    "Data Upload & Summary", 
    "ML Data Cleaning", 
    "Deep Dive Analytics",
    "Strategic Business Theory"
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

# --- PAGE 5: STRATEGIC BUSINESS THEORY (NEW OPTION) ---
elif page == "Strategic Business Theory":
    st.title("📖 Logistics & Strategy Theory")
    st.write("Detailed theoretical framework for the ITM Skills Case Study.")
    
    # Question 1: NPS
    st.subheader("1. Understanding Net Promoter Score (NPS)")
    
    st.write("""
    The **Net Promoter Score** is a gold-standard metric used by Seashells Logistics to measure customer loyalty. 
    It is calculated based on the question: *'On a scale of 0-10, how likely are you to recommend our service?'*
    """)
    
    st.code("""
# Python Formula for NPS
def calculate_nps(promoters, detractors, total_responses):
    nps_score = ((promoters - detractors) / total_responses) * 100
    return nps_score
    """, language="python")
    
    st.info("**Strategy:** During the festive surge, even if promoters increase, a sharper rise in detractors (due to late deliveries) will cause the total NPS to plummet.")

    # Question 2: RTO & Impact
    st.subheader("2. Return-To-Origin (RTO) Dynamics")
    
    st.write("""
    **RTO** occurs when a shipment cannot be delivered and is sent back to the warehouse. 
    In Tier-2 cities like Nagpur, the RTO rate is high due to:
    - **Address Inaccuracy:** High failure rates in non-digitized addresses.
    - **Customer Availability:** Multiple failed attempts during working hours.
    - **Cash on Delivery (COD) Refusals:** Festive impulse buys leading to cancellations at the doorstep.
    """)
    
    # Cost Impact Chart
    rto_data = pd.DataFrame({
        'Stage': ['Forward Freight', 'Processing', 'Reverse Freight', 'Repackaging'],
        'Cost Impact (%)': [40, 10, 35, 15]
    })
    fig_rto = px.bar(rto_data, x='Stage', y='Cost Impact (%)', title="Cost Breakdown of an RTO Order")
    st.plotly_chart(fig_rto)

    # Question 3: SLA & Courier Performance
    st.subheader("3. Service Level Agreement (SLA) & Delay Rates")
    
    st.write("""
    An **SLA Breach** occurs when the `delivery_date` exceeds the `promised_date`. 
    In logistics, we track the **Delay Rate** using this logic:
    """)
    
    st.code("""
# Calculating Delay Rate
delay_rate = (orders[orders['delivery_date'] > orders['promised_date']].count()) / total_orders
    """, language="python")

    st.write("""
    ### Root Causes of the Festive Decline:
    1. **Hub Congestion:** Tier-2 hubs (Indore/Nagpur) are not designed for 3x volume surges.
    2. **Last-Mile Exhaustion:** Courier partners like 'QuickShip' showed a 32% SLA breach, indicating they reached their maximum capacity limit.
    3. **Information Gap:** "Wrong Status" complaints indicate a failure in the real-time API sync between couriers and Seashells Logistics.
    """)

    st.image("https://images.unsplash.com/photo-1594122230689-45899d9e6f69?auto=format&fit=crop&q=80&w=1000", 
             caption="Optimizing the Supply Chain Journey")

    st.success("Theoretical Review Complete. Combine these insights with the 'Deep Dive' charts for the final presentation.")
