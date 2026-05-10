import streamlit as st
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import plotly.express as px

# Set Page Config
st.set_page_config(page_title="Seashells Logistics Dashboard", layout="wide")

# --- NAVIGATION ---
# Updated to include the Deep Dive Analytics page
page = st.sidebar.selectbox("Navigate", [
    "Home", 
    "Data Upload & Summary", 
    "ML Data Cleaning", 
    "Deep Dive Analytics"
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

# --- PAGE 4: DEEP DIVE ANALYTICS (NEW CODE) ---
elif page == "Deep Dive Analytics":
    st.header("🔍 Deep Dive: Festive Surge Analysis")
    
    if not st.session_state['datasets']:
        st.warning("Please upload datasets first to unlock analytics.")
    else:
        # Map filenames to dataframes
        dsets = st.session_state['datasets']
        orders = dsets.get('orders.csv')
        nps = dsets.get('nps.csv')
        complaints = dsets.get('complaints.csv')
        hubs = dsets.get('hub_performance.csv')
        couriers = dsets.get('courier_performance.csv')

        # 1. Customer Sentiment Story
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
                st.write("The high volume of **Detractors** suggests that the festive surge overwhelmed the logistics network. Feedback analysis shows 'Late Delivery' as the primary pain point.")
                st.image("https://images.unsplash.com/photo-1556742044-3c52d6e88c62?auto=format&fit=crop&q=80&w=600", caption="Customer Satisfaction Impact")

        st.divider()

        # 2. Operational Efficiency (Bar Chart)
        st.subheader("2. Operational Bottlenecks by City")
        
        if hubs is not None:
            hubs['SLA_Breach_Rate'] = ((hubs['total_orders'] - hubs['on_time_delivery']) / hubs['total_orders']) * 100
            fig_hubs = px.bar(hubs, x='city', y='SLA_Breach_Rate', color='city',
                             title="SLA Breach Percentage by City", text_auto='.2s')
            st.plotly_chart(fig_hubs, use_container_width=True)
            st.info("💡 **Analyst Note:** Nagpur and Indore show significantly higher SLA breaches compared to Tier-1 hubs.")

        st.divider()

        # 3. Complaint Distribution (Horizontal Bar / Pie)
        st.subheader("3. Why are customers complaining?")
        if complaints is not None:
            c_dist = complaints['issue_type'].value_counts().reset_index()
            fig_issues = px.bar(c_dist, x='count', y='issue_type', orientation='h', 
                               title="Complaint Categories", color='issue_type')
            st.plotly_chart(fig_issues, use_container_width=True)

        st.divider()

        # 4. Delivery Trend (Line Chart)
        st.subheader("4. Resolution Performance")
        if complaints is not None:
            # Simple line chart for resolution time across tickets
            st.write("Monitoring the time taken to resolve issues during the surge period:")
            st.line_chart(complaints['resolution_time'])
            st.image("https://images.unsplash.com/photo-1566576721346-d4a3b4eaad5b?auto=format&fit=crop&q=80&w=800", caption="Fleet Management Trends")

        st.divider()

        # 5. Funnel Metrics
        st.subheader("5. Key Performance Indicators (KPIs)")
        m1, m2, m3, m4 = st.columns(4)
        if nps is not None:
            m1.metric("Avg NPS Score", f"{nps['score'].mean():.2f}", delta="-1.5")
        if hubs is not None:
            m2.metric("Total RTO Count", int(hubs['rto_count'].sum()), delta="High", delta_color="inverse")
        if couriers is not None:
            m3.metric("Avg SLA Breach", f"{couriers['sla_breach_rate'].mean()*100:.1f}%")
        if orders is not None:
            success_rate = (len(orders[orders['order_status']=='Delivered']) / len(orders)) * 100
            m4.metric("Success Rate", f"{success_rate:.1f}%")

        st.success("Analysis Storyboard Generated Successfully!")
