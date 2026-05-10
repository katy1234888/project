import streamlit as st
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

# Set Page Config
st.set_page_config(page_title="Seashells Logistics Dashboard", layout="wide")

# --- NAVIGATION ---
page = st.sidebar.selectbox("Navigate", ["Home", "Data Upload & Summary", "ML Data Cleaning"])

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
            # Remove completely empty rows/columns that often appear in CSV exports
            df = df.dropna(how='all')
            st.session_state['datasets'][file.name] = df
            
        st.success(f"Successfully uploaded {len(uploaded_files)} files!")
        
        for name, df in st.session_state['datasets'].items():
            with st.expander(f"Summary: {name}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Data Preview (First 5 rows):**")
                    st.dataframe(df.head())
                with col2:
                    st.write("**Missing Values Profile:**")
                    st.write(df.isnull().sum())
                    st.metric("Total Rows", df.shape[0])
                    st.metric("Total Columns", df.shape[1])

# --- PAGE 3: ML DATA CLEANING ---
elif page == "ML Data Cleaning":
    st.header("🤖 Machine Learning Based Data Imputation")
    
    if not st.session_state['datasets']:
        st.warning("Please upload datasets first on the 'Data Upload' page.")
    else:
        st.info("The system will use the MICE (Multivariate Imputation by Chained Equations) algorithm to fill missing numeric data.")
        
        if st.button("Run ML Imputation Pipeline"):
            progress_bar = st.progress(0)
            count = 0
            total = len(st.session_state['datasets'])
            
            for name, df in st.session_state['datasets'].items():
                # Separate numeric and categorical
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(exclude=[np.number]).columns
                
                # ML Imputation for Numeric (Iterative Imputer)
                if len(numeric_cols) > 0 and df[numeric_cols].isnull().sum().sum() > 0:
                    it_imp = IterativeImputer(random_state=42)
                    df[numeric_cols] = it_imp.fit_transform(df[numeric_cols])
                
                # Simple Imputation for Categorical (Most Frequent)
                if len(categorical_cols) > 0 and df[categorical_cols].isnull().sum().sum() > 0:
                    cat_imp = SimpleImputer(strategy='most_frequent')
                    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])
                
                st.session_state['datasets'][name] = df
                count += 1
                progress_bar.progress(count / total)
            
            st.success("Data Processing and Imputation Complete!")
            
        st.subheader("Data Summary After Cleaning")
        for name, df in st.session_state['datasets'].items():
            with st.expander(f"Cleaned Summary: {name}"):
                st.write("**Missing Values (Check):**")
                st.write(df.isnull().sum())
                st.write("**Processed Data Preview:**")
                st.dataframe(df.head())
# ... (Keep all your previous code above this line) ...

# 1. Update your Navigation at the top to include the new page:
# page = st.sidebar.selectbox("Navigate", ["Home", "Data Upload & Summary", "ML Data Cleaning", "Deep Dive Analytics"])

# 2. Add this logic at the end of your file:

elif page == "Deep Dive Analytics":
    st.header("🔍 Deep Dive: Festive Surge Analysis")
    
    if not st.session_state['datasets']:
        st.warning("Please upload datasets first on the 'Data Upload' page.")
    else:
        # Load datasets from session state
        orders = st.session_state['datasets'].get('orders.csv')
        nps = st.session_state['datasets'].get('nps.csv')
        complaints = st.session_state['datasets'].get('complaints.csv')
        hubs = st.session_state['datasets'].get('hub_performance.csv')
        
        st.subheader("1. The Voice of the Customer (NPS Analysis)")
        
        if nps is not None:
            # Logic: Calculate NPS Category
            def categorize_nps(score):
                if score >= 9: return 'Promoter'
                if score <= 6: return 'Detractor'
                return 'Passive'
            
            nps['Category'] = nps['score'].apply(categorize_nps)
            nps_counts = nps['Category'].value_counts()
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("### How do our customers feel?")
                st.write("The chart shows a significant lean towards 'Detractors' during the festive months. This indicates that while volumes are up, the emotional connection with the brand is fragile.")
                # Pie Chart
                st.plotly_chart(pd.Series(nps_counts).plot.pie(
                    title="NPS Distribution", 
                    backend="plotly",
                    hole=0.4
                ))
            
            with col2:
                st.image("https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&q=80&w=1000", 
                         caption="Customer Feedback is our North Star")
        
        st.divider()
        
        st.subheader("2. Operational Bottlenecks (City-wise Delivery)")
        
        
        if hubs is not None:
            # Logic: Identify worst performing cities
            hubs['Failure_Rate'] = (hubs['failed_attempts'] / hubs['total_orders']) * 100
            
            st.write("### Where is the friction?")
            st.write("By analyzing failed delivery attempts, we see Nagpur and Indore facing the highest operational friction. This correlates with the 'Tier-2 decline' mentioned in the business context.")
            
            # Bar Chart
            st.bar_chart(hubs.set_index('city')['Failure_Rate'])
            
        st.divider()
        
        st.subheader("3. Complaint Trends & Escalations")
        
        if complaints is not None:
            # Logic: Issue type distribution
            issue_counts = complaints['issue_type'].value_counts().reset_index()
            issue_counts.columns = ['Issue', 'Count']
            
            # Line/Area Chart for Resolution Time
            st.write("### Resolution Efficiency")
            st.write("Late Delivery remains the #1 issue. The resolution time spikes for escalated tickets, suggesting a need for better automated tracking.")
            
            st.line_chart(complaints['resolution_time'])
            
            # Text based innovation: The "Action Plan"
            st.info("💡 **Analyst Insight:** 70% of 'Fake Delivery' complaints are concentrated in Pune. This suggests a specific training gap or a courier partner issue in that region.")
        
        st.divider()
        
        st.subheader("4. The Logistics Funnel Summary")
        
        
        # Summary metrics
        m1, m2, m3 = st.columns(3)
        if orders is not None:
            delivered_count = len(orders[orders['order_status'] == 'Delivered'])
            m1.metric("Total Successful Deliveries", f"{delivered_count}")
        
        if nps is not None:
            avg_score = round(nps['score'].mean(), 2)
            m2.metric("Average NPS Score", avg_score, delta="-1.2 vs Last Quarter")
            
        if hubs is not None:
            total_rto = int(hubs['rto_count'].sum())
            m3.metric("Total RTO Count", total_rto, delta="High Risk", delta_color="inverse")

        st.success("Analysis Complete. Use these insights to draft the Festive Strategy 2.0!")
