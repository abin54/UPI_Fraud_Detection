"""
Streamlit Dashboard for UPI Fraud Detection
Interactive fraud monitoring and analysis dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os

# Page config
st.set_page_config(
    page_title="UPI Fraud Detection Dashboard",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #fd7e14; font-weight: bold; }
    .risk-critical { color: #dc3545; font-weight: bold; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    try:
        model = joblib.load('../models/fraud_model.joblib')
        engineer = joblib.load('../models/feature_engineer.joblib')
        config = joblib.load('../models/config.joblib')
        return model, engineer, config['threshold']
    except:
        return None, None, 0.5

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../data/synthetic_upi_transactions.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except:
        return None

model, engineer, threshold = load_model()
df = load_data()

# Title
st.title("🔒 UPI Fraud Detection Dashboard")
st.markdown("Real-time monitoring and analysis of UPI transaction fraud")

# Sidebar
st.sidebar.header("Settings")
date_range = st.sidebar.selectbox(
    "Time Period",
    ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"]
)

risk_filter = st.sidebar.multiselect(
    "Risk Level Filter",
    ["Low", "Medium", "High", "Critical"],
    default=["High", "Critical"]
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Transaction Monitor", "📈 Analytics", "⚡ Real-time Check"])

with tab1:
    st.header("Fraud Detection Overview")

    if df is not None:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        total_txns = len(df)
        fraud_count = df['is_fraud'].sum()
        fraud_rate = (fraud_count / total_txns) * 100
        total_amount = df['amount'].sum()
        fraud_amount = df[df['is_fraud'] == 1]['amount'].sum()

        with col1:
            st.metric("Total Transactions", f"{total_txns:,}")
        with col2:
            st.metric("Fraud Detected", f"{fraud_count:,}", f"{fraud_rate:.2f}%")
        with col3:
            st.metric("Total Volume", f"₹{total_amount/1e6:.2f}M")
        with col4:
            st.metric("Fraud Amount", f"₹{fraud_amount/1e6:.2f}M")

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fraud by Merchant Category")
            fraud_by_category = df.groupby('merchant_category').agg({
                'is_fraud': ['sum', 'count']
            }).reset_index()
            fraud_by_category.columns = ['category', 'fraud_count', 'total']
            fraud_by_category['fraud_rate'] = fraud_by_category['fraud_count'] / fraud_by_category['total'] * 100

            fig = px.bar(
                fraud_by_category,
                x='category',
                y='fraud_rate',
                color='fraud_rate',
                color_continuous_scale='Reds',
                title="Fraud Rate by Category (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Fraud by Hour of Day")
            fraud_by_hour = df.groupby('hour').agg({
                'is_fraud': ['sum', 'count']
            }).reset_index()
            fraud_by_hour.columns = ['hour', 'fraud_count', 'total']
            fraud_by_hour['fraud_rate'] = fraud_by_hour['fraud_count'] / fraud_by_hour['total'] * 100

            fig = px.line(
                fraud_by_hour,
                x='hour',
                y='fraud_rate',
                markers=True,
                title="Fraud Rate by Hour (%)"
            )
            fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Fraud Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

        # Transaction amount distribution
        st.subheader("Transaction Amount Distribution: Fraud vs Normal")
        fig = px.histogram(
            df,
            x='amount',
            color='is_fraud',
            nbins=50,
            color_discrete_map={0: 'blue', 1: 'red'},
            labels={'is_fraud': 'Is Fraud'},
            log_y=True
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Please run data_generator.py first to generate transaction data.")

with tab2:
    st.header("Transaction Monitor")

    if df is not None:
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            min_amount = st.number_input("Min Amount (₹)", value=0)
        with col2:
            max_amount = st.number_input("Max Amount (₹)", value=int(df['amount'].max()))
        with col3:
            show_fraud_only = st.checkbox("Show Fraud Only", value=False)

        # Filter data
        filtered_df = df[(df['amount'] >= min_amount) & (df['amount'] <= max_amount)]
        if show_fraud_only:
            filtered_df = filtered_df[filtered_df['is_fraud'] == 1]

        # Display transactions
        st.subheader(f"Transactions ({len(filtered_df):,} records)")

        display_cols = ['transaction_id', 'timestamp', 'sender_upi', 'receiver_upi',
                       'amount', 'merchant_category', 'location_city', 'is_fraud']
        st.dataframe(
            filtered_df[display_cols].head(100),
            use_container_width=True,
            hide_index=True
        )

        # Download option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "Download Filtered Data (CSV)",
            csv,
            "transactions.csv",
            "text/csv"
        )

with tab3:
    st.header("Fraud Analytics")

    if df is not None:
        col1, col2 = st.columns(2)

        with col1:
            # City-wise fraud
            st.subheader("Top 10 Cities by Fraud Count")
            city_fraud = df[df['is_fraud'] == 1]['location_city'].value_counts().head(10)
            fig = px.bar(
                x=city_fraud.values,
                y=city_fraud.index,
                orientation='h',
                labels={'x': 'Fraud Count', 'y': 'City'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Device type fraud
            st.subheader("Fraud by Device Type")
            device_fraud = df.groupby('device_type')['is_fraud'].agg(['sum', 'count']).reset_index()
            device_fraud['rate'] = device_fraud['sum'] / device_fraud['count'] * 100
            fig = px.pie(
                device_fraud,
                values='sum',
                names='device_type',
                title="Fraud Distribution by Device"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Heatmap: Hour vs Day of Week
        st.subheader("Fraud Heatmap: Hour vs Day of Week")
        heatmap_data = df[df['is_fraud'] == 1].groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        fig = px.imshow(
            heatmap_data,
            labels={'x': 'Hour', 'y': 'Day of Week', 'color': 'Fraud Count'},
            aspect='auto',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Real-time Fraud Check")
    st.markdown("Enter transaction details to get instant fraud risk assessment")

    col1, col2 = st.columns(2)

    with col1:
        user_id = st.text_input("User ID", "USER_00001")
        sender_upi = st.text_input("Sender UPI", "user123@paytm")
        receiver_upi = st.text_input("Receiver UPI", "merchant@sbi")
        amount = st.number_input("Amount (₹)", value=1000.0, min_value=1.0)
        merchant_category = st.selectbox("Merchant Category", [
            'grocery', 'restaurant', 'fuel', 'electronics', 'clothing',
            'pharmacy', 'utilities', 'entertainment', 'travel', 'education'
        ])

    with col2:
        location = st.selectbox("Location", [
            'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai',
            'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow'
        ])
        device_type = st.selectbox("Device Type", ['android', 'ios'])
        hour = st.slider("Transaction Hour", 0, 23, datetime.now().hour)
        is_new_device = st.checkbox("New Device?")

    if st.button("🔍 Check Fraud Risk", type="primary"):
        # Simulate prediction
        risk_score = np.random.beta(2, 10)  # Placeholder - use actual model in production

        # Adjust based on inputs
        if hour < 5 or hour > 23:
            risk_score += 0.15
        if amount > 50000:
            risk_score += 0.2
        if is_new_device:
            risk_score += 0.1
        if merchant_category in ['electronics', 'travel']:
            risk_score += 0.05

        risk_score = min(risk_score, 1.0)

        st.markdown("---")
        st.subheader("Risk Assessment Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Risk Score", f"{risk_score*100:.1f}%")

        with col2:
            if risk_score < 0.3:
                st.success("Risk Level: LOW")
            elif risk_score < 0.6:
                st.warning("Risk Level: MEDIUM")
            elif risk_score < 0.8:
                st.error("Risk Level: HIGH")
            else:
                st.error("Risk Level: CRITICAL")

        with col3:
            if risk_score < 0.5:
                st.success("Recommendation: APPROVE")
            elif risk_score < 0.7:
                st.warning("Recommendation: REVIEW")
            else:
                st.error("Recommendation: BLOCK")

        # Risk factors
        st.subheader("Risk Factors Detected")
        factors = []
        if hour < 5 or hour > 23:
            factors.append("⚠️ Unusual transaction time")
        if amount > 50000:
            factors.append("⚠️ High transaction amount")
        if is_new_device:
            factors.append("⚠️ New device detected")
        if not factors:
            factors.append("✅ No significant risk factors")

        for factor in factors:
            st.write(factor)

# Footer
st.markdown("---")
st.markdown("**UPI Fraud Detection System** | Built with Streamlit | © 2024")
