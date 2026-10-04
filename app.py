import streamlit as st
import pandas as pd
from ml_predictor import train_risk_model, predict_customer_risk
from llm_summarizer import generate_targeted_campaign

st.set_page_config(page_title="Customer Intelligence Hub", layout="wide")
st.title("🎯 Customer Intelligence & Marketing Hub")

# Cache the model training so it only runs once when the app starts
@st.cache_resource
def load_ml_model():
    return train_risk_model("data/restaurant_users_confirm.csv")

rf_model, expected_columns = load_ml_model()

st.markdown("### 🔮 Predictive Risk Assessment")
st.markdown("Enter a hypothetical customer's data to predict their churn risk using the Random Forest model.")

# Create input fields for the ML model
col1, col2, col3 = st.columns(3)
with col1:
    sim_amount = st.number_input("Amount Spent (₹)", min_value=100, max_value=5000, value=800)
with col2:
    sim_payment = st.selectbox("Payment Method", ["UPI", "Credit Card", "Debit Card", "Cash"])
with col3:
    sim_dish = st.selectbox("Favorite Dish", ["Masala Dosa", "Pasta", "Biryani", "Butter Chicken", "Paneer Tikka"])

if st.button("Predict Customer Risk"):
    # Run the ML Prediction
    risk_status = predict_customer_risk(rf_model, expected_columns, sim_amount, sim_payment, sim_dish)
    
    if risk_status == "High Risk of Churn":
        st.error(f"⚠️ Model Prediction: {risk_status}")
        
        # Trigger the GenAI Marketing Engine
        st.markdown("#### ⚡ AI Action Required: Generating Win-Back Campaign...")
        with st.spinner("Drafting targeted SMS with Gemini..."):
            sms_copy = generate_targeted_campaign(sim_dish, sim_payment, sim_amount)
            st.success("Campaign Drafted!")
            st.info(f"📱 **SMS Copy:** {sms_copy}")
    else:
        st.success(f"✅ Model Prediction: {risk_status}. No immediate intervention required.")