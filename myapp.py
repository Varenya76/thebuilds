import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("loan_approval_model1.pkl")

st.title(" Loan Approval Predictor")

st.write("Enter applicant details below:")

# User inputs
income = st.number_input("Income (in USD)", min_value=0.0, value=50000.0, step=1000.0)
credit_score = st.slider("Credit Score", 300, 850, 700)
loan_amount = st.number_input("Loan Amount (in USD)", min_value=1000.0, value=10000.0, step=500.0)
dti_ratio = st.slider("Debt-to-Income Ratio (%)", 0.0, 100.0, 25.0)
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed"])

# Map employment status to 0/1
employment_status_encoded = 1 if employment_status == "Employed" else 0

# Prediction
if st.button("Predict"):
    input_data = np.array([[income, credit_score, loan_amount, dti_ratio, employment_status_encoded]])
    prediction = model.predict(input_data)[0]
    result = " Approved" if prediction == 1 else " Rejected"
    st.subheader("Result:")
    st.success(result)
