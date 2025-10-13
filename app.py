# app.py
import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="🏦 Customer Churn Predictor", page_icon="💼", layout="centered")

st.title("🏦 Customer Churn Prediction App")
st.write("Enter customer details below to predict whether they will churn or stay.")

# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0)
num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

# Convert categorical values to numerical
geo_map = {"France": 0, "Germany": 1, "Spain": 2}
gender_map = {"Male": 1, "Female": 0}

has_credit_card = 1 if has_credit_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

# Prepare input array
data = np.array([[credit_score,
                  geo_map[geography],
                  gender_map[gender],
                  age,
                  tenure,
                  balance,
                  num_products,
                  has_credit_card,
                  is_active_member,
                  estimated_salary]])

# Scale input
scaled_data = scaler.transform(data)

# Predict churn
if st.button("Predict Churn 💡"):
    prediction = model.predict(scaled_data)[0]
    result = "Customer is likely to CHURN ❌" if prediction >= 0.5 else "Customer will STAY ✅"
    st.subheader(result)
    st.write(f"Predicted Score: **{prediction:.2f}**")

st.markdown("---")
st.caption("Model: Ridge Regression | Built with Streamlit")
