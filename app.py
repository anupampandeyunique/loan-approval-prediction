import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# Load Model
model = pickle.load(open('loan_model.pkl', 'rb'))

# Header
st.title("🏦 Loan Approval Prediction System")
st.markdown(
    "Fill in the applicant details below to predict whether the loan is likely to be approved."
)

st.divider()

# User Inputs
gender_option = st.selectbox(
    "Gender",
    ["Male", "Female"]
)
gender = 1 if gender_option == "Male" else 0

married_option = st.selectbox(
    "Marital Status",
    ["No", "Yes"]
)
married = 1 if married_option == "Yes" else 0

dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=5,
    value=0
)

education_option = st.selectbox(
    "Education",
    ["Not Graduate", "Graduate"]
)
education = 1 if education_option == "Graduate" else 0

self_emp_option = st.selectbox(
    "Self Employed",
    ["No", "Yes"]
)
self_employed = 1 if self_emp_option == "Yes" else 0

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=150
)

loan_term = st.number_input(
    "Loan Amount Term (Months)",
    min_value=0,
    value=360
)

credit_option = st.selectbox(
    "Credit History",
    ["Poor", "Good"]
)
credit_history = 1 if credit_option == "Good" else 0

area_option = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

property_area = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}[area_option]

st.divider()

# Prediction Button
if st.button("Predict Loan Status"):

    data = np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")

st.caption("Developed using Machine Learning and Streamlit")
