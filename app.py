
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('loan_model.pkl', 'rb'))

st.title("Loan Approval Prediction System")

gender = st.selectbox("Gender", [0,1])
married = st.selectbox("Married", [0,1])
dependents = st.number_input("Dependents", 0, 5, 0)
education = st.selectbox("Education", [0,1])
self_employed = st.selectbox("Self Employed", [0,1])

applicant_income = st.number_input("Applicant Income", 0)
coapplicant_income = st.number_input("Coapplicant Income", 0)

loan_amount = st.number_input("Loan Amount", 0)
loan_term = st.number_input("Loan Amount Term", 0)

credit_history = st.selectbox("Credit History", [0,1])

property_area = st.selectbox("Property Area", [0,1,2])

if st.button("Predict"):
    data = np.array([[gender, married, dependents, education,
                      self_employed, applicant_income,
                      coapplicant_income, loan_amount,
                      loan_term, credit_history,
                      property_area]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
