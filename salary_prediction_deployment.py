import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("salary_prediction_rfr_model.pkl")
encoder = joblib.load("label_encoder_salary.pkl")

st.title("Salary Prediction Model")

# Inputs
age = st.number_input("Enter Age", min_value=18, max_value=65)

gender = st.selectbox(
    "Select Gender",
    encoder["Gender"].classes_
)

education = st.selectbox(
    "Select Education Level",
    encoder["Education Level"].classes_
)

job = st.selectbox(
    "Select Job Title",
    encoder["Job Title"].classes_
)

# Encode values
gender_encoded = encoder["Gender"].transform([gender])[0]
education_encoded = encoder["Education Level"].transform([education])[0]
job_encoded = encoder["Job Title"].transform([job])[0]

if st.button("Predict Salary"):

    # Create dataframe
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_encoded],
        "Education Level": [education_encoded],
        "Job Title": [job_encoded]
    })

    # Ensure column order matches training
    input_data = input_data[model.feature_names_in_]

    # Prediction
    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")
