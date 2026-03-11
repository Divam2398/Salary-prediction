import streamlit as st
import pandas as pd
import joblib

model = joblib.load('salary_prediction_rfr_model.pkl')
encoder = joblib.load('label_encoder_Salary.pkl')

st.title('Salary Prediction model')

age = st.number_input("Enter your age", 18, 65)
gender = st.selectbox("Select your gender", encoder["Gender"].classes_)
education = st.selectbox("Select your education", encoder["Education Level"].classes_)
job_title = st.selectbox("Select your job title", encoder["Job Title"].classes_)
experience = st.number_input("Enter your experiene(in years)", 0, 50)

# IMPORTANT: column names must match training dataset
df = pd.DataFrame({
    "Age":[age],
    "Gender":[gender],
    "Education Level":[education],
    "Job Title":[job_title],
    "Years of Experience":[experience]
})

if st.button("predict salary"):

    for col in ["Gender","Education Level","Job Title"]:
        df[col] = encoder[col].transform(df[col])

    prediction = model.predict(df)

    st.success(f"predicted salary: {prediction[0]}")
