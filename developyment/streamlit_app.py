import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Config
st.set_page_config(page_title="AI Replacement Prediction", layout="centered")

st.title("AI Replacement Prediction App")
st.write("Predict whether a job is likely to be replaced by AI.")

# Load Model
model = pickle.load(open("model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# Load dataset (optional for reference)
data = pd.read_csv("AIREPLACEMENT.csv")

st.subheader("Enter Job Details")

# Example Input Fields (Modify according to your dataset columns)

education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
experience = st.slider("Years of Experience", 0, 30, 1)
salary = st.number_input("Salary", min_value=0)
automation_risk = st.slider("Automation Risk Score", 0.0, 1.0, 0.5)

# Create DataFrame from input
input_data = pd.DataFrame({
    "Education": [education],
    "Experience": [experience],
    "Salary": [salary],
    "Automation_Risk": [automation_risk]
})

# Prediction
if st.button("Predict"):
    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)

    if prediction[0] == 1:
        st.error(" High Risk: This job may be replaced by AI.")
    else:
        st.success(" Low Risk: This job is less likely to be replaced by AI.")
