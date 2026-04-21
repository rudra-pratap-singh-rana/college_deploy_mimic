import streamlit as st
import pandas as pd
import numpy as np
import pickle

# LOAD MODEL
with open("model1.pkl", "rb") as f:
    model, scaler, le, columns = pickle.load(f)

st.title("🌡️ Temperature Prediction App")

user_input = []

for col in columns:
    val = st.number_input(f"{col}", value=0.0)
    user_input.append(val)

if st.button("Predict"):
    input_data = pd.DataFrame([user_input], columns=columns)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = le.inverse_transform(prediction)

    st.success(f"Prediction: {result[0]}")
