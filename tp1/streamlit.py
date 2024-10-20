import streamlit as st
from tp1.model_utils import load_model
import numpy as np

model = load_model('regression.joblib')

st.title("Prediction App")

size = st.number_input("Enter the size:", min_value=0)
nb_rooms = st.number_input("Enter the number of rooms:", min_value=0)
garden = st.number_input("Enter 1 if there is a garden, otherwise 0:", min_value=0, max_value=1)

if st.button("Predict"):
    input_array = np.array([[size, nb_rooms, garden]])
    prediction = model.predict(input_array)
    st.write(f'Price is {round(prediction[0], 2)} $')
