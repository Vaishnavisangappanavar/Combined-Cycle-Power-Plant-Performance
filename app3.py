# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:47:15 2024

@author: palth
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model
load_model = pickle.load(open('D:/project_3_deployment/trained_model.sav', 'rb'))

# Define a function to predict energy production
def energy_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = load_model.predict(input_data_reshaped)
    return prediction

# Define the Streamlit app layout
def main():
    st.title('Combined-Cycle Power Plant Energy Production Prediction')
    st.sidebar.header('Input Parameters')

    # Define input fields using number_input for manual input
    temperature = st.sidebar.number_input('Temperature (°C)', min_value=0.0, max_value=100.0, value=25.0)
    exhaust_vacuum = st.sidebar.number_input('Exhaust Vacuum (cm Hg)', min_value=0.0, max_value=100.0, value=50.0)
    amb_pressure = st.sidebar.number_input('Ambient Pressure (millibar)', min_value=0.0, max_value=1500.0, value=1000.0)
    r_humidity = st.sidebar.number_input('Relative Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)

    # Combine input data
    input_data = [temperature, exhaust_vacuum, amb_pressure, r_humidity]

    # Predict energy production
    if st.sidebar.button('Predict'):
        prediction = energy_prediction(input_data)
        
        # Output the prediction
        st.subheader('Prediction:')
        st.write('Predicted Energy Production (MW):', prediction[0])

if __name__ == '__main__':
    main()
