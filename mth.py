import os
import streamlit as st
import pickle
import numpy as np

# Construct the absolute path for the model and scaler
model_path =  'finalized_model_ice.pickle'
scaler_path =  'scaler-ice-pickle'

# Load the trained model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Title for the app
st.title('User Prediction App')

# Input fields
st.write("Please enter the following details:")

# Get input data from the user
Temperature = st.number_input('Temperature', min_value=0.0, step=0.1)


# When the user clicks the Predict button
if st.button('Predict'):
    try:
        # Prepare the data for prediction as a NumPy array
        input_data = np.array([[Temperature]])

        # Scale the input data
        scaled_input = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(scaled_input)[0]

        # Display the result
        st.success(f"The Ice Cream profit: {round(prediction, 2)}")  # Display as currency if desired
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")