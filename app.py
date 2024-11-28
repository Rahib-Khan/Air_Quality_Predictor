import streamlit as st
import pickle
import joblib
import numpy as np

# Streamlit application title
st.title("Air Quality Map")

# Tableau Public URL
tableau_public_url = "https://public.tableau.com/views/Book1_17323934199370/Sheet2?:embed=yes&:display_count=no&:showVizHome=no"

# Embed the Tableau Public dashboard
st.components.v1.iframe(src=tableau_public_url, width=1100, height=850)

# Load the model
with open('src/saved_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Add input fields to get user data
st.subheader("AQI Prediction Input")
state_name = st.text_input("Enter State Name")
population = st.number_input("Enter Population", min_value=0, step=1)
non_renewable_vehicles = st.number_input("Enter Number of Non-Renewable Vehicles", min_value=0, step=1)

# Button to make predictions
if st.button("Predict AQI"):
    # Ensure that the user has entered a state name
    if not state_name:
        st.warning("Please enter a state name")
    else:
        # Prepare input data for prediction
        input_features = np.array([population, non_renewable_vehicles]).reshape(1, -1)

        # Make prediction using the model
        prediction = loaded_model.predicted_aqi(input_features)

        # Display the predicted AQI
        st.success(f"The predicted Air Quality Index (AQI) for {state_name} is: {prediction[0]}")
