import streamlit as st
import joblib
import pandas as pd

# Load model
loaded_model = joblib.load("src/saved_model.pkl")

# Tableau Public URL
tableau_public_url = "https://public.tableau.com/views/Book1_17323934199370/Sheet2?:embed=yes&:display_count=no&:showVizHome=no"

# Set Streamlit page configuration
st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌎",
    layout="centered"
)

# Apply CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #eef2f3;
    }
    .main {
        background-color: transparent;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        max-width: 800px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    h1 {
        color: white;
        text-align: center;
        font-family: Arial, sans-serif;
        margin-bottom: 20px;
    }
    h3 {
        color: white;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Wrap everything in a centered container
with st.container():
    # App title
    st.title("Air Quality Prediction")

    # Embed Tableau Public dashboard
    st.markdown("### Visualization of Air Quality Trends")
    st.components.v1.iframe(src=tableau_public_url, width=800, height=600)

    # User inputs section
    st.markdown("### 📋 Input Details")
    states = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
        "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
        "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maryland",
        "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
        "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
        "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
        "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
        "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
    ]

    # User inputs
    state_name = st.selectbox('🌐 Select State', states)
    population = st.number_input("👨‍👩‍👧‍👦 Enter Population", min_value=0, step=1)
    non_renewable_vehicles = st.number_input("🚗 Enter Number of Non-Renewable Vehicles", min_value=0, step=1)

    # Prediction function
    def predict_aqi(state_name, population, non_renewable_vehicles, model):
        # Check the columns expected by the model
        expected_columns = model.feature_names_in_

        # Scale the population data
        population = population / 1000

        # Create input features with all expected columns
        input_data = {col: 0 for col in expected_columns}  # Initialize all with 0
        input_data['Population'] = population
        input_data['Non-Renewable Vehicles'] = non_renewable_vehicles

        # Set the one-hot encoded column for the selected state
        state_column = f"State Name_{state_name}"
        if state_column in input_data:
            input_data[state_column] = 1
        else:
            print(f"Warning: State '{state_name}' not found in training data. Defaulting to 0.")

        # Convert to DataFrame for the model
        input_data_df = pd.DataFrame([input_data])

        # Make the prediction
        predicted_aqi = model.predict(input_data_df)
        return predicted_aqi[0]

    # Predict button
    if st.button("🚀 Predict AQI"):
        if not state_name:
            st.warning("⚠️ Please select a state.")
        else:
            try:
                prediction = predict_aqi(state_name, population, non_renewable_vehicles, loaded_model)
                st.success(f"✅ The predicted Air Quality Index (AQI) for {state_name} is: {prediction:.2f}")
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
