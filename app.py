import streamlit as st
import joblib
import pandas as pd

# Load model
loaded_model = joblib.load("src/saved_model.pkl")

# Tableau Public URL
tableau_public_url = "https://public.tableau.com/views/Book1_17323934199370/Sheet2?:embed=yes&:display_count=no&:showVizHome=no"

# Set Streamlit page configuration
st.set_page_config(
    page_title="Air Quality App",
    page_icon="🌎",
    layout="centered"
)

# App title
st.title("🌎 Air Quality Dashboard")

# Create tabs
tabs = st.tabs(["📖 Introduction", "🌍 Map Visualization", "📊 Predict AQI", "🧍 About Us"])

# Introduction Tab
with tabs[0]:
    st.subheader("📖 What is AQI?")
    st.markdown("""
        The **Air Quality Index (AQI)** is a measure used to communicate how polluted the air currently is or how polluted it is forecast to become. 
        AQI values range from 0 to 500, with higher values indicating worse air quality. Key pollutants include:
        - Particulate matter (PM2.5 and PM10)
        - Ozone (O3)
        - Nitrogen dioxide (NO2)
        - Sulfur dioxide (SO2)
        - Carbon monoxide (CO)
        
        The AQI is divided into categories, each corresponding to a different level of health concern:
        - **0–50:** Good
        - **51–100:** Moderate
        - **101–150:** Unhealthy for sensitive groups
        - **151–200:** Unhealthy
        - **201–300:** Very Unhealthy
        - **301–500:** Hazardous
    """)
    
    st.subheader("📊 About Our Model")
    st.markdown("""
        This application uses a **Random Forest Tree** to predict AQI values based on:
        - **Population trends** in different states.
        - **Number of non-renewable vehicles**, which contribute to pollution.
        
        The model was trained using a dataset spanning **10 years (2013–2023)**, including AQI data, population statistics, and vehicle registrations. 
        Predictions are state-specific and consider factors such as historical AQI trends and population growth. The main goal is to provide actionable insights for reducing air pollution.
    """)
    st.image('Images/output.png', caption='Model Prediction vs Actual')

# Map Visualization Tab
with tabs[1]:
    st.subheader("🌟 Visualization of Air Quality Trends")
    st.components.v1.iframe(src=tableau_public_url, width=1000, height=800)

# Prediction Tab
with tabs[2]:
    st.subheader("📋 Predict Air Quality Index")
    
    # List of states
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



with tabs[3]:
    st.subheader("👨‍👩‍👧‍👦 About Us")
    st.markdown("""
        Welcome to our **Air Quality Prediction App**!  
        We are a team of three passionate individuals who collaborated to build this application to raise awareness about air quality trends and empower data-driven decisions.

        ### 👥 Meet the Team
        - **Rahib Khandaker**  
          💡 Computer Science Student at CUNY Queens College with a minor in Data Analytics.  
          [Visit Rahib's LinkedIn Profile](https://www.linkedin.com/in/rahib-khandaker/)
                
          [Visit Rahib's Portfolio Website](https://rahib-khan.github.io/)
        
        - **Mohamed Anas Aaffoute**  
          🌍 Computer Science Student at CUNY Queens College and an aspiring software engineer.  
          [Visit Anas's Portfolio Website](https://main.d3imjygzv65w7j.amplifyapp.com/)  
        
        - **Arnan Khan**  
          📚 Computer Science student at CUNY Queens College with expertise in data analysis and web development.  
          [Visit Arnan's Portfolio Website](https://arnank.github.io/Portfolio-Website/)  

        ### 💡 Our Vision
        Our mission is to leverage technology to promote cleaner air and a healthier environment by providing valuable insights into air quality data.

        **Thank you for exploring our app!**
    """)
