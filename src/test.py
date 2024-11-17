# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the data
safedata = pd.read_csv('../Processed_Data/AQI_Final.csv')
data = pd.read_csv('../Processed_Data/AQI_Final.csv')

def randomRF(State_Name):
    safedata = pd.read_csv('../Processed_Data/AQI_Final.csv')
    data = pd.read_csv('../Processed_Data/AQI_Final.csv')
    # Filter the state
    data = data[data['State Name'] == State_Name]
    
    # Drop non-numeric and irrelevant columns for prediction (e.g., 'State Name' and 'Year')
    safedata = safedata.drop(columns=[ 'Unknown Fuel','Ethanol/Flex (E85)', 'Diesel', 'Hybrid Electric (HEV)', 
                            'Electric (EV)','Biodiesel' ,'Compressed Natural Gas (CNG)', 'Gasoline', 'Plug-In Hybrid Electric (PHEV)', ])

    data = data.drop(columns=[ 'Year','State Name', 'Unknown Fuel','Ethanol/Flex (E85)', 'Diesel', 'Hybrid Electric (HEV)', 
                            'Electric (EV)','Biodiesel' ,'Compressed Natural Gas (CNG)', 'Gasoline', 'Plug-In Hybrid Electric (PHEV)', ])

    # Check for missing values and fill or drop them as needed
    if data.isnull().sum().sum() > 0:
        data = data.fillna(data.mean())  # Fill missing values with column means (alternative: data.dropna())

    # Split data into features (X) and target (y)
    X = data.drop(columns=['Overall AQI'])  # Features
    y = data['Overall AQI']  # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features to improve model performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance for {State_Name}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")

    # Optional: Feature Importance
    feature_importances = pd.DataFrame(model.feature_importances_, index=data.drop(columns=['Overall AQI']).columns, columns=['Importance']).sort_values('Importance', ascending=False)
    #print("\nFeature Importances:")
    #print(feature_importances)
    merge = pd.merge(safedata,data, how='inner')
    #print(merge)
    return(mse)


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

AQI_error = []
AQI_r = []
AQI_error = []
for state in states:
    AQI_error.append(randomRF(state))
print(f"Average error {np.mean(AQI_error)}")
