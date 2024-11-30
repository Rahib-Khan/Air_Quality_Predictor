import joblib

# Load the saved model
rf_model = joblib.load('src/saved_model.pkl')

# The model is now restored and ready for use
print((rf_model))