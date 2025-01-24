import streamlit as st
import joblib
import pandas as pd

# Load the saved Logistic Regression model
model_filename = "logistic_regression_model.pkl"
feature_names_filename = "training_feature_names.pkl"  # Save this file during training
try:
    model = joblib.load(model_filename)
    st.sidebar.success(f"Model '{model_filename}' loaded successfully.")
    # Load the feature names
    training_features = joblib.load(feature_names_filename)
except FileNotFoundError as e:
    st.sidebar.error(f"Error loading file: {e}")
    st.stop()

st.title("Chronic Kidney Disease Classification")
st.write("""
This tool allows you to input patient data and predict the likelihood of Chronic Kidney Disease.
""")

# Input fields for patient data
st.sidebar.header("Enter Patient Details")

# Collect user inputs
user_input = {
    'age': st.sidebar.number_input("Age (Years):", min_value=0, max_value=120, step=1, value=30),
    'blood_pressure': st.sidebar.number_input("Blood Pressure (mmHg):", min_value=50, max_value=200, step=1, value=120),
    'specific_gravity': st.sidebar.selectbox("Specific Gravity:", [1.005, 1.010, 1.015, 1.020, 1.025]),
    'albumin': st.sidebar.slider("Albumin (0-5):", min_value=0, max_value=5, step=1, value=0),
    'sugar': st.sidebar.slider("Sugar (0-5):", min_value=0, max_value=5, step=1, value=0),
    'red_blood_cells_normal': st.sidebar.selectbox("Red Blood Cells (Normal):", ['No', 'Yes']),
    'pus_cell_normal': st.sidebar.selectbox("Pus Cell (Normal):", ['No', 'Yes']),
    'pus_cell_clumps_present': st.sidebar.selectbox("Pus Cell Clumps (Present):", ['No', 'Yes']),
    'bacteria_present': st.sidebar.selectbox("Bacteria (Present):", ['No', 'Yes']),
    'blood_glucose_random': st.sidebar.number_input("Blood Glucose Random:", min_value=0.0, step=0.1, value=120.0),
    'blood_urea': st.sidebar.number_input("Blood Urea:", min_value=0.0, step=0.1, value=20.0),
    'serum_creatinine': st.sidebar.number_input("Serum Creatinine:", min_value=0.0, step=0.1, value=1.2),
    'sodium': st.sidebar.number_input("Sodium:", min_value=0.0, step=0.1, value=135.0),
    'potassium': st.sidebar.number_input("Potassium:", min_value=0.0, step=0.1, value=4.5),
    'hemoglobin': st.sidebar.number_input("Hemoglobin:", min_value=0.0, step=0.1, value=13.5),
    'packed_cell_volume': st.sidebar.number_input("Packed Cell Volume:", min_value=0.0, step=0.1, value=45.0),
    'white_blood_cell_count': st.sidebar.number_input("White Blood Cell Count:", min_value=0.0, step=0.1, value=8000.0),
    'red_blood_cell_count': st.sidebar.number_input("Red Blood Cell Count:", min_value=0.0, step=0.1, value=5.0),
    'hypertension_yes': st.sidebar.selectbox("Hypertension (Yes):", ['No', 'Yes']),
    'diabetes_mellitus_yes': st.sidebar.selectbox("Diabetes Mellitus (Yes):", ['No', 'Yes']),
    'coronary_artery_disease_yes': st.sidebar.selectbox("Coronary Artery Disease (Yes):", ['No', 'Yes']),
    'appetite_poor': st.sidebar.selectbox("Appetite (Poor):", ['No', 'Yes']),
    'pedal_edema_yes': st.sidebar.selectbox("Pedal Edema (Yes):", ['No', 'Yes']),
    'anemia_yes': st.sidebar.selectbox("Anemia (Yes):", ['No', 'Yes']),
}

# Convert user inputs to a DataFrame
user_input_df = pd.DataFrame([user_input])

# Convert binary inputs (Yes/No) to 0/1 for the model
user_input_df = user_input_df.replace({'Yes': 1, 'No': 0})

# Ensure all training features are included in the input DataFrame
for col in training_features:
    if col not in user_input_df.columns:
        user_input_df[col] = 0  # Add missing columns with default values (e.g., 0 for unseen features)

# Align columns to match training data
user_input_df = user_input_df[training_features]

# Display user input data
st.write("### Input Data for Prediction:")
st.dataframe(user_input_df)

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(user_input_df)
        prediction_proba = model.predict_proba(user_input_df)
        result = "CKD Detected" if prediction[0] == 1 else "No CKD"
        st.success(f"Prediction: {result}")
        st.write(f"Prediction Probability: {prediction_proba[0][1]:.2f} (CKD), {prediction_proba[0][0]:.2f} (No CKD)")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
