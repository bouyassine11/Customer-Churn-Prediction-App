import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Streamlit app title
st.title("Customer Churn Prediction App")

# Load the saved model and preprocessing objects
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load training dataset to get valid categories for LabelEncoder
training_data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
training_data = training_data.drop(columns=['customerID'])
training_data['TotalCharges'] = pd.to_numeric(training_data['TotalCharges'], errors='coerce')
categorical_cols = training_data.select_dtypes(include=['object']).columns
categorical_cols = [col for col in categorical_cols if col != 'Churn']  # Exclude target

# Create dictionary to store LabelEncoders for each categorical column
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(training_data[col])
    label_encoders[col] = le

# Create input fields for each feature (example features from Telco dataset)
st.header("Enter Customer Information")

# Numerical features
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)

# Categorical features (example options based on Telco dataset)
gender = st.selectbox("Gender", options=['Male', 'Female'])
senior_citizen = st.selectbox("Senior Citizen", options=[0, 1])
partner = st.selectbox("Partner", options=['Yes', 'No'])
dependents = st.selectbox("Dependents", options=['Yes', 'No'])
phone_service = st.selectbox("Phone Service", options=['Yes', 'No'])
multiple_lines = st.selectbox("Multiple Lines", options=['Yes', 'No', 'No phone service'])
internet_service = st.selectbox("Internet Service", options=['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox("Online Security", options=['Yes', 'No', 'No internet service'])
# Add other features as needed (e.g., OnlineBackup, DeviceProtection, etc.)

# Create DataFrame for new data
new_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
    # Add other columns to match training data
})

# Ensure all training columns are present (fill missing with mode or mean)
training_columns = training_data.drop('Churn', axis=1).columns
for col in training_columns:
    if col not in new_data.columns:
        if training_data[col].dtype == 'object':
            new_data[col] = training_data[col].mode()[0]
        else:
            new_data[col] = training_data[col].mean()

# Reorder columns to match training data
new_data = new_data[training_columns]

# Encode categorical variables using saved LabelEncoders
for col in categorical_cols:
    try:
        new_data[col] = label_encoders[col].transform(new_data[col])
    except ValueError:
        st.error(f"Invalid value for {col}. Please select from {label_encoders[col].classes_}")
        st.stop()

# Scale numerical features
new_data_scaled = scaler.transform(new_data)

# Make prediction
prediction = model.predict(new_data_scaled)[0]
probability = model.predict_proba(new_data_scaled)[0, 1]

# Display results
st.header("Prediction Results")
st.write(f"**Predicted Churn**: {'Yes' if prediction == 1 else 'No'}")
st.write(f"**Churn Probability**: {probability:.2%}")

# Visualize probability
st.subheader("Churn Probability Visualization")
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=['No Churn', 'Churn'], y=[1 - probability, probability], palette=['#4CAF50', '#F44336'])
ax.set_title('Churn Probability')
ax.set_ylabel('Probability')
st.pyplot(fig)

# Save results to session state for download
st.session_state['predictions'] = pd.DataFrame({
    'Predicted_Churn': ['Yes' if prediction == 1 else 'No'],
    'Churn_Probability': [probability]
})
st.download_button(
    label="Download Prediction",
    data=st.session_state['predictions'].to_csv(index=False),
    file_name="churn_prediction_result.csv",
    mime="text/csv"
)
