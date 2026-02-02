import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="sheerazzulfi/Tourism-Package-Prediction", filename="best_machine_failure_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a tourist to opt for a package, please enter the required fields.
Get the prediction by clicking the predict button.
""")

# User input
Age = st.number_input("Age of the person", min_value=0, max_value=100, value=30, step=1)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch", min_value=0, max_value=150, value=20, step=1)
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender =  st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=2, step=1)
NumberOfFollowups = st.number_input("Number of Followups", min_value=1, max_value=10, value=2, step=1)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe","Super Deluxe","King"])
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced","Unmarried"])
NumberOfTrips = st.number_input("Number of Trips", min_value=1, max_value=25, value=2)
Passport = st.selectbox("Passport", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=2)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager","AVP","VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=100000, value=50000, step=100)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

input_data = input_data.replace({'Yes': 1, 'No': 0})

if st.button("Predict result"):
    prediction = model.predict(input_data)[0]
    result = "Customer is likely to purchase" if prediction == 1 else "Customer is not likely to purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
