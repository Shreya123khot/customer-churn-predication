import streamlit as st
import pandas as pd
import pickle


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# -------------------------------
# Load Model
# -------------------------------

@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    return model


model = load_model()


# -------------------------------
# Title
# -------------------------------

st.title("📊 Customer Churn Prediction System")

st.write(
    "Machine Learning based system to predict whether customer will leave or stay."
)


st.divider()


# -------------------------------
# Input Section
# -------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )


    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0,1]
    )


    Partner = st.selectbox(
        "Partner",
        ["Yes","No"]
    )


    Dependents = st.selectbox(
        "Dependents",
        ["Yes","No"]
    )


    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )


with col2:

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes","No"]
    )


    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["Yes","No","No phone service"]
    )


    InternetService = st.selectbox(
        "Internet Service",
        ["DSL","Fiber optic","No"]
    )


    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes","No","No internet service"]
    )


    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes","No","No internet service"]
    )


with col3:


    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes","No","No internet service"]
    )


    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes","No","No internet service"]
    )


    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes","No","No internet service"]
    )


    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes","No","No internet service"]
    )



# -------------------------------
# Payment Details
# -------------------------------

st.subheader("💳 Payment Information")


col4,col5,col6 = st.columns(3)


with col4:

    Contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


with col5:

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes","No"]
    )


with col6:

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )



MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)


TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)



# -------------------------------
# Prediction
# -------------------------------


if st.button("🔍 Predict Churn"):


    input_data = pd.DataFrame({

        "gender":[gender],

        "SeniorCitizen":[SeniorCitizen],

        "Partner":[Partner],

        "Dependents":[Dependents],

        "tenure":[tenure],

        "PhoneService":[PhoneService],

        "MultipleLines":[MultipleLines],

        "InternetService":[InternetService],

        "OnlineSecurity":[OnlineSecurity],

        "OnlineBackup":[OnlineBackup],

        "DeviceProtection":[DeviceProtection],

        "TechSupport":[TechSupport],

        "StreamingTV":[StreamingTV],

        "StreamingMovies":[StreamingMovies],

        "Contract":[Contract],

        "PaperlessBilling":[PaperlessBilling],

        "PaymentMethod":[PaymentMethod],

        "MonthlyCharges":[MonthlyCharges],

        "TotalCharges":[TotalCharges]

    })


    prediction = model.predict(input_data)


    probability = model.predict_proba(input_data)


    st.divider()


    if prediction[0] == 1:

        st.error("⚠️ Customer is likely to Churn")

        st.metric(
            "Churn Probability",
            f"{probability[0][1]*100:.2f}%"
        )


    else:

        st.success("✅ Customer will Stay")

        st.metric(
            "Stay Probability",
            f"{probability[0][0]*100:.2f}%"
        )


st.divider()

st.caption(
    "Developed using Machine Learning | Random Forest | Streamlit"
)
