import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background:#f4f8fb;
}

.title{
    font-size:40px;
    font-weight:bold;
    color:#0066cc;
}

.sub{
    font-size:18px;
    color:gray;
}

.stButton>button{
    width:100%;
    background:#0066cc;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
}

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 0px 10px rgba(0,0,0,0.2);
}

</style>
""",unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model=pickle.load(open("customer_churn_model.pkl","rb"))

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📊 Customer Churn")

st.sidebar.info("""
Machine Learning Project



st.markdown("<p class='title'>Customer Churn Prediction System</p>",unsafe_allow_html=True)

st.markdown("<p class='sub'>Predict whether customer will leave the company or not.</p>",unsafe_allow_html=True)

st.write("---")

col1,col2=st.columns(2)

with col1:

    SeniorCitizen=st.selectbox(
        "Senior Citizen",
        [0,1]
    )

    tenure=st.number_input(
        "Tenure",
        0,
        100,
        10
    )

    MonthlyCharges=st.number_input(
        "Monthly Charges",
        0.0,
        500.0,
        70.0
    )

    TotalCharges=st.number_input(
        "Total Charges",
        0.0,
        10000.0,
        1000.0
    )

    gender_Male=st.selectbox(
        "Gender",
        [0,1],
        format_func=lambda x:"Female" if x==0 else "Male"
    )

    Partner_Yes=st.selectbox(
        "Partner",
        [0,1]
    )

    Dependents_Yes=st.selectbox(
        "Dependents",
        [0,1]
    )

    PhoneService_Yes=st.selectbox(
        "Phone Service",
        [0,1]
    )

    MultipleLines_No_phone_service=st.selectbox(
        "Multiple Lines (No Phone Service)",
        [0,1]
    )

    MultipleLines_Yes=st.selectbox(
        "Multiple Lines",
        [0,1]
    )

    InternetService_Fiber_optic=st.selectbox(
        "Fiber Optic",
        [0,1]
    )

    InternetService_No=st.selectbox(
        "No Internet",
        [0,1]
    )

    OnlineSecurity_No_internet_service=st.selectbox(
        "Online Security (No Internet)",
        [0,1]
    )

    OnlineSecurity_Yes=st.selectbox(
        "Online Security",
        [0,1]
    )

    OnlineBackup_No_internet_service=st.selectbox(
        "Online Backup (No Internet)",
        [0,1]
    )
    with col2:

    OnlineBackup_Yes=st.selectbox(
        "Online Backup",
        [0,1]
    )

    DeviceProtection_No_internet_service=st.selectbox(
        "Device Protection (No Internet)",
        [0,1]
    )

    DeviceProtection_Yes=st.selectbox(
        "Device Protection",
        [0,1]
    )

    TechSupport_No_internet_service=st.selectbox(
        "Tech Support (No Internet)",
        [0,1]
    )

    TechSupport_Yes=st.selectbox(
        "Tech Support",
        [0,1]
    )

    StreamingTV_No_internet_service=st.selectbox(
        "Streaming TV (No Internet)",
        [0,1]
    )

    StreamingTV_Yes=st.selectbox(
        "Streaming TV",
        [0,1]
    )

    StreamingMovies_No_internet_service=st.selectbox(
        "Streaming Movies (No Internet)",
        [0,1]
    )

    StreamingMovies_Yes=st.selectbox(
        "Streaming Movies",
        [0,1]
    )

    Contract_One_year=st.selectbox(
        "One Year Contract",
        [0,1]
    )

    Contract_Two_year=st.selectbox(
        "Two Year Contract",
        [0,1]
    )

    PaperlessBilling_Yes=st.selectbox(
        "Paperless Billing",
        [0,1]
    )

    PaymentMethod_Credit_card_automatic=st.selectbox(
        "Credit Card (Automatic)",
        [0,1]
    )

    PaymentMethod_Electronic_check=st.selectbox(
        "Electronic Check",
        [0,1]
    )

    PaymentMethod_Mailed_check=st.selectbox(
        "Mailed Check",
        [0,1]
    )

st.write("")

if st.button("🔍 Predict Customer Churn"):

    data=[[

        SeniorCitizen,
        tenure,
        MonthlyCharges,
        TotalCharges,
        gender_Male,
        Partner_Yes,
        Dependents_Yes,
        PhoneService_Yes,
        MultipleLines_No_phone_service,
        MultipleLines_Yes,
        InternetService_Fiber_optic,
        InternetService_No,
        OnlineSecurity_No_internet_service,
        OnlineSecurity_Yes,
        OnlineBackup_No_internet_service,
        OnlineBackup_Yes,
        DeviceProtection_No_internet_service,
        DeviceProtection_Yes,
        TechSupport_No_internet_service,
        TechSupport_Yes,
        StreamingTV_No_internet_service,
        StreamingTV_Yes,
        StreamingMovies_No_internet_service,
        StreamingMovies_Yes,
        Contract_One_year,
        Contract_Two_year,
        PaperlessBilling_Yes,
        PaymentMethod_Credit_card_automatic,
        PaymentMethod_Electronic_check,
        PaymentMethod_Mailed_check

    ]]

    prediction=model.predict(data)[0]

    probability=model.predict_proba(data)[0]

        st.write("---")

    if prediction == 1:

        st.error("❌ Customer is likely to Churn")

    else:

        st.success("✅ Customer is likely to Stay")

    st.subheader("Prediction Probability")

    churn_probability = probability[1] * 100
    stay_probability = probability[0] * 100

    st.write("### Customer Stay Probability")
    st.progress(int(stay_probability))
    st.write(f"{stay_probability:.2f}%")

    st.write("### Customer Churn Probability")
    st.progress(int(churn_probability))
    st.write(f"{churn_probability:.2f}%")

    st.write("---")

    st.subheader("📋 Customer Summary")

    summary = pd.DataFrame({

        "Feature":[
            "Senior Citizen",
            "Tenure",
            "Monthly Charges",
            "Total Charges",
            "Gender (Male)",
            "Partner",
            "Dependents",
            "Phone Service"
        ],

        "Value":[
            SeniorCitizen,
            tenure,
            MonthlyCharges,
            TotalCharges,
            gender_Male,
            Partner_Yes,
            Dependents_Yes,
            PhoneService_Yes
        ]

    })

    st.dataframe(summary, use_container_width=True)

    st.write("---")

    st.metric(
        label="Churn Probability",
        value=f"{churn_probability:.2f}%"
    )

    st.metric(
        label="Stay Probability",
        value=f"{stay_probability:.2f}%"
    )

    st.balloons()

