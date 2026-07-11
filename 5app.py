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

try:
    model = pickle.load(open("model.pkl", "rb"))
    

except Exception as e:
    st.error("Model file load error")
    st.write(e)
    st.stop()



# -------------------------------
# Custom CSS
# -------------------------------

st.markdown("""
<style>

body{
background-color:#0f172a;
}

.main{
background-color:#0f172a;
}

.card{
background:rgba(255,255,255,0.1);
padding:25px;
border-radius:15px;
box-shadow:0px 0px 15px #000;
}

h1{
color:#38bdf8;
text-align:center;
}

</style>

""",unsafe_allow_html=True)



# -------------------------------
# Title
# -------------------------------

st.title("📊 Customer Churn Prediction System")

st.write(
"""
### AI Based Customer Retention Analysis
✔ Machine Learning  
✔ Random Forest Algorithm  
✔ Streamlit Application
"""
)


st.divider()



# -------------------------------
# User Input
# -------------------------------


st.subheader("Enter Customer Details")


col1,col2,col3 = st.columns(3)



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



with col2:

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )


    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes","No"]
    )


    InternetService = st.selectbox(
        "Internet Service",
        ["DSL","Fiber optic","No"]
    )


    Contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )



with col3:

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        value=50.0
    )


    TotalCharges = st.number_input(
        "Total Charges",
        value=500.0
    )


    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card"
        ]
    )



st.divider()



# -------------------------------
# Prediction Button
# -------------------------------


if st.button("🔍 Predict Churn",use_container_width=True):


    input_data = {

        "SeniorCitizen":SeniorCitizen,
        "tenure":tenure,
        "MonthlyCharges":MonthlyCharges,
        "TotalCharges":TotalCharges,

        "gender_Male":1 if gender=="Male" else 0,

        "Partner_Yes":1 if Partner=="Yes" else 0,

        "Dependents_Yes":1 if Dependents=="Yes" else 0,

        "PhoneService_Yes":1 if PhoneService=="Yes" else 0,

        "InternetService_Fiber optic":
        1 if InternetService=="Fiber optic" else 0,

        "Contract_One year":
        1 if Contract=="One year" else 0,

        "Contract_Two year":
        1 if Contract=="Two year" else 0

    }


    df=pd.DataFrame([input_data])


    # missing columns add

    for col in columns:

        if col not in df.columns:
            df[col]=0


    df=df[columns]



    prediction=model.predict(df)



    probability=model.predict_proba(df)[0][1]



    st.divider()



    if prediction[0]==1:

        st.error(
        "⚠ Customer is likely to Churn"
        )


        st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
        )


    else:

        st.success(
        "✅ Customer is Not Likely to Churn"
        )


        st.metric(
        "Retention Probability",
        f"{(1-probability)*100:.2f}%"
        )



st.sidebar.title("About Project")

st.sidebar.info(
"""
Customer Churn Prediction

Algorithm:
Random Forest Classifier

Technology:
Python
Machine Learning
Streamlit
"""
)







