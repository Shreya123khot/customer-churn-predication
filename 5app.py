import streamlit as st
import pandas as pd
import pickle



# Page

st.set_page_config(

page_title="Customer Churn Prediction",
page_icon="📊",
layout="wide"

)



# Load Model

model = pickle.load(
    open("model.pkl","rb")
)



st.title(
"📊 Customer Churn Prediction System"
)


st.write(
"""
AI Based Customer Retention System

✔ Machine Learning  
✔ Random Forest  
✔ Streamlit
"""
)



st.divider()



# Inputs


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
"Tenure Months",
0,
100,
12
)


PhoneService = st.selectbox(
"Phone Service",
["Yes","No"]
)


InternetService = st.selectbox(
"Internet Service",
[
"DSL",
"Fiber optic",
"No"
]
)


Contract = st.selectbox(
"Contract",
[
"Month-to-month",
"One year",
"Two year"
]
)


MonthlyCharges = st.number_input(
"Monthly Charges",
0.0,
500.0,
50.0
)


TotalCharges = st.number_input(
"Total Charges",
0.0,
10000.0,
500.0
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



if st.button(
"🔍 Predict Churn"
):


    data = {

    "gender":gender,
    "SeniorCitizen":SeniorCitizen,
    "Partner":Partner,
    "Dependents":Dependents,
    "tenure":tenure,
    "PhoneService":PhoneService,
    "InternetService":InternetService,
    "Contract":Contract,
    "MonthlyCharges":MonthlyCharges,
    "TotalCharges":TotalCharges,
    "PaymentMethod":PaymentMethod

    }



    input_df = pd.DataFrame(
        [data]
    )



    result = model.predict(
        input_df
    )


    probability = model.predict_proba(
        input_df
    )[0][1]



    if result[0]=="Yes" or result[0]==1:


        st.error(
        "⚠ Customer Will Churn"
        )


        st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
        )


    else:


        st.success(
        "✅ Customer Will Stay"
        )


        st.metric(
        "Retention Probability",
        f"{(1-probability)*100:.2f}%"
        )
