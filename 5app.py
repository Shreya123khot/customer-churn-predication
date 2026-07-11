import streamlit as st
import pandas as pd
import pickle


# Load Model

model = pickle.load(open("model.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))


# Page Config

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# CSS

st.markdown("""
<style>

body{
background:#0f172a;
}

h1{
color:#38bdf8;
text-align:center;
}

.stButton>button{
width:100%;
height:50px;
font-size:20px;
border-radius:10px;
}

</style>
""",unsafe_allow_html=True)



# Header

st.title("📊 Customer Churn Prediction System")

st.write(
"""
### AI Based Customer Retention System

✅ Machine Learning  
✅ Random Forest Algorithm  
✅ Streamlit Application
"""
)


st.divider()



# Input

st.subheader("Enter Customer Details")


user_input={}


col1,col2=st.columns(2)


for i,feature in enumerate(columns):

    if i%2==0:

        with col1:

            user_input[feature]=st.number_input(
                feature,
                value=0.0
            )

    else:

        with col2:

            user_input[feature]=st.number_input(
                feature,
                value=0.0
            )



# Convert Input

input_df=pd.DataFrame(
    [user_input]
)



st.subheader("Customer Information")

st.dataframe(
    input_df,
    use_container_width=True
)



# Prediction

if st.button("🔮 Predict Customer Churn"):


    result=model.predict(input_df)

    probability=model.predict_proba(input_df)



    churn_prob=probability[0][1]*100
    stay_prob=probability[0][0]*100



    if result[0]==1:

        st.error(
            "⚠ Customer May Leave (Churn)"
        )

        st.metric(
            "Churn Probability",
            f"{churn_prob:.2f}%"
        )

        st.progress(
            int(churn_prob)
        )


    else:

        st.success(
            "✅ Customer Will Stay"
        )

        st.metric(
            "Retention Probability",
            f"{stay_prob:.2f}%"
        )

        st.progress(
            int(stay_prob)
        )



st.divider()

st.caption(
"Developed using Python | Machine Learning | Streamlit"
)
