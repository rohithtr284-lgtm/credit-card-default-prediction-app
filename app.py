import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# =========================
# Load Model and Scaler
# =========================
@st.cache_resource
def load_files():
    model = pickle.load(open("model (6).sav", "rb"))
    scaler = pickle.load(open("scaler (6).sav", "rb"))
    return model, scaler

model, scaler = load_files()

# =========================
# Header
# =========================
st.title("💳 Credit Card Default Risk Prediction")

st.markdown("""
This Machine Learning model predicts whether a customer is likely to  
**default on their credit card payment**.
""")

# Optional Image
try:
    img = Image.open("uci_card.jpg.png")
    st.image(img, use_container_width=True)
except:
    pass

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("Enter Customer Details")

# Dummy ID required because scaler was trained with ID
ID = 1

LIMIT_BAL = st.sidebar.number_input(
    "Credit Limit",
    min_value=0,
    max_value=1000000,
    value=20000
)

gender_map = {"Male": 1, "Female": 2, "Other": 3}
SEX = st.sidebar.selectbox("Gender", list(gender_map.keys()))
SEX = gender_map[SEX]

education_map = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}
EDUCATION = st.sidebar.selectbox("Education", list(education_map.keys()))
EDUCATION = education_map[EDUCATION]

marriage_map = {"Married": 1, "Single": 2, "Others": 3}
MARRIAGE = st.sidebar.selectbox("Marital Status", list(marriage_map.keys()))
MARRIAGE = marriage_map[MARRIAGE]

AGE = st.sidebar.number_input("Age", 18, 100, 30)

# =========================
# Repayment Status
# =========================
repay_labels = {
    -2: "No consumption",
    -1: "Paid duly",
     0: "Paid on time",
     1: "1 month delay",
     2: "2 months delay",
     3: "3 months delay",
     4: "4 months delay",
     5: "5 months delay",
     6: "6 months delay",
     7: "7 months delay",
     8: "8 months delay",
     9: "9 months delay",
}

st.sidebar.subheader("Repayment History")

PAY_0 = st.sidebar.selectbox("Last Month", repay_labels.keys(), format_func=lambda x: repay_labels[x])
PAY_2 = st.sidebar.selectbox("2 Months Ago", repay_labels.keys(), format_func=lambda x: repay_labels[x])
PAY_3 = st.sidebar.selectbox("3 Months Ago", repay_labels.keys(), format_func=lambda x: repay_labels[x])
PAY_4 = st.sidebar.selectbox("4 Months Ago", repay_labels.keys(), format_func=lambda x: repay_labels[x])
PAY_5 = st.sidebar.selectbox("5 Months Ago", repay_labels.keys(), format_func=lambda x: repay_labels[x])
PAY_6 = st.sidebar.selectbox("6 Months Ago", repay_labels.keys(), format_func=lambda x: repay_labels[x])

# =========================
# Billing Information
# =========================
st.subheader("Billing Information")

col1, col2, col3 = st.columns(3)

with col1:
    BILL_AMT1 = st.number_input("Bill Amount Sep", 0, 1000000, 0)
    BILL_AMT4 = st.number_input("Bill Amount Jun", 0, 1000000, 0)

with col2:
    BILL_AMT2 = st.number_input("Bill Amount Aug", 0, 1000000, 0)
    BILL_AMT5 = st.number_input("Bill Amount May", 0, 1000000, 0)

with col3:
    BILL_AMT3 = st.number_input("Bill Amount Jul", 0, 1000000, 0)
    BILL_AMT6 = st.number_input("Bill Amount Apr", 0, 1000000, 0)

# =========================
# Payment Information
# =========================
st.subheader("Payment Information")

col1, col2, col3 = st.columns(3)

with col1:
    PAY_AMT1 = st.number_input("Payment Sep", 0, 1000000, 0)
    PAY_AMT4 = st.number_input("Payment Jun", 0, 1000000, 0)

with col2:
    PAY_AMT2 = st.number_input("Payment Aug", 0, 1000000, 0)
    PAY_AMT5 = st.number_input("Payment May", 0, 1000000, 0)

with col3:
    PAY_AMT3 = st.number_input("Payment Jul", 0, 1000000, 0)
    PAY_AMT6 = st.number_input("Payment Apr", 0, 1000000, 0)

# =========================
# Create DataFrame (INCLUDING ID)
# =========================
input_df = pd.DataFrame([[

    ID,
    LIMIT_BAL,
    SEX,
    EDUCATION,
    MARRIAGE,
    AGE,
    PAY_0,
    PAY_2,
    PAY_3,
    PAY_4,
    PAY_5,
    PAY_6,
    BILL_AMT1,
    BILL_AMT2,
    BILL_AMT3,
    BILL_AMT4,
    BILL_AMT5,
    BILL_AMT6,
    PAY_AMT1,
    PAY_AMT2,
    PAY_AMT3,
    PAY_AMT4,
    PAY_AMT5,
    PAY_AMT6

]], columns=[

    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"

])

# =========================
# Prediction
# =========================
if st.button("Predict Default Risk"):

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Default")
    else:
        st.success("✅ Low Risk of Default")

    st.metric("Default Probability", f"{probability:.2%}")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("Developed by Rohith TR | Machine Learning Project")
