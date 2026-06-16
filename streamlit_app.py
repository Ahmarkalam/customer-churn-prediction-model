import streamlit as st
import pickle
import numpy as np

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD MODEL + SCALER
# =========================

model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
    <style>

    .main {
        background-color: #0E1117;
        color: white;
    }

    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        border: none;
    }

    .stButton>button:hover {
        background-color: #ff1e1e;
        color: white;
    }

    .css-1d391kg {
        background-color: #161A23;
    }

    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("""
Predict whether a customer is likely to leave the subscription service using Machine Learning.
""")

# =========================
# SIDEBAR
# =========================

st.sidebar.header("📝 Customer Information")

# =========================
# INPUTS
# =========================

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ]
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges",
    0.0,
    200.0,
    50.0
)

total_charges = st.sidebar.slider(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)

# =========================
# ENCODING
# =========================

gender = 1 if gender == "Male" else 0

partner = 1 if partner == "Yes" else 0

dependents = 1 if dependents == "Yes" else 0

phone_service = 1 if phone_service == "Yes" else 0

multiple_lines = 1 if multiple_lines == "Yes" else 0

online_security = 1 if online_security == "Yes" else 0

online_backup = 1 if online_backup == "Yes" else 0

device_protection = 1 if device_protection == "Yes" else 0

tech_support = 1 if tech_support == "Yes" else 0

streaming_tv = 1 if streaming_tv == "Yes" else 0

streaming_movies = 1 if streaming_movies == "Yes" else 0

paperless_billing = 1 if paperless_billing == "Yes" else 0

internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

internet_service = internet_map[internet_service]

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

contract = contract_map[contract]

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}

payment_method = payment_map[payment_method]

# Extra dummy feature
tech_dummy = 0

# =========================
# FINAL INPUT
# =========================

input_data = np.array([[
    gender,
    senior,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
    tech_dummy
]])

# =========================
# PREDICTION BUTTON
# =========================

if st.button("🚀 Predict Churn"):

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)[0][1]

    # =========================
    # RESULT SECTION
    # =========================

    st.subheader("📈 Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
    )

    st.progress(int(probability * 100))

    # Risk category
    if probability > 0.7:

        st.error("⚠️ HIGH RISK CUSTOMER")

        st.markdown("""
        ### Recommended Actions
        - Offer discounts
        - Improve customer support
        - Provide loyalty rewards
        """)

    elif probability > 0.4:

        st.warning("🟡 MEDIUM RISK CUSTOMER")

        st.markdown("""
        ### Recommended Actions
        - Send engagement emails
        - Offer better plans
        """)

    else:

        st.success("✅ LOW RISK CUSTOMER")

        st.markdown("""
        ### Recommended Actions
        - Maintain service quality
        - Upsell premium plans
        """)

# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown("""
Made with ❤️ using Streamlit + Machine Learning
""")