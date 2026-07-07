import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Income Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================
# Custom CSS - Clean & Professional
# ==========================
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean light background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Clean card design */
    .card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        border: 1px solid #e8ecef;
    }
    
    /* Professional button */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white !important;
        border: none;
        padding: 14px 32px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
        transform: translateY(-2px);
    }
    
    /* Section titles */
    .section-title {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #2563eb;
    }
    
    /* Main title */
    .main-title {
        color: #1e293b;
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    .sub-title {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
    }
    
    /* Input labels */
    label {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Input fields */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #f8fafc !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }
    
    .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
        border-color: #2563eb !important;
    }
    
    /* Slider */
    .stSlider > div > div {
        background: #2563eb !important;
    }
    
    /* Info box */
    .stAlert {
        background: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 10px !important;
        color: #1e40af !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #2563eb, transparent);
        margin: 2rem 0;
    }
    
    /* Result cards */
    .result-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
    }
    
    .result-fail {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# Load Files
# ==========================
@st.cache_resource
def load_assets():
    model = load_model("income_model.keras")
    scaler = joblib.load("scaler.pkl")
    training_columns = joblib.load("training_columns.pkl")
    return model, scaler, training_columns

model, scaler, training_columns = load_assets()

# ==========================
# Header Section
# ==========================
st.markdown("<h1 class='main-title'>💼 Adult Income Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict whether an individual's income exceeds $50K/year</p>", unsafe_allow_html=True)
st.divider()

# ==========================
# Inputs Section
# ==========================

# Row 1
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>👤 Personal Details</h3>", unsafe_allow_html=True)
    age = st.number_input("Age", 18, 100, 30, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    education_num = st.slider("Education Number (Years)", 1, 16, 10)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>💼 Employment Details</h3>", unsafe_allow_html=True)
    workclass = st.selectbox(
        "Workclass",
        ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    )
    hours = st.slider("Hours per Week", 1, 100, 40)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 0, 1000000, 100000, step=1000)
    st.markdown("</div>", unsafe_allow_html=True)

# Row 2
col3, col4 = st.columns(2)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>💰 Financial Indicators</h3>", unsafe_allow_html=True)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0, step=100)
    capital_loss = st.number_input("Capital Loss", 0, 5000, 0, step=100)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>ℹ️ Quick Guide</h3>", unsafe_allow_html=True)
    st.info("""
    **Age:** Person's age (18-100)  
    **Education Num:** Years of education (1-16)  
    **Fnlwgt:** Census sampling weight  
    **Capital:** Asset gains and losses  
    **Hours:** Weekly working hours
    """)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================
# Prediction Button
# ==========================
if st.button("🚀 Predict Income"):
    
    # Prepare Data
    data = pd.DataFrame(
        np.zeros((1, len(training_columns))),
        columns=training_columns
    )

    data["age"] = age
    data["fnlwgt"] = fnlwgt
    data["education.num"] = education_num
    data["hours.per.week"] = hours
    data["capital.gain"] = capital_gain
    data["capital.loss"] = capital_loss

    if f"workclass_{workclass}" in data.columns:
        data[f"workclass_{workclass}"] = 1

    if f"sex_{sex}" in data.columns:
        data[f"sex_{sex}"] = 1

    # Scale and Predict
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)
    probability = pred[0][0]

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display Results
    if probability > 0.5:
        st.markdown(f"""
        <div class='result-success'>
            <h1 style='margin:0; font-size: 2.5rem;'>✅ Income > 50K</h1>
            <p style='font-size: 20px; margin-top: 15px;'>
                Confidence Score: <b style='font-size: 28px;'>{probability*100:.2f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='result-fail'>
            <h1 style='margin:0; font-size: 2.5rem;'>❌ Income ≤ 50K</h1>
            <p style='font-size: 20px; margin-top: 15px;'>
                Confidence Score: <b style='font-size: 28px;'>{(1-probability)*100:.2f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)