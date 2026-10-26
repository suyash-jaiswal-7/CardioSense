import streamlit as st
import pandas as pd
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="CardioSense | Suyash Jaiswal",
    page_icon="❤️",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    /* Premium charcoal background */
    .stApp {
        background: #101714;
        color: #E8F0EA;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #151F1A;
        border-right: 1px solid #293B31;
    }

    [data-testid="stSidebar"] * {
        color: #DDE9DF;
    }

    /* Hero banner */
    .hero {
        padding: 34px 36px;
        border-radius: 22px;
        background: linear-gradient(120deg, #173C2D, #205843 65%, #26705A);
        border: 1px solid #356A50;
        margin-bottom: 26px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.20);
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 2.65rem;
        font-weight: 750;
        letter-spacing: -1px;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #D5E9DB !important;
        font-size: 1.02rem;
        line-height: 1.7;
        margin: 0;
    }

    /* Headings */
    h1, h2, h3 {
        color: #EDF5EF;
        letter-spacing: -0.4px;
    }

    .section-heading {
        font-size: 1.25rem;
        font-weight: 700;
        color: #9BD8B0;
        margin: 24px 0 14px 0;
    }

    /* Labels */
    label, [data-testid="stWidgetLabel"] p {
        color: #D9E6DC !important;
        font-weight: 600 !important;
    }

    /* Input fields */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #1C2821;
        border: 1px solid #35483B;
        border-radius: 10px;
        color: #F0F5F1;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: #64B985;
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="input"] input {
        color: #F0F5F1;
    }

    /* Dropdown menu */
    div[data-baseweb="popover"] {
        background: #1C2821;
    }

    /* Slider accent */
    div[data-testid="stSlider"] {
        color: #75C994;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        min-height: 3.2rem;
        background: linear-gradient(100deg, #26734D, #31966A);
        color: #FFFFFF;
        border: 1px solid #3B9C70;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(100deg, #31966A, #42AA7B);
        border-color: #67C793;
        color: #FFFFFF;
        box-shadow: 0 5px 18px rgba(49, 150, 106, 0.24);
        transform: translateY(-1px);
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 13px;
    }

    /* Expandable details */
    [data-testid="stExpander"] {
        background: #18231D;
        border: 1px solid #304438;
        border-radius: 12px;
    }

    /* Data table */
    [data-testid="stDataFrame"] {
        border: 1px solid #304438;
        border-radius: 12px;
    }

    /* Dividers */
    hr {
        border-color: #2B3D31;
    }

    /* Captions */
    [data-testid="stCaptionContainer"] {
        color: #A4B8A9;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #91A89A;
        font-size: 0.85rem;
        line-height: 1.8;
        padding: 22px 0 8px 0;
    }

    /* Responsive layout */
    @media (max-width: 768px) {
        .hero {
            padding: 24px 20px;
        }

        .hero h1 {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD SAVED MODEL --------------------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
    <h1>♥ CardioSense</h1>
    <p>
        A machine-learning approach to understanding heart health.<br>
        Enter your health measurements to explore a model-generated prediction.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## ♥ CardioSense")
    st.caption("HEART HEALTH · MACHINE LEARNING")

    st.divider()

    st.markdown("### About the Project")
    st.write(
        "CardioSense uses a trained K-Nearest Neighbors (KNN) model "
        "to predict a heart disease class from selected health measurements."
    )

    st.markdown("### How It Works")
    st.write(
        "1. Enter the requested health details.\n"
        "2. Click Predict.\n"
        "3. Review the model output."
    )

    st.divider()

    st.markdown("### Project Author")
    st.write("**Suyash Jaiswal**")
    st.caption("B.Tech CSE · Data Science")

# -------------------- DISCLAIMER --------------------
st.info(
    "Educational project only. This model output is not a medical diagnosis "
    "and should not replace advice from a qualified healthcare professional."
)

# -------------------- PERSONAL INFORMATION --------------------
st.markdown(
    '<div class="section-heading">👤 Personal Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 18, 100, 40)

with col2:
    sex = st.selectbox("Sex", ["M", "F"])

# -------------------- HEALTH MEASUREMENTS --------------------
st.markdown(
    '<div class="section-heading">🩺 Health Measurements</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    resting_bp = st.number_input(
        "Blood Pressure (mm Hg)", 80, 200, 120
    )

    cholesterol = st.number_input(
        "Blood Cholesterol (mg/dL)", 100, 600, 200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar Above 120 mg/dL?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:
    resting_ecg = st.selectbox(
        "Heart's Electrical Activity (ECG)",
        ["Normal", "ST", "LVH"],
        format_func=lambda x: {
            "Normal": "Normal",
            "ST": "ST — ST-T wave abnormality",
            "LVH": "LVH — Left ventricular hypertrophy"
        }[x]
    )

    max_hr = st.slider(
        "Maximum Heart Rate (bpm)", 60, 220, 150
    )

    exercise_angina = st.selectbox(
        "Chest Discomfort During Exercise",
        ["Y", "N"],
        format_func=lambda x: "Yes" if x == "Y" else "No"
    )

# -------------------- HEART-RELATED INDICATORS --------------------
st.markdown(
    '<div class="section-heading">🫀 Heart-Related Indicators</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    chest_pain = st.selectbox(
        "Type of Chest Pain",
        ["ATA", "NAP", "TA", "ASY"],
        format_func=lambda x: {
            "ATA": "Atypical Angina",
            "NAP": "Non-Anginal Pain",
            "TA": "Typical Angina",
            "ASY": "Asymptomatic"
        }[x]
    )

with col2:
    st_slope = st.selectbox(
        "ECG Pattern",
        ["Up", "Flat", "Down"],
        format_func=lambda x: {
            "Up": "Upward",
            "Flat": "Flat",
            "Down": "Downward"
        }[x]
    )

oldpeak = st.slider(
    "ECG Change Measurement",
    0.0, 6.0, 1.0
)

st.caption(
    "This measurement is usually recorded during an ECG test. "
    "Enter a value provided by a healthcare professional if available."
)

# -------------------- PREDICTION --------------------
st.markdown("---")

if st.button("♥  Predict Heart Disease"):

    # Original input dictionary — unchanged feature names and values
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Match expected feature order
    input_df = input_df[expected_columns]

    # Scale input using saved scaler
    scaled_input = scaler.transform(input_df)

    # Predict using saved model
    prediction = model.predict(scaled_input)[0]

    # -------------------- DISPLAY RESULT --------------------
    st.markdown(
        '<div class="section-heading">Prediction Result</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:
        st.error(
            "**⚠️ Heart Disease Indicated**\n\n"
            "Based on the information entered, the model identified a "
            "pattern associated with heart disease. This is not a medical "
            "diagnosis. Please consult a qualified healthcare professional "
            "for an appropriate assessment."
        )
    else:
        st.success(
            "**♥ Heart Disease Not Indicated**\n\n"
            "Based on the information entered, the model did not identify "
            "a pattern associated with heart disease. This does not rule "
            "out heart disease or guarantee that your heart is healthy."
        )

    with st.expander("View Submitted Information"):
        summary = {
            "Age": age,
            "Sex": "Male" if sex == "M" else "Female",
            "Type of Chest Pain": {
                "ATA": "Atypical angina",
                "NAP": "Non-anginal pain",
                "TA": "Typical angina",
                "ASY": "Asymptomatic"
            }[chest_pain],
            "Blood Pressure (mm Hg)": resting_bp,
            "Blood Cholesterol (mg/dL)": cholesterol,
            "Fasting Blood Sugar Above 120 mg/dL": "Yes" if fasting_bs == 1 else "No",
            "Heart's Electrical Activity": {
                "Normal": "Normal",
                "ST": "ST-T wave abnormality",
                "LVH": "Left ventricular hypertrophy"
            }[resting_ecg],
            "Maximum Heart Rate (bpm)": max_hr,
            "Chest Discomfort During Exercise": "Yes" if exercise_angina == "Y" else "No",
            "ECG Change Measurement": oldpeak,
            "ECG Pattern": {
                "Up": "Upward",
                "Flat": "Flat",
                "Down": "Downward"
            }[st_slope]
        }

        st.dataframe(
            pd.DataFrame(
                summary.items(),
                columns=["Measurement", "Entered Value"]
            ),
            use_container_width=True,
            hide_index=True
        )

# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer">
    <strong>CardioSense</strong> · Developed by Suyash Jaiswal<br>
    Machine Learning Project · For Educational Purposes Only
</div>
""", unsafe_allow_html=True)
