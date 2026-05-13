import streamlit as st
import tensorflow as tf
import pickle
import base64
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Analisis Sentimen Timnas Indonesia",
    page_icon="⚽",
    layout="centered"
)

# =====================================================
# FUNCTION BASE64 IMAGE
# =====================================================
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# =====================================================
# LOAD BACKGROUND IMAGE
# =====================================================
bg_image = get_base64("timnas.png")

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown(
    f"""
    <style>

    .stApp {{
        background-image:
            linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
            url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .main-container {{
        background-color: rgba(0,0,0,0.70);
        padding: 35px;
        border-radius: 20px;
        margin-top: 30px;
    }}

    .title {{
        text-align: center;
        color: white;
        font-size: 38px;
        font-weight: bold;
        margin-top: 10px;
    }}

    .subtitle {{
        text-align: center;
        color: #dddddd;
        font-size: 18px;
        margin-bottom: 25px;
    }}

    .logo-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }}

    textarea {{
        background-color: rgba(255,255,255,0.95) !important;
        color: black !important;
        border-radius: 12px !important;
    }}

    div[data-testid="stTextArea"] label {{
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }}

    .result-positive {{
        background-color: #00c853;
        padding: 18px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        margin-top: 20px;
    }}

    .result-negative {{
        background-color: #d50000;
        padding: 18px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        margin-top: 20px;
    }}

    .prediction-score {{
        text-align: center;
        color: white;
        font-size: 20px;
        margin-top: 15px;
        font-weight: bold;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD LOGO
# =====================================================
logo = get_base64("logo.png")

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_my_model():
    return load_model("sentiment_analysis_model.h5", compile=False)

model = load_my_model()

# =====================================================
# LOAD TOKENIZER
# =====================================================
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

MAXLEN = 100

# =====================================================
# MAIN CONTAINER
# =====================================================
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# =====================================================
# LOGO
# =====================================================
st.markdown(
    f"""
    <div class="logo-container">
        <img src="data:image/png;base64,{logo}" width="120">
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TITLE
# =====================================================
st.markdown(
    """
    <div class='title'>
        ANALISIS SENTIMEN TIMNAS INDONESIA
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# SUBTITLE
# =====================================================
st.markdown(
    """
    <div class='subtitle'>
        Model BiLSTM untuk mendeteksi sentimen positif dan negatif
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# INPUT TEXT
# =====================================================
text = st.text_area(
    "Masukkan Kalimat:",
    height=180,
    placeholder="Contoh: Timnas Indonesia bermain sangat bagus malam ini"
)

# =====================================================
# BUTTON
# =====================================================
if st.button("🚀 Prediksi Sentimen"):

    if text.strip() == "":
        st.warning("Masukkan teks terlebih dahulu!")
    else:

        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAXLEN)

        prediction = model.predict(padded)[0][0]

        prob_pos = float(prediction)
        prob_neg = float(1 - prediction)

        # =================================================
        # POSITIVE
        # =================================================
        if prediction >= 0.5:

            st.markdown(
                """
                <div class='result-positive'>
                    POSITIF
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class='prediction-score'>
                    Nilai Prediksi Positif: {prob_pos:.4f}
                </div>
                """,
                unsafe_allow_html=True
            )

        # =================================================
        # NEGATIVE
        # =================================================
        else:

            st.markdown(
                """
                <div class='result-negative'>
                    NEGATIF
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class='prediction-score'>
                    Nilai Prediksi Negatif: {prob_neg:.4f}
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("</div>", unsafe_allow_html=True)
