import os

import requests
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chest X-Ray Pneumonia Classifier",
    page_icon="🩺",
    layout="wide",
)


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


# ------------------------------------------------------------
# Custom CSS for modern portfolio style
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="stApp"] {
        font-family: 'Inter', sans-serif;
        background: #06131f;
        color: #eaf2ff;
    }

    .stApp {
        background: radial-gradient(circle at top left, rgba(76, 141, 255, 0.18), transparent 30%),
                    radial-gradient(circle at bottom right, rgba(0, 204, 182, 0.12), transparent 25%),
                    #06131f;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .portfolio-shell {
        background: rgba(9, 20, 33, 0.7);
        border: 1px solid rgba(140, 170, 220, 0.18);
        border-radius: 22px;
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.35);
        padding: 1.25rem 1.25rem 1.5rem;
        backdrop-filter: blur(6px);
    }

    .hero-badge {
        display: inline-block;
        margin-bottom: 0.8rem;
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.14);
        border: 1px solid rgba(129, 140, 248, 0.35);
        color: #a5b4fc;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: clamp(2.4rem, 4vw, 4rem);
        font-weight: 900;
        line-height: 1.05;
        margin: 0;
        letter-spacing: -0.06em;
        color: #f4f8ff;
    }

    .hero-subtitle {
        margin-top: 0.8rem;
        font-size: 1.08rem;
        color: #b7c6da;
        max-width: 760px;
    }

    .section-card {
        background: rgba(12, 27, 42, 0.88);
        border: 1px solid rgba(168, 186, 214, 0.14);
        border-radius: 18px;
        padding: 1rem 1rem 1.1rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .sidebar-box {
        background: rgba(16, 27, 39, 0.8);
        border: 1px solid rgba(133, 156, 193, 0.18);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .sidebar-title {
        color: #edf5ff;
        font-weight: 800;
        font-size: 1.15rem;
        margin-bottom: 0.8rem;
    }

    .sidebar-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        margin: 0.45rem 0;
        color: #dfeaff;
        font-size: 0.94rem;
    }

    .sidebar-key {
        color: #98a9c3;
        font-weight: 600;
    }

    .upload-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 220px;
        border: 1.5px dashed rgba(136, 161, 205, 0.4);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(20,36,52,0.95), rgba(13,23,35,0.8));
    }

    .stFileUploader > div {
        background: rgba(18, 32, 46, 0.9);
        border: 1px solid rgba(150, 172, 210, 0.18);
        border-radius: 16px;
        padding: 0.5rem 0.5rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2d8cff, #11c4d9);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.3rem;
        font-weight: 700;
        font-size: 0.98rem;
        transition: 0.2s ease;
        box-shadow: 0 8px 18px rgba(17, 196, 217, 0.28);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(45, 140, 255, 0.32);
    }

    .result-card {
        background: linear-gradient(135deg, rgba(13, 41, 52, 0.9), rgba(11, 22, 35, 0.92));
        border: 1px solid rgba(99, 207, 182, 0.22);
        border-radius: 18px;
        padding: 1rem;
        margin-top: 0.8rem;
    }

    .result-badge {
        display: inline-flex;
        width: fit-content;
        margin-bottom: 0.85rem;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        background: rgba(10, 165, 121, 0.12);
        border: 1px solid rgba(68, 214, 159, 0.35);
        color: #7cecc0;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .result-title {
        color: #edf6ff;
        margin: 0;
        font-size: 1.3rem;
        font-weight: 800;
    }

    .confidence-card {
        background: rgba(16, 34, 48, 0.9);
        border: 1px solid rgba(157, 178, 220, 0.2);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-top: 1rem;
    }

    .confidence-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.2rem 0 0;
        color: #7ad6ff;
    }

    .probability-row {
        margin-top: 0.9rem;
    }

    .probability-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0.25rem;
        border-bottom: 1px solid rgba(156, 176, 205, 0.12);
        color: #eaf3ff;
        font-size: 0.95rem;
    }

    .disclaimer {
        color: #b7c6da;
        font-size: 0.92rem;
        text-align: center;
        margin-top: 1rem;
    }

    @media (max-width: 768px) {
        .main .block-container { padding-top: 1rem; }
        .hero-title { font-size: 2.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GLOBAL LAYOUT
# ============================================================

st.markdown('<div class="portfolio-shell">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-badge">Portfolio ML Project</div>
    <h1 class="hero-title">Chest X-Ray Pneumonia<br>Classifier</h1>
    <p class="hero-subtitle">
        A deep learning application powered by EfficientNet-B0 for chest X-ray analysis,
        designed to classify pneumonia patterns from medical images.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-title'>Model Overview</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='sidebar-item'><span class='sidebar-key'>Model</span><span>EfficientNet-B0</span></div>
        <div class='sidebar-item'><span class='sidebar-key'>Framework</span><span>FastAPI + PyTorch</span></div>
        <div class='sidebar-item'><span class='sidebar-key'>Input Size</span><span>224 × 224</span></div>
        <div class='sidebar-item'><span class='sidebar-key'>Classes</span><span>NORMAL, PNEUMONIA</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-title'>Controls</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='sidebar-item'><span class='sidebar-key'>API</span><span>Local FastAPI</span></div>
        <div class='sidebar-item'><span class='sidebar-key'>Decision</span><span>&lt; 50% = Unknown</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col_left, col_right = st.columns([1.05, 1.05])

    with col_left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Uploaded Image")
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        if st.button("Analyze Image", use_container_width=True):
            try:
                with st.spinner("Sending image to FastAPI backend..."):
                    response = requests.post(
                        API_URL,
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream",
                            )
                        },
                        timeout=120,
                    )

                if response.status_code != 200:
                    st.error("Prediction request failed.")
                    try:
                        st.json(response.json())
                    except Exception:
                        st.write(response.text)
                else:
                    result = response.json()
                    predicted_label = result.get("predicted_label", "Unknown")
                    confidence = float(result.get("confidence_value", 0.0))
                    probabilities = result.get("probability_list", {})

                    st.markdown(
                        f"""
                        <div class='result-card'>
                            <div class='result-badge'>Prediction Result</div>
                            <p class='result-title'>{predicted_label}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if predicted_label == "Unknown":
                        st.warning("Prediction rejected: confidence is below 50%. The model is unsure and labeled the case as Unknown.")
                    else:
                        st.success(f"Prediction: {predicted_label}")

                    st.markdown(
                        f"""
                        <div class='confidence-card'>
                            <div style='color:#98a9c3; font-weight:600; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em;'>Confidence</div>
                            <p class='confidence-value'>{confidence * 100:.2f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("<div class='section-card probability-row'>", unsafe_allow_html=True)
                    st.subheader("Class Probabilities")
                    for class_name, probability in probabilities.items():
                        probability_percentage = float(probability) * 100
                        st.markdown(
                            f"<div class='probability-item'><span>{class_name}</span><strong>{probability_percentage:.2f}%</strong></div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error("An error occurred while sending the request to the backend.")
                st.exception(e)
        else:
            st.info("👆 Click the button to run the chest X-ray prediction.")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class='upload-wrapper'>
            <div style='text-align: center; color: #dfeafc;'>
                <div style='font-size: 2.8rem; margin-bottom: 0.5rem;'>🩺</div>
                <div style='font-size: 1.1rem; font-weight: 600;'>Upload a chest X-ray image to begin</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='disclaimer'>⚠️ This application is an educational/research prototype and is not intended for medical diagnosis.</div>",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
