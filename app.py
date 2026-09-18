import os

import streamlit as st
from PIL import Image

from src.predict import load_model, predict_image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "pneumonia_classifier.pth"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    model, class_names = load_model(
        MODEL_PATH
    )

    return model, class_names


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        f"Model file not found: {MODEL_PATH}"
    )

    st.info(
        "Make sure pneumonia_classifier.pth "
        "is inside the models folder."
    )

    st.stop()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:

    model, class_names = get_model()

except Exception as e:

    st.error("Failed to load the trained model.")

    st.exception(e)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Chest X-Ray Pneumonia Classifier")

st.markdown(
    """
    Upload a chest X-ray image and the trained
    **EfficientNet-B0** model will predict the class.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.write("**Model:** EfficientNet-B0")
    st.write("**Framework:** PyTorch")
    st.write("**Input Size:** 224 × 224")
    st.write(
        f"**Classes:** {', '.join(class_names)}"
    )

    st.divider()

    st.write("### Supported Images")

    st.write(
        "JPG, JPEG and PNG"
    )


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# IMAGE PREVIEW
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True
    )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔍 Predict",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Analyzing chest X-ray..."
            ):

                predicted_label, confidence, probabilities = (
                    predict_image(
                        model,
                        image,
                        class_names
                    )
                )


            # =================================================
            # PREDICTION RESULT
            # =================================================

            st.divider()

            st.subheader("Prediction Result")

            st.success(
                f"Prediction: {predicted_label}"
            )

            st.metric(
                label="Confidence",
                value=f"{confidence * 100:.2f}%"
            )


            # =================================================
            # CLASS PROBABILITIES
            # =================================================

            st.subheader("Class Probabilities")

            for class_name, probability in zip(
                class_names,
                probabilities
            ):

                probability_percentage = (
                    float(probability) * 100
                )

                st.write(
                    f"**{class_name}**: "
                    f"{probability_percentage:.2f}%"
                )

                st.progress(
                    float(probability)
                )


        except Exception as e:

            st.error(
                "An error occurred while making the prediction."
            )

            st.exception(e)


# ============================================================
# INSTRUCTIONS
# ============================================================

else:

    st.info(
        "👆 Upload a chest X-ray image to start prediction."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "⚠️ This application is an educational/research "
    "prototype and is not intended for medical diagnosis."
)