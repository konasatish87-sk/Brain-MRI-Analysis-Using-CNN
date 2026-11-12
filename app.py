
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain MRI Analysis",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #07111F, #10243B);
    color: white;
}

header {
    background: transparent !important;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1100px;
}

/* SECTION TITLES */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #65B8FF;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* UPLOAD AREA */
[data-testid="stFileUploader"] {
    background: #102A43;
    border: 2px dashed #3985B9;
    border-radius: 15px;
    padding: 20px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #45D6B5;
}

/* UPLOAD BUTTON */
[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, #2878E5, #12B8A6);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
}

/* ANALYZE BUTTON */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2878E5, #12B8A6);
    color: white;
    border: none;
    border-radius: 10px;
    height: 50px;
    font-size: 17px;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #12B8A6, #2878E5);
    color: white;
    border: none;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #8196AC;
    font-size: 13px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "cnn_simple_from_scratch.keras"
    )

model = load_model()


# ---------------- LOAD CLASS NAMES ----------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)


# ---------------- HEADER ----------------
st.markdown("""
<div style="
    text-align:center;
    padding:35px 0 40px 0;
">
    <h1 style="
        font-size:38px;
        font-weight:800;
        color:#EAF4FF;
        letter-spacing:1.5px;
        margin:0;
        text-shadow:0 0 18px rgba(69,214,181,0.25);
    ">
        🧠 Brain MRI Analysis
    </h1>
</div>
""", unsafe_allow_html=True)


# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1, 1.5], gap="large")


# ---------------- LEFT SECTION ----------------
with left:

    st.markdown(
        '<div class="section-title">📁 Upload Brain MRI Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an MRI image from your device",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="MRI Image Preview",
            width=150
        )

    predict_button = st.button(
        "✨ Analyze MRI Image",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------
if uploaded_file is not None and predict_button:

    with st.spinner("AI is analyzing your MRI image..."):

        # Preprocessing
        img = image.resize((224, 224))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)

        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        confidence = predictions[0][predicted_index] * 100

    # ---------------- RIGHT SECTION ----------------
    with right:

        st.markdown(
            '<div class="section-title">📊 Class Probabilities</div>',
            unsafe_allow_html=True
        )

        for i, name in enumerate(class_names):

            probability = float(predictions[0][i] * 100)

            st.write(
                f"**{name}** — {probability:.2f}%"
            )

            st.progress(
                min(max(probability / 100, 0.0), 1.0)
            )


elif predict_button:

    st.warning("Please upload an MRI image first.")


# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown("""
<div class="footer">
    Brain MRI Analysis
    <br>
    Built with TensorFlow, CNN & Streamlit
    <br><br>
    AI-assisted classification for educational and research purposes.
    Not a substitute for professional medical diagnosis.
</div>
""", unsafe_allow_html=True)
