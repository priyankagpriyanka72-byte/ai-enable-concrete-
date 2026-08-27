import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Concrete Defect AI",
    page_icon="🏗️",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🏗️ Concrete Defect AI")
st.write("AI-powered concrete defect detection")

# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = "concrete_defect_model.keras"

# Check model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found: {MODEL_PATH}")
    st.info("Make sure concrete_defect_model.keras is in the main project folder.")
    st.stop()

# -----------------------------
# Load Keras model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.success("✅ AI model loaded successfully!")

except Exception as e:
    st.error("❌ Could not load the AI model.")
    st.exception(e)
    st.stop()

# -----------------------------
# Class names
# -----------------------------
# Change these if your trained model has different classes.
CLASS_NAMES = [
    "Crack",
    "Spalling",
    "Surface Defect",
    "No Defect"
]

# -----------------------------
# Image upload
# -----------------------------
st.subheader("📷 Upload Concrete Image")

uploaded_file = st.file_uploader(
    "Choose a concrete image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("🔍 Detect Defects", type="primary"):

        with st.spinner("Analyzing concrete..."):

            # Get model input size
            input_shape = model.input_shape

            if isinstance(input_shape, list):
                input_shape = input_shape[0]

            height = input_shape[1]
            width = input_shape[2]

            # Resize image
            resized_image = image.resize((width, height))

            # Convert to NumPy
            img_array = np.array(resized_image).astype("float32")

            # Normalize
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array, verbose=0)

        # -----------------------------
        # Process prediction
        # -----------------------------
        prediction = np.array(prediction)

        # Binary classification
        if prediction.shape[-1] == 1:

            probability = float(prediction[0][0])

            if probability >= 0.5:
                class_id = 1
                confidence = probability
            else:
                class_id = 0
                confidence = 1 - probability

        # Multi-class classification
        else:
            probabilities = prediction[0]

            class_id = int(np.argmax(probabilities))
            confidence = float(probabilities[class_id])

        # Get class name safely
        if class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = f"Class {class_id}"

        # -----------------------------
        # Display result
        # -----------------------------
        with col2:
            st.subheader("Detection Result")

            if class_name.lower() == "no defect":
                st.success(f"✅ {class_name}")
            else:
                st.warning(f"⚠️ {class_name}")

            st.info(f"Confidence: **{confidence:.2%}**")

        st.subheader("📊 Detected Defect")

        st.write(
            f"🔴 **{class_name}** — "
            f"Confidence: **{confidence:.2%}**"
        )

        # Show probabilities for multi-class model
        if prediction.shape[-1] > 1:

            st.subheader("📈 Class Probabilities")

            for i, probability in enumerate(prediction[0]):

                if i < len(CLASS_NAMES):
                    name = CLASS_NAMES[i]
                else:
                    name = f"Class {i}"

                st.write(
                    f"**{name}**: {float(probability):.2%}"
                )

st.divider()

st.caption("Concrete Defect AI • Powered by TensorFlow")
