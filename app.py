import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# Page configuration
st.set_page_config(
    page_title="Concrete Defect AI",
    page_icon="🏗️",
    layout="wide"
)

# Title
st.title("🏗️ Concrete Defect AI")
st.write("AI-powered concrete defect detection")

# Model path
MODEL_PATH = "models/best.pt"

# Check model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}")
    st.info("Make sure best.pt is inside the models folder.")
    st.stop()

# Load model
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
    st.success("✅ AI model loaded successfully!")
except Exception as e:
    st.error("❌ Could not load the AI model.")
    st.exception(e)
    st.stop()

# Upload image
st.subheader("📷 Upload Concrete Image")

uploaded_file = st.file_uploader(
    "Choose a concrete image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Detection
    if st.button("🔍 Detect Defects", type="primary"):

        with st.spinner("Analyzing concrete..."):

            results = model.predict(
                source=image,
                conf=0.25
            )

        result = results[0]

        # Annotated image
        annotated_image = result.plot()

        with col2:
            st.subheader("Detection Result")
            st.image(
                annotated_image,
                channels="BGR",
                use_container_width=True
            )

        # Detection information
        if len(result.boxes) > 0:

            st.success(
                f"⚠️ {len(result.boxes)} defect(s) detected!"
            )

            st.subheader("📊 Detected Defects")

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                st.write(
                    f"🔴 **{class_name}** — "
                    f"Confidence: {confidence:.2%}"
                )

        else:
            st.success("✅ No concrete defects detected.")

st.divider()

st.caption("Concrete Defect AI • Powered by YOLO")
