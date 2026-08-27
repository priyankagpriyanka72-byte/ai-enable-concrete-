import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Concrete Defect AI",
    page_icon="🏗️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 20px;
    color: #666;
}

.feature-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-bottom: 15px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    border: 2px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🏗️ Concrete Defect AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered concrete defect detection, prevention and recommendation system</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# MODEL
# =========================================================

MODEL_PATH = "concrete_defect_model.keras"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found.")
    st.info(
        "Make sure concrete_defect_model.keras is in the main GitHub repository folder."
    )
    st.stop()


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

# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "bleeding",
    "carbonation",
    "cold_joint",
    "cracks",
    "delamination",
    "discoloration",
    "efflorescence",
    "honeycombing",
    "improper_finishing",
    "low_workability",
    "plastic_settlement",
    "reinforcement_corrosion",
    "scaling",
    "segregation",
    "spalling"
]

# =========================================================
# DEFECT INFORMATION
# =========================================================

DEFECT_INFO = {

"bleeding": {
    "severity": "Medium",
    "cause": "Excess water rises to the concrete surface after placement.",
    "prevention": [
        "Use the correct water-cement ratio.",
        "Avoid excessive vibration.",
        "Use suitable concrete mix proportions.",
        "Provide proper drainage and finishing."
    ],
    "recommendation": [
        "Remove weak surface material if necessary.",
        "Allow the surface to dry properly.",
        "Improve concrete mix design for future work.",
        "Monitor the affected area for surface deterioration."
    ]
},

"carbonation": {
    "severity": "Medium",
    "cause": "Carbon dioxide reacts with cement compounds and reduces concrete alkalinity.",
    "prevention": [
        "Use good quality concrete.",
        "Maintain adequate concrete cover.",
        "Use a suitable water-cement ratio.",
        "Provide protective coatings where appropriate."
    ],
    "recommendation": [
        "Inspect reinforcement condition.",
        "Measure carbonation depth if required.",
        "Repair damaged concrete cover.",
        "Protect the surface against further carbonation."
    ]
},

"cold_joint": {
    "severity": "High",
    "cause": "A weak interface forms when fresh concrete is placed against concrete that has already started setting.",
    "prevention": [
        "Maintain continuous concrete placement.",
        "Plan construction joints properly.",
        "Ensure proper surface preparation.",
        "Avoid unnecessary delays during placement."
    ],
    "recommendation": [
        "Inspect the joint carefully.",
        "Remove loose or weak material.",
        "Use suitable repair material.",
        "Consider professional structural assessment for significant joints."
    ]
},

"cracks": {
    "severity": "High",
    "cause": "Cracking can result from shrinkage, temperature changes, loading, settlement or other causes.",
    "prevention": [
        "Use proper mix proportions.",
        "Provide adequate curing.",
        "Control shrinkage and temperature effects.",
        "Provide appropriate reinforcement and joints."
    ],
    "recommendation": [
        "Measure crack width and monitor changes.",
        "Identify the underlying cause.",
        "Seal suitable non-structural cracks.",
        "Obtain professional assessment for significant or growing cracks."
    ]
},

"delamination": {
    "severity": "High",
    "cause": "A layer near the concrete surface separates from the underlying concrete.",
    "prevention": [
        "Avoid finishing while bleed water is present.",
        "Use proper vibration.",
        "Follow correct finishing procedures.",
        "Provide adequate curing."
    ],
    "recommendation": [
        "Identify hollow or detached areas.",
        "Remove loose material.",
        "Repair the affected surface.",
        "Inspect the surrounding concrete."
    ]
},

"discoloration": {
    "severity": "Low",
    "cause": "Uneven color may result from curing conditions, materials, finishing or environmental exposure.",
    "prevention": [
        "Use consistent concrete materials.",
        "Maintain uniform curing conditions.",
        "Use consistent finishing procedures.",
        "Avoid contamination during construction."
    ],
    "recommendation": [
        "Determine whether discoloration is only cosmetic.",
        "Clean the surface appropriately.",
        "Use suitable surface treatment if required.",
        "Monitor for changes."
    ]
},

"efflorescence": {
    "severity": "Low",
    "cause": "Soluble salts migrate to the surface and form a white deposit.",
    "prevention": [
        "Control water penetration.",
        "Use suitable concrete materials.",
        "Provide proper drainage.",
        "Ensure adequate curing."
    ],
    "recommendation": [
        "Clean deposits using an appropriate method.",
        "Identify and reduce the moisture source.",
        "Improve drainage or waterproofing.",
        "Monitor recurrence."
    ]
},

"honeycombing": {
    "severity": "High",
    "cause": "Voids occur because concrete does not properly fill the formwork, often due to inadequate compaction or poor placement.",
    "prevention": [
        "Use proper concrete workability.",
        "Ensure adequate vibration.",
        "Place concrete carefully.",
        "Use properly sealed formwork."
    ],
    "recommendation": [
        "Assess the depth and extent of honeycombing.",
        "Remove loose concrete.",
        "Repair shallow defects with suitable repair mortar.",
        "Seek professional assessment for deep or structural defects."
    ]
},

"improper_finishing": {
    "severity": "Medium",
    "cause": "Incorrect timing or technique during concrete finishing can damage the surface.",
    "prevention": [
        "Finish concrete at the correct time.",
        "Avoid finishing while bleed water is present.",
        "Use appropriate finishing tools.",
        "Follow proper site procedures."
    ],
    "recommendation": [
        "Evaluate surface strength and condition.",
        "Remove weak surface material where necessary.",
        "Use suitable surface repair methods.",
        "Improve finishing procedures in future work."
    ]
},

"low_workability": {
    "severity": "Medium",
    "cause": "Concrete has insufficient ease of placement and compaction.",
    "prevention": [
        "Design the mix appropriately.",
        "Maintain suitable water-cement ratio.",
        "Use approved admixtures where required.",
        "Avoid excessive loss of workability during transport."
    ],
    "recommendation": [
        "Check concrete mix design.",
        "Verify placement and compaction procedures.",
        "Avoid adding uncontrolled water at site.",
        "Use suitable admixtures where appropriate."
    ]
},

"plastic_settlement": {
    "severity": "Medium",
    "cause": "Fresh concrete settles while reinforcement or obstructions restrict movement, causing cracks.",
    "prevention": [
        "Use suitable concrete consistency.",
        "Ensure proper placement and vibration.",
        "Pay attention to reinforcement arrangement.",
        "Provide appropriate curing."
    ],
    "recommendation": [
        "Inspect settlement cracks.",
        "Monitor crack development.",
        "Repair suitable surface cracks.",
        "Review placement procedures."
    ]
},

"reinforcement_corrosion": {
    "severity": "Critical",
    "cause": "Reinforcement corrosion can occur when moisture, oxygen and aggressive substances reach the steel.",
    "prevention": [
        "Provide adequate concrete cover.",
        "Use durable concrete.",
        "Control water penetration.",
        "Protect exposed reinforcement."
    ],
    "recommendation": [
        "Inspect the reinforcement condition.",
        "Remove loose and damaged concrete where appropriate.",
        "Treat or replace affected reinforcement according to engineering requirements.",
        "Obtain professional structural assessment for significant corrosion."
    ]
},

"scaling": {
    "severity": "Medium",
    "cause": "The concrete surface gradually flakes or peels due to environmental exposure, poor finishing or inadequate durability.",
    "prevention": [
        "Use durable concrete.",
        "Provide proper curing.",
        "Avoid excessive surface water.",
        "Use suitable finishing techniques."
    ],
    "recommendation": [
        "Remove loose surface material.",
        "Clean and prepare the surface.",
        "Apply a suitable repair system.",
        "Improve drainage and protection."
    ]
},

"segregation": {
    "severity": "Medium",
    "cause": "Concrete ingredients separate during handling, transportation or placement.",
    "prevention": [
        "Use suitable mix proportions.",
        "Avoid excessive vibration.",
        "Handle concrete carefully.",
        "Maintain appropriate workability."
    ],
    "recommendation": [
        "Inspect affected areas.",
        "Remove weak or segregated material where necessary.",
        "Repair the surface appropriately.",
        "Improve concrete handling procedures."
    ]
},

"spalling": {
    "severity": "High",
    "cause": "Concrete surface material breaks away, often because of corrosion, freeze-thaw exposure, impact or other deterioration.",
    "prevention": [
        "Provide adequate reinforcement cover.",
        "Control moisture penetration.",
        "Use durable concrete.",
        "Perform regular inspection and maintenance."
    ],
    "recommendation": [
        "Remove loose concrete safely.",
        "Inspect reinforcement if exposed.",
        "Repair damaged concrete with suitable materials.",
        "Seek professional assessment for extensive spalling."
    ]
}

}

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Feature",
    [
        "🏠 Home",
        "🔍 Defect Detection",
        "🛡️ Prevention",
        "💡 Recommendations"
    ]
)

# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.header("🏗️ Concrete Defect AI")

    st.write(
        "An AI-based system designed to identify common concrete defects "
        "from images and provide prevention and maintenance recommendations."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Classes", "15")

    with col2:
        st.metric("Images / Class", "500")

    with col3:
        st.metric("Total Images", "7,500")

    st.divider()

    st.subheader("🔍 Supported Defects")

    for i in range(0, len(CLASS_NAMES), 3):

        cols = st.columns(3)

        for j, col in enumerate(cols):

            index = i + j

            if index < len(CLASS_NAMES):

                with col:
                    st.info(
                        f"{index + 1}. "
                        f"{CLASS_NAMES[index].replace('_', ' ').title()}"
                    )

# =========================================================
# DETECTION
# =========================================================

elif page == "🔍 Defect Detection":

    st.header("🔍 Concrete Defect Detection")

    uploaded_file = st.file_uploader(
        "Upload a concrete image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Concrete Image",
            use_container_width=True
        )

        if st.button(
            "🔍 Detect Defect",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("AI is analyzing the concrete image..."):

                input_shape = model.input_shape

                if isinstance(input_shape, list):
                    input_shape = input_shape[0]

                height = input_shape[1]
                width = input_shape[2]

                resized_image = image.resize((width, height))

                img_array = np.array(resized_image).astype("float32")

                img_array = img_array / 255.0

                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(
                    img_array,
                    verbose=0
                )

            prediction = np.array(prediction)

            # -----------------------------
            # Prediction processing
            # -----------------------------

            if prediction.shape[-1] == 1:

                probability = float(prediction[0][0])

                class_id = 1 if probability >= 0.5 else 0

                confidence = (
                    probability
                    if class_id == 1
                    else 1 - probability
                )

            else:

                probabilities = prediction[0]

                class_id = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    probabilities[class_id]
                )

            class_name = CLASS_NAMES[
                class_id
            ]

            info = DEFECT_INFO[
                class_name
            ]

            # -----------------------------
            # Results
            # -----------------------------

            st.divider()

            st.subheader("📊 AI Detection Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Detected Defect",
                    class_name.replace(
                        "_", " "
                    ).title()
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2%}"
                )

            with col3:
                st.metric(
                    "Severity",
                    info["severity"]
                )

            st.divider()

            st.subheader("🔎 Possible Cause")

            st.write(info["cause"])

            st.subheader("📈 Prediction Probabilities")

            if prediction.shape[-1] > 1:

                for i, probability in enumerate(
                    prediction[0]
                ):

                    if i < len(CLASS_NAMES):

                        name = CLASS_NAMES[i]

                        st.write(
                            f"**{name.replace('_', ' ').title()}**"
                        )

                        st.progress(
                            float(probability)
                        )

# =========================================================
# PREVENTION
# =========================================================

elif page == "🛡️ Prevention":

    st.header("🛡️ Defect Prevention")

    defect = st.selectbox(
        "Select a concrete defect",
        CLASS_NAMES
    )

    info = DEFECT_INFO[defect]

    st.subheader(
        f"🛡️ Prevention for {defect.replace('_', ' ').title()}"
    )

    st.write("### Possible Cause")

    st.write(info["cause"])

    st.write("### Prevention Measures")

    for item in info["prevention"]:

        st.success(
            f"✓ {item}"
        )

# =========================================================
# RECOMMENDATIONS
# =========================================================

elif page == "💡 Recommendations":

    st.header("💡 Repair & Maintenance Recommendations")

    defect = st.selectbox(
        "Select detected defect",
        CLASS_NAMES
    )

    info = DEFECT_INFO[defect]

    st.subheader(
        f"Recommendations for {defect.replace('_', ' ').title()}"
    )

    st.write("### Severity")

    if info["severity"] == "Critical":

        st.error(
            "🚨 Critical: professional engineering assessment is recommended."
        )

    elif info["severity"] == "High":

        st.warning(
            "⚠️ High: the affected area should be inspected carefully."
        )

    elif info["severity"] == "Medium":

        st.info(
            "ℹ️ Medium: monitor and address the defect appropriately."
        )

    else:

        st.success(
            "✅ Low: primarily monitor and maintain the affected surface."
        )

    st.write("### 💡 Recommended Actions")

    for item in info["recommendation"]:

        st.info(
            f"🔧 {item}"
        )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🏗️ Concrete Defect AI | "
    "AI-based Concrete Inspection & Maintenance Assistant"
)
