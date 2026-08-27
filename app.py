import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Concrete Defect AI",
    page_icon="🏗️",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🏗️ Concrete Defect AI")
st.write(
    "AI-powered concrete defect detection, prevention and "
    "recommendation system"
)

st.divider()

# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "concrete_defect_model.keras"

if not os.path.exists(MODEL_PATH):
    st.error("❌ AI model not found!")
    st.info(
        "Make sure 'concrete_defect_model.keras' is in the "
        "main folder of your GitHub repository."
    )
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

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


# ============================================================
# 15 MODEL CLASSES
# ============================================================

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


# ============================================================
# DEFECT INFORMATION
# ============================================================

DEFECT_INFO = {

    "bleeding": {
        "severity": "Medium",
        "cause": (
            "Excess water rises to the concrete surface after "
            "placement."
        ),
        "prevention": [
            "Use a suitable water-cement ratio.",
            "Use proper concrete mix proportions.",
            "Avoid excessive vibration.",
            "Provide proper finishing and curing."
        ],
        "recommendation": [
            "Inspect the affected surface.",
            "Allow the surface to dry properly.",
            "Remove weak surface material if necessary.",
            "Improve mix design and placement practices."
        ]
    },

    "carbonation": {
        "severity": "Medium",
        "cause": (
            "Carbon dioxide reacts with concrete compounds and "
            "reduces the alkalinity of concrete."
        ),
        "prevention": [
            "Use good-quality concrete.",
            "Maintain adequate concrete cover.",
            "Use a suitable water-cement ratio.",
            "Protect concrete from excessive moisture and exposure."
        ],
        "recommendation": [
            "Inspect the concrete and reinforcement condition.",
            "Check carbonation depth when required.",
            "Repair damaged concrete cover.",
            "Use suitable protective treatment where appropriate."
        ]
    },

    "cold_joint": {
        "severity": "High",
        "cause": (
            "A weak interface can form when fresh concrete is "
            "placed against concrete that has already started setting."
        ),
        "prevention": [
            "Maintain continuous concrete placement.",
            "Plan construction joints properly.",
            "Avoid unnecessary delays during placement.",
            "Prepare construction joint surfaces correctly."
        ],
        "recommendation": [
            "Inspect the joint carefully.",
            "Check for cracking or weak material.",
            "Remove loose material where necessary.",
            "Use an appropriate repair system.",
            "Seek professional assessment if the joint is significant."
        ]
    },

    "cracks": {
        "severity": "High",
        "cause": (
            "Cracks may result from shrinkage, temperature changes, "
            "settlement, loading or other causes."
        ),
        "prevention": [
            "Use proper concrete mix proportions.",
            "Provide adequate curing.",
            "Control shrinkage and temperature effects.",
            "Provide appropriate reinforcement and joints."
        ],
        "recommendation": [
            "Inspect and measure the cracks.",
            "Monitor whether the cracks are changing.",
            "Identify the underlying cause.",
            "Seal suitable non-structural cracks.",
            "Seek professional assessment for significant or growing cracks."
        ]
    },

    "delamination": {
        "severity": "High",
        "cause": (
            "A layer near the concrete surface separates from the "
            "underlying concrete."
        ),
        "prevention": [
            "Avoid finishing while bleed water is present.",
            "Use proper concrete vibration.",
            "Follow correct finishing procedures.",
            "Provide adequate curing."
        ],
        "recommendation": [
            "Inspect the surface for hollow or detached areas.",
            "Remove loose material.",
            "Prepare the surface properly.",
            "Repair using a suitable concrete repair material."
        ]
    },

    "discoloration": {
        "severity": "Low",
        "cause": (
            "Uneven concrete color can result from curing conditions, "
            "materials, finishing or environmental exposure."
        ),
        "prevention": [
            "Use consistent concrete materials.",
            "Maintain uniform curing conditions.",
            "Use consistent finishing procedures.",
            "Avoid contamination during construction."
        ],
        "recommendation": [
            "Determine whether the problem is mainly cosmetic.",
            "Clean the surface using a suitable method.",
            "Use appropriate surface treatment if required.",
            "Monitor the area for further changes."
        ]
    },

    "efflorescence": {
        "severity": "Low",
        "cause": (
            "Soluble salts move through moisture and form deposits "
            "on the concrete surface."
        ),
        "prevention": [
            "Control water penetration.",
            "Provide proper drainage.",
            "Use suitable concrete materials.",
            "Ensure adequate curing."
        ],
        "recommendation": [
            "Clean the deposits using an appropriate method.",
            "Identify and reduce the moisture source.",
            "Improve drainage or waterproofing.",
            "Monitor whether deposits return."
        ]
    },

    "honeycombing": {
        "severity": "High",
        "cause": (
            "Voids occur when concrete does not properly fill the "
            "formwork, commonly due to inadequate compaction or poor placement."
        ),
        "prevention": [
            "Use suitable concrete workability.",
            "Ensure adequate vibration and compaction.",
            "Place concrete carefully.",
            "Use properly sealed formwork."
        ],
        "recommendation": [
            "Assess the depth and extent of honeycombing.",
            "Remove loose or weak concrete.",
            "Repair shallow defects with suitable repair material.",
            "Seek professional assessment for deep or structural defects."
        ]
    },

    "improper_finishing": {
        "severity": "Medium",
        "cause": (
            "Incorrect timing or technique during concrete finishing "
            "can weaken or damage the surface."
        ),
        "prevention": [
            "Finish concrete at the correct time.",
            "Avoid finishing while bleed water is present.",
            "Use appropriate finishing tools.",
            "Follow proper site procedures."
        ],
        "recommendation": [
            "Evaluate the surface condition.",
            "Remove weak surface material when necessary.",
            "Use suitable surface repair methods.",
            "Improve finishing procedures for future work."
        ]
    },

    "low_workability": {
        "severity": "Medium",
        "cause": (
            "Concrete has insufficient ease of placement and "
            "compaction."
        ),
        "prevention": [
            "Design the concrete mix appropriately.",
            "Maintain a suitable water-cement ratio.",
            "Use approved admixtures when required.",
            "Avoid excessive loss of workability during transport."
        ],
        "recommendation": [
            "Check the concrete mix design.",
            "Verify placement and compaction procedures.",
            "Avoid uncontrolled addition of water at the site.",
            "Use suitable admixtures where appropriate."
        ]
    },

    "plastic_settlement": {
        "severity": "Medium",
        "cause": (
            "Fresh concrete settles while reinforcement or other "
            "obstructions restrict movement."
        ),
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
            "Review concrete placement procedures."
        ]
    },

    "reinforcement_corrosion": {
        "severity": "Critical",
        "cause": (
            "Corrosion can occur when moisture, oxygen and aggressive "
            "substances reach reinforcement."
        ),
        "prevention": [
            "Provide adequate concrete cover.",
            "Use durable concrete.",
            "Control water penetration.",
            "Protect exposed reinforcement."
        ],
        "recommendation": [
            "Inspect the reinforcement condition.",
            "Identify the extent of concrete damage.",
            "Repair affected concrete using suitable methods.",
            "Obtain professional structural assessment for significant corrosion."
        ]
    },

    "scaling": {
        "severity": "Medium",
        "cause": (
            "The concrete surface flakes or peels because of exposure, "
            "poor finishing or inadequate durability."
        ),
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
            "Improve drainage and surface protection."
        ]
    },

    "segregation": {
        "severity": "Medium",
        "cause": (
            "Concrete ingredients separate during handling, transportation "
            "or placement."
        ),
        "prevention": [
            "Use suitable mix proportions.",
            "Avoid excessive vibration.",
            "Handle concrete carefully.",
            "Maintain appropriate workability."
        ],
        "recommendation": [
            "Inspect the affected concrete.",
            "Remove weak material where necessary.",
            "Repair the affected surface appropriately.",
            "Improve concrete handling and placement procedures."
        ]
    },

    "spalling": {
        "severity": "High",
        "cause": (
            "Concrete surface material breaks away due to deterioration, "
            "corrosion, environmental exposure or other causes."
        ),
        "prevention": [
            "Provide adequate reinforcement cover.",
            "Control moisture penetration.",
            "Use durable concrete.",
            "Perform regular inspection and maintenance."
        ],
        "recommendation": [
            "Inspect the affected area.",
            "Remove loose concrete safely.",
            "Inspect reinforcement if exposed.",
            "Repair damaged concrete using suitable materials.",
            "Seek professional assessment for extensive spalling."
        ]
    }
}


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📷 Upload Concrete Image")

uploaded_file = st.file_uploader(
    "Choose a concrete image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# DETECTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )

    st.divider()

    if st.button(
        "🔍 Detect Concrete Defect",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("🤖 AI is analyzing the concrete image..."):

            try:

                # Get model input size
                input_shape = model.input_shape

                if isinstance(input_shape, list):
                    input_shape = input_shape[0]

                height = input_shape[1]
                width = input_shape[2]

                # Resize
                resized_image = image.resize(
                    (width, height)
                )

                # Convert image to NumPy
                img_array = np.array(
                    resized_image
                ).astype("float32")

                # Normalize
                img_array = img_array / 255.0

                # Add batch dimension
                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # Model prediction
                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                prediction = np.array(prediction)

                # ==================================================
                # MULTI-CLASS MODEL
                # ==================================================

                if prediction.shape[-1] == len(CLASS_NAMES):

                    probabilities = prediction[0]

                    class_id = int(
                        np.argmax(probabilities)
                    )

                    confidence = float(
                        probabilities[class_id]
                    )

                # ==================================================
                # BINARY MODEL
                # ==================================================

                elif prediction.shape[-1] == 1:

                    probability = float(
                        prediction[0][0]
                    )

                    if probability >= 0.5:
                        class_id = 1
                        confidence = probability
                    else:
                        class_id = 0
                        confidence = 1 - probability

                else:

                    st.error(
                        "❌ The model output does not match "
                        "the 15 defect classes."
                    )

                    st.stop()

                # ==================================================
                # DETECTED DEFECT
                # ==================================================

                defect_name = CLASS_NAMES[class_id]

                info = DEFECT_INFO[defect_name]

                display_name = defect_name.replace(
                    "_", " "
                ).title()

                # ==================================================
                # RESULT HEADER
                # ==================================================

                st.divider()

                st.header("🔍 Detection Result")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Detected Defect",
                        display_name
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

                # ==================================================
                # CAUSE
                # ==================================================

                st.divider()

                st.subheader("🔎 Possible Cause")

                st.info(info["cause"])

                # ==================================================
                # PREVENTION
                # ==================================================

                st.subheader(
                    "🛡️ Prevention Measures"
                )

                for item in info["prevention"]:

                    st.success(
                        f"✓ {item}"
                    )

                # ==================================================
                # RECOMMENDATIONS
                # ==================================================

                st.subheader(
                    "💡 Recommended Actions"
                )

                for item in info["recommendation"]:

                    st.info(
                        f"🔧 {item}"
                    )

                # ==================================================
                # PROBABILITIES
                # ==================================================

                if prediction.shape[-1] == len(CLASS_NAMES):

                    st.divider()

                    st.subheader(
                        "📊 AI Prediction Probabilities"
                    )

                    for i, probability in enumerate(
                        probabilities
                    ):

                        name = CLASS_NAMES[i].replace(
                            "_", " "
                        ).title()

                        st.write(
                            f"**{name}: "
                            f"{float(probability):.2%}**"
                        )

                        st.progress(
                            min(
                                max(
                                    float(probability),
                                    0.0
                                ),
                                1.0
                            )
                        )

            except Exception as e:

                st.error(
                    "❌ Error while analyzing the image."
                )

                st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏗️ Concrete Defect AI | "
    "AI-based Concrete Defect Detection & Maintenance Assistant"
)
