import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Concrete Defect AI",
    page_icon="🏗️",
    layout="wide"
)

MODEL_PATH = "concrete_defect_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
