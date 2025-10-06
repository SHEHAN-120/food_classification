import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# --- Page setup ---
st.set_page_config(
    page_title="🍽️ Food Classifier",
    page_icon="🍕",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
        color: #2f2f2f;
        font-family: 'Poppins', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 40px;
        color: #ffffff;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        padding: 18px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .result-card {
        background-color: #ffffff;
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        margin-top: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<div class='title'>🍽️ Food Classifier 🍽️</div>", unsafe_allow_html=True)

# --- Sidebar: show available food classes ---
st.sidebar.header("🍴 Food Categories")
st.sidebar.markdown(
    """
    **Available Classes:**
    - burger  
    - butter_naan  
    - chai  
    - chapati  
    - chole_bhature  
    - dal_makhani  
    - dhokla  
    - fried_rice  
    - idli  
    - jalebi  
    - kaathi_rolls  
    - kadai_paneer  
    - kulfi  
    - masala_dosa  
    - momos  
    - paani_puri  
    - pakode  
    - pav_bhaji  
    - pizza  
    - samosa
    """
)

# --- Load model and processor ---
@st.cache_resource
def load_model():
    model_ckpt = "food_classification"  # your saved model folder
    model = AutoModelForImageClassification.from_pretrained(model_ckpt)
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    return model, processor

model, processor = load_model()

# --- Upload section ---
uploaded_file = st.file_uploader("📸 Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)


    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    label = model.config.id2label[predicted_class_idx]

    # --- Display result ---
    st.markdown(f"""
        <div class='result-card'>
            <h3>🍔 Predicted Food: <span style="color:#0072ff;">{label}</span></h3>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("👆 Upload a food image to get the prediction.")
