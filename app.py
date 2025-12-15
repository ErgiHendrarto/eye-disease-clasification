import streamlit as st
import joblib
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Eye Health Screening",
    page_icon="👁️",
    layout="centered"
)

# =========================
# CUSTOM CSS (Clean Medical UI)
# =========================
st.markdown("""
<style>
    body {
        background-color: #f7f9fc;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .result-box {
        background: #ffffff;
        border-radius: 14px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        text-align: center;
    }
    .result-label {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .confidence {
        font-size: 1.4rem;
        color: #495057;
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, tfidf, label_encoder

model, tfidf, label_encoder = load_models()

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(image):
    img = cv2.resize(image, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # LBP
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)

    # HOG
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    # GLCM
    glcm = graycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    return np.hstack([
        lbp_hist,
        hog_features,
        contrast,
        homogeneity,
        energy
    ])

# =========================
# HEADER
# =========================
st.markdown("<div class='main-title'>Eye Disease Screening</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Image-based eye condition assessment with symptom support</div>", unsafe_allow_html=True)

# =========================
# INPUT SECTION
# =========================
st.markdown("<div class='section-title'>1. Upload Eye Image</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a clear eye image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

st.markdown("<div class='section-title'>2. Describe Symptoms (optional)</div>", unsafe_allow_html=True)
symptoms_text = st.text_area(
    "Describe symptoms such as redness, pain, blurry vision, discharge, etc.",
    height=90,
    placeholder="Example: red eye, blurry vision, eye pain",
    label_visibility="collapsed"
)

# =========================
# IMAGE PREVIEW
# =========================
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Uploaded Eye Image",
        use_container_width=True
    )

    # =========================
    # ANALYSIS BUTTON
    # =========================
    if st.button("Analyze", use_container_width=True):
        with st.spinner("Processing data..."):
            try:
                visual_features = extract_features(image)
                symptom_features = tfidf.transform([symptoms_text]).toarray()[0]
                input_features = np.hstack([visual_features, symptom_features])

                pred = model.predict([input_features])[0]
                proba = model.predict_proba([input_features])[0]

                label = label_encoder.inverse_transform([pred])[0]
                confidence = proba[pred] * 100

                # =========================
                # RESULT DISPLAY
                # =========================
                st.markdown("<div class='section-title'>Result</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">{label}</div>
                    <div class="confidence">Confidence: {confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # =========================
                # PROBABILITY DISTRIBUTION
                # =========================
                st.markdown("<div class='section-title'>Detailed Probabilities</div>", unsafe_allow_html=True)
                for i, cls in enumerate(label_encoder.classes_):
                    st.write(cls)
                    st.progress(float(proba[i]))

                # =========================
                # DISCLAIMER
                # =========================
                st.markdown("---")
                st.markdown("""
                <div class="disclaimer">
                This application is intended for screening and educational purposes only.
                It does not replace professional medical diagnosis.
                Please consult a qualified ophthalmologist for clinical decisions.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error("An error occurred during analysis. Please try again.")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### About")
    st.write("""
    This application assists in identifying potential eye conditions
    using image features and symptom descriptions.
    """)

    st.markdown("### Supported Conditions")
    st.write("""
    - Normal  
    - Cataract  
    - Conjunctivitis  
    - Uveitis  
    - Eyelid Disorders  
    """)

    st.markdown("---")
    st.caption("Clinical Decision Support Prototype")
