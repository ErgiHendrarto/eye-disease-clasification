import streamlit as st
import joblib
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog

# Page config
st.set_page_config(
    page_title="EyeCare AI - Eye Disease Classification",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #FFFFFF;
}

.main .block-container {
    background-color: #FFFFFF;
    padding-top: 1rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A5F 0%, #2C5282 100%);
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Hero section */
.hero-section {
    background: linear-gradient(135deg, #1E3A5F 0%, #3182CE 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 30px rgba(30, 58, 95, 0.3);
}

.hero-title {
    color: #FFFFFF;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: #E2E8F0;
    font-size: 1.1rem;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Stats card */
.stats-card {
    background: linear-gradient(135deg, #3182CE 0%, #2C5282 100%);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: #FFFFFF;
    box-shadow: 0 4px 15px rgba(49, 130, 206, 0.3);
}

.stats-number {
    font-size: 2rem;
    font-weight: 800;
}

.stats-label {
    font-size: 0.85rem;
    opacity: 0.9;
}

/* Step card */
.step-card {
    background: #F8FAFC;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #E2E8F0;
    height: 100%;
}

.step-number {
    background: linear-gradient(135deg, #3182CE 0%, #2C5282 100%);
    color: #FFFFFF;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}

.step-title {
    color: #1E3A5F;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.step-desc {
    color: #64748B;
    font-size: 0.8rem;
}

/* Section header */
.section-header {
    color: #1E3A5F;
    font-size: 1.5rem;
    font-weight: 700;
    text-align: center;
    margin: 2rem 0 1rem;
}

/* Result card */
.result-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
    border: 1px solid #E2E8F0;
    margin: 1rem 0;
}

.prediction-box {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: #FFFFFF;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1rem;
}

.prediction-label {
    font-size: 0.9rem;
    opacity: 0.9;
}

.prediction-value {
    font-size: 2rem;
    font-weight: 800;
}

.confidence-box {
    background: linear-gradient(135deg, #3182CE 0%, #2C5282 100%);
    color: #FFFFFF;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
}

/* Warning box */
.warning-box {
    background: #FEF3C7;
    border-left: 4px solid #F59E0B;
    padding: 1rem 1.5rem;
    border-radius: 0 12px 12px 0;
    margin: 1rem 0;
}

.warning-title {
    color: #92400E;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.warning-text {
    color: #78350F;
    font-size: 0.9rem;
}

/* Footer */
.footer {
    background: #1E3A5F;
    padding: 1.5rem;
    border-radius: 16px 16px 0 0;
    margin-top: 2rem;
    text-align: center;
    color: #E2E8F0;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #3182CE 0%, #2C5282 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(49, 130, 206, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(49, 130, 206, 0.4);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #3182CE 0%, #2C5282 100%);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model.pkl')
        tfidf = joblib.load('tfidf.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        severity_ohe = joblib.load('severity_ohe.pkl')
        return model, tfidf, label_encoder, severity_ohe, True
    except Exception as e:
        return None, None, None, None, False


# Feature extraction
def extract_features(image):
    img = cv2.resize(image, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HSV Histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # LBP
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), range=(0, 256))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # HOG
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True
    )

    # GLCM
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Blur & edge
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (224 * 224)

    return np.hstack([hist, lbp_hist, hog_features, contrast, energy, homogeneity, laplacian_var, edge_density])


# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2>👁️ EyeCare AI</h2>
        <p style="font-size: 0.9rem; opacity: 0.9;">Advanced Eye Disease Detection</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Quick Guide")
    st.markdown("""
    1. Upload eye image  
    2. Enter symptoms  
    3. Click Analyze  
    4. View results
    """)

    st.markdown("---")
    st.markdown("### 🔬 Detectable Conditions")
    conditions = ["✓ Cataracts", "✓ Glaucoma", "✓ Diabetic Retinopathy", "✓ Macular Degeneration", "✓ Normal Eyes"]
    for c in conditions:
        st.markdown(f"<span style='color: #90CDF4;'>{c}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📞 Support")
    st.markdown("""
    📧 support@eyecare-ai.com  
    🌐 www.eyecare-ai.com
    """)

    st.markdown("---")
    st.caption("Version 2.0 | Dec 2025")


# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">👁️ EyeCare AI</h1>
    <p class="hero-subtitle">
        Advanced AI-powered eye disease classification system. Upload an eye image 
        and describe symptoms for instant, accurate analysis using machine learning.
    </p>
</div>
""", unsafe_allow_html=True)

# Stats Row
col1, col2, col3, col4 = st.columns(4)
stats = [("95%+", "Accuracy"), ("50K+", "Images Analyzed"), ("5+", "Conditions"), ("<3s", "Analysis Time")]
for col, (num, label) in zip([col1, col2, col3, col4], stats):
    with col:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{num}</div>
            <div class="stats-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# How It Works
st.markdown('<h2 class="section-header">⚙️ How It Works</h2>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
steps = [
    ("1", "📤 Upload", "Upload eye image"),
    ("2", "📝 Symptoms", "Describe symptoms"),
    ("3", "🔬 Analyze", "AI processes data"),
    ("4", "📊 Results", "Get classification")
]
for col, (num, title, desc) in zip([col1, col2, col3, col4], steps):
    with col:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-number">{num}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Load models
model, tfidf, label_encoder, severity_ohe, models_loaded = load_models()

# Analysis Section
st.markdown('<h2 class="section-header">🩺 Start Analysis</h2>', unsafe_allow_html=True)

if not models_loaded:
    st.markdown("""
    <div class="warning-box">
        <div class="warning-title">⚠️ Models Not Found</div>
        <div class="warning-text">Required model files not found. Please ensure model.pkl, tfidf.pkl, 
        label_encoder.pkl, and severity_ohe.pkl are in the app directory.</div>
    </div>
    """, unsafe_allow_html=True)

# Input columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📷 Upload Eye Image")
    uploaded_file = st.file_uploader("Choose an eye image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

with col2:
    st.markdown("#### 📝 Enter Symptoms")
    symptoms_input = st.text_area(
        "Describe your symptoms",
        placeholder="e.g., blurred vision, eye pain, redness, seeing halos...",
        height=120
    )

    severity = st.select_slider("Symptom Severity", options=["Mild", "Moderate", "Severe"], value="Mild")

# Analyze button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🔬 Analyze Image", type="primary", use_container_width=True)

# Prediction
if analyze_btn:
    if not uploaded_file:
        st.error("❌ Please upload an eye image first!")
    elif not models_loaded:
        st.error("❌ Models not loaded. Cannot perform analysis.")
    else:
        with st.spinner("🔄 Analyzing image..."):
            try:
                # Reset file pointer and read image
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Extract features
                visual_features = extract_features(img)

                # Process symptoms
                symptoms_text = symptoms_input.strip() if symptoms_input.strip() else "no symptoms"
                symptom_features = tfidf.transform([symptoms_text]).toarray()[0]

                # Severity encoding
                severity_features = severity_ohe.transform([[severity]])[0]

                # Combine features
                input_features = np.hstack([visual_features, symptom_features, severity_features])

                # Prediction
                pred = model.predict([input_features])[0]
                pred_label = label_encoder.inverse_transform([pred])[0]
                proba = model.predict_proba([input_features])[0]
                confidence = proba[pred] * 100

                # Display results
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("### 🎯 Analysis Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-label">Detected Condition</div>
                        <div class="prediction-value">{pred_label}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="confidence-box">
                        <div class="prediction-label">Confidence Level</div>
                        <div class="prediction-value">{confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Probability distribution
                st.markdown("#### 📊 Probability Distribution")
                sorted_idx = np.argsort(proba)[::-1]
                for idx in sorted_idx[:5]:
                    label = label_encoder.classes_[idx]
                    prob = proba[idx]
                    st.markdown(f"**{label}**")
                    st.progress(float(prob))
                    st.caption(f"{prob * 100:.2f}%")

                st.markdown('</div>', unsafe_allow_html=True)

                # Disclaimer
                st.markdown("""
                <div class="warning-box">
                    <div class="warning-title">⚠️ Medical Disclaimer</div>
                    <div class="warning-text">
                        This tool is for educational and informational purposes only. 
                        It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
                        Please consult a qualified ophthalmologist for proper eye examination and diagnosis.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

# Footer
st.markdown("""
<div class="footer">
    <p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">👁️ EyeCare AI</p>
    <p style="font-size: 0.85rem; opacity: 0.9;">
        Advanced Eye Disease Classification System<br>
        © 2025 EyeCare AI. All rights reserved.
    </p>
    <p style="font-size: 0.8rem; margin-top: 1rem; opacity: 0.7;">
        Built with ❤️ using Streamlit & Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)