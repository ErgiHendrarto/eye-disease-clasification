import streamlit as st
import joblib
import numpy as np
import cv2
from pathlib import Path
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog

# Page config
st.set_page_config(
    page_title="Eye Disease Classification",
    page_icon="👁️",
    layout="wide"
)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('model.pkl')
        tfidf = joblib.load('tfidf.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        severity_ohe = joblib.load('severity_ohe.pkl')
        
        return model, tfidf, label_encoder, severity_ohe
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

def extract_features(image):
    """Extract visual features from image"""
    img = cv2.resize(image, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Color Histogram (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # LBP
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(257), 
                                  range=(0, 256))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    # HOG
    hog_features = hog(
        gray, orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True
    )
    
    # GLCM
    glcm = graycomatrix(
        gray,
        distances=[5],
        angles=[5],
        levels=256,
        symmetric=True,
        normed=True
    )
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # BLUR + EDGE
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (224 * 224)
    
    # Concatenate
    features = np.hstack([
        hist,
        lbp_hist,
        hog_features,
        contrast,
        energy,
        homogeneity,
        laplacian_var,
        edge_density
    ])
    
    return features

# Load models
model, tfidf, label_encoder, severity_ohe = load_models()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
    .prediction-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .prediction-confidence {
        font-size: 2rem;
        margin-top: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">👁️ Eye Disease Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Klasifikasi penyakit mata menggunakan Random Forest. Kami fokus pada feature engineering citra fundus untuk solusi skrining oftalmologis yang ringan dan andal. Memfasilitasi diagnosis cepat dan mendukung pengambilan keputusan klinis.</div>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 Upload Eye Image")
    uploaded_file = st.file_uploader(
        "Choose an eye image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of an eye"
    )
    
    if uploaded_file:
        # Read and display image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                 caption="Uploaded Image", 
                 use_container_width=True)

with col2:
    st.markdown("### 📝 Input Symptoms")
    
    # Symptoms input
    st.markdown("#### Symptoms")
    symptoms_input = st.text_area(
        "Enter symptoms (comma separated)",
        placeholder="e.g., eye pain, blurred vision, redness, sensitivity to light",
        help="Enter multiple symptoms separated by commas",
        height=100
    )
    
    
    # Info boxes
    st.info("💡 **Tip:** Be as specific as possible with symptoms for better accuracy")

# Predict button
st.markdown("---")

if uploaded_file:
    if st.button("🔍 Analyze & Predict Disease", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing image and processing data..."):
            try:
                # Extract visual features
                visual_features = extract_features(img)
                
                # Encode symptoms
                if symptoms_input.strip():
                    symptoms_list = [s.strip() for s in symptoms_input.split(",")]
                    symptoms_text = " ".join(symptoms_list)
                else:
                    symptoms_text = ""
                
                symptom_features = tfidf.transform([symptoms_text]).toarray()[0]
                
                # Encode severity
                severity_features = severity_ohe.transform([[severity]])[0]
                
                # Combine all features
                input_features = np.hstack([
                    visual_features,
                    symptom_features,
                    severity_features
                ])
                
                # Predict
                pred = model.predict([input_features])[0]
                pred_label = label_encoder.inverse_transform([pred])[0]
                proba = model.predict_proba([input_features])[0]
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Main prediction with custom styling
                confidence = proba[pred] * 100
                
                st.markdown(f"""
                <div class="prediction-box">
                    <p style='font-size: 1.2rem; margin: 0; opacity: 0.9;'>Predicted Disease</p>
                    <h1 class="prediction-title">{pred_label}</h1>
                    <h2 class="prediction-confidence">{confidence:.1f}%</h2>
                    <p style='margin: 0; opacity: 0.9;'>Confidence Level</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed results in columns
                st.markdown("---")
                st.markdown("### 📊 Detailed Analysis")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("#### 🎯 Input Summary")
                    st.write(f"**Image:** Uploaded ✓")
                    st.write(f"**Symptoms:** {symptoms_input if symptoms_input else 'None provided'}")
                
                with col_b:
                    st.markdown("#### 📈 Top 3 Predictions")
                    sorted_indices = np.argsort(proba)[::-1][:3]
                    
                    for idx in sorted_indices:
                        disease = label_encoder.classes_[idx]
                        prob = proba[idx] * 100
                        st.write(f"**{disease}**: {prob:.2f}%")
                
                # Probability distribution
                st.markdown("---")
                st.markdown("### 📊 Complete Probability Distribution")
                
                # Sort by probability
                sorted_indices = np.argsort(proba)[::-1]
                
                for idx in sorted_indices:
                    disease = label_encoder.classes_[idx]
                    prob = proba[idx]
                    
                    col_label, col_bar = st.columns([1, 4])
                    with col_label:
                        st.write(f"**{disease}**")
                    with col_bar:
                        st.progress(prob)
                        st.caption(f"{prob * 100:.2f}%")
                
                # Medical disclaimer
                st.markdown("---")
                st.warning("""
                    **⚠️ Medical Disclaimer:** 
                    This is an AI-powered diagnostic tool for reference and educational purposes only. 
                    It is NOT a substitute for professional medical advice, diagnosis, or treatment.
                    
                    **Always consult** with a qualified ophthalmologist for proper diagnosis and treatment 
                    of any eye condition. Do not rely solely on this tool for medical decisions.
                """)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("Please ensure the image is clear and properly formatted, and check your inputs.")
else:
    st.info("👆 Please upload an eye image to begin analysis")

# Sidebar info
with st.sidebar:
    st.markdown("### ℹ️ About This App")
    st.info("""
        This application uses a **multimodal machine learning approach** to classify eye diseases.
        
        **Model:** Random Forest Classifier (Optimized)
        
        **Features:**
        - Visual: HOG, LBP, GLCM, Color Histogram
        - Textual: TF-IDF encoded symptoms
        - Categorical: One-hot encoded severity
        
        **Accuracy:** ~95% (Validation)
    """)
    
    st.markdown("### 🏥 Detectable Conditions")
    st.success("""
        - **Normal** - Healthy eye
        - **Cataract** - Katarak
        - **Conjunctivitis** - Mata merah
        - **Uveitis** - Peradangan uvea
        - **Eyelid Disorders** - Kelainan kelopak mata
    """)
    
    st.markdown("### 📌 Usage Tips")
    st.info("""
        **For Best Results:**
        - Use clear, well-lit images
        - Focus on the eye area
        - Avoid blurry or dark photos
        - Provide detailed symptoms
    """)
    
    st.markdown("---")
    st.caption("Powered by Random Forest & Computer Vision")
    st.caption("© 2024 Eye Disease Classification System")