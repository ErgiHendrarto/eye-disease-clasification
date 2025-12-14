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
    layout="centered"
)

@st.cache_resource
def load_models():
    try:
        # Load model utama (Random Forest)
        model = joblib.load('model.pkl')
        
        # Load TFIDF untuk symptoms (jika model masih memerlukannya)
        tfidf = joblib.load('tfidf.pkl')
        
        # Load Label Encoder
        label_encoder = joblib.load('label_encoder.pkl')
        
        # severity_ohe SUDAH DIHAPUS
        
        return model, tfidf, label_encoder
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
model, tfidf, label_encoder = load_models()

# UI
st.title("👁️ Eye Disease Classification")
st.markdown("### Upload an eye image to detect potential diseases")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an eye image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear image of an eye"
)

if uploaded_file:
    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                 caption="Uploaded Image", 
                 use_container_width=True)
    
    # Predict button
    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                # Extract visual features
                visual_features = extract_features(img)
                
                # --- LOGIC UPDATE: Hapus Severity ---
                
                # Default symptoms
                default_symptoms = ""
                symptom_features = tfidf.transform([default_symptoms]).toarray()[0]
                
                # Combine features (Visual + Symptoms saja)
                # Pastikan urutan ini sesuai dengan saat training Random Forest
                input_features = np.hstack([
                    visual_features,
                    symptom_features
                ])
                
                # Predict
                pred = model.predict([input_features])[0]
                pred_label = label_encoder.inverse_transform([pred])[0]
                proba = model.predict_proba([input_features])[0]
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Main prediction
                st.markdown("---")
                st.markdown("### 🎯 Detected Condition")
                
                confidence = proba[pred] * 100
                
                # Color based on prediction
                if pred_label == "Normal":
                    color = "green"
                elif confidence > 70:
                    color = "red"
                elif confidence > 50:
                    color = "orange"
                else:
                    color = "blue"
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 30px; 
                            border-radius: 15px; 
                            text-align: center;
                            box-shadow: 0 10px 25px rgba(0,0,0,0.1);'>
                    <h1 style='color: white; margin: 0; font-size: 3em;'>{pred_label}</h1>
                    <h2 style='color: #f0f0f0; margin-top: 10px; font-size: 2em;'>{confidence:.1f}%</h2>
                    <p style='color: #e0e0e0; margin-top: 10px;'>Confidence Level</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("---")
                st.markdown("### 📊 Probability Distribution")
                
                # Sort by probability
                sorted_indices = np.argsort(proba)[::-1]
                
                for idx in sorted_indices:
                    disease = label_encoder.classes_[idx]
                    prob = proba[idx] * 100
                    
                    st.write(f"**{disease}**")
                    st.progress(proba[idx])
                    st.caption(f"{prob:.2f}%")
                
                # Medical disclaimer
                st.markdown("---")
                st.warning("""
                    **⚠️ Medical Disclaimer:** This is an AI-powered diagnostic tool for reference purposes only. 
                    Please consult with a qualified ophthalmologist for proper diagnosis and treatment.
                """)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("Please ensure your model is trained without severity input.")

# Sidebar info
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info("""
        This app uses machine learning to classify eye diseases from images.
        
        **Detectable Conditions:**
        - Normal
        - Cataract
        - Conjunctivitis
        - Uveitis
        - Eyelid disorders
    """)
    
    st.markdown("### 📌 Tips")
    st.success("""
        - Use clear, well-lit images
        - Focus on the eye area
        - Avoid blurry photos
        - Front-facing images work best
    """)
    
    st.markdown("---")
    # UPDATED: Menampilkan Random Forest
    st.caption("Powered by Random Forest")