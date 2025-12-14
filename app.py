import streamlit as st
import pickle
import joblib
import numpy as np
import cv2
from pathlib import Path
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog

@st.cache_resource
def load_all():
    try:
        # Load dengan joblib (lebih robust daripada pickle)
        model = joblib.load('model.pkl')
        tfidf = joblib.load('tfidf.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        severity_ohe = joblib.load('severity_ohe.pkl')
        
        st.success("✅ Models loaded successfully!")
        return model, tfidf, label_encoder, severity_ohe
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("📝 Make sure all .pkl files are uploaded with Git LFS")
        st.stop()

def extract_features(image):
    """Extract features dari gambar (sama seperti di training)"""
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
model, tfidf, label_encoder, severity_ohe = load_all()

# Streamlit UI
st.title("🏥 Eye Disease Classification")
st.write("Upload gambar mata dan input gejala untuk diagnosis")

# Upload image
uploaded_file = st.file_uploader("Upload Eye Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image")
    
    # Symptoms input
    st.subheader("Input Symptoms")
    symptoms_input = st.text_area(
        "Enter symptoms (comma separated)",
        placeholder="e.g., pain, blurred vision, redness"
    )
    
    # Severity input
    severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe", "None"])
    
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            try:
                # Extract visual features
                visual_features = extract_features(img)
                
                # Encode symptoms
                symptoms_list = [s.strip() for s in symptoms_input.split(",")]
                symptoms_text = " ".join(symptoms_list)
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
                st.success(f"**Predicted Disease:** {pred_label}")
                st.write("**Probability Distribution:**")
                
                for i, label in enumerate(label_encoder.classes_):
                    st.progress(proba[i], text=f"{label}: {proba[i]:.2%}")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")