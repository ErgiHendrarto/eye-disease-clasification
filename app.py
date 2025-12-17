import streamlit as st
import joblib
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog

# Page config
st.set_page_config(
    page_title="Eye Disease Classification",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS to enhance UI
st.markdown("""
    <style>
        body {
            background-color: white;
            color: #333333;
        }
        .stButton > button {
            background-color: #667eea;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            padding: 12px 24px;
            border: none;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton > button:hover {
            background-color: #4f65c1;
        }
        .stTitle {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            color: #4e4e4e;
        }
        .stMarkdown h2 {
            font-size: 2rem;
            color: #667eea;
            font-weight: 600;
        }
        .stMarkdown h1 {
            color: #4e4e4e;
        }
        .stTextInput textarea {
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
            width: 100%;
            margin-bottom: 20px;
            border: 1px solid #ccc;
        }
        .stImage {
            border-radius: 10px;
        }
        .stSidebar {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 12px;
        }
        .stProgress > div {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

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
    img = cv2.resize(image, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HSV Histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
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
    glcm = graycomatrix(gray, distances=[5], angles=[5],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Blur & edge
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (224 * 224)

    return np.hstack([
        hist,
        lbp_hist,
        hog_features,
        contrast,
        energy,
        homogeneity,
        laplacian_var,
        edge_density
    ])

# Load models
model, tfidf, label_encoder, severity_ohe = load_models()

# UI
st.title("👁️ Eye Disease Classification")
st.markdown("### Upload an eye image and enter symptoms")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an eye image...",
    type=['jpg', 'jpeg', 'png']
)

# Symptoms input
symptoms_input = st.text_area(
    "Enter symptoms (comma separated)",
    placeholder="e.g., blurred vision, eye pain, redness",
    height=100
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption="Uploaded Image",
                 use_container_width=True)

    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                # Image features
                visual_features = extract_features(img)

                # Symptoms → TF-IDF
                symptoms_text = symptoms_input.strip()
                symptom_features = tfidf.transform([symptoms_text]).toarray()[0]

                # Severity DEFAULT (hidden from UI)
                default_severity = "Mild"
                severity_features = severity_ohe.transform([[default_severity]])[0]

                # Combine features
                input_features = np.hstack([
                    visual_features,
                    symptom_features,
                    severity_features
                ])

                # Prediction
                pred = model.predict([input_features])[0]
                pred_label = label_encoder.inverse_transform([pred])[0]
                proba = model.predict_proba([input_features])[0]
                confidence = proba[pred] * 100

                st.success("✅ Analysis Complete!")
                st.markdown("### 🎯 Detected Condition")

                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 30px;
                            border-radius: 15px;
                            text-align: center;'>
                    <h1 style='color: white; font-size: 3em;'>{pred_label}</h1>
                    <h2 style='color: #f0f0f0;'>{confidence:.1f}%</h2>
                    <p style='color: #e0e0e0;'>Confidence Level</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📊 Probability Distribution")
                sorted_idx = np.argsort(proba)[::-1]
                for idx in sorted_idx:
                    st.write(f"**{label_encoder.classes_[idx]}**")
                    st.progress(proba[idx])
                    st.caption(f"{proba[idx]*100:.2f}%")

                st.warning(""" ⚠️ Medical Disclaimer: This tool is for educational purposes only. Please consult an ophthalmologist for diagnosis. """)

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info("""
    Eye disease classification using image features and symptoms.
    
    **Model:** Random Forest  
    **Input:** Eye image + symptoms
    """)
