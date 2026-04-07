import streamlit as st
import joblib
import numpy as np

# page configuration
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide"
)

# custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# model loading
@st.cache_resource
def load_my_model():
    # Ensure this file exists in your directory!
    return joblib.load('iris_model.joblib')

try:
    model = load_my_model()
except FileNotFoundError:
    st.error("⚠️ Model file not found. Please run your training script first!")
    st.stop()

# main dashboard
st.title("🌸 Iris Flower Classification")
st.markdown("Aljhun R. Angala | ")

col1, col2 = st.columns([2, 1])

# flower prediction side
with col1:
    sepal_l = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_w = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_l = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_w = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
    
    predict_btn = st.button("🚀 Predict Species")

# species result side
with col2:
    st.subheader("🎯 Prediction Result")
    if predict_btn:
        features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
        prediction = model.predict(features)[0]
        
        # Map numerical predictions to names if necessary
        species_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        result = species_names.get(prediction, prediction)

        st.markdown(f"""
            <div class="prediction-card">
                <h3>Identified Species:</h3>
                <h1 style="color: #ff4b4b;">{result}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        st.balloons()
    else:
        st.write("Click 'Predict Species' to see results.")


