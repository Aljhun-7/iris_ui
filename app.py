import streamlit as st
import joblib
import numpy as np

# ✨ Use cache so the model stays in RAM and doesn't reload on every click
@st.cache_resource
def load_my_model():
    return joblib.load('iris_model.joblib')

# Load the static model
model = load_my_model()

st.title("🌸 Iris Predictor")
st.write("Enter values to get a prediction without retraining!")

# Create inputs for the user
sepal_l = st.number_input("Sepal Length", value=5.1)
sepal_w = st.number_input("Sepal Width", value=3.5)
petal_l = st.number_input("Petal Length", value=1.4)
petal_w = st.number_input("Petal Width", value=0.2)

if st.button("Predict"):
    # Format the input for the model
    features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    
    # Make the prediction
    prediction = model.predict(features)
    
    st.success(f"The predicted class is: {prediction[0]}")
