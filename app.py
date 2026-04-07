import streamlit as st
import pandas as pd

# --- USER INPUT ---
st.sidebar.header("Input Features")

sepal_length = st.number_input("Sepal Length", 4.0, 8.0)
sepal_width  = st.number_input("Sepal Width", 2.0, 5.0)
petal_length = st.number_input("Petal Length", 1.0, 7.0)
petal_width  = st.number_input("Petal Width", 0.1, 2.5)

# --- PREDICTION ---
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=X.columns)

prediction = model.predict(input_data)

predictBtn = st.button("Predict")

if predictBtn:
    st.write("### Prediction:")
    st.write(prediction[0])
    st.success("Successfully Predict")
