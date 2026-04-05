import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.title("CSV File Uploader")

# 1. Create the file uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # 2. Read the file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # 3. Display the DataFrame
    st.subheader("Data Preview")
    st.write(df)
else:
    st.info("Please upload a CSV file to begin.")
# --- PREPARE DATA ---
# Example assumes Iris dataset format
X = df.drop(columns=["Id","Species"])   # features
y = df["Species"]                 # target

# --- TRAIN MODEL ---
model = DecisionTreeClassifier()
model.fit(X, y)

st.success("Model trained successfully!")

# --- USER INPUT ---
st.sidebar.header("Input Features")

sepal_length = st.sidebar.number_input("Sepal Length", 4.0, 8.0)
sepal_width  = st.sidebar.number_input("Sepal Width", 2.0, 5.0)
petal_length = st.sidebar.number_input("Petal Length", 1.0, 7.0)
petal_width  = st.sidebar.number_input("Petal Width", 0.1, 2.5)

# --- PREDICTION ---
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=X.columns)

prediction = model.predict(input_data)

predictBtn = st.sidebar.button("Predict")

if predictBtn:
    st.write("### Prediction:")
    st.write(prediction[0])
