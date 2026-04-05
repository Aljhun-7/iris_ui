import streamlit as st
import pandas as pd

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
