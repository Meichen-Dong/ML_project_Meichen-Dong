import streamlit as st
import joblib
import pandas as pd
import numpy as np
import math
import sys 

st.set_page_config(
    page_title="Diamond price prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUT_MAPPING = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
COLOR_MAPPING = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
CLARITY_MAPPING = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

model_xgb = joblib.load('XGBoost_model.joblib')

# --- Page user can see---

st.title("Welcome To Predict The diamond price")
st.markdown('The result is based on the XGBoost model') 

st.sidebar.header("Input diamond features")

carat = st.sidebar.slider("Carat", min_value=0.2, max_value=3.65, value=0.7, step=0.01)
depth = st.sidebar.slider("Depth % (= Height / average diameter)", min_value=43.0, max_value=79.0, value=61.8, step=0.1)
table = st.sidebar.slider("Table %(= table width / maximum diameter)", min_value=43.0, max_value=95.0, value=57.0, step=0.1)
cut_str = st.sidebar.selectbox("Cut", options=list(CUT_MAPPING.keys()), index=0)
color_str = st.sidebar.selectbox("Color", options=list(COLOR_MAPPING.keys()), index=1)
clarity_str = st.sidebar.selectbox("Clarity", options=list(CLARITY_MAPPING.keys()), index=3)
x = st.sidebar.slider("Width (X) mm", min_value=0.0, max_value=10.74, value=5.7, step=0.01)
y = st.sidebar.slider("Length (Y) mm", min_value=0.0, max_value=58.9, value=5.7, step=0.01)
z = st.sidebar.slider("Height (Z) mm", min_value=0.0, max_value=31.8, value=3.5, step=0.01)

cut = CUT_MAPPING[cut_str]
color = COLOR_MAPPING[color_str]
clarity = CLARITY_MAPPING[clarity_str]

volume = x * y * z
density = carat / volume if volume != 0 else 0.0
xy_ratio = x / y if y != 0 else 0.0

input_data = pd.DataFrame({
    'carat': [carat],
    'cut': [cut],
    'color': [color],
    'clarity': [clarity],
    'depth': [depth],
    'table': [table],
    'x': [x],
    'y': [y],
    'z': [z],
    'volume': [volume],
    'density': [density],
    'xy_ratio': [xy_ratio]
})

# --- result ---
st.subheader("The diamond feature you entered")
st.dataframe(input_data)
st.markdown("---")

if st.button("Click me to predict"):
    log_price_pred = model_xgb.predict(input_data)[0]
    price_pred = np.exp(log_price_pred)

    st.subheader("Prediction results")
    st.balloons()
    st.write(f"## **${price_pred:,.2f}**")
    st.caption(f"---")
    st.info(f"Note: The model predicts logarithmic prices, then convert back to the actual price。")