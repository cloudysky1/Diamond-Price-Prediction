import streamlit as st
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Diamond Price Prediction", layout="centered")

st.title("💎 Diamond Price Prediction App")

st.write("Enter the details of the diamond to predict its price:")

# --- Form for inputs ---
with st.form("diamond_form"):
    carat = st.number_input("Carat", min_value=0.0, format="%.2f")
    depth = st.number_input("Depth", min_value=0.0, format="%.2f")
    table = st.number_input("Table", min_value=0.0, format="%.2f")
    x = st.number_input("Length (x)", min_value=0.0, format="%.2f")
    y = st.number_input("Width (y)", min_value=0.0, format="%.2f")
    z = st.number_input("Height (z)", min_value=0.0, format="%.2f")

    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

    submitted = st.form_submit_button("Predict Price")

# --- Prediction ---
if submitted:
    data = CustomData(
        carat=carat,
        depth=depth,
        table=table,
        x=x,
        y=y,
        z=z,
        cut=cut,
        color=color,
        clarity=clarity
    )
    final_new_data = data.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(final_new_data)

    result = round(pred[0], 2)

    st.success(f"💰 Predicted Diamond Price: ${result}")
