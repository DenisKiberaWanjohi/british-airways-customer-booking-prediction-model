import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.io import load_pipeline
from src.schema import FeatureSchema
from src.features import prepare_inference_df, enforce_types

MODEL_DIR = Path("models")
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"
FEATURE_INFO_PATH = MODEL_DIR / "feature_info.json"

st.set_page_config(page_title="BA Booking Prediction", page_icon="🧠")
st.title("British Airways Booking Prediction")

if not PIPELINE_PATH.exists() or not FEATURE_INFO_PATH.exists():
    st.info("Model artifacts not found. Train the model first by running `python src/train_pipeline.py`.")
else:
    pipe = load_pipeline(PIPELINE_PATH)
    schema = FeatureSchema.load(FEATURE_INFO_PATH)

    tab1, tab2 = st.tabs(["Single prediction", "Batch from CSV"])

    with tab1:
        st.write("Enter feature values. The form uses the saved schema.")
        inputs = {}
        for col in schema.numeric:
            inputs[col] = st.number_input(col, value=0.0)
        for col in schema.categorical:
            inputs[col] = st.text_input(col, value="")

        if st.button("Predict"):
            X = prepare_inference_df(inputs, schema.numeric, schema.categorical)
            pred = pipe.predict(X)[0]
            proba = getattr(pipe, "predict_proba", lambda X: None)(X)
            st.write({"prediction": int(pred)})
            if proba is not None:
                st.write({"probabilities": proba[0].tolist()})

    with tab2:
        st.write("Upload a CSV with the same columns used in training, excluding the target.")
        f = st.file_uploader("CSV file", type=["csv"])
        if f:
            df = pd.read_csv(f)
            df = enforce_types(df, schema.numeric, schema.categorical)
            st.dataframe(df.head())
            preds = pipe.predict(df)
            prob = getattr(pipe, "predict_proba", lambda X: None)(df)
            out = df.copy()
            out["prediction"] = preds
            if prob is not None and hasattr(prob, "shape"):
                if prob.shape[1] == 2:
                    out["prob_positive"] = prob[:, 1]
            st.dataframe(out.head(50))
            st.download_button("Download predictions", data=out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")
