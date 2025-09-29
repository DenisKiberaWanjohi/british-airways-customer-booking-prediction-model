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

st.set_page_config(page_title="BA Booking Prediction", page_icon="🧠", layout="wide")
st.title("British Airways Booking Prediction")

def field_help(schema: FeatureSchema, col: str):
    return schema.descriptions.get(col, "") if hasattr(schema, "descriptions") else ""

if not PIPELINE_PATH.exists() or not FEATURE_INFO_PATH.exists():
    st.info("Model artifacts not found. Train the model first by running `python -m src.train_pipeline`.")
else:
    pipe = load_pipeline(PIPELINE_PATH)
    schema = FeatureSchema.load(FEATURE_INFO_PATH)

    tab1, tab2 = st.tabs(["Single prediction", "Batch from CSV"])

    with tab1:
        st.write("Enter feature values. The form uses the saved schema and shows friendly descriptions.")
        col_left, col_right = st.columns(2)
        inputs = {}

        with col_left:
            for col in schema.numeric:
                inputs[col] = st.number_input(
                    label=col,
                    value=0.0,
                    step=1.0,
                    help=field_help(schema, col),
                )
        with col_right:
            for col in schema.categorical:
                opts = getattr(schema, "categorical_options", {}).get(col)
                desc = field_help(schema, col)
                if opts and len(opts) > 0:
                    inputs[col] = st.selectbox(col, options=opts, help=desc)
                else:
                    inputs[col] = st.text_input(col, value="", help=desc)

        if st.button("Predict"):
            X = prepare_inference_df(inputs, schema.numeric, schema.categorical)
            pred = pipe.predict(X)[0]
            proba = getattr(pipe, "predict_proba", lambda X: None)(X)

            st.subheader("Prediction")
            st.write({"prediction": int(pred)})
            if proba is not None:
                probs = proba[0]
                st.write({"probabilities": probs.tolist()})
                fig = plt.figure()
                x = [str(c) for c in schema.class_labels]
                plt.bar(x, probs)
                plt.title("Class probabilities")
                plt.xlabel("class")
                plt.ylabel("probability")
                st.pyplot(fig)

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

            st.subheader("Preview of predictions")
            st.dataframe(out.head(50))

            st.subheader("Prediction distribution")
            counts = pd.Series(preds).value_counts().sort_index()
            fig2 = plt.figure()
            counts.plot(kind="bar")
            plt.title("Predicted class counts")
            plt.xlabel("class")
            plt.ylabel("count")
            st.pyplot(fig2)

            st.download_button(
                "Download predictions",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
            )