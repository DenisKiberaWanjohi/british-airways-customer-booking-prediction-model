# app.py
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

def desc(schema: FeatureSchema, col: str) -> str:
    return getattr(schema, "descriptions", {}).get(col, "")

def hint(schema: FeatureSchema, col: str) -> dict:
    return getattr(schema, "ui_hints", {}).get(col, {})

if not PIPELINE_PATH.exists() or not FEATURE_INFO_PATH.exists():
    st.info("Model artifacts not found. Train the model first by running `python -m src.train_pipeline`.")
else:
    pipe = load_pipeline(PIPELINE_PATH)
    schema = FeatureSchema.load(FEATURE_INFO_PATH)

    numeric = list(schema.numeric)
    categorical = list(schema.categorical)

    tab1, tab2 = st.tabs(["Single prediction", "Batch from CSV"])

    with tab1:
        st.write("Enter feature values. Hover the ⓘ icon for guidance on each field.")
        left, right = st.columns(2)
        inputs = {}

        with left:
            # numeric and binary
            for col in numeric:
                h = hint(schema, col)
                kind = h.get("kind", "float")
                if col == "flight_hour" or (kind == "int" and h.get("min") == 0 and h.get("max") == 23):
                    val = st.slider(col, min_value=0, max_value=23, value=0, help=desc(schema, col))
                    inputs[col] = int(val)
                elif kind == "int":
                    val = st.number_input(col, value=0, step=1, help=desc(schema, col))
                    inputs[col] = int(val)
                elif kind == "binary":
                    val = st.checkbox(col + " (check for Yes)", value=False, help=desc(schema, col))
                    inputs[col] = 1 if val else 0
                else:
                    val = st.number_input(col, value=0.0, step=0.1, help=desc(schema, col))
                    inputs[col] = float(val)

        with right:
            # categoricals as dropdowns
            for col in categorical:
                opts = getattr(schema, "categorical_options", {}).get(col, [])
                if len(opts) == 0:
                    # fallback to text when no options are stored
                    inputs[col] = st.text_input(col, value="", help=desc(schema, col))
                else:
                    # ensure string options and sorted for stability
                    opts = sorted([str(x) for x in opts])
                    inputs[col] = st.selectbox(col, options=opts, help=desc(schema, col))

        if st.button("Predict"):
            X = prepare_inference_df(inputs, numeric, categorical)
            pred = int(pipe.predict(X)[0])
            proba = getattr(pipe, "predict_proba", lambda X: None)(X)

            st.subheader("Prediction")
            if len(schema.class_labels) == 2 and proba is not None:
                pos_index = schema.class_labels.index(1) if 1 in schema.class_labels else 1
                pos_prob = float(proba[0][pos_index])
                message = "The customer is likely to complete the booking." if pred == 1 else "The customer is unlikely to complete the booking."
                st.markdown(f"**{message}**  (Estimated probability of completion: {pos_prob:.2%})")
            else:
                st.markdown("**Predicted class:** " + str(pred))

            if proba is not None:
                probs = proba[0]
                labels = [str(c) for c in schema.class_labels]
                fig = plt.figure()
                plt.pie(probs, labels=labels, autopct="%1.1f%%", startangle=90)
                plt.title("Prediction probabilities")
                plt.axis("equal")
                st.pyplot(fig)

    with tab2:
        st.write("Upload a CSV with the same columns used in training, excluding the target.")
        f = st.file_uploader("CSV file", type=["csv"])
        if f:
            df = pd.read_csv(f)
            df = enforce_types(df, numeric, categorical)
            st.dataframe(df.head())

            preds = pipe.predict(df)
            out = df.copy()
            out["prediction"] = preds

            st.subheader("Preview of predictions")
            st.dataframe(out.head(50))

            st.subheader("Prediction distribution")
            counts = pd.Series(preds).value_counts().sort_index()
            fig2 = plt.figure()
            plt.pie(counts.values, labels=[str(k) for k in counts.index], autopct="%1.1f%%", startangle=90)
            plt.title("Predicted class share")
            plt.axis("equal")
            st.pyplot(fig2)

            st.download_button("Download predictions", data=out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")