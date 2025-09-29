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
    return getattr(schema, "descriptions", {}).get(col, "")

if not PIPELINE_PATH.exists() or not FEATURE_INFO_PATH.exists():
    st.info("Model artifacts not found. Train the model first by running `python -m src.train_pipeline`.")
else:
    pipe = load_pipeline(PIPELINE_PATH)
    schema = FeatureSchema.load(FEATURE_INFO_PATH)

    numeric_cols = list(schema.numeric)
    categorical_cols = list(schema.categorical)

    tab1, tab2 = st.tabs(["Single prediction", "Batch from CSV"])

    with tab1:
        st.write("Enter feature values. Hover the ⓘ icon for guidance on each field.")
        left, right = st.columns(2)
        inputs = {}

        with left:
            for col in numeric_cols:
                inputs[col] = st.number_input(
                    label=col,
                    value=0.0,
                    step=1.0,
                    help=field_help(schema, col),
                )
        with right:
            for col in categorical_cols:
                opts = getattr(schema, "categorical_options", {}).get(col, [])
                desc = field_help(schema, col)
                if isinstance(opts, list) and len(opts) > 0:
                    inputs[col] = st.selectbox(col, options=opts, help=desc)
                else:
                    inputs[col] = st.text_input(col, value="", help=desc)

        if st.button("Predict"):
            X = prepare_inference_df(inputs, numeric_cols, categorical_cols)
            pred = int(pipe.predict(X)[0])
            proba = getattr(pipe, "predict_proba", lambda X: None)(X)

            st.subheader("Prediction")
            if len(schema.class_labels) == 2 and proba is not None:
                pos_index = schema.class_labels.index(1) if 1 in schema.class_labels else 1
                pos_prob = float(proba[0][pos_index])
                message = "The customer is likely to complete the booking." if pred == 1 else "The customer is unlikely to complete the booking."
                st.markdown(f"**{message}**  (estimated probability of completion: {pos_prob:.2%})")
            else:
                st.markdown("**Predicted class:** " + str(pred))

            if proba is not None:
                probs = proba[0]
                labels = [str(c) for c in schema.class_labels]
                fig = plt.figure()
                plt.pie(probs, labels=labels, autopct='%1.1f%%', startangle=90)
                plt.title("Prediction probabilities")
                plt.axis('equal')
                st.pyplot(fig)

    with tab2:
        st.write("Upload a CSV with the same columns used in training, excluding the target.")
        f = st.file_uploader("CSV file", type=["csv"])
        if f:
            df = pd.read_csv(f)
            df = enforce_types(df, numeric_cols, categorical_cols)
            st.dataframe(df.head())

            preds = pipe.predict(df)
            out = df.copy()
            out["prediction"] = preds

            st.subheader("Preview of predictions")
            st.dataframe(out.head(50))

            st.subheader("Prediction distribution")
            counts = pd.Series(preds).value_counts().sort_index()
            sizes = counts.values
            labels = [str(k) for k in counts.index]
            fig2 = plt.figure()
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title("Predicted class share")
            plt.axis('equal')
            st.pyplot(fig2)

            st.download_button(
                "Download predictions",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
            )