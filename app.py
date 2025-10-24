# app.py
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.io import load_pipeline
from src.schema import FeatureSchema
from src.features import prepare_inference_df, enforce_types

# -------------------- Config --------------------
MODEL_DIR = Path("models")
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"
FEATURE_INFO_PATH = MODEL_DIR / "feature_info.json"

st.set_page_config(page_title="BA Booking Prediction", page_icon="🧠", layout="wide")
st.title("British Airways Booking Prediction")

# Style: blue Predict button
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #1f77b4;
        color: white;
    }
    div.stButton > button:first-child:hover {
        background-color: #135c91;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hard UI rules
INT_FIELDS = {"num_passengers", "purchase_lead", "length_of_stay", "flight_duration"}
BINARY_FIELDS = {"wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals"}

# -------------------- Helpers --------------------
def desc(schema: FeatureSchema, col: str) -> str:
    return getattr(schema, "descriptions", {}).get(col, "")

def load_artifacts():
    if not PIPELINE_PATH.exists() or not FEATURE_INFO_PATH.exists():
        return None, None
    pipe = load_pipeline(PIPELINE_PATH)
    schema = FeatureSchema.load(FEATURE_INFO_PATH)
    return pipe, schema

# Synonym mapping for headers
SYNONYMS = {
    "num passengers": "num_passengers",
    "passengers": "num_passengers",
    "sales channel": "sales_channel",
    "trip type": "trip_type",
    "purchase lead": "purchase_lead",
    "length of stay": "length_of_stay",
    "flight hour": "flight_hour",
    "flight day": "flight_day",
    "booking origin": "booking_origin",
    "wants extra baggage": "wants_extra_baggage",
    "wants preferred seat": "wants_preferred_seat",
    "wants in flight meals": "wants_in_flight_meals",
    "wants inflight meals": "wants_in_flight_meals",
    "flight duration": "flight_duration",
}

def _clean_names(cols):
    out = []
    for c in cols:
        c2 = str(c).strip()
        c2 = c2.replace("\ufeff", "")      # BOM
        c2 = c2.replace("-", " ")
        c2 = " ".join(c2.split())
        c2 = c2.lower().replace(" ", "_")
        c2 = SYNONYMS.get(c2, c2)
        out.append(c2)
    return out

def read_csv_flexible(file):
    """Try multiple encodings and header strategies."""
    tried = []
    for enc in ["utf-8", "utf-8-sig", "ISO-8859-1", "latin1"]:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc, header=0)
            df.columns = _clean_names(df.columns)
            return df
        except Exception as e:
            tried.append((enc, "header=0", str(e)))

        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc, header=None)
            df.columns = _clean_names(df.iloc[0].tolist())
            df = df.iloc[1:].reset_index(drop=True)
            return df
        except Exception as e:
            tried.append((enc, "header=None", str(e)))

    try:
        file.seek(0)
        df = pd.read_excel(file, header=0)
        df.columns = _clean_names(df.columns)
        return df
    except Exception:
        pass

    raise ValueError(f"Could not read the file. Tried: {tried}")

def validate_and_order(df, expected_cols):
    df = df.rename(columns={c: SYNONYMS.get(c, c) for c in df.columns})
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns are missing: {set(missing)}. Found columns: {list(df.columns)}")
    return df[[c for c in expected_cols]]

# -------------------- Load artifacts --------------------
pipe, schema = load_artifacts()
if pipe is None or schema is None:
    st.info("Artifacts not found. Train first with `python -m src.train_pipeline` then restart the app.")
    st.stop()

numeric_all = list(schema.numeric)
categorical_all = list(schema.categorical)
categorical_ui = [c for c in categorical_all if c not in BINARY_FIELDS]
numeric_ui = [n for n in numeric_all if n not in BINARY_FIELDS]

EXPECTED_COLS = numeric_all + categorical_all
class_labels = list(schema.class_labels)
label_map = {0: "Booking not likely to be Completed", 1: "Booking likely to be completed"}

tab1, tab2 = st.tabs(["Single prediction", "Batch from CSV"])

# ---------------- Single prediction ----------------
with tab1:
    st.write("Enter feature values. Hover the info icon for guidance on each field.")
    left, right = st.columns(2)
    inputs = {}

    with left:
        # Binary fields as explicit 0 or 1
        for col in sorted(BINARY_FIELDS):
            help_txt = desc(schema, col)
            yn = st.selectbox(col, options=["No", "Yes"], help=help_txt)
            inputs[col] = 1 if yn == "Yes" else 0

        # Categorical fields
        for col in categorical_ui:
            help_txt = desc(schema, col)
            opts = getattr(schema, "categorical_options", {}).get(col, [])
            if len(opts) > 0:
                # Keep original dtypes for the selected value
                options = sorted(opts, key=lambda x: str(x))
                inputs[col] = st.selectbox(
                    col,
                    options=options,
                    format_func=lambda x: str(x),
                    help=help_txt,
                )
            else:
                inputs[col] = st.text_input(col, value="", help=help_txt)

    with right:
        # Numeric fields with proper types
        for col in numeric_ui:
            help_txt = desc(schema, col)
            if col == "flight_hour":
                val = st.slider(col, min_value=0, max_value=23, value=0, help=help_txt)
                inputs[col] = int(val)
            elif col in INT_FIELDS:
                val = st.number_input(col, value=0, step=1, help=help_txt, format="%d")
                inputs[col] = int(val)
            else:
                val = st.number_input(col, value=0.0, step=0.1, help=help_txt)
                inputs[col] = float(val)

        if st.button("Predict"):
            # Build one row, enforce schema types, then predict
            X = prepare_inference_df(inputs, numeric_all, categorical_all)
            X = enforce_types(X, numeric_all, categorical_all)

            try:
                pred = int(pipe.predict(X)[0])
                proba_fn = getattr(pipe, "predict_proba", None)
                proba = proba_fn(X) if callable(proba_fn) else None

                st.subheader("Prediction")
                if len(class_labels) == 2 and proba is not None:
                    pos_index = class_labels.index(1) if 1 in class_labels else 1
                    pos_prob = float(proba[0][pos_index])
                    msg = "The customer is likely to complete the booking." if pred == 1 else "The customer is unlikely to complete the booking."
                    st.markdown(f"**{msg}**  Estimated probability of completion: **{pos_prob:.2%}**")
                else:
                    st.markdown(f"**Predicted class:** {label_map.get(pred, str(pred))}")

                if proba is not None:
                    probs = proba[0]
                    labels = [label_map.get(c, str(c)) for c in class_labels]
                    fig = plt.figure()
                    plt.pie(probs, labels=labels, autopct="%1.1f%%", startangle=90)
                    plt.title("Prediction probabilities")
                    plt.axis("equal")
                    st.pyplot(fig)

            except Exception as e:
                st.error("Prediction failed")
                st.exception(e)
                st.write("Debug dtypes:", X.dtypes.astype(str).to_dict())
                st.write("Debug row:", X.iloc[0].to_dict())
                st.stop()

# ---------------- Batch prediction ----------------
with tab2:
    st.write("Upload a CSV. The first row must contain the column names shown below, in any order.")
    st.caption(", ".join(EXPECTED_COLS))
    f = st.file_uploader("CSV file", type=["csv", "xls", "xlsx"])
    if f:
        try:
            df_raw = read_csv_flexible(f)
            df = validate_and_order(df_raw, EXPECTED_COLS)
            df = enforce_types(df, numeric_all, categorical_all)

            st.subheader("Preview")
            st.dataframe(df.head())

            preds = pipe.predict(df)
            proba_fn = getattr(pipe, "predict_proba", None)
            probas = proba_fn(df) if callable(proba_fn) else None

            out = df.copy()
            out["prediction"] = [label_map.get(int(p), str(p)) for p in preds]

            if probas is not None:
                if len(class_labels) == 2:
                    pos_index = class_labels.index(1) if 1 in class_labels else 1
                    out["probability"] = (probas[:, pos_index] * 100).round(1).astype(str) + "%"
                else:
                    pred_indices = [list(class_labels).index(p) for p in preds]
                    out["probability"] = [f"{probas[i, j]*100:.1f}%" for i, j in enumerate(pred_indices)]
            else:
                out["probability"] = np.nan

            st.subheader("Preview of predictions")
            st.dataframe(out.head(50))

            st.subheader("Prediction distribution")
            st.caption("This pie chart shows the proportion of rows predicted as completed vs not completed.")
            counts = pd.Series(preds).value_counts().sort_index()
            labels = [label_map.get(k, str(k)) for k in counts.index]
            fig2 = plt.figure()
            plt.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90)
            plt.title("Predicted class share")
            plt.axis("equal")
            st.pyplot(fig2)

            st.download_button(
                "Download predictions",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
            )
        except Exception as e:
            st.error(f"Could not process the file. {e}")
