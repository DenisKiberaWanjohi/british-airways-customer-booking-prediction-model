import pandas as pd
import numpy as np

def detect_feature_types(df: pd.DataFrame, target: str) -> tuple[list, list]:
    X = df.drop(columns=[target])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

def enforce_types(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in numeric:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in categorical:
        if c in out.columns:
            out[c] = out[c].astype("string").fillna("")
    return out

def prepare_inference_df(inputs: dict, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    # Build a one-row DataFrame in schema order
    row = {}
    for c in numeric:
        row[c] = float(inputs.get(c, 0.0)) if inputs.get(c) not in [None, ""] else 0.0
    for c in categorical:
        row[c] = str(inputs.get(c, ""))
    return pd.DataFrame([row])
