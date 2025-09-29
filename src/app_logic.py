from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from .features import detect_feature_types, enforce_types, prepare_inference_df
from .schema import FeatureSchema
from .io import save_pipeline, load_pipeline

def build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])

def fit_and_save(df: pd.DataFrame, target: str):
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    numeric, categorical = detect_feature_types(df, target)
    X = enforce_types(X, numeric, categorical)

    pipe = build_pipeline(numeric, categorical)
    pipe.fit(X, y)
    save_pipeline(pipe)

    schema = FeatureSchema(
        numeric=numeric,
        categorical=categorical,
        target=target,
        class_labels=sorted(pd.unique(y).tolist()),
        n_rows=int(len(df)),
    )
    schema.save()
    return pipe, schema
