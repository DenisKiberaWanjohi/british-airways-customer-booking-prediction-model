from pathlib import Path
import joblib
import pandas as pd

PIPELINE_PATH = Path("models/pipeline.pkl")

def save_pipeline(pipe, path: Path = PIPELINE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)

def load_pipeline(path: Path = PIPELINE_PATH):
    return joblib.load(path)

def read_csv(path: Path, encoding: str | None = None) -> pd.DataFrame:
    # Try common encodings gracefully
    for enc in [encoding, "utf-8", "ISO-8859-1", "latin1"]:
        if enc is None:
            continue
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # Final attempt with pandas default
    return pd.read_csv(path)
