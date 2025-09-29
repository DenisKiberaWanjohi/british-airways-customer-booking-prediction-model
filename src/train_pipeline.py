from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

from src.io import read_csv
from src.app_logic import fit_and_save
from src.schema import FeatureSchema
from src.io import load_pipeline

DATA_PATH = Path("data/customer_booking.csv")
TARGET_COL = "booking_complete"  # Change if needed

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected {DATA_PATH}. Place the dataset in the data folder.")

    df = read_csv(DATA_PATH, encoding="ISO-8859-1")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in CSV.")

    df = df.dropna(subset=[TARGET_COL])
    pipe, schema = fit_and_save(df, TARGET_COL)

    # Simple in-sample report for quick feedback
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    y_pred = pipe.predict(X)
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    main()
