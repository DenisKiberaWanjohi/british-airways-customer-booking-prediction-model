from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

from src.io import read_csv
from src.app_logic import fit_and_save
from src.schema import FeatureSchema

DATA_PATH = Path("data/customer_booking.csv")
TARGET_COL = "booking_complete"
MAX_CAT_OPTIONS = 50

HUMAN_READABLE = {
    "num_passengers": "Number of passengers on the booking (integer)",
    "sales_channel": "Sales channel used to book (e.g. Web, Mobile, Agent)",
    "trip_type": "Trip type (e.g. OneWay or Return)",
    "purchase_lead": "Days between purchase date and flight date (integer)",
    "length_of_stay": "Days between outbound and return (integer)",
    "flight_hour": "Hour of flight departure (0–23)",
    "flight_day": "Day of week of the flight (e.g. Monday)",
    "route": "Route or origin–destination pair",
    "booking_origin": "Country or market where booking was made",
    "wants_extra_baggage": "Customer wants extra baggage (0 or 1)",
    "wants_preferred_seat": "Customer wants preferred seat (0 or 1)",
    "wants_in_flight_meals": "Customer wants in-flight meals (0 or 1)",
    "flight_duration": "Flight duration in hours",
}

def build_schema_extras(df: pd.DataFrame, schema: FeatureSchema):
    dtypes = {}
    for c in schema.numeric:
        dtypes[c] = "number"
    for c in schema.categorical:
        dtypes[c] = "string"
    schema.dtypes = dtypes

    desc = {}
    for c in list(df.columns):
        if c == schema.target:
            continue
        desc[c] = HUMAN_READABLE.get(c, f"Value for '{c}'")
    schema.descriptions = desc

    opts = {}
    for c in schema.categorical:
        vals = df[c].astype(str).dropna().unique().tolist()
        if len(vals) > MAX_CAT_OPTIONS:
            top = df[c].astype(str).value_counts().head(MAX_CAT_OPTIONS).index.tolist()
            vals = top
        opts[c] = vals
    schema.categorical_options = opts

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected {DATA_PATH}. Place the dataset in the data folder.")

    df = read_csv(DATA_PATH, encoding="ISO-8859-1")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in CSV.")

    df = df.dropna(subset=[TARGET_COL])
    pipe, schema = fit_and_save(df, TARGET_COL)

    build_schema_extras(df, schema)
    schema.save()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    y_pred = pipe.predict(X)
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    main()
