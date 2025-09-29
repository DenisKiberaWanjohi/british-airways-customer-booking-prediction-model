# src/train_pipeline.py
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

from src.io import read_csv
from src.app_logic import fit_and_save
from src.schema import FeatureSchema

DATA_PATH = Path("data/customer_booking.csv")
TARGET_COL = "booking_complete"
MAX_CAT_OPTIONS = 50

# Human-friendly help text for YOUR columns
HUMAN_READABLE = {
    "num_passengers": "Number of passengers on the booking. Integer.",
    "sales_channel": "Sales channel used to book. For example Web, Mobile, Agent.",
    "trip_type": "Trip type. For example OneWay or RoundTrip.",
    "purchase_lead": "Days between purchase date and flight date. Integer in days.",
    "length_of_stay": "Days between outbound and return. Integer in days.",
    "flight_hour": "Hour of departure. Integer from 0 to 23.",
    "flight_day": "Day of week of the flight. For example Mon, Tue, Sat.",
    "route": "Route or origin–destination pair. For example LHR-JFK.",
    "booking_origin": "Country or market where the booking was made.",
    "wants_extra_baggage": "Customer wants extra baggage. 1 = Yes, 0 = No.",
    "wants_preferred_seat": "Customer wants a preferred seat. 1 = Yes, 0 = No.",
    "wants_in_flight_meals": "Customer wants in-flight meals. 1 = Yes, 0 = No.",
    "flight_duration": "Flight duration in hours. Decimal number.",
}

def build_schema_extras(df: pd.DataFrame, schema: FeatureSchema):
    # dtype hints for UI
    ui_hints = {}
    for col in df.columns:
        if col == schema.target:
            continue
        s = df[col]
        if col in ("wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals"):
            ui_hints[col] = {"kind": "binary"}                       # checkbox
        elif col == "flight_hour":
            ui_hints[col] = {"kind": "int", "min": 0, "max": 23}     # slider
        elif col in ("purchase_lead", "length_of_stay", "num_passengers"):
            ui_hints[col] = {"kind": "int"}                           # integer
        elif col == "flight_duration":
            ui_hints[col] = {"kind": "float"}                         # float
        elif col in schema.categorical:
            ui_hints[col] = {"kind": "category"}                      # dropdown
        else:
            # fallback based on pandas dtype
            if pd.api.types.is_integer_dtype(s):
                ui_hints[col] = {"kind": "int"}
            elif pd.api.types.is_float_dtype(s):
                ui_hints[col] = {"kind": "float"}
            else:
                ui_hints[col] = {"kind": "category"}

    schema.descriptions = {c: HUMAN_READABLE.get(c, f"Value for '{c}'.")
                           for c in df.columns if c != schema.target}
    schema.dtypes = {c: ("number" if ui_hints.get(c, {}).get("kind") in {"int", "float", "binary"} else "string")
                     for c in df.columns if c != schema.target}

    # category options
    opts = {}
    for c in schema.categorical:
        vals = df[c].astype(str).dropna()
        # keep the most frequent MAX_CAT_OPTIONS
        top = vals.value_counts().head(MAX_CAT_OPTIONS).index.tolist()
        opts[c] = top
    schema.categorical_options = opts

    # store ui hints
    # add field in FeatureSchema first if you have not done so yet
    try:
        # if your schema has ui_hints, set it
        schema.ui_hints = ui_hints                         # type: ignore[attr-defined]
    except Exception:
        # if your schema class does not yet define ui_hints, you can ignore this line
        pass

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