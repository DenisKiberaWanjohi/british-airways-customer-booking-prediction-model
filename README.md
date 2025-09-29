# British Airways Customer Booking — Predictive Model

This repository contains a classification model that predicts booking completion based on historical booking attributes, built for the British Airways Data Science job simulation on Forage. It also includes a Streamlit UI for single predictions and CSV batch scoring.

## Project structure

```
ba-booking-prediction/
├── app.py                         # Streamlit UI for interactive predictions
├── requirements.txt
├── notebooks/
│   └── BApredictivemodel.ipynb
├── src/
│   ├── train_pipeline.py          # Train and export a Sklearn Pipeline
│   └── utils.py                   # Shared helpers
├── data/
│   └── customer_booking.csv       # Place the raw dataset here (not tracked in git)
└── models/
    ├── pipeline.pkl               # Exported model + preprocessing (created after training)
    └── feature_info.json          # Saved at training time to drive the UI
```

## Data

Expected file: `data/customer_booking.csv`. The default target is `booking_complete` with values 0 or 1. Adjust the `TARGET_COL` in `src/train_pipeline.py` if your column differs.

## Train the model

```bash
python src/train_pipeline.py
```
This script:
- auto-detects numeric and categorical features
- builds a ColumnTransformer + RandomForestClassifier inside a single Pipeline
- saves `models/pipeline.pkl` and `models/feature_info.json`

## Run the Streamlit app

```bash
streamlit run app.py
```

You can enter values in a form for a single prediction or upload a CSV for batch predictions. The app uses the saved pipeline, so run training first if the artifacts are missing.

## Included artifacts

This repo includes a trained example model so the app runs immediately:

- `models/pipeline.pkl`
- `models/feature_info.json`

To retrain on your real dataset, replace `data/customer_booking.csv` and run:

```bash
python src/train_pipeline.py
```
