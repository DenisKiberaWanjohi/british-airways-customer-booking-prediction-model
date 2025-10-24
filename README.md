# British Airways Customer Booking — Predictive Model With Streamlit UI

## 🧭 Overview / Problem Statement

British Airways aims to improve conversion rates by identifying customers likely to complete a booking.  
This model analyzes historical booking data to predict completion likelihood, helping the airline target follow-up campaigns or special offers more effectively.

---

## 🎯 Objectives

- Build a machine learning model to predict booking completion  
- Identify key drivers influencing booking decisions  
- Provide an interactive interface for real-time and batch predictions  

---

## 🧩 Features

- End-to-end training pipeline (data preprocessing → model training → artifact export)  
- Streamlit dashboard for real-time scoring and visualization  
- Automated feature type detection and transformation  
- Configurable target and feature columns  


## Streamlit App

Access the live demo here:
https://british-airways-customer-booking-prediction-model.streamlit.app/

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

## Using the UI

You can enter values in a form for a single prediction or upload a CSV for batch predictions. The app uses the saved pipeline, so run training first if the artifacts are missing.

## Included artifacts

This repo includes a trained example model so the app runs immediately:

- `models/pipeline.pkl`
- `models/feature_info.json`

**Author:** Denis Wanjohi (deniskibera7@gmail.com)
Built as part of the British Airways Data Science Job Simulation on Forage.


