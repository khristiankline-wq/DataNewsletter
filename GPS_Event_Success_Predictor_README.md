# GPS Event Success Predictor (Streamlit)

## What it does
A lightweight, explainable predictor that estimates:
- Expected unique attendees
- Probability the event is a "success" (default threshold: top-quartile attendees in training history)

Trained on FY24–FY26 ON24 exports:
- webcast prereg (7_All_Webcastpreregs_*.csv) for event schedule + prereg counts
- webcast views (9_All_detailedwebcastviews_*.csv) for attendee counts

## How to run
From a terminal (with Streamlit installed):

```bash
cd /mnt/data
streamlit run gps_event_success_app.py
```

If your data/models are stored elsewhere, set:
```bash
export GPS_DATA_DIR=/path/to/dir
```

## Files generated
- gps_event_success_regressor.joblib
- gps_event_success_classifier.joblib
- gps_event_success_meta.json
- event_level_training_table_FY24_FY25_FY26.csv
