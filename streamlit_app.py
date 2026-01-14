import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

DATA_DIR = os.environ.get("GPS_DATA_DIR", ".")
REG_PATH = os.path.join(DATA_DIR, "gps_event_success_regressor.joblib")
CLF_PATH = os.path.join(DATA_DIR, "gps_event_success_classifier.joblib")
META_PATH = os.path.join(DATA_DIR, "gps_event_success_meta.json")
TRAIN_TABLE_PATH = os.path.join(DATA_DIR, "event_level_training_table_FY24_FY25_FY26.csv")

st.set_page_config(page_title="GPS Event Success Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    reg = joblib.load(REG_PATH)
    clf = joblib.load(CLF_PATH)
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            meta = json.load(f)
    return reg, clf, meta

def lab_flags_from_title(title: str):
    t = (title or "").lower()
    is_lab_support = ("lab support" in t) or (("q&a" in t) and ("lab" in t))
    is_lab = (
        ("hands-on lab" in t) or ("hands on lab" in t) or
        ("interactive simulated lab" in t) or ("simulated lab" in t) or
        (("hands-on" in t) and ("lab" in t))
    ) and not is_lab_support
    return float(is_lab), float(is_lab_support)

def parse_day_info(title: str):
    if not title:
        return (np.nan, np.nan)
    m = re.search(r"day\s*(\d+)\s*(?:of\s*(\d+))?", title, flags=re.IGNORECASE)
    if not m:
        return (np.nan, np.nan)
    day = float(m.group(1))
    tot = float(m.group(2)) if m.group(2) else np.nan
    return (day, tot)

def build_feature_row(title, event_dt, prereg_count, timezone, campaign_code, is_lab_override=None, is_lab_support_override=None):
    month = float(event_dt.month)
    dow = float(event_dt.weekday())
    hour = float(event_dt.hour)

    is_lab, is_lab_support = lab_flags_from_title(title)
    if is_lab_override is not None:
        is_lab = float(is_lab_override)
    if is_lab_support_override is not None:
        is_lab_support = float(is_lab_support_override)

    day_num, day_total = parse_day_info(title)

    return pd.DataFrame([{
        "title": title or "",
        "month": month,
        "dow": dow,
        "hour": hour,
        "timezone": timezone or "UTC",
        "campaign_code": campaign_code or "Unknown",
        "prereg_count": float(prereg_count) if prereg_count is not None else np.nan,
        "is_lab": float(is_lab),
        "is_lab_support": float(is_lab_support),
        "day_num": day_num,
        "day_total": day_total
    }])

def predict(reg, clf, X_row):
    # reg predicts log1p(attendees)
    pred_log = float(reg.predict(X_row)[0])
    pred_attendees = float(np.expm1(pred_log))
    pred_attendees = max(0.0, pred_attendees)

    prob_success = float(clf.predict_proba(X_row)[0][1])
    return pred_attendees, prob_success

st.title("GPS Event Success Predictor")
st.caption("Predicts expected unique attendees and success probability using FY24–FY26 historical ON24 prereg + view patterns.")

reg, clf, meta = load_artifacts()

with st.expander("Model info", expanded=False):
    st.write({
        "trained_at": meta.get("trained_at"),
        "n_events": meta.get("n_events"),
        "default_success_threshold_attendees": meta.get("default_success_threshold_attendees"),
        "holdout_mae_attendees": meta.get("mae_holdout"),
        "holdout_r2": meta.get("r2_holdout"),
        "notes": "Regressor predicts log1p(attendees). Classifier predicts probability(event attendees >= threshold)."
    })
    if os.path.exists(TRAIN_TABLE_PATH):
        st.write("Training table:", TRAIN_TABLE_PATH)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Planned event inputs")
    title = st.text_input("Event title", value="Agentic AI Accelerator (Day 1 of 4)")
    event_date = st.date_input("Event date", value=datetime(2026, 2, 5).date())
    event_time = st.time_input("Event start time (local)", value=datetime(2026, 2, 5, 9, 0).time())
    timezone = st.selectbox("Timezone label (for reporting)", options=["UTC","PST","MST","CST","EST","GMT","CET","IST","Other"], index=4)
    if timezone == "Other":
        timezone = st.text_input("Enter timezone label", value="Other")

    prereg_count = st.number_input("Expected prereg / registrants (if known)", min_value=0, value=150, step=10)

    campaign_code = st.text_input("Campaign code (optional)", value="Unknown")

    st.markdown("**Lab classification** (auto-detected from title; override if needed)")
    auto_is_lab, auto_is_lab_support = lab_flags_from_title(title)
    st.write({"auto_is_lab": bool(auto_is_lab), "auto_is_lab_support": bool(auto_is_lab_support)})

    override = st.checkbox("Override lab flags", value=False)
    if override:
        is_lab_override = st.checkbox("Treat as TRUE LAB", value=bool(auto_is_lab))
        is_lab_support_override = st.checkbox("Treat as Lab Support & Q&A (NOT a lab)", value=bool(auto_is_lab_support))
    else:
        is_lab_override = None
        is_lab_support_override = None

    run = st.button("Predict success", type="primary")

with col2:
    st.subheader("Prediction")
    if run:
        event_dt = datetime.combine(event_date, event_time)
        X_row = build_feature_row(title, event_dt, prereg_count, timezone, campaign_code, is_lab_override, is_lab_support_override)
        pred_attendees, prob = predict(reg, clf, X_row)

        thr = meta.get("default_success_threshold_attendees", 75)
        st.metric("Predicted unique attendees", f"{pred_attendees:,.0f}")
        st.metric("Success probability", f"{prob*100:.1f}%")
        st.caption(f"Success = attendees ≥ {thr} (default). You can change this threshold in a future version or by retraining.")

        st.markdown("### Drivers (explainable, lightweight)")
        st.write({
            "Timing": {"month": int(X_row.loc[0,'month']), "day_of_week": int(X_row.loc[0,'dow']), "hour": int(X_row.loc[0,'hour'])},
            "Lab flags": {"is_lab": bool(X_row.loc[0,'is_lab']), "is_lab_support": bool(X_row.loc[0,'is_lab_support'])},
            "Prereg count (input)": int(prereg_count),
            "Title keywords": title[:140] + ("..." if len(title) > 140 else "")
        })

        st.markdown("### What-if quick checks")
        st.write("Try modifying prereg count or adding/removing 'Simulated Lab' / 'Hands-on Lab' in the title and rerun.")

st.divider()
st.markdown("### Batch scoring (optional)")
st.write("If you have a CSV of planned events, you can score them in bulk with a small script. Ask and I'll generate the template + batch scorer.")
