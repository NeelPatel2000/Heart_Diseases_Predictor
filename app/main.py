
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# loading the trained model
MODEL_PATH = "model/model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("Heart Disease Risk (XGBoost)")
st.caption("Educational demo — not medical advice.")

#  encodings  
SEX_MAP        = {"Female": 0, "Male": 1}  # training used {'Male':1,'Female':0}
CHEST_PAIN_MAP = {"typical angina": 0, "asymptomatic": 1, "non-anginal": 2, "atypical angina": 3}
REST_ECG_MAP   = {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2}
SLOPE_MAP      = {"downsloping": 0, "flat": 1, "upsloping": 2}
THAL_MAP       = {"normal": 0, "fixed defect": 1, "reversable defect": 2}
BOOL_MAP       = {"No": 0, "Yes": 1}       # for fasting_bs, exang

# column order in the same order as the model was trained 
FEATURE_ORDER = [
    "age", "sex", "chest_pain", "resting_bs", "cholesterol",
    "fasting_bs", "rest_ecg", "thalch", "exang", "oldpeak",
    "slope", "major_vessels", "thal_defect"
]

# creating sidebar
st.sidebar.header("Patient inputs") # sidebar heading
c1, c2 = st.sidebar.columns(2) # creating 2 columns for sidebar

with c1: # first set of inputs
    age = st.number_input("Age", min_value=18, max_value=120, value=55, step=1)
    resting_bs = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=220, value=130, step=1)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240, step=1)
    thalch = st.number_input("Max HR (thalch)", min_value=60, max_value=220, value=150, step=1)
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    major_vessels = st.number_input("Major vessels colored (0-3)", min_value=0, max_value=3, value=0, step=1)

with c2: # second set of inputs
    sex = SEX_MAP[st.selectbox("Sex", list(SEX_MAP.keys()))]
    chest_pain = CHEST_PAIN_MAP[st.selectbox("Chest pain type", list(CHEST_PAIN_MAP.keys()))]
    fasting_bs = BOOL_MAP[st.selectbox("Fasting blood sugar > 120 mg/dl?", list(BOOL_MAP.keys()))]
    rest_ecg = REST_ECG_MAP[st.selectbox("Resting ECG", list(REST_ECG_MAP.keys()))]
    exang = BOOL_MAP[st.selectbox("Exercise-induced angina?", list(BOOL_MAP.keys()))]
    slope = SLOPE_MAP[st.selectbox("Slope of peak exercise ST segment", list(SLOPE_MAP.keys()))]
    thal_defect = THAL_MAP[st.selectbox("Thal defect", list(THAL_MAP.keys()))]
    

# creating a slider so that user can decide the decision threshold
threshold = st.sidebar.slider(
    "Decision threshold (classify as disease if probability ≥ threshold)",
    min_value=0.10, max_value=0.90, value=0.50, step=0.01
)

# this will format selected values by the user in a row of data so model can understand
def make_feature_row():
    row = {
        "age": int(age),
        "sex": int(sex),
        "chest_pain": int(chest_pain),
        "cholesterol": float(cholesterol),
        "fasting_bs": int(fasting_bs),
        "rest_ecg": int(rest_ecg),
        "thalch": float(thalch),
        "exang": int(exang),
        "oldpeak": float(oldpeak),
        "resting_bs": float(resting_bs),
        "slope": int(slope),
        "major_vessels": int(major_vessels),
        "thal_defect": int(thal_defect),
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)

# prediction  
st.markdown("---")
if st.button("Predict"): # a predict button when press will trigger all the following actions
    X = make_feature_row()
    # probability of positive class (1 = disease)
    pos_idx = list(model.classes_).index(1) if hasattr(model, "classes_") else 1
    prob = float(model.predict_proba(X)[:, pos_idx][0])
    pred = int(prob >= threshold) # applying threshold

    st.subheader("🧠 Prediction Results")
    if pred == 1:
        st.error("🟥 **High Risk: Likely Heart Disease**") # inbuilt streamlit function to show error
    else:
        st.success("🟩 **Low Risk: No Heart Disease Detected**") # inbuilt streamlit function to show success

    st.metric("Predicted Probability of Heart Disease", f"{prob:.3f}") # shows probability
    st.caption(f"Decision threshold: {threshold:.2f} — adjust to change sensitivity/specificity.") # shows threshold selected

    # show all the features send to the model to make predictions
    with st.expander("Show features sent to the model"):
        st.dataframe(X)

    st.caption("Tip: lower the threshold to increase sensitivity (catch more positives) at the cost of more false alarms.")