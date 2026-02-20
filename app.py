"""
ABC Credit — Loan Application Chatbot
======================================
Streamlit-native conversational chatbot for real-time loan decisioning.

Run:
    streamlit run app.py
"""

import csv
import os
from datetime import datetime, date

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

matplotlib.use("Agg")  # Non-interactive backend for server rendering

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_loan_model.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")
SESSION_LOG = "session_log.csv"

LOG_COLUMNS = [
    "timestamp",
    "Age",
    "Emp_Level",
    "Education_Level",
    "Net_Sal",
    "Loan_Amt",
    "Asset_Value",
    "LTV_Perc",
    "Housing_Category",
    "Existing_Liabilities",
    "Model_Probability",
    "Decision",
]

VALID_EMP_LEVELS = ["AG", "SA", "NR", "ST", "SE"]

# Map step index → sidebar field label
FIELD_LABELS = [
    "Date of Birth",
    "Employment Type",
    "Education Level",
    "Monthly Net Salary",
    "Loan Amount",
    "Asset / Property Value",
    "Housing Status",
    "Existing Liabilities",
]

# Question prompts (indexed 1–8 matching step numbers)
QUESTIONS = {
    1: (
        "Welcome to ABC Credit! I will guide you through a quick 8-question loan "
        "application — it takes under 5 minutes.\n\n"
        "**Question 1 of 8**\n"
        "Please enter your **date of birth** in DD/MM/YYYY format."
    ),
    2: (
        "**Question 2 of 8**\n"
        "What is your **employment type**?\n\n"
        "Please enter one of the following codes:\n"
        "- **AG** — Agriculture\n"
        "- **SA** — Salaried\n"
        "- **NR** — Non-Resident\n"
        "- **ST** — Student\n"
        "- **SE** — Self-Employed"
    ),
    3: (
        "**Question 3 of 8**\n"
        "What is your **highest education level**?\n\n"
        "Enter a number from **0 to 5**:\n"
        "- 0 = No formal education\n"
        "- 1 = Primary\n"
        "- 2 = Secondary\n"
        "- 3 = Diploma\n"
        "- 4 = Graduate\n"
        "- 5 = Post-Graduate / Professional"
    ),
    4: (
        "**Question 4 of 8**\n"
        "What is your **monthly net salary** (take-home pay in ₹)?\n\n"
        "Enter a positive number, e.g. 45000"
    ),
    5: (
        "**Question 5 of 8**\n"
        "What is the **loan amount** you are requesting (in ₹)?\n\n"
        "Enter a positive number, e.g. 500000"
    ),
    6: (
        "**Question 6 of 8**\n"
        "What is the current **market value of the asset or property** "
        "you are pledging as collateral (in ₹)?\n\n"
        "Enter a positive number, e.g. 2000000"
    ),
    7: (
        "**Question 7 of 8**\n"
        "What is your **housing status**?\n\n"
        "Enter **Owner** or **Rent**"
    ),
    8: (
        "**Question 8 of 8 — Final question!**\n"
        "Do you have any **existing liabilities** (outstanding loans or EMIs)?\n\n"
        "Enter **Y** (Yes) or **N** (No)"
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (cached — loaded once per session)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_artifacts():
    """Load trained XGBoost model, label encoders, and feature columns."""
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, label_encoders, feature_columns


# ──────────────────────────────────────────────────────────────────────────────
# Input validators
# ──────────────────────────────────────────────────────────────────────────────

def validate_dob(raw: str):
    """
    Parse DD/MM/YYYY and compute age.
    Returns (age: int, error: str | None)
    """
    raw = raw.strip()
    try:
        dob = datetime.strptime(raw, "%d/%m/%Y").date()
    except ValueError:
        return None, "Invalid format. Please use **DD/MM/YYYY** (e.g. 15/06/1990)."

    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    if age < 18:
        return age, f"AGE_POLICY_UNDER"
    if age > 60:
        return age, f"AGE_POLICY_OVER"

    return age, None


def validate_emp_level(raw: str):
    """
    Validate employment type code.
    Returns (value: str, error: str | None)
    """
    val = raw.strip().upper()
    if val not in VALID_EMP_LEVELS:
        return None, (
            f"**{raw}** is not a valid code. "
            f"Please enter one of: **AG, SA, NR, ST, SE**."
        )
    return val, None


def validate_education(raw: str):
    """
    Validate education level integer 0–5.
    Returns (value: int, error: str | None)
    """
    raw = raw.strip()
    try:
        val = int(raw)
    except ValueError:
        return None, "Please enter a whole number between **0 and 5**."
    if val < 0 or val > 5:
        return None, f"**{val}** is out of range. Enter a number from **0 to 5**."
    return val, None


def validate_positive_number(raw: str, field: str):
    """
    Validate a positive numeric value (strip commas/₹ signs).
    Returns (value: float, error: str | None)
    """
    cleaned = raw.strip().replace(",", "").replace("₹", "").replace(" ", "")
    try:
        val = float(cleaned)
    except ValueError:
        return None, f"Please enter a valid positive number for **{field}** (digits only, e.g. 45000)."
    if val <= 0:
        return None, f"**{field}** must be greater than zero. Please re-enter."
    return val, None


def validate_housing(raw: str):
    """
    Validate housing status.
    Returns (value: str, error: str | None)  — 'Owner' or 'Rent'
    """
    val = raw.strip().lower()
    if val == "owner":
        return "Owner", None
    if val == "rent":
        return "Rent", None
    return None, f"Please enter **Owner** or **Rent** (you entered: *{raw}*)."


def validate_liabilities(raw: str):
    """
    Validate existing liabilities.
    Returns (value: str, error: str | None)  — 'Y' or 'N'
    """
    val = raw.strip().upper()
    if val == "Y":
        return "Y", None
    if val == "N":
        return "N", None
    return None, f"Please enter **Y** (Yes) or **N** (No) (you entered: *{raw}*)."


# ──────────────────────────────────────────────────────────────────────────────
# Feature row builder
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_row(answers: dict, label_encoders: dict, feature_columns: list):
    """
    Convert chatbot answers dict into a 1-row DataFrame matching training schema.
    """
    le_emp = label_encoders["Emp_Level"]
    housing_map = label_encoders["Housing_Category"]
    liabilities_map = label_encoders["Existing_Liabilities"]

    # Encode Emp_Level — default -1 if unseen category
    emp_val = answers["Emp_Level"]
    if emp_val in le_emp.classes_:
        emp_encoded = int(le_emp.transform([emp_val])[0])
    else:
        emp_encoded = -1  # unknown — edge case guard

    row = {
        "Age": answers["Age"],
        "Education Level": answers["Education Level"],
        "Emp_Level": emp_encoded,
        "Loan_Amt": answers["Loan_Amt"],
        "LTV_Perc": min(answers["LTV_Perc"], 100.0),  # cap at 100%
        "Housing_Category": housing_map.get(answers["Housing_Category"], 0),
        "Net_Sal": answers["Net_Sal"],
        "Existing_Liabilities": liabilities_map.get(answers["Existing_Liabilities"], 0),
    }

    df_row = pd.DataFrame([row], columns=feature_columns)
    return df_row


# ──────────────────────────────────────────────────────────────────────────────
# Session logging
# ──────────────────────────────────────────────────────────────────────────────

def log_session(answers: dict, probability: float, decision: str):
    """
    Append one row to session_log.csv.
    Creates the file with headers if it does not exist yet.
    """
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": answers["Age"],
        "Emp_Level": answers["Emp_Level"],
        "Education_Level": answers["Education Level"],
        "Net_Sal": answers["Net_Sal"],
        "Loan_Amt": answers["Loan_Amt"],
        "Asset_Value": answers["Asset_Value"],
        "LTV_Perc": round(answers["LTV_Perc"], 4),
        "Housing_Category": answers["Housing_Category"],
        "Existing_Liabilities": answers["Existing_Liabilities"],
        "Model_Probability": round(probability, 4),
        "Decision": decision,
    }

    file_exists = os.path.exists(SESSION_LOG)
    try:
        with open(SESSION_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except PermissionError:
        st.warning(
            "Could not write to session_log.csv — it may be open in another application. "
            "Please close it and try again."
        )


# ──────────────────────────────────────────────────────────────────────────────
# SHAP chart
# ──────────────────────────────────────────────────────────────────────────────

def render_shap_chart(model, feature_row: pd.DataFrame, feature_columns: list):
    """
    Generate and render a SHAP horizontal bar chart inside the chat.
    Top 5 features, green = favourable, red = unfavourable.
    """
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values — handle both old and new SHAP API
    try:
        explanation = explainer(feature_row)
        shap_vals = explanation.values[0]
        if shap_vals.ndim == 2:
            shap_vals = shap_vals[:, 1]  # class-1 (Approved)
    except Exception:
        raw = explainer.shap_values(feature_row)
        if isinstance(raw, list):
            shap_vals = raw[1][0]
        else:
            shap_vals = raw[0]

    # Build sorted DataFrame — top 5 by absolute SHAP value
    shap_df = pd.DataFrame(
        {"Feature": feature_columns, "SHAP Value": shap_vals}
    )
    shap_df["abs"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.nlargest(5, "abs").drop(columns="abs")
    shap_df = shap_df.sort_values("SHAP Value")

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in shap_df["SHAP Value"]]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (impact on approval probability)", fontsize=9)
    ax.set_title(
        "Top Factors — Positive = Favourable, Negative = Unfavourable",
        fontsize=10,
        pad=10,
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Session state initialiser
# ──────────────────────────────────────────────────────────────────────────────

def init_session():
    """Initialise all session_state keys on first load or after Start Over."""
    defaults = {
        "step": 1,          # current question (1–8); 9 = done; 0 = exited
        "messages": [],     # list of {role, content}
        "answers": {},      # collected answers keyed by feature name
        "exited": False,    # True when age policy triggers early exit
        "logged": False,    # True once log_session() has been called
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_session():
    """Clear all session state for a clean restart."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session()
    # Seed the first bot message
    st.session_state.messages.append(
        {"role": "assistant", "content": QUESTIONS[1]}
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render progress bar, field checklist, and Start Over button."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/ABC-logo.svg/200px-ABC-logo.svg.png",
            width=80,
        ) if False else None  # placeholder — replace with real logo path if available
        st.title("ABC Credit")
        st.caption("Loan Application Progress")
        st.divider()

        step = st.session_state.step
        exited = st.session_state.exited

        # Progress bar
        progress = min(step - 1, 8) / 8 if not exited else 1.0
        st.progress(progress)
        st.caption(f"{'Complete' if step > 8 or exited else f'Step {step} of 8'}")
        st.divider()

        # Field checklist
        answers = st.session_state.answers
        label_map = {
            "Date of Birth":            "Age" in answers,
            "Employment Type":          "Emp_Level" in answers,
            "Education Level":          "Education Level" in answers,
            "Monthly Net Salary":       "Net_Sal" in answers,
            "Loan Amount":              "Loan_Amt" in answers,
            "Asset / Property Value":   "Asset_Value" in answers,
            "Housing Status":           "Housing_Category" in answers,
            "Existing Liabilities":     "Existing_Liabilities" in answers,
        }
        for label, done in label_map.items():
            icon = "✅" if done else "⬜"
            st.markdown(f"{icon} {label}")

        st.divider()
        if st.button("🔄 Start Over", use_container_width=True):
            reset_session()
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Decision card + SHAP
# ──────────────────────────────────────────────────────────────────────────────

def render_decision(model, label_encoders, feature_columns):
    """Score the application and render Approve/Decline card + SHAP chart."""
    answers = st.session_state.answers
    feature_row = build_feature_row(answers, label_encoders, feature_columns)

    with st.spinner("Analysing your application... please wait."):
        import time
        time.sleep(1.5)  # UX delay — actual inference is <100ms
        proba = float(model.predict_proba(feature_row)[0, 1])
        decision = "APPROVED" if proba >= 0.5 else "DECLINED"

    # Log session (once)
    if not st.session_state.logged:
        log_session(answers, proba, decision)
        st.session_state.logged = True

    with st.chat_message("assistant"):
        if decision == "APPROVED":
            st.success(
                f"## Loan Application — **APPROVED** ✅\n\n"
                f"Confidence score: **{proba * 100:.1f}%** approval probability\n\n"
                f"Congratulations! Based on your profile, ABC Credit is pleased to "
                f"proceed with your loan application. A representative will contact "
                f"you within 2 business days."
            )
        else:
            st.error(
                f"## Loan Application — **DECLINED** ❌\n\n"
                f"Confidence score: **{(1 - proba) * 100:.1f}%** decline probability\n\n"
                f"Unfortunately, based on the information provided, we are unable to "
                f"approve your loan application at this time. You may re-apply after "
                f"90 days or contact our branch for further guidance."
            )

        st.markdown("---")
        st.markdown("### Key Factors That Influenced This Decision")
        render_shap_chart(model, feature_row, feature_columns)

        st.markdown("---")
        st.caption(
            f"Decision recorded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
            "Session logged. Use the **Start Over** button in the sidebar to begin a new application."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Input processor — validates and advances state machine
# ──────────────────────────────────────────────────────────────────────────────

def process_input(user_text: str):
    """
    Validate user input for the current step, update answers, advance step.
    Returns (ack_message: str | None, error_message: str | None)
    """
    step = st.session_state.step
    answers = st.session_state.answers

    if step == 1:
        age, err = validate_dob(user_text)
        if err == "AGE_POLICY_UNDER":
            st.session_state.exited = True
            return None, (
                f"We're sorry — applicants must be **at least 18 years old** to apply. "
                f"Our records indicate you are **{age} year(s) old**.\n\n"
                "Thank you for your interest in ABC Credit. We hope to serve you in the future!"
            )
        if err == "AGE_POLICY_OVER":
            st.session_state.exited = True
            return None, (
                f"We're sorry — ABC Credit's eligibility policy covers applicants up to **60 years of age**. "
                f"Our records indicate you are **{age} year(s) old**.\n\n"
                "Thank you for your interest. Please visit a branch for alternative options."
            )
        if err:
            return None, err
        answers["Age"] = age
        return (
            f"Thank you! You are **{age} years old** — you meet our age eligibility criteria. "
            "Let's continue.",
            None,
        )

    if step == 2:
        val, err = validate_emp_level(user_text)
        if err:
            return None, err
        answers["Emp_Level"] = val
        labels = {
            "AG": "Agriculture", "SA": "Salaried", "NR": "Non-Resident",
            "ST": "Student", "SE": "Self-Employed",
        }
        return f"Got it — **{val}** ({labels[val]}). Moving on.", None

    if step == 3:
        val, err = validate_education(user_text)
        if err:
            return None, err
        answers["Education Level"] = val
        descs = {
            0: "No formal education", 1: "Primary", 2: "Secondary",
            3: "Diploma", 4: "Graduate", 5: "Post-Graduate / Professional",
        }
        return f"Noted — Education Level **{val}** ({descs[val]}).", None

    if step == 4:
        val, err = validate_positive_number(user_text, "Monthly Net Salary")
        if err:
            return None, err
        answers["Net_Sal"] = val
        return f"Monthly net salary recorded: **₹{val:,.0f}**.", None

    if step == 5:
        val, err = validate_positive_number(user_text, "Loan Amount")
        if err:
            return None, err
        answers["Loan_Amt"] = val
        return f"Loan amount requested: **₹{val:,.0f}**.", None

    if step == 6:
        val, err = validate_positive_number(user_text, "Asset / Property Value")
        if err:
            return None, err
        answers["Asset_Value"] = val
        loan = answers["Loan_Amt"]
        ltv = min((loan / val) * 100, 100.0)
        answers["LTV_Perc"] = ltv
        ltv_comment = (
            "This is within a healthy range." if ltv <= 80
            else "This is a high LTV — it may affect your eligibility."
        )
        return (
            f"Asset value recorded: **₹{val:,.0f}**.\n\n"
            f"Your computed **Loan-to-Value (LTV) ratio is {ltv:.1f}%**. {ltv_comment}",
            None,
        )

    if step == 7:
        val, err = validate_housing(user_text)
        if err:
            return None, err
        answers["Housing_Category"] = val
        return f"Housing status: **{val}**.", None

    if step == 8:
        val, err = validate_liabilities(user_text)
        if err:
            return None, err
        answers["Existing_Liabilities"] = val
        label = "Yes — existing liabilities noted." if val == "Y" else "No existing liabilities."
        return (
            f"{label}\n\nThank you — that was the last question! "
            "I am now scoring your application…",
            None,
        )

    return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="ABC Credit — Loan Chatbot",
        page_icon="🏦",
        layout="centered",
    )

    # ── Guard: model must exist ────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        st.error(
            "**Model not found.** Please train the model first by running:\n\n"
            "```\npython train_model.py --data Data_FINAL.xlsx\n```"
        )
        st.stop()

    # ── Load model ────────────────────────────────────────────────────────────
    model, label_encoders, feature_columns = load_model_artifacts()

    # ── Initialise session ────────────────────────────────────────────────────
    init_session()

    # Seed initial bot message (only once)
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append(
            {"role": "assistant", "content": QUESTIONS[1]}
        )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    render_sidebar()

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("## 🏦 ABC Credit — Loan Application")
    st.caption("Powered by XGBoost · SHAP Explainability · Real-time Decisioning")
    st.divider()

    # ── Render all past messages ───────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Decision render (step 9) ───────────────────────────────────────────────
    step = st.session_state.step
    exited = st.session_state.exited

    if step == 9:
        render_decision(model, label_encoders, feature_columns)
        return  # No chat_input after decision

    if exited:
        return  # No chat_input after age-gate exit

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input("Type your answer and press Enter…")

    if user_input:
        # Display user's message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Validate and process
        ack, err = process_input(user_input)

        if err:
            # Validation failed — re-ask same question
            if st.session_state.exited:
                # Age policy exit
                st.session_state.messages.append({"role": "assistant", "content": err})
            else:
                error_msg = f"⚠️ {err}\n\n{QUESTIONS[step]}"
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
        else:
            # Success — acknowledge and advance
            if ack:
                st.session_state.messages.append(
                    {"role": "assistant", "content": ack}
                )
            st.session_state.step += 1
            next_step = st.session_state.step

            if next_step <= 8:
                st.session_state.messages.append(
                    {"role": "assistant", "content": QUESTIONS[next_step]}
                )
            # step == 9 → decision rendered on next rerun

        st.rerun()


if __name__ == "__main__":
    main()
