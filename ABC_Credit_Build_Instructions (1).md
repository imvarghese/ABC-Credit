# ABC Credit — Loan Chatbot: Complete Build Instructions
> A step-by-step guide for building the ABC Credit AI-powered loan decisioning chatbot using an AI assistant (Claude / Cursor / Copilot etc.)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Business Context & Problem Statement](#2-business-context--problem-statement)
3. [Architecture Decisions (Finalised)](#3-architecture-decisions-finalised)
4. [Model Decision — Why XGBoost](#4-model-decision--why-xgboost)
5. [Feature Engineering — Columns to Drop & Keep](#5-feature-engineering--columns-to-drop--keep)
6. [Chatbot Design — Questions, Flow & UX](#6-chatbot-design--questions-flow--ux)
7. [Tech Stack](#7-tech-stack)
8. [Folder Structure](#8-folder-structure)
9. [Step-by-Step Build Instructions](#9-step-by-step-build-instructions)
10. [Prompts to Give Your AI Assistant](#10-prompts-to-give-your-ai-assistant)
11. [Running the App](#11-running-the-app)
12. [Evaluation Benchmarks](#12-evaluation-benchmarks)
13. [Session Logging Spec](#13-session-logging-spec)
14. [Policy Rules](#14-policy-rules)

---

## 1. Project Overview

**Client:** ABC Credit — an NBFC (Non-Banking Financial Company) with 10+ years of lending experience across 10+ loan categories.

**What we are building:** A Streamlit-based conversational chatbot that:
- Collects 8 key applicant inputs through a guided chat interface
- Scores the application in real time using a trained XGBoost classification model
- Returns an **Approve** or **Decline** decision at the end of the session
- Displays a SHAP-based explanation of the top factors that drove the decision
- Logs every session (inputs, model score, decision) to a CSV repository

**Dataset:** `DATA_FINAL.xlsx` — ~100,000+ rows of historical loan application data with 14 columns.

---

## 2. Business Context & Problem Statement

ABC Credit wants to:
- **Minimise form-fill time** for the customer — too many questions cause drop-off
- **Reduce credit risk** — avoid approving applicants likely to default
- **Automate decisioning** — approve/decline in real time, within seconds of final input
- **Log everything** — every input, feature value, and model score must be recorded

**Key Constraints:**
- Time to completion: **≤ 5 minutes** (8 questions × ~30 seconds each)
- Real-time decision SLA: **≤ 3 seconds** after final input
- Policy: Loans only for applicants **aged 18–60**

---

## 3. Architecture Decisions (Finalised)

| Component | Decision | Reason |
|---|---|---|
| **ML Model** | XGBoost (XGBClassifier) | Best accuracy on tabular financial data, SHAP explainability, handles missing values and class imbalance natively |
| **Benchmark Model** | Logistic Regression | Used to set accuracy floor — XGBoost must beat it by ≥5% |
| **Chatbot Framework** | Streamlit Native (`st.chat_message`, `st.chat_input`, `st.session_state`) | No external chatbot library needed — the flow is a fixed 8-question state machine, not open-ended AI conversation |
| **Explainability** | SHAP (TreeExplainer) | Regulatory requirement for NBFCs — every decline must be explainable |
| **Session Logging** | CSV (`session_log.csv`) | Lightweight, portable, no database setup required |
| **Inference Speed** | XGBoost single-row scoring (~10ms) | Comfortably within the 3-second SLA |

---

## 4. Model Decision — Why XGBoost

### Why NOT other models

| Model | Reason Eliminated |
|---|---|
| **LightGBM** | Leaf-wise growth risks overfitting on correlated financial features (Loan_Amt ↔ LTV_Perc). XGBoost is more battle-tested specifically in credit scoring globally. At 100K rows, LightGBM's speed advantage is irrelevant. |
| **Random Forest** | Uses bagging (averaging), not boosting (sequential error correction). Cannot natively handle missing values. Weaker on minority class (declined loans) even with class weighting. |
| **Logistic Regression** | Assumes a linear decision boundary. Loan risk is inherently non-linear — a high-salary applicant with 90% LTV is a completely different risk to a low-salary applicant with 90% LTV. Cannot capture interaction effects without manual feature engineering. |
| **Neural Networks** | Overkill for 13 structured features at 100K rows. Black box — legally untenable for RBI-regulated credit decisioning. Requires heavy preprocessing, architecture tuning, and produces no explainability. |
| **SVM** | Inference time scales with training set size — at 100K rows, scoring a single new applicant at runtime takes seconds, violating the real-time SLA. No native feature importance. |
| **KNN** | Extremely slow at inference on 100K rows. No feature importance. Cannot handle class imbalance. |
| **Naive Bayes** | Assumes all features are independent — completely false for financial data where salary, loan amount, and LTV are deeply correlated. |

### Why XGBoost Wins

1. **Gradient Boosting** — Each tree corrects the errors of the previous one. This sequential learning is fundamentally more powerful than averaging (Random Forest) on imbalanced classification problems.
2. **Class Imbalance** — `scale_pos_weight` parameter directly corrects for skewed approve/decline ratios without any data resampling (SMOTE etc.).
3. **Missing Values** — Handles blank/null entries natively (your `Region_Level` column has missing entries). No imputation preprocessing needed.
4. **SHAP Explainability** — TreeExplainer produces exact SHAP values in milliseconds. Every decision can be explained as: *"Your loan was declined primarily because your LTV was 92% and you have existing liabilities."* This is required for RBI Fair Practice Code compliance.
5. **Real-time Inference** — Single-row scoring takes ~10ms. Well within the 3-second SLA.
6. **Mixed Feature Types** — Handles numerical and label-encoded categorical features without normalisation.
7. **Proven in Financial Credit Scoring** — XGBoost is the dominant model in credit risk, FICO scoring, and lending decision systems globally.

### XGBoost Hyperparameters to Use

```python
XGBClassifier(
    n_estimators      = 300,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = <neg_count / pos_count>,  # computed from data
    eval_metric       = "logloss",
    random_state      = 42,
    n_jobs            = -1,
)
```

---

## 5. Feature Engineering — Columns to Drop & Keep

### Dataset Columns (Original 14)

| Column | Keep / Drop | Reason |
|---|---|---|
| `Cust Id` | ❌ **Drop** | Identifier only. Zero predictive signal. Must never be a model feature. |
| `Education Level` | ✅ **Keep** | Ordinal 0–5, proxy for income stability and earning potential. Already numerical, use as-is. |
| `Emp_Level` | ✅ **Keep** | Strongest categorical predictor of repayment capacity. Label encode: AG/SA/NR/ST/SE → 0–4. |
| `Birth Year` | ⚠️ **Transform** | Convert to **Age** = 2025 − Birth Year. Raw birth year is meaningless to the model. |
| `Zip` | ❌ **Drop** | Extremely high cardinality (thousands of unique zip codes across 100K rows). Encoding creates noise. Geography is already partially captured by Region_Level. |
| `Demographic_Category` | ❌ **Drop** | This is gender (M/F). **Using gender in credit decisioning is illegal under RBI Fair Practice Code.** Non-negotiable drop. |
| `Product_Cate` | ❌ **Drop** | Represents the loan product type — set by ABC Credit, not the applicant. It is a downstream business field, not an upstream applicant attribute. |
| `Loan_Amt` | ✅ **Keep** | Core input. Direct signal for repayment burden. |
| `LTV_Perc` | ✅ **Keep** | Loan-to-Value ratio — critical credit risk metric. In the chatbot, computed as `(Loan_Amt / Asset_Value) × 100`. |
| `Housing_Category` | ✅ **Keep** | Owner vs. Rent. Homeowners have lower default risk. Binary encode: Owner=1, Rent=0. |
| `Net_Sal` | ✅ **Keep** | Take-home salary. Direct repayment capacity signal. |
| `Region_Level` | ❌ **Drop** | Has blank/missing entries throughout the dataset. Coarser and redundant geography proxy. Two weak geography columns (Zip + Region_Level) together add noise, not signal. |
| `Existing_Liabilities` | ✅ **Keep** | Binary Y/N. Existing debt dramatically changes risk profile. Encode: Y=1, N=0. |
| `Loan_Given` | 🎯 **Target** | Binary target variable: 1 = Approved, 0 = Declined. |

### Final Feature Set Entering XGBoost (8 features)

| Feature | Type | Transformation |
|---|---|---|
| `Age` | Numerical | Derived: `2025 - Birth Year` |
| `Education Level` | Ordinal | Already numerical 0–5, use as-is |
| `Emp_Level` | Categorical | Label encode: AG/SA/NR/ST/SE |
| `Loan_Amt` | Numerical | Use as-is |
| `LTV_Perc` | Numerical | Use as-is (or compute from Loan_Amt / Asset_Value in chatbot) |
| `Housing_Category` | Binary | Owner=1, Rent=0 |
| `Net_Sal` | Numerical | Use as-is |
| `Existing_Liabilities` | Binary | Y=1, N=0 |

**Target variable:** `Loan_Given` (1 = Approved, 0 = Declined)

---

## 6. Chatbot Design — Questions, Flow & UX

### Why Streamlit Native (No External Chatbot Library)

The chatbot is a **fixed state machine**, not an open-ended AI conversation. It asks 8 predetermined questions in a fixed order. There is no ambiguity to resolve, no intent classification needed, and no LLM required. External libraries (LangChain, Rasa, Dialogflow, Botpress) all add enormous dependency weight and complexity with zero functional benefit for this use case.

Streamlit Native components used:
- `st.chat_message("assistant")` — renders bot question bubbles
- `st.chat_message("user")` — renders applicant response bubbles
- `st.chat_input()` — text input bar at the bottom
- `st.session_state` — stores conversation history, current question index, and collected values
- `st.spinner()` — UX delay simulation during model scoring
- `st.progress()` — sidebar progress bar
- `st.pyplot()` — renders SHAP bar chart

### The 8 Questions and Mapping

| # | Question | Maps To Feature | Validation Rule |
|---|---|---|---|
| 1 | Date of birth | Age | DD/MM/YYYY format. Age must be 18–60. Hard exit if outside range. |
| 2 | Employment type (AG/SA/NR/ST/SE) | Emp_Level | Must be one of the 5 valid codes. |
| 3 | Highest education level (0–5) | Education Level | Integer 0–5 only. |
| 4 | Monthly net salary (₹) | Net_Sal | Positive number only. |
| 5 | Loan amount requested (₹) | Loan_Amt | Positive number only. |
| 6 | Current market value of asset/property (₹) | Computes LTV_Perc | Positive number. LTV = (Loan_Amt / Asset_Value) × 100. Show LTV% to applicant immediately. |
| 7 | Housing status (Owner / Rent) | Housing_Category | Must be "Owner" or "Rent" (case-insensitive). |
| 8 | Existing liabilities (Y / N) | Existing_Liabilities | Must be Y or N. |

### Chatbot Session Flow (3 Phases)

```
Phase 1 — Welcome + Age Gate
  └─ Ask DOB
  └─ If age < 18 or > 60 → politely exit, stop session
  └─ If valid → acknowledge age, continue

Phase 2 — Financial Profile (Questions 2–8)
  └─ Ask each question sequentially
  └─ Validate every input before proceeding
  └─ On validation failure → show error message, re-ask same question
  └─ On success → acknowledge with contextual confirmation, advance
  └─ After Q6 (asset value) → compute and display LTV% live

Phase 3 — Scoring + Decision
  └─ Build feature row from 8 answers
  └─ Load pre-trained XGBoost model
  └─ Score with model.predict_proba() 
  └─ Display Approve or Decline card with confidence score
  └─ Display SHAP bar chart (top 5 factors)
  └─ Log session to session_log.csv
```

### Time to Completion Justification
**Target: ≤ 5 minutes**
- 8 questions × ~30 seconds per question (read + type) = 4 minutes
- Model scoring = <1 second
- Total = ~4 minutes with comfortable buffer

### Real-time Decisioning SLA
**Target: ≤ 3 seconds** from final input to decision
- XGBoost single-row inference: ~10ms
- SHAP explanation generation: ~200ms
- Streamlit state update + render: ~500ms
- Session CSV logging: ~50ms
- Total: well under 1 second actual compute; 1.5s artificial spinner for UX

---

## 7. Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | 1.35.0 | Frontend chatbot UI |
| `xgboost` | 2.0.3 | Primary ML model |
| `scikit-learn` | 1.4.2 | LabelEncoder, train_test_split, metrics |
| `pandas` | 2.2.2 | Data manipulation |
| `numpy` | 1.26.4 | Numerical operations |
| `shap` | 0.45.0 | Model explainability (SHAP values) |
| `matplotlib` | 3.8.4 | SHAP bar chart rendering |
| `joblib` | 1.4.2 | Model serialisation (save/load) |
| `openpyxl` | latest | Reading DATA_FINAL.xlsx |

---

## 8. Folder Structure

```
abc_credit_chatbot/
│
├── DATA_FINAL.xlsx            ← Your raw dataset (place here before training)
│
├── train_model.py             ← Run once to train and save the model
├── app.py                     ← Main Streamlit chatbot application
├── requirements.txt           ← All Python dependencies
│
├── model/                     ← Auto-created by train_model.py
│   ├── xgb_loan_model.joblib  ← Trained XGBoost model
│   ├── label_encoders.joblib  ← Fitted LabelEncoders for categorical columns
│   └── feature_columns.joblib ← Ordered list of feature column names
│
└── session_log.csv            ← Auto-created by app.py on first session
```

---

## 9. Step-by-Step Build Instructions

### Step 1 — Set Up Your Environment

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

### Step 2 — Place Your Dataset

Put `DATA_FINAL.xlsx` in the root project folder (same level as `train_model.py`).

Verify the dataset has these column names exactly:

```
Cust Id, Education Level, Emp_Level, Birth Year, Zip,
Demographic_Category, Product_Cate, Loan_Amt, LTV_Perc,
Housing_Category, Net_Sal, Region_Level, Existing_Liabilities, Loan_Given
```

### Step 3 — Train the Model

```bash
python train_model.py --data DATA_FINAL.xlsx
```

This will:
- Load and preprocess the dataset
- Drop: `Cust Id`, `Zip`, `Region_Level`, `Demographic_Category`, `Product_Cate`
- Derive `Age` from `Birth Year`
- Label encode: `Emp_Level`, `Housing_Category`, `Existing_Liabilities`
- Compute `scale_pos_weight` from class distribution
- Train XGBoost with 80/20 train-test split
- Print accuracy, ROC-AUC, confusion matrix, classification report, 5-fold CV
- Save model artifacts to `model/`

Expected output benchmarks:
- Accuracy: ≥ 80%
- ROC-AUC: ≥ 0.85
- 5-Fold CV AUC: within ±0.03 of test AUC (confirms no overfitting)

### Step 4 — Launch the Chatbot

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Step 5 — Test the App

Run through at least 3 test sessions:
1. An applicant **below 18** → should exit immediately with policy message
2. An applicant **above 60** → should exit immediately with policy message
3. A normal applicant → should complete all 8 questions and return a decision

---

## 10. Prompts to Give Your AI Assistant

Use these exact prompts when building with Claude, Cursor, GitHub Copilot, or any AI coding assistant. Feed them in sequence.

---

### Prompt 1 — Project Context (Give this first, always)

```
I am building a loan application chatbot for ABC Credit, an NBFC in India.
The chatbot is a Streamlit app that:
- Collects 8 inputs from a loan applicant through a chat interface
- Scores the application using a pre-trained XGBoost model
- Returns Approve or Decline with a SHAP explanation
- Logs every session to a CSV file

Dataset: DATA_FINAL.xlsx (~100K rows, 14 columns)
Target column: Loan_Given (1=Approved, 0=Declined)
Policy: Only applicants aged 18–60 are eligible.

Key decisions already made:
- Model: XGBoost (XGBClassifier)
- Frontend: Streamlit native only (no LangChain, no Rasa)
- Explainability: SHAP TreeExplainer
- Columns to DROP: Cust Id, Zip, Region_Level, Demographic_Category, Product_Cate
- Birth Year to be converted to Age (2025 - Birth Year)
- Categoricals to encode: Emp_Level, Housing_Category, Existing_Liabilities
```

---

### Prompt 2 — Training Script

```
Write train_model.py that:
1. Reads DATA_FINAL.xlsx using pandas (openpyxl engine)
2. Drops these columns: [Cust Id, Zip, Region_Level, Demographic_Category, Product_Cate]
3. Derives Age = 2025 - Birth Year, then drops Birth Year
4. Label encodes: Emp_Level, Housing_Category, Existing_Liabilities
5. Splits 80/20 with stratify=y and random_state=42
6. Trains XGBClassifier with n_estimators=300, max_depth=6, learning_rate=0.05,
   subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg/pos ratio, eval_metric=logloss
7. Prints: accuracy, ROC-AUC, classification report, confusion matrix, 5-fold CV AUC
8. Saves to model/ folder: xgb_loan_model.joblib, label_encoders.joblib, feature_columns.joblib
Use argparse for --data flag. Add clear print statements at each step.
```

---

### Prompt 3 — Streamlit Chatbot App

```
Write app.py as a Streamlit chatbot for ABC Credit loan applications.

Use ONLY Streamlit native components: st.chat_message, st.chat_input, st.session_state.
Do NOT use LangChain, Rasa, or any external chatbot library.

The chatbot asks exactly 8 questions in this order:
1. Date of birth (DD/MM/YYYY) → derive Age, enforce 18–60 policy (exit if outside)
2. Employment type → AG/SA/NR/ST/SE only
3. Education level → integer 0–5 only
4. Monthly net salary → positive number in ₹
5. Loan amount → positive number in ₹
6. Asset/property value → positive number, compute LTV% = (loan/asset)*100, show to user
7. Housing status → Owner or Rent only
8. Existing liabilities → Y or N only

After Q8:
- Load model/xgb_loan_model.joblib, model/label_encoders.joblib, model/feature_columns.joblib
- Build a single feature row matching training column order
- Score with model.predict_proba()
- Display Approve or Decline card with confidence score
- Show SHAP TreeExplainer bar chart (top 5 factors, green=positive, red=negative)
- Log session to session_log.csv with: timestamp, all 8 inputs, LTV_Perc, model probability, decision

Add a sidebar with: progress bar, checklist of 8 fields, Start Over button.
Handle validation errors by re-asking the same question with an error message.
```

---

### Prompt 4 — Validation Logic

```
Add input validation functions to app.py:
- DOB: parse DD/MM/YYYY, compute age, reject if age < 18 or > 60 with specific message
- Emp_Level: must be in [AG, SA, NR, ST, SE] (case-insensitive)
- Education: must be integer in range 0–5
- Net_Sal, Loan_Amt, Asset_Value: must be positive numbers, strip commas, reject non-numeric
- Housing_Category: must be "owner" or "rent" (case-insensitive), store as "Owner"/"Rent"
- Existing_Liabilities: must be "y" or "n" (case-insensitive), store as "Y"/"N"
Each validator returns (value, error_message) where error_message is None on success.
```

---

### Prompt 5 — Session Logging

```
Add a log_session() function to app.py that appends a row to session_log.csv.
Columns to log:
timestamp, Age, Emp_Level, Education_Level, Net_Sal, Loan_Amt, Asset_Value,
LTV_Perc, Housing_Category, Existing_Liabilities, Model_Probability, Decision

Create the CSV with headers on first run. Append on subsequent runs.
Call this function immediately after the model scores the application.
```

---

### Prompt 6 — Bug Fixes and Edge Cases

```
Review app.py for these edge cases:
1. What if the model/ folder doesn't exist? Show a clear warning and st.stop()
2. What if an encoder key is missing for a new category? Default to -1 (unknown)
3. What if LTV_Perc exceeds 100%? Cap it at 100%
4. What if the user refreshes mid-session? st.session_state handles this — confirm it resets cleanly
5. What if session_log.csv is open in Excel when writing? Wrap in try/except and show a warning
6. Ensure st.rerun() is called after every validated input to update the chat display
```

---

### Prompt 7 — SHAP Chart Rendering

```
Add a SHAP explanation chart to app.py after the Approve/Decline decision:
- Use shap.TreeExplainer(model) with the trained XGBoost model
- Extract shap_values for the single applicant feature row
- Build a DataFrame of Feature vs SHAP Value, sorted by absolute value, top 5 rows
- Plot a horizontal bar chart using matplotlib:
  - Green bars for positive SHAP values (favourable)
  - Red bars for negative SHAP values (unfavourable)
  - Vertical line at 0
  - Title: "Top Factors — Positive = Favourable, Negative = Unfavourable"
- Render with st.pyplot() inside st.chat_message("assistant")
- Add heading: "Key Factors That Influenced This Decision"
```

---

## 11. Running the App

```bash
# Fresh install
pip install -r requirements.txt

# Train model (run once — or re-run if dataset changes)
python train_model.py --data DATA_FINAL.xlsx

# Start chatbot
streamlit run app.py
```

App will be available at: `http://localhost:8501`

To run on a specific port:
```bash
streamlit run app.py --server.port 8080
```

---

## 12. Evaluation Benchmarks

The training script prints all metrics. Here is what to expect and what to flag:

| Metric | Acceptable | Good | Flag if Below |
|---|---|---|---|
| Test Accuracy | ≥ 75% | ≥ 85% | < 70% |
| ROC-AUC Score | ≥ 0.80 | ≥ 0.90 | < 0.75 |
| 5-Fold CV AUC | Within ±0.05 of test AUC | Within ±0.02 | Gap > 0.08 (overfitting) |
| Precision (Declined) | ≥ 0.70 | ≥ 0.85 | < 0.65 |
| Recall (Declined) | ≥ 0.65 | ≥ 0.80 | < 0.60 |

If ROC-AUC is below 0.75, check:
1. Is `Loan_Given` actually balanced? Print `df['Loan_Given'].value_counts()`
2. Are there data leakage columns still in features? (e.g., any column derived from the target)
3. Is the label encoding consistent between training and inference?

---

## 13. Session Logging Spec

Every completed chatbot session writes one row to `session_log.csv` with these columns:

| Column | Description |
|---|---|
| `timestamp` | Format: `YYYY-MM-DD HH:MM:SS` |
| `Age` | Derived from DOB input |
| `Emp_Level` | Raw code: AG / SA / NR / ST / SE |
| `Education_Level` | Integer 0–5 |
| `Net_Sal` | Monthly take-home salary in ₹ |
| `Loan_Amt` | Loan amount requested in ₹ |
| `Asset_Value` | Asset/property value entered by applicant in ₹ |
| `LTV_Perc` | Computed: `(Loan_Amt / Asset_Value) × 100`, capped at 100 |
| `Housing_Category` | Owner or Rent |
| `Existing_Liabilities` | Y or N |
| `Model_Probability` | XGBoost `predict_proba` score for class 1 (Approved), rounded to 4 decimal places |
| `Decision` | APPROVED or DECLINED |

Sessions that exit early (age policy violation) are **not** logged.

---

## 14. Policy Rules

These rules must be hardcoded and cannot be overridden by the model:

| Rule | Implementation |
|---|---|
| Minimum age: 18 years | Validate in DOB step. If age < 18, show message, set `exited=True`, stop session. |
| Maximum age: 60 years | Validate in DOB step. If age > 60, show message, set `exited=True`, stop session. |
| Gender cannot be used | `Demographic_Category` column dropped before training and never collected in chatbot. |
| All sessions logged | `log_session()` called immediately after every scored decision, before rendering result. |

---

*Document prepared for ABC Credit Loan Chatbot project — Mesa School of Business, PGP Startup Leadership 2025–2026.*
