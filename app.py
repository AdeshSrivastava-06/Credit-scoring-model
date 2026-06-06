# ============================================================
#  CreditIQ — Streamlit Credit Score Predictor
#  Run: streamlit run app.py
# ============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

.stApp {
    background: #f0f4f8;
    color: #1a202c;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1a365d 0%, #2a4a7f 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e0 !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 60%, #3182ce 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(26,54,93,0.3);
}
.hero::after {
    content: '💳';
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.15;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: white;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: #bee3f8;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* Cards */
.card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: #2b6cb0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #ebf8ff;
}

/* Result cards */
.result-section {
    width: 100%;
    margin-top: 0.5rem;
}
.result-panel {
    width: 100%;
    min-height: 260px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 1rem;
}
.result-good {
    background: linear-gradient(135deg, #f0fff4, #c6f6d5);
    border: 2px solid #48bb78;
    border-radius: 16px;
    padding: 2.2rem 1.8rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(72,187,120,0.2);
}
.result-standard {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 2px solid #f6ad55;
    border-radius: 16px;
    padding: 2.2rem 1.8rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(246,173,85,0.2);
}
.result-poor {
    background: linear-gradient(135deg, #fff5f5, #fed7d7);
    border: 2px solid #fc8181;
    border-radius: 16px;
    padding: 2.2rem 1.8rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(252,129,129,0.2);
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0.3rem 0;
}
.result-emoji { font-size: 3rem; line-height: 1; }
.result-conf {
    font-size: 0.85rem;
    color: #718096;
    margin-top: 0.3rem;
}

/* Prob bar */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.7rem;
}
.prob-label {
    font-size: 0.82rem;
    font-weight: 500;
    width: 70px;
    color: #4a5568;
}
.prob-track {
    flex: 1;
    background: #edf2f7;
    border-radius: 50px;
    height: 8px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 50px;
}
.prob-pct {
    font-size: 0.8rem;
    font-weight: 600;
    width: 42px;
    text-align: right;
}

/* Advice box */
.advice {
    background: #ebf8ff;
    border-left: 4px solid #3182ce;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.87rem;
    color: #2c5282;
    margin-top: 1rem;
    line-height: 1.5;
}

/* Metric pill */
.metric-pill {
    display: inline-block;
    background: #ebf8ff;
    color: #2b6cb0;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.3rem 0.8rem;
    border-radius: 50px;
    margin-bottom: 0.2rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #3182ce) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(49,130,206,0.35) !important;
}

/* Inputs */
div[data-testid="stSlider"] label,
.stNumberInput label,
.stSelectbox label {
    font-size: 0.82rem !important;
    color: #4a5568 !important;
    font-weight: 500 !important;
}

hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model     = joblib.load('credit_scoring_rf_model.pkl')
        prep      = joblib.load('preprocessor.pkl')
        le        = joblib.load('label_encoder.pkl')
        return model, prep, le, None
    except FileNotFoundError as e:
        return None, None, None, str(e)

model, preprocessor, label_encoder, load_error = load_artifacts()


# ─────────────────────────────────────────────
# EXACT COLUMN ORDER (must match training)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'Age',
    'Occupation',
    'Annual_Income',
    'Monthly_Inhand_Salary',
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',
    'Num_of_Loan',
    'Delay_from_due_date',
    'Num_of_Delayed_Payment',
    'Changed_Credit_Limit',
    'Num_Credit_Inquiries',
    'Credit_Mix',
    'Outstanding_Debt',
    'Credit_Utilization_Ratio',
    'Payment_of_Min_Amount',
    'Total_EMI_per_month',
    'Amount_invested_monthly',
    'Payment_Behaviour',
    'Monthly_Balance',
    'Debt_to_Income',
    'EMI_to_Salary_Ratio',
    'Savings_Rate',
    'Delay_per_Loan',
    'Cards_per_Bank',
    'Credit_History_Months'
]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.2rem 0 0.8rem">
        <div style="font-size:2.8rem">💳</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.5rem;
                    font-weight:800; color:white; margin-top:0.3rem">
            CreditIQ
        </div>
        <div style="color:#90cdf4; font-size:0.75rem; margin-top:4px">
            ML Credit Assessment
        </div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.15)!important; margin:0.5rem 0 1rem">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Predict", "📊  Model Info", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15)!important'>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#90cdf4; line-height:1.8">
        <b style="color:white">Model</b><br>Random Forest<br><br>
        <b style="color:white">Accuracy</b><br>79.05%<br><br>
        <b style="color:white">Dataset</b><br>100,000 rows<br><br>
        <b style="color:white">Classes</b><br>Good · Standard · Poor
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────
if "Predict" in page:

    st.markdown("""
    <div class="hero">
        <div class="hero-title">Credit Score Predictor</div>
        <div class="hero-sub">
            Enter financial details to assess creditworthiness
            using a Random Forest model trained on 100,000 records
        </div>
    </div>
    """, unsafe_allow_html=True)

    if load_error:
        st.error(f"⚠️ Could not load model: {load_error}")
        st.info("Make sure these files are in the same folder as app.py:\n"
                "- credit_scoring_rf_model.pkl\n"
                "- preprocessor.pkl\n- label_encoder.pkl")
        st.stop()

    # ── Section 1: Personal ──
    st.markdown('<div class="card"><div class="card-title">👤 Personal Information</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 70, 30)
        occupation = st.selectbox("Occupation", [
            'Scientist', 'Teacher', 'Engineer', 'Entrepreneur',
            'Developer', 'Lawyer', 'Media_Manager', 'Doctor',
            'Journalist', 'Manager', 'Accountant', 'Musician',
            'Mechanic', 'Writer', 'Architect', 'Nurse'
        ])
    with c2:
        annual_income = st.number_input("Annual Income ($)", 7000, 2500000, 45000, step=1000)
        monthly_salary = st.number_input("Monthly In-hand Salary ($)", 500, 15000, 3500, step=100)
    with c3:
        monthly_balance = st.number_input("Monthly Balance ($)", 0.0, 1700.0, 350.0, step=10.0)
        amount_invested = st.number_input("Amount Invested Monthly ($)", 0.0, 2000.0, 150.0, step=10.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2: Banking ──
    st.markdown('<div class="card"><div class="card-title">🏦 Banking & Loans</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        num_bank_accounts = st.slider("Bank Accounts", 0, 13, 4)
        num_credit_cards  = st.slider("Credit Cards", 1, 12, 4)
    with c2:
        interest_rate = st.slider("Interest Rate (%)", 1, 38, 14)
        num_loans     = st.slider("Number of Loans", 0, 9, 3)
    with c3:
        num_loan_types      = st.slider("Loan Types", 1, 9, 3)
        num_credit_inquiries = st.slider("Credit Inquiries", 0, 18, 5)
    with c4:
        outstanding_debt     = st.number_input("Outstanding Debt ($)", 0.0, 5000.0, 1000.0, step=50.0)
        changed_credit_limit = st.number_input("Credit Limit Change (%)", 0.0, 40.0, 10.0, step=0.5)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3: Payment History ──
    st.markdown('<div class="card"><div class="card-title">📅 Payment History & Behaviour</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        delay_from_due      = st.slider("Avg Days Late on Payment", 0, 55, 15)
        num_delayed_payments = st.slider("No. of Delayed Payments", 0, 31, 10)
    with c2:
        credit_utilization = st.slider("Credit Utilization Ratio (%)", 22, 48, 32)
        total_emi          = st.number_input("Total EMI per Month ($)", 0.0, 360.0, 80.0, step=5.0)
    with c3:
        payment_of_min = st.selectbox("Pays Minimum Amount?", ['Yes', 'No', 'NM'])
        payment_behaviour = st.selectbox("Payment Behaviour", [
            'High_spent_Small_value_payments',
            'Low_spent_Large_value_payments',
            'Low_spent_Medium_value_payments',
            'High_spent_Medium_value_payments',
            'Low_spent_Small_value_payments',
            'High_spent_Large_value_payments'
        ])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 4: Credit Profile ──
    st.markdown('<div class="card"><div class="card-title">📊 Credit Profile</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        credit_mix            = st.selectbox("Credit Mix", ['Good', 'Standard', 'Bad'])
        credit_history_months = st.slider("Credit History (months)", 0, 400, 180)
    with c2:
        st.markdown("*These are auto-calculated from your inputs above.*")
        debt_to_income      = outstanding_debt / (annual_income + 1)
        emi_to_salary_ratio = total_emi / (monthly_salary + 1)
        savings_rate        = amount_invested / (monthly_salary + 1)
        delay_per_loan      = num_delayed_payments / (num_loans + 1)
        cards_per_bank      = num_credit_cards / (num_bank_accounts + 1)

        st.markdown(f"""
        <div style="font-size:0.82rem; color:#4a5568; line-height:2">
            Debt-to-Income Ratio : <b>{debt_to_income:.4f}</b><br>
            EMI-to-Salary Ratio  : <b>{emi_to_salary_ratio:.4f}</b><br>
            Savings Rate         : <b>{savings_rate:.4f}</b><br>
            Delay per Loan       : <b>{delay_per_loan:.4f}</b><br>
            Cards per Bank       : <b>{cards_per_bank:.4f}</b>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict Button ──
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        predict = st.button("🔍 Predict Credit Score")

    if predict:
        # Build input dataframe — exact column order must match FEATURE_COLS
        
        input_dict = {
    'Age': age,
    'Occupation': occupation,
    'Annual_Income': annual_income,
    'Monthly_Inhand_Salary': monthly_salary,
    'Num_Bank_Accounts': num_bank_accounts,
    'Num_Credit_Card': float(num_credit_cards),
    'Interest_Rate': interest_rate,
    'Num_of_Loan': float(num_loans),
    'Delay_from_due_date': delay_from_due,
    'Num_of_Delayed_Payment': float(num_delayed_payments),
    'Changed_Credit_Limit': changed_credit_limit,
    'Num_Credit_Inquiries': num_credit_inquiries,
    'Credit_Mix': credit_mix,
    'Outstanding_Debt': outstanding_debt,
    'Credit_Utilization_Ratio': credit_utilization,
    'Payment_of_Min_Amount': payment_of_min,
    'Total_EMI_per_month': total_emi,
    'Amount_invested_monthly': amount_invested,
    'Payment_Behaviour': payment_behaviour,
    'Monthly_Balance': monthly_balance,

    # Engineered Features
    'Debt_to_Income': outstanding_debt / (annual_income + 1),
    'EMI_to_Salary_Ratio': total_emi / (monthly_salary + 1),
    'Savings_Rate': amount_invested / (monthly_salary + 1),
    'Delay_per_Loan': num_delayed_payments / (num_loans + 1),
    'Cards_per_Bank': num_credit_cards / (num_bank_accounts + 1),

    'Credit_History_Months': credit_history_months
}

        input_df = pd.DataFrame([input_dict])[FEATURE_COLS]

        try:
            X_enc      = preprocessor.transform(input_df)
            pred_idx   = model.predict(X_enc)[0]
            proba      = model.predict_proba(X_enc)[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = proba[pred_idx] * 100

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card-title" style="font-family:Syne,sans-serif;'
                        'font-size:0.9rem;color:#2b6cb0;font-weight:700;'
                        'text-transform:uppercase;letter-spacing:0.08em">'
                        '🎯 Prediction Result</div>', unsafe_allow_html=True)

            if pred_label == 'Good':
                css   = 'result-good'
                emoji = '✅'
                color = '#276749'
                advice = ("Your credit profile looks strong. Keep maintaining "
                          "timely payments and low debt utilization to stay here.")
            elif pred_label == 'Standard':
                css   = 'result-standard'
                emoji = '⚠️'
                color = '#744210'
                advice = ("Average credit profile. Reduce outstanding debt and "
                          "avoid delayed payments to improve your score.")
            else:
                css   = 'result-poor'
                emoji = '❌'
                color = '#742a2a'
                advice = ("Below-average credit profile. Prioritise clearing "
                          "delayed payments and reducing credit utilization urgently.")

            st.markdown(f"""
            <div class="result-section">
                <div class="result-panel">
                    <div class="{css}">
                        <div class="result-emoji">{emoji}</div>
                        <div class="result-label" style="color:{color}">{pred_label}</div>
                    </div>
                    <div class="advice">💡 {advice}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Check that preprocessor.pkl was saved after your final "
                    "ColumnTransformer fit.")


# ─────────────────────────────────────────────
# PAGE: MODEL INFO
# ─────────────────────────────────────────────
elif "Model" in page:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Model Information</div>
        <div class="hero-sub">Random Forest performance metrics and configuration</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [("Accuracy","79.05%"), ("F1 Score","79.03%"),
               ("Trees","200"), ("Features","41")]
    for col, (label, val) in zip([c1,c2,c3,c4], metrics):
        col.markdown(f"""
        <div class="card" style="text-align:center">
            <div class="metric-pill">{val}</div>
            <div style="font-size:0.78rem;color:#718096;
                        text-transform:uppercase;letter-spacing:0.06em">
                {label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">Best Parameters Found</div>',
                unsafe_allow_html=True)
    params = {
        'n_estimators'     : 200,
        'max_depth'        : 'None (unlimited)',
        'max_features'     : 'log2',
        'min_samples_split': 2,
        'min_samples_leaf' : 1,
        'bootstrap'        : True,
        'class_weight'     : 'balanced'
    }
    param_df = pd.DataFrame(params.items(), columns=['Parameter', 'Value'])
    st.dataframe(param_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">Classification Report</div>',
                unsafe_allow_html=True)
    report_data = {
        'Class'    : ['Good (0)', 'Poor (1)', 'Standard (2)', 'Weighted Avg'],
        'Precision': [0.75, 0.78, 0.81, 0.79],
        'Recall'   : [0.72, 0.80, 0.81, 0.79],
        'F1-Score' : [0.73, 0.79, 0.81, 0.79],
        'Support'  : [3566, 5799, 10635, 20000]
    }
    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">Feature Groups</div>',
                unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown("**Numerical (passthrough)**")
        num = ['Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts',
               'Num_Credit_Card','Interest_Rate','Num_of_Loan','Delay_from_due_date',
               'Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries',
               'Outstanding_Debt','Credit_Utilization_Ratio','Total_EMI_per_month',
               'Amount_invested_monthly','Monthly_Balance','Debt_to_Income',
               'EMI_to_Salary_Ratio','Savings_Rate','Delay_per_Loan','Cards_per_Bank']
        for f in num: st.markdown(f"• `{f}`")
    with fc2:
        st.markdown("**Ordinal Encoded**")
        for f in ['Credit_Mix','Payment_of_Min_Amount','Credit_History_Months']:
            st.markdown(f"• `{f}`")
    with fc3:
        st.markdown("**One-Hot Encoded**")
        for f in ['Occupation','Payment_Behaviour']:
            st.markdown(f"• `{f}`")
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
elif "About" in page:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">About This Project</div>
        <div class="hero-sub">Credit Score Classification using Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">Project Overview</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        This app predicts whether a person's credit score is
        **Good**, **Standard**, or **Poor** based on their financial history.

        **Dataset:** Kaggle Credit Score Classification

        **Problem Type:** Multi-class Classification (3 classes)

        **Training Data:** 80,000 rows · **Test Data:** 20,000 rows
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">Tech Stack</div>',
                    unsafe_allow_html=True)
        stack = {
            "🐍 Python 3.10"       : "Core language",
            "🤖 Scikit-learn"      : "Random Forest + Pipeline",
            "🌊 Streamlit"         : "Frontend UI",
            "🐼 Pandas / NumPy"    : "Data processing",
            "💾 Joblib"            : "Model persistence",
        }
        for k, v in stack.items():
            st.markdown(f"**{k}** — {v}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">ML Pipeline</div>',
                unsafe_allow_html=True)
    st.code("""
Raw CSV (100k rows)
    ↓ Drop: ID, Name, SSN, Customer_ID, Month, Type_of_Loan
    ↓ Fix dirty values (negative age, garbage strings)
    ↓ Winsorization (IQR-based outlier capping)
    ↓ Feature Engineering (Debt_to_Income, EMI_to_Salary, etc.)
    ↓ ColumnTransformer:
        ├── Numerical (21 cols) → passthrough
        ├── Ordinal  (3 cols)   → OrdinalEncoder
        └── OneHot   (2 cols)   → OneHotEncoder  [41 total features]
    ↓ Train/Test Split (80/20, stratified)
    ↓ RandomForestClassifier (200 trees, class_weight=balanced)
    ↓ RandomizedSearchCV (50 iterations, cv=3)
    ↓ Final Accuracy: 79.05% | F1: 79.03%
    """, language="text")
    st.markdown('</div>', unsafe_allow_html=True)