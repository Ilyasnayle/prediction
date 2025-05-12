# --------------------- Page Config ---------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from PIL import Image
from dotenv import load_dotenv
import os
from langchain_fireworks import ChatFireworks
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain_core.messages import SystemMessage

load_dotenv()

st.set_page_config(
    page_title="Smart Subscription Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------- Custom CSS Styling ---------------------
st.markdown("""
    <style>
        .main {background-color: #fafafa;}
        h1, h2, h4 {color: #333333; font-family: 'Segoe UI', sans-serif;}
        .stButton>button {background-color: #4CAF50; color: white; font-weight: bold;}
        .stMetricLabel, .stMetricValue {color: #333333;}
    </style>
""", unsafe_allow_html=True)

# --------------------- Sidebar ---------------------
st.sidebar.title("🔍 Choose Mode")
mode = st.sidebar.radio(
    "What would you like to do?",
    ["🏠 Home", "📞 Predict: Before Call", "📊 Predict: After Call", "🤖 Ask the Assistant"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ by Ilyas Nayle")

# --------------------- Header ---------------------
st.markdown("""
<h1 style='text-align: center;'>🧠 Smart Subscription Predictor</h1>
<h4 style='text-align: center; color: gray;'>Bank Marketing | ADS542 Final Project</h4>
---
""", unsafe_allow_html=True)

# --------------------- Home ---------------------
if mode == "🏠 Home":
    st.markdown("""
    ## 📘 About the Project
    This application is developed as part of the ADS542 Final Project. It is designed to predict whether a customer is likely to subscribe to a term deposit based on various input features.

    ### 🔍 What you can do here:
    - **📞 Predict: Before Call** → Predict customer subscription likelihood before calling.
    - **📊 Predict: After Call** → Predict based on call duration and outcomes.
    - **🤖 Ask the Assistant** → Chat with an AI-powered assistant trained on your dataset.

    ### 🧠 Under the Hood:
    - Two ensemble models used: **CatBoost** and **XGBoost**
    - Data preprocessed and engineered into two modes: realistic (before call) and full (after call)
    - Assistant is powered by **Groq LLMs** and understands both dataset and project context

    ### 🧾 Feature Descriptions:
    - **Age**: Age of the customer.
    - **Job**: Type of job (e.g., admin., student, technician).
    - **Marital**: Marital status of the customer.
    - **Education**: Education level.
    - **Housing/Loan**: Does the customer have housing or personal loans?
    - **Contact Type**: Communication type during the campaign.
    - **Month/Day of Week**: Timing of the last contact.
    - **Campaign/Pdays/Previous**: Contact history details.
    - **Economic Indicators**: Key macroeconomic variables affecting outcome.
    - **Call Duration**: Duration of contact (only for after-call predictions).

    ### 👤 Created by:
    Ilyas Nayle — ML Engineer & Data Scientist
    """)

# --------------------- Load Models and Preprocessors ---------------------
@st.cache_resource
def load_models():
    xgb_real = joblib.load("models/xgboost_real.pkl")
    cb_real = CatBoostClassifier()
    cb_real.load_model("models/catboost_real.cbm")
    prep_real = joblib.load("preprocessors/preprocessor_real.pkl")
    xgb_full = joblib.load("models/xgboost_full.pkl")
    cb_full = CatBoostClassifier()
    cb_full.load_model("models/catboost_full.cbm")
    prep_full = joblib.load("preprocessors/preprocessor_full.pkl")
    return xgb_real, cb_real, prep_real, xgb_full, cb_full, prep_full

xgb_real, cb_real, prep_real, xgb_full, cb_full, prep_full = load_models()

# --------------------- Prediction: Before Call ---------------------
if mode == "📞 Predict: Before Call":
    st.subheader("📞 Predict if a customer will subscribe — before the call")
    st.info("This uses the Realistic Model (no call duration features).")
    st.caption("ℹ️ Hover over the (i) icon next to each input field to learn more.")
    with st.expander("ℹ️ Feature Guide"):
     st.markdown("""
    **Features used in this model:**

    - **AGE**: age — Age of the customer
    - **JOB**: job — Type of job (e.g., admin., technician)
    - **MARITAL**: marital — Marital status
    - **EDUCATION**: education — Education level
    - **HOUSING**: housing — Has housing loan?
    - **LOAN**: loan — Has personal loan?
    - **CONTACT**: contact — Type of communication used during the campaign
    - **MONTH**: month — Month of last contact
    - **DAY_OF_WEEK**: day_of_week — Day of week of last contact
    - **CAMPAIGN**: campaign — Number of contacts during this campaign
    - **PDAYS**: pdays — Days since last contact (999 means never contacted)
    - **PREVIOUS**: previous — Number of contacts before this campaign
    - **POUTCOME**: poutcome — Outcome of previous campaign
    - **EMP.VAR.RATE**: emp.var.rate — Quarterly employment variation rate
    - **CONS.PRICE.IDX**: cons.price.idx — Monthly consumer price index
    - **CONS.CONF.IDX**: cons.conf.idx — Consumer confidence index
    - **EURIBOR3M**: euribor3m — 3-month Euribor rate
    - **NR.EMPLOYED**: nr.employed — Average number of employees in the economy
    - **HAS_PREVIOUS_CONTACT**: has_previous_contact — 1 if previous > 0 (engineered feature)
    - **DURATION** *(After Call only)*: duration — Duration of the last call in seconds
    - **CALL_SUCCESS** *(After Call only)*: call_success — 1 if call duration > 0 (engineered)
    """)


    with st.form("realistic_form"):
        st.markdown("### ✍️ Customer Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 18, 100, step=1, help="Customer's age in years.")
            job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
                                       'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'],
                                 help="Occupation type (e.g., admin., student, technician).")
            marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'],
                                   help="Customer's marital status.")
            education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                                   'professional.course', 'university.degree', 'illiterate', 'unknown'],
                                     help="Customer's highest level of education.")

        with col2:
            housing = st.selectbox("Housing Loan", ['yes', 'no', 'unknown'],
                                   help="Does the customer have a housing loan?")
            loan = st.selectbox("Personal Loan", ['yes', 'no', 'unknown'],
                                help="Does the customer have a personal loan?")
            contact = st.selectbox("Contact Type", ['cellular', 'telephone'],
                                   help="Type of communication used during the campaign.")
            month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                                 help="Month of last campaign contact.")

        with col3:
            day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'],
                                       help="Day of week of last contact.")
            campaign = st.number_input("Number of Contacts in Campaign", 1, 100, step=1,
                                       help="Number of contacts performed during this campaign.")
            pdays = st.number_input("Days Since Last Contact", 0, 999, step=1,
                                    help="Days since the client was last contacted (999 means never).")
            previous = st.number_input("Previous Contacts", 0, 50, step=1,
                                       help="Number of contacts performed before this campaign.")

        st.markdown("### 📊 Economic Indicators")
        col4, col5 = st.columns(2)
        with col4:
            emp_var_rate = st.number_input("Employment Variation Rate", -3.0, 2.0, step=0.1,
                                          help="Quarterly employment variation (economic indicator).")
            cons_price_idx = st.number_input("Consumer Price Index", 92.0, 95.0, step=0.01,
                                             help="Monthly consumer price index.")
        with col5:
            cons_conf_idx = st.number_input("Consumer Confidence Index", -50.0, -20.0, step=0.1,
                                           help="Consumer confidence indicator.")
            euribor3m = st.number_input("3-Month Euribor Rate", 0.0, 5.0, step=0.01,
                                       help="Interest rate for 3-month Euribor.")
            nr_employed = st.number_input("Number of Employees", 4800.0, 5500.0, step=1.0,
                                          help="Average number of employees in the economy.")

        poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'],
                                help="Outcome of the previous marketing campaign.")


        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        input_data = pd.DataFrame([{ 'age': age, 'job': job, 'marital': marital, 'education': education,
                                     'housing': housing, 'loan': loan, 'contact': contact, 'month': month, 'day_of_week': day_of_week,
                                     'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
                                     'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
                                     'euribor3m': euribor3m, 'nr.employed': nr_employed,
                                     'call_success': 0, 'has_previous_contact': 1 if previous > 0 else 0 }
                                   ])

        X_input = prep_real.transform(input_data)
        pred_xgb = xgb_real.predict_proba(X_input)[:, 1]
        pred_cb = cb_real.predict_proba(X_input)[:, 1]
        ensemble_pred = (pred_xgb + pred_cb) / 2
        result = int(ensemble_pred[0] >= 0.40)

        st.markdown("---")
        st.subheader("📈 Prediction Result:")
        if result == 1:
            st.success("✅ The customer is likely to subscribe!")
        else:
            st.error("❌ The customer is unlikely to subscribe.")
        st.metric("Confidence Score", f"{ensemble_pred[0]*100:.2f}%")

        # ── NEW: Realistic Model Feature Importance ──
        st.markdown("### 🔍 Realistic Model Top 10 Features")
        feat_names = prep_real.get_feature_names_out()
        importances = xgb_real.feature_importances_
        imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}) \
            .sort_values('importance', ascending=False).head(10)
        st.bar_chart(imp_df.set_index('feature'))

       # 💡 Why & How to Improve
        st.markdown("#### 💡 Why & How to Improve the result!!S")
        df_in = input_data.iloc[0]

        if result == 1:
            st.write("✅ This prediction is positive due to favorable customer characteristics and marketing context.")
        else:
            st.write("❌ This prediction is negative likely due to low historical engagement or weak campaign targeting.")

        # 1. Contact history
        if df_in['has_previous_contact'] == 0:
            st.write("- **Reach out at least once before launching the main campaign** to improve trust and response rate.")

        # 2. Past campaign result
        if df_in['poutcome'] != 'success':
            st.write("- **Analyze what didn’t work in past attempts** and refine your approach or timing.")

        # 3. Call intensity
        if df_in['campaign'] > 5:
            st.write("- **Limit repeated calls** — too many attempts may reduce customer interest.")

        # 4. Recency
        if df_in['pdays'] == 999:
            st.write("- **Customer hasn’t been contacted recently** — build fresh interest before the offer.")

        # 5. Customer education
        if df_in['education'] in ['basic.4y', 'basic.6y', 'illiterate']:
            st.write("- **Simplify the message** and highlight benefits clearly.")

        # 6. Job type
        if df_in['job'] in ['unemployed', 'student', 'housemaid']:
            st.write("- **Focus on low-entry, flexible deposit options** to appeal to this segment.")

        # 7. Economic indicators
        if df_in['emp.var.rate'] < 0:
            st.write("- **Emphasize security and low-risk savings**, especially in uncertain job markets.")

        if df_in['cons.conf.idx'] < -40:
            st.write("- **Boost consumer trust** by providing successful case examples or testimonials.")

# --------------------- After Call ---------------------
elif mode == "📊 Predict: After Call":
    st.subheader("📊 Predict if a customer will subscribe — after the call")
    st.info("This uses the Full Model (with duration, call success, etc).")
    st.caption("ℹ️ Hover over the (i) icon next to each input field to learn more.")

    with st.expander("ℹ️ Feature Guide"):
        st.markdown("""
    **Features used in this model:**
    - **AGE**: age — Age of the customer
    - **JOB**: job — Type of job (e.g., admin., technician)
    - **MARITAL**: marital — Marital status
    - **EDUCATION**: education — Education level
    - **HOUSING**: housing — Has housing loan?
    - **LOAN**: loan — Has personal loan?
    - **CONTACT**: contact — Type of communication used during the campaign
    - **MONTH**: month — Month of last contact
    - **DAY_OF_WEEK**: day_of_week — Day of week of last contact
    - **CAMPAIGN**: campaign — Number of contacts during this campaign
    - **PDAYS**: pdays — Days since last contact (999 means never contacted)
    - **PREVIOUS**: previous — Number of contacts before this campaign
    - **POUTCOME**: poutcome — Outcome of previous campaign
    - **EMP.VAR.RATE**: emp.var.rate — Quarterly employment variation rate
    - **CONS.PRICE.IDX**: cons.price.idx — Monthly consumer price index
    - **CONS.CONF.IDX**: cons.conf.idx — Consumer confidence index
    - **EURIBOR3M**: euribor3m — 3-month Euribor rate
    - **NR.EMPLOYED**: nr.employed — Average number of employees in the economy
    - **HAS_PREVIOUS_CONTACT**: has_previous_contact — 1 if previous > 0 (engineered feature)
    - **DURATION** *(After Call only)*: duration — Duration of the last call in seconds
    - **CALL_SUCCESS** *(After Call only)*: call_success — 1 if call duration > 0 (engineered)
    """)   


    @st.cache_resource
    def load_full_models():
        xgb_model = joblib.load("models/xgboost_full.pkl")
        cb_model = CatBoostClassifier()
        cb_model.load_model("models/catboost_full.cbm")
        preprocessor = joblib.load("preprocessors/preprocessor_full.pkl")
        return xgb_model, cb_model, preprocessor

    xgb_full, cb_full, prep_full = load_full_models()

    with st.form("full_model_form"):
        st.markdown("### 🖍️ Customer Information (After Call)")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 18, 100, step=1, help="Customer's age in years.")
            job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                                       'management', 'retired', 'self-employed', 'services',
                                       'student', 'technician', 'unemployed', 'unknown'],
                                 help="Occupation type (e.g., admin., student, technician).")
            marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'],
                                   help="Customer's marital status.")
            education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                                   'professional.course', 'university.degree',
                                                   'illiterate', 'unknown'],
                                     help="Customer's highest level of education.")

        with col2:
            housing = st.selectbox("Housing Loan", ['yes', 'no', 'unknown'], help="Does the customer have a housing loan?")
            loan = st.selectbox("Personal Loan", ['yes', 'no', 'unknown'], help="Does the customer have a personal loan?")
            contact = st.selectbox("Contact Type", ['cellular', 'telephone'], help="Type of communication used during the campaign.")
            month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                                 help="Month of last campaign contact.")

        with col3:
            day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'], help="Day of week of last contact.")
            duration = st.number_input("Call Duration (seconds)", 0, 2000, step=1,
                                      help="Duration of the marketing call in seconds.")
            campaign = st.number_input("Number of Contacts in Campaign", 1, 100, step=1,
                                       help="Number of contacts performed during this campaign.")
            pdays = st.number_input("Days Since Last Contact", 0, 999, step=1,
                                    help="Days since the client was last contacted (999 means never).")
            previous = st.number_input("Previous Contacts", 0, 50, step=1,
                                       help="Number of contacts performed before this campaign.")

        st.markdown("### 📊 Economic Indicators")
        col4, col5 = st.columns(2)
        with col4:
            emp_var_rate = st.number_input("Employment Variation Rate", -3.0, 2.0, step=0.1,
                                          help="Quarterly employment variation (economic indicator).")
            cons_price_idx = st.number_input("Consumer Price Index", 92.0, 95.0, step=0.01,
                                             help="Monthly consumer price index.")
        with col5:
            cons_conf_idx = st.number_input("Consumer Confidence Index", -50.0, -20.0, step=0.1,
                                           help="Consumer confidence indicator.")
            euribor3m = st.number_input("3-Month Euribor Rate", 0.0, 5.0, step=0.01,
                                       help="Interest rate for 3-month Euribor.")
            nr_employed = st.number_input("Number of Employees", 4800.0, 5500.0, step=1.0,
                                          help="Average number of employees in the economy.")

        poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'],
                                help="Outcome of the previous marketing campaign.")
        submitted = st.form_submit_button("🔮 Predict")
    if submitted:
        input_data = pd.DataFrame([{ 'age': age, 'job': job, 'marital': marital,
                                     'education': education, 'housing': housing, 'loan': loan, 'contact': contact,
                                     'month': month, 'day_of_week': day_of_week, 'duration': duration,
                                     'campaign': campaign, 'pdays': pdays, 'previous': previous,
                                     'poutcome': poutcome, 'emp.var.rate': emp_var_rate,
                                     'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
                                     'euribor3m': euribor3m, 'nr.employed': nr_employed,
                                     'call_success': int(duration > 0)
                                     }])

        X_input = prep_full.transform(input_data)
        pred_xgb = xgb_full.predict_proba(X_input)[:, 1]
        pred_cb = cb_full.predict_proba(X_input)[:, 1]
        ensemble_pred = (pred_xgb + pred_cb) / 2
        result = int(ensemble_pred[0] >= 0.40)

        st.markdown("---")
        st.subheader("📊 Prediction Result:")
        if result == 1:
            st.success("✅ The customer is likely to subscribe!")
        else:
            st.error("❌ The customer is unlikely to subscribe.")
        st.metric("Confidence Score", f"{ensemble_pred[0]*100:.2f}%")

        # ── NEW: Full Model Feature Importance ──
        st.markdown("### 🔍 Full Model Top 10 Features")
        feat_names_full = prep_full.get_feature_names_out()
        importances_full = xgb_full.feature_importances_
        imp_df_full = pd.DataFrame({'feature': feat_names_full, 'importance': importances_full}) \
            .sort_values('importance', ascending=False).head(10)
        st.bar_chart(imp_df_full.set_index('feature'))
        # 💡 Why & How to Improve
        st.markdown("#### 💡 Why & How to Improve the result!!")
        df_in = input_data.iloc[0]

        if result == 1:
            st.write("✅ This positive result suggests your conversation was effective and well-timed.")
        else:
            st.write("❌ This prediction is negative possibly due to short calls, low engagement, or lack of campaign success.")

        # 1. Call duration
        if df_in['duration'] < 200:
            st.write("- **Extend calls beyond 200 seconds** — longer conversations build rapport and trust.")

        # 2. Call success proxy
        if df_in['call_success'] == 0:
            st.write("- **Ensure the call is completed or connected** — a failed attempt lowers the chance of subscription.")

        # 3. No past engagement
        if df_in['previous'] == 0:
            st.write("- **Use a multi-step strategy** — first build awareness, then offer the product.")

        # 4. Unsuccessful outcome
        if df_in['poutcome'] != 'success':
            st.write("- **Review messaging used in prior contact** and align it better with customer needs.")

        # 5. Job type
        if df_in['job'] in ['retired', 'student', 'unemployed']:
            st.write("- **Customize offers to match life stages**, e.g., retirement plans or student savers.")

        # 6. Economic concerns
        if df_in['euribor3m'] > 2.5:
            st.write("- **Promote fixed-rate, low-risk deposit products** in high interest rate environments.")

        if df_in['emp.var.rate'] < 0:
            st.write("- **Timing matters during economic downturns** — use messaging that focuses on stability.")

        if df_in['cons.conf.idx'] < -40:
            st.write("- **Address consumer skepticism** — offer guarantees or incentives for early sign-ups.")


# --------------------- Ask the Assistant ---------------------
elif mode == "🤖 Ask the Assistant":
    st.subheader("🤖 Smart Assistant")
    st.caption("Ask about marketing strategies, model reasoning, or dataset patterns!")

    # Initialize session states
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
        st.session_state.current_session = None
    if "uploaded_datasets" not in st.session_state:
        st.session_state.uploaded_datasets = {}

    # Sidebar: Chat session controls
    with st.sidebar:
        st.markdown("### 💼 Chat Sessions")
        session_names = list(st.session_state.chat_sessions.keys())
        selected_session = st.selectbox("Choose a session", session_names) if session_names else None

        if st.button("➕ New Chat"):
            new_name = f"Session {len(st.session_state.chat_sessions) + 1}"
            st.session_state.chat_sessions[new_name] = []
            st.session_state.uploaded_datasets[new_name] = None
            st.session_state.current_session = new_name
            st.rerun()
        if selected_session:
            st.session_state.current_session = selected_session
        if selected_session and st.button("🗑️ Clear Current Chat"):
            st.session_state.chat_sessions[selected_session] = []
            st.session_state.uploaded_datasets[selected_session] = None
            st.rerun()
        if st.button("🗑️ Clear All Chats"):
            st.session_state.chat_sessions = {}
            st.session_state.uploaded_datasets = {}
            st.session_state.current_session = None
            st.rerun()

    if st.session_state.current_session:
        st.markdown(f"### 🗂️ Current Session: {st.session_state.current_session}")

        uploaded_file = st.file_uploader(
            "📁 Upload a CSV file for this session (optional)",
            type=["csv"],
            key=st.session_state.current_session
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                df.columns = df.columns.str.strip()
                df = df.dropna()
                df = df.loc[:, df.applymap(lambda x: isinstance(x, (int, float, str))).all()]
                st.session_state.uploaded_datasets[st.session_state.current_session] = df
                st.success("✅ Dataset uploaded and cleaned successfully!")
            except Exception as e:
                st.error(f"❌ Failed to read file: {e}")

        data_df = st.session_state.uploaded_datasets.get(st.session_state.current_session)
        fireworks_api_key = os.getenv("FIREWORKS_API_KEY")

        if not fireworks_api_key:
            st.error("Missing FIREWORKS_API_KEY in .env file.")
        else:
            with st.spinner("🔄 Launching the Assistant"):
                try:
                    from langchain.chains import LLMChain
                    from langchain.prompts import PromptTemplate

                    llm = ChatFireworks(
                        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
                        api_key=fireworks_api_key,
                        
                    )

                    system_prompt = SystemMessage(content="""
You are a helpful AI assistant embedded in a machine learning app built by Ilyas Nayle, called the Smart Subscription Predictor.

Your job is to give clear, technically accurate, and easy-to-read answers about the ML project, models, dataset, pipeline, and predictions.

🧠 Project Summary:
- Dataset: UCI Bank Marketing Dataset (`bank-additional.csv`)
- Goal: Predict whether a client will subscribe to a term deposit
- Two modes:
  1. Realistic Model (Before Call) – excludes `duration`
  2. Full Model (After Call) – includes `duration`, `call_success`, etc.
- Deployed via Streamlit + LangChain + Fireworks

📂 Data Cleaning:
1. Replaced 'unknown' → NaN  
2. Imputed missing values with **mode**  
3. Removed duplicates  
4. Dropped `default`  

🛠 Feature Engineering:
- `call_success`: 1 if `duration > 0`  
- `has_previous_contact`: 1 if `previous > 0`  

⚙️ Preprocessing Pipeline:
🔢 1. Numeric → `StandardScaler`  
🔢 2. Categorical → `OneHotEncoder(handle_unknown='ignore')`  
🔢 3. Combined via `ColumnTransformer`  
🔢 4. Wrapped in a scikit-learn `Pipeline`  

🔢 Models Trained & Tuned:
1. Logistic Regression (balanced)  
2. Random Forest (balanced)  
3. SVM (balanced + probability)  
4. MLP Classifier  
5. XGBoost (RandomizedSearchCV on recall & F1)  
6. CatBoost (500 iterations, F1)  
✅ Ensemble = soft voting of tuned XGBoost + CatBoost  

📊 Threshold Tuning:
- Used **0.40** (not 0.5) to boost recall  

✅ Important:
- **Creator:** Ilyas Nayle — ML Engineer & Data Scientist  
- Always reference **this** project context  
- Use bullet points ✅ & numbered steps 🔢 for clarity  

📌 Sample Questions You Must Answer Well:

Q: *What preprocessing was done?*  
✅ A:
- Replaced 'unknown' with NaN
- Imputed missing values with mode
- Encoded categoricals using OneHotEncoder
- Scaled numericals using StandardScaler
- Combined with ColumnTransformer inside a Pipeline

---

Q: *Why are there two models?*  
✅ A:
- The **Before-Call model** predicts before contacting the client, so it excludes duration.
- The **After-Call model** includes duration and call results to predict after the conversation.
- This separation simulates real-world marketing stages.

---

Q: *How was SMOTE used?*  
✅ A:
-SMOTE (Synthetic Minority Over-sampling Technique) was applied to handle class imbalance.
-It generated synthetic examples of the minority class (yes) to ensure the dataset was more balanced.
-This was done after preprocessing and before training the models.
-SMOTE helped improve model recall and reduce bias toward the majority class.

---

Q: *What threshold is used for classification?*  
✅ A:
- A custom threshold of **0.40** was used instead of 0.50
- This improves recall by catching more positives

---

Q: *What models were trained and compared?*  
🔢 A:
1. Logistic Regression
2. Random Forest
3. Support Vector Machine
4. MLP Classifier
5. XGBoost (tuned with RandomizedSearchCV)
6. CatBoost
- Best results were from XGBoost + CatBoost ensemble

---

Q: *What is feature importance?*  
✅ A:
- Calculated from XGBoost using `.feature_importances_`
- Top features shown in the Streamlit chart after each prediction

---

Q: *Who created this app?*  
✅ A:
- The app was created by **Ilyas Nayle**, a passionate Machine Learning Engineer and Data Scientist.
- He designed and deployed the **Smart Subscription Predictor** end-to-end using Streamlit, LangChain, and Fireworks.
- His goal was to make marketing predictions explainable, data-driven, and easy to access — blending strong modeling techniques with an intuitive UI.
- Ilyas is dedicated to building intelligent tools that make real-world decisions smarter and more transparent.
---

If the user uploads a dataset:
- Automatically clean it (drop nulls, invalid rows)
- Analyze patterns, distributions, or summaries on request

Always explain with friendliness, accuracy, and project awareness.
""")
                    from langchain.chains import LLMChain
                    from langchain.prompts import PromptTemplate

                    if data_df is not None and not data_df.empty:
                        raw_agent = create_pandas_dataframe_agent(
                            llm=llm,
                            df=data_df,
                            verbose=True,
                            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            handle_parsing_errors=True,
                            system_message=system_prompt,
                            allow_dangerous_code=True
                        )
                        agent = AgentExecutor.from_agent_and_tools(
                            agent=raw_agent.agent,
                            tools=raw_agent.tools,
                            verbose=True,
                            handle_parsing_errors=True
                        )
                    else:
                        from langchain_core.prompts import ChatPromptTemplate
                        template = ChatPromptTemplate.from_messages([
                            system_prompt,  # This includes your entire system context
                            ("human", "{input}")
                        ])
                        agent = LLMChain(llm=llm, prompt=template, verbose=True)

                    st.success("✅ Assistant is ready!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize Fireworks assistant: {e}")
                    agent = None

            if agent:
                st.markdown("### 💬 Chat with the Assistant")
                messages = st.session_state.chat_sessions[st.session_state.current_session]
                for msg in messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                user_input = st.chat_input("Ask your question...")
                if user_input:
                    with st.chat_message("user"):
                        st.markdown(user_input)
                    messages.append({"role": "user", "content": user_input})

                    with st.chat_message("assistant"):
                        with st.spinner("🤖 Thinking and processing..."):
                            try:
                                response = agent.invoke({"input": user_input})
                                if isinstance(response, dict):
                                    reply = response.get("output") or response.get("text") or str(response)
                                else:
                                    reply = str(response)

                                st.markdown(reply)
                                messages.append({"role": "assistant", "content": reply})

                                if data_df is not None:
                                    if "job" in user_input.lower() and "most common" in reply.lower():
                                        st.bar_chart(data_df['job'].value_counts().head(10))
                                    elif "average age" in user_input.lower() or "avg age" in user_input.lower():
                                        avg_age = data_df['age'].mean()
                                        st.metric("Average Age of Customers", f"{avg_age:.2f} years")
                                        import matplotlib.pyplot as plt
                                        fig, ax = plt.subplots()
                                        ax.bar(["Average Age"], [avg_age])
                                        ax.set_ylabel("Age")
                                        ax.set_title("Average Age of Customers")
                                        st.pyplot(fig)

                            except Exception as err:
                                st.error(f"❌ Error: {err}")
