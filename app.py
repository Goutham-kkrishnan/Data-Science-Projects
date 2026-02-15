import streamlit as st
import os
import pandas as pd
import pickle
import warnings
from io import BytesIO

# Import your model functions
from models.logistic_regression import logistic_regression
from models.decision_tree import decision_tree
from models.knn import knn
from models.naive_bayes import naive_bayes
from models.random_forest import random_forest
from models.xgboost import xgboost_model
from models.preprocessing import (
    preprocess_features,
    encode_categorical_columns,
    TARGET_MAPPING
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Obesity Prediction Dashboard",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS with the provided color theme
# --------------------------------------------------
st.markdown("""
<style>
    /* Color theme variables */
    :root {
        --color-1: #F5276C;
        --color-2: #F54927;
        --color-3: #F5B027;
        --color-4: #27F5B0;
        --color-5: #276CF5;
        --bg-dark: #0f172a;
        --card-bg: #1e293b;
        --text-light: #f8fafc;
        --text-muted: #cbd5e1;
    }

    /* Hide Streamlit branding */
    .stAppDeployButton {display:none;}
    
    /* Main title gradient using theme colors */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--color-1), var(--color-3), var(--color-5));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    /* Subtitle */
    .subtitle {
        color: var(--text-muted);
        font-size: 1.1rem;
        margin-top: 0;
    }
    
    /* Card styling */
    .dashboard-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 6px solid var(--color-1);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        color: var(--text-light);
        margin-bottom: 1.5rem;
    }
    
    /* Input card for single prediction */
    .input-card {
        background: var(--card-bg);
        color: var(--text-light) !important;
        padding: 1.8rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
    }
    .input-card label, .input-card .stSelectbox, .input-card .stNumberInput {
        color: var(--text-light) !important;
    }
    
    /* Metric containers */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        border-bottom: 4px solid var(--color-4);
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    div[data-testid="metric-container"] label {
        color: var(--text-muted) !important;
        font-weight: 500;
    }
    div[data-testid="metric-container"] div {
        color: white !important;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--color-1), var(--color-2));
        color: white;
        border-radius: 40px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border: none;
        box-shadow: 0 4px 15px rgba(245, 39, 108, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(245, 39, 108, 0.5);
    }
    
    /* Download button specific */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--color-5), var(--color-4));
        box-shadow: 0 4px 15px rgba(39, 108, 245, 0.3);
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1wrcr25 {
        background-color: var(--bg-dark);
    }
    .sidebar .sidebar-content {
        background: var(--bg-dark);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--card-bg);
        border-radius: 40px;
        padding: 8px 20px;
        color: var(--text-muted);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--color-3), var(--color-4));
        color: black !important;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--card-bg);
        color: var(--text-light);
        border-radius: 10px;
        border-left: 4px solid var(--color-3);
    }
    
    /* Success/warning/info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 6px solid;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--color-4), var(--color-5));
    }
 

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar – Page Selection (Batch first)
# --------------------------------------------------
st.sidebar.markdown("## 🧠 **Navigation**")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Batch Model Evaluation", "Single Prediction", "Test Data Evaluation"],
    index=0,
    format_func=lambda x: f"📌 {x}"
)


st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<p style='color:#27F5B0; font-size:0.9rem;'>"
    "⚡ Batch: Retrain & compare models<br>"
    "🎯 Single: Use pre‑trained models"
    "</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Helper: Load performance.csv (for Single Prediction only)
# --------------------------------------------------
def load_performance():
    if os.path.exists("performance.csv"):
        return pd.read_csv("performance.csv")
    elif os.path.exists("performace.csv"):
        return pd.read_csv("performace.csv")
    return None

# --------------------------------------------------
# PAGE 0 : DASHBOARD
# --------------------------------------------------
def dashboard_page():
    st.markdown("<h1 class='main-title'>📊 Welcome</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Know the app and overall statistics</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    AVAILABLE_MODELS = [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
    # Load performance data (if any)
    perf_df = load_performance()

    # ------------------------------------------------------------
    # Available Models (with count)
    # ------------------------------------------------------------
    st.subheader(f"🤖 Available Models ({len(AVAILABLE_MODELS)} total)")

    # Display models in a 3‑column grid with checkmarks
    cols = st.columns(3)
    for i, model in enumerate(AVAILABLE_MODELS):
        with cols[i % 3]:
            st.markdown(f"✅ {model}")

    st.markdown("---")


    # ------------------------------------------------------------
    # Instructions for using the app
    # ------------------------------------------------------------
    st.markdown("""
    <div style="background-color:#1e293b; padding:1.5rem; border-radius:20px; border-left:6px solid #27F5B0; margin-bottom:2rem;">
        <h3 style="color:#F5B027; margin-top:0;">🧭 How to Use This App</h3>
        <ul style="color:#f8fafc; font-size:1.05rem; line-height:1.8;">
            <li><strong style="color:#F5276C;">1. Dashboard (you are here)</strong> – Overview and quick access to model performance.</li>
            <li><strong style="color:#F5276C;">2. Batch Model Evaluation</strong> – Upload your dataset (CSV). Then the uploaded data will be trained and tested will return the performance of multiple machine learning models at once. You will see performance metrics, classification reports, and per‑class correct/incorrect counts. You can download a sample dataset also.</li>
            <li><strong style="color:#F5276C;">3. Single Prediction</strong> – Use a pre‑trained model to predict the obesity level for one individual. Enter the person’s features and click <em>Predict</em>.</li>
            <li><strong style="color:#F5276C;">4. Test Data Evaluation</strong> – Evaluate a pre-trained model on uploaded test data and view performance metrics and classification reports. Includes sample test data download.</li>

        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Performance summary (if data exists)
    # ------------------------------------------------------------
    if perf_df is not None and not perf_df.empty:
        st.subheader("📈 Pretrained Model Performance Summary")
        st.dataframe(
            perf_df[['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score']]
            .style.background_gradient(cmap='coolwarm', subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score'])
            .format(precision=3),
            width="stretch"

        )

        st.subheader("📊 Accuracy Comparison")
        chart_data = perf_df.set_index('Model')['Accuracy'].sort_values()
        st.bar_chart(chart_data)
    else:
        # If no data, still show a friendly reminder
        st.info("👆 Head over to **Batch Model Evaluation** to train models and generate performance data.")

# --------------------------------------------------
# PAGE 1 : BATCH MODEL EVALUATION (with sample data download)
# --------------------------------------------------
def batch_evaluation_page():
    # Header with dashboard feel
    st.markdown("<h1 class='main-title'>📁 Batch Model Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload your dataset and compare multiple ML models at once</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Two columns for sample download and upload area
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        # ---- Download sample data button ----
        if os.path.exists("test_data.csv"):
            with open("test_data.csv", "rb") as f:
                sample_data = f.read()
            st.download_button(
                label="📥 Download Sample CSV",
                data=sample_data,
                file_name="test_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Sample file 'test_data.csv' not found.")
    
    with col_right:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ **Loaded:** {uploaded_file.name} – {df.shape[0]} rows, {df.shape[1]} columns")

        with st.expander("🔍 Preview uploaded data"):
            st.dataframe(df.head(10), width="stretch")

        # Required columns check
        required_cols = [
            'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
            'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
            'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
            return

        st.markdown("---")
        st.markdown("## 🎯 Select Models to Evaluate")

        # Model dictionary
        model_dict = {
            "Logistic Regression": logistic_regression,
            "Decision Tree": decision_tree,
            "KNN": knn,
            "Naive Bayes": naive_bayes,
            "Random Forest": random_forest,
            "XGBoost": xgboost_model
        }

        # Session state for selection
        if "selected_models" not in st.session_state:
            st.session_state.selected_models = []

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            selected_names = st.multiselect(
                "Choose one or more models:",
                options=list(model_dict.keys()),
                default=st.session_state.selected_models
            )

        with col2:
            st.markdown("### &nbsp;")
            if st.button("✔️ Select All"):
                st.session_state.selected_models = list(model_dict.keys())
                st.rerun()

        with col3:
            st.markdown("### &nbsp;")
            compute_clicked = st.button("🚀 Compute", type="primary")

        st.session_state.selected_models = selected_names

        if not compute_clicked:
            st.info("ℹ️ Select models and click **Compute** to start evaluation.")
            return

        if not selected_names:
            st.warning("⚠️ Please select at least one model.")
            return

        # Filter selected models
        selected_models = [(name, model_dict[name]) for name in selected_names]

        st.markdown("---")
        st.markdown("## 🔄 Running selected models ...")

        all_results = []
        all_reports = {}
        all_per_class = {}   # new dictionary to store per‑class counts

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (name, func) in enumerate(selected_models):
            status_text.text(f"Training **{name}** ...")
            try:
                # Now expecting three return values
                result_row, report_dict, per_class_counts = func(df)
                all_results.append(result_row)
                all_reports[name] = report_dict
                all_per_class[name] = per_class_counts   # store them
            except Exception as e:
                st.error(f"⚠️ Error in {name}: {str(e)}")

            progress_bar.progress((i + 1) / len(selected_models))

        status_text.text("✅ **All selected models completed!**")
        progress_bar.empty()

        if all_results:
            metrics_df = pd.concat(all_results, ignore_index=True)
            metrics_df = metrics_df[
                ['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
            ]

            st.markdown("---")
            st.markdown("## 📊 Performance Comparison")

            # Use a container to hold the dataframe with a nice background
            with st.container():
                st.dataframe(
                    metrics_df.style
                    .background_gradient(
                        cmap='coolwarm',
                        subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
                    )
                    .format(precision=3),
                    width="stretch"
                )

            csv = metrics_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Metrics CSV",
                csv,
                "batch_performance.csv",
                "text/csv"
            )

            st.markdown("---")
            st.markdown("## 📋 Classification Reports")

            tabs = st.tabs(selected_names)
            for tab, name in zip(tabs, selected_names):
                with tab:
                    # Get the standard report and the per‑class counts
                    report_dict = all_reports[name]
                    per_class = all_per_class[name]

                    # Convert report dict to DataFrame
                    report_df = pd.DataFrame(report_dict).transpose()

                    # Add empty columns for correct/incorrect
                    report_df['Correct'] = None
                    report_df['Incorrect'] = None

                    # Fill in values for class rows only
                    for class_name, counts in per_class.items():
                        if class_name in report_df.index:
                            report_df.loc[class_name, 'Correct'] = counts['correct']
                            report_df.loc[class_name, 'Incorrect'] = counts['incorrect']

                    # --- NEW: Replace None/NaN in Correct/Incorrect columns with "---" ---
                    report_df['Correct'] = report_df['Correct'].fillna('---')
                    report_df['Incorrect'] = report_df['Incorrect'].fillna('---')

                    # Reorder columns to put correct/incorrect after 'support'
                    desired_order = ['precision', 'recall', 'f1-score', 'support', 'Correct', 'Incorrect']
                    # Keep only columns that actually exist in the DataFrame
                    existing_cols = [col for col in desired_order if col in report_df.columns]
                    report_df = report_df[existing_cols]

                    # Display with formatting
                    st.dataframe(
                        report_df.style.format(precision=3),
                        width="stretch"
                    )

        else:
            st.error("❌ No models could be evaluated successfully.")


# --------------------------------------------------
# PAGE 2 : SINGLE PREDICTION
# --------------------------------------------------
def single_prediction_page():
    st.markdown("<h1 class='main-title'>🎯 Single Obesity Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Use a pre‑trained model to classify one individual</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Model selection in sidebar (already there) but we also show it prominently
    MODEL_DIR = "models"
    model_files = [
        f.replace(".pkl", "")
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".pkl") and f != "scaler.pkl"
    ]
    
    # Use columns to place model selector more visibly
    col_model, _ = st.columns([2, 3])
    with col_model:
        selected_model = st.selectbox(
            "🧠 **Select ML Model**",
            ["-- Select Model --"] + model_files
        )

    # Load metrics
    results_df = load_performance()

    if selected_model != "-- Select Model --":
        if results_df is not None:
            row = results_df[results_df["Model"].str.lower() == selected_model.lower()]
            if not row.empty:
                st.markdown("### 📈 Model Performance")
                acc, auc, prec, rec, f1 = row[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score']].values[0]
                cols = st.columns(5)
                metrics = [("Accuracy", acc), ("AUC", auc), ("Precision", prec), ("Recall", rec), ("F1 Score", f1)]
                for col, (label, value) in zip(cols, metrics):
                    with col:
                        st.metric(label, f"{value:.3f}")

        st.markdown("---")
        st.markdown("## 🔧 Enter Input Features")

        # Mappings
        gender_map = {"Male": 1, "Female": 0}
        yes_no_map = {"Yes": 1, "No": 0}
        caec_map = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
        calc_map = {"No": 0, "Sometimes": 1, "Frequently": 2}
        mtrans_map = {"Automobile": 0, "Bike": 1, "Motorbike": 2,
                      "Public Transportation": 3, "Walking": 4}

        # Input form in a card
        with st.container():
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**👤 Demographics**")
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.number_input("Age", 10, 100, 21)
                height = st.number_input("Height (m)", 1.0, 2.5, 1.70, step=0.01)
                weight = st.number_input("Weight (kg)", 20.0, 200.0, 70.0, step=0.1)
                
            with col2:
                st.markdown("**🍽️ Eating Habits**")
                family = st.selectbox("Family History Overweight", ["Yes", "No"])
                favc = st.selectbox("High Calorie Food (FAVC)", ["Yes", "No"])
                fcvc = st.slider("Vegetable Intake (FCVC)", 1, 3, 2)
                ncp = st.slider("Meals per Day (NCP)", 1, 4, 3)
                caec = st.selectbox("Snacking (CAEC)", ["No", "Sometimes", "Frequently", "Always"])
                
            with col3:
                st.markdown("**🏃 Lifestyle**")
                smoke = st.selectbox("Smoking", ["Yes", "No"])
                ch2o = st.slider("Water Intake (CH2O)", 1, 3, 2)
                faf = st.slider("Physical Activity (FAF)", 0, 3, 1)
                tue = st.slider("Technology Use (TUE)", 0, 3, 1)
                calc = st.selectbox("Alcohol Consumption (CALC)", ["No", "Sometimes", "Frequently"])
                mtrans = st.selectbox("Transport Mode (MTRANS)", list(mtrans_map.keys()))
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Predict button centered
        col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
        with col_b2:
            predict_clicked = st.button("🚀 **Predict Obesity Level**", width="stretch")

        if predict_clicked:
            model_path = f"models/{selected_model}.pkl"
            scaler_path = "models/scaler.pkl"
            if not os.path.exists(model_path):
                st.error("❌ Model file not found.")
            elif not os.path.exists(scaler_path):
                st.error("❌ Scaler file not found.")
            else:
                model = pickle.load(open(model_path, "rb"))
                scaler = pickle.load(open(scaler_path, "rb"))

                input_df = pd.DataFrame([{
                    "Gender": gender_map[gender],
                    "Age": age,
                    "Height": height,
                    "Weight": weight,
                    "family_history_with_overweight": yes_no_map[family],
                    "FAVC": yes_no_map[favc],
                    "FCVC": fcvc,
                    "NCP": ncp,
                    "CAEC": caec_map[caec],
                    "SMOKE": yes_no_map[smoke],
                    "CH2O": ch2o,
                    "SCC": 0,
                    "FAF": faf,
                    "TUE": tue,
                    "CALC": calc_map[calc],
                    "MTRANS": mtrans_map[mtrans]
                }])
                input_df = preprocess_features(input_df)
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                
                # Show prediction in a nice card
                st.balloons()
                st.success(f"### ✅ **Predicted Obesity Category:** **{prediction}**")


# --------------------------------------------------
# Page 3 : Test Data Evaluation
# --------------------------------------------------

def test_data_evaluation_page():
    st.markdown("<h1 class='main-title'>📊 Test Data Evaluation</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Evaluate pre-trained models on uploaded test data</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # --------------------------------------------------
    # Download sample test data
    # --------------------------------------------------
    col_left, col_right = st.columns([1, 2])

    with col_left:
        if os.path.exists("test_data.csv"):
            with open("test_data.csv", "rb") as f:
                st.download_button(
                    label="📥 Download Sample Test CSV",
                    data=f,
                    file_name="test_data.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ Sample file 'test_data.csv' not found.")

    with col_right:
        uploaded_file = st.file_uploader(
            "Upload Test CSV",
            type="csv"
        )

    if uploaded_file is None:
        st.info("ℹ️ Upload a test CSV file to continue.")
        return

    # --------------------------------------------------
    # Load & preview data
    # --------------------------------------------------
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    with st.expander("🔍 Preview uploaded test data"):
        st.dataframe(df.head(), use_container_width=True)

    # --------------------------------------------------
    # Required columns check
    # --------------------------------------------------
    required_cols = [
        'Gender', 'Age', 'Height', 'Weight',
        'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP',
        'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
        'CALC', 'MTRANS', 'NObeyesdad'
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        return

    # --------------------------------------------------
    # Select pretrained model
    # --------------------------------------------------
    MODEL_DIR = "models"
    model_files = [
        f.replace(".pkl", "")
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".pkl") and f != "scaler.pkl"
    ]

    selected_model = st.selectbox(
        "🧠 Select Pre-trained Model",
        ["-- Select Model --"] + model_files
    )

    if selected_model == "-- Select Model --":
        return

    # --------------------------------------------------
    # Evaluate model
    # --------------------------------------------------
    if st.button("🚀 Evaluate Model", type="primary"):
        model_path = f"{MODEL_DIR}/{selected_model}.pkl"
        scaler_path = f"{MODEL_DIR}/scaler.pkl"

        if not os.path.exists(model_path):
            st.error("❌ Model file not found.")
            return

        if not os.path.exists(scaler_path):
            st.error("❌ Scaler file not found.")
            return

        # Load model & scaler
        model = pickle.load(open(model_path, "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))

        # --------------------------------------------------
        # Split X / y (ENCODE y_true)
        # --------------------------------------------------
        X = df.drop(columns=["NObeyesdad"])
        y_true = df["NObeyesdad"].map(TARGET_MAPPING)

        if y_true.isna().any():
            unseen = set(df["NObeyesdad"]) - set(TARGET_MAPPING.keys())
            st.error(f"❌ Unseen target labels in test data: {unseen}")
            return

        # --------------------------------------------------
        # Apply SAME preprocessing as training
        # --------------------------------------------------
        X = preprocess_features(X)
        X = encode_categorical_columns(X)

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        y_pred = model.predict(X_scaled)

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            classification_report
        )

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        st.markdown("## 📈 Performance Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1 Score", f"{f1:.3f}")

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------
        st.markdown("---")
        st.markdown("## 📋 Classification Report")

        report_df = pd.DataFrame(
            classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0
            )
        ).transpose()

        st.dataframe(
            report_df.style.format(precision=3),
            use_container_width=True
        )


# --------------------------------------------------
# Main router
# --------------------------------------------------
if page == "Dashboard":
    dashboard_page()
elif page == "Batch Model Evaluation":
    batch_evaluation_page()
elif page == "Single Prediction":
    single_prediction_page()
else:
    test_data_evaluation_page()
