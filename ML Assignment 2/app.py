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
from models.preprocessing import preprocess_features
warnings.filterwarnings("ignore")

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Obesity Prediction Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS (unchanged)
# --------------------------------------------------
st.markdown("""
<style>
.stAppDeployButton {display:none;}
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.input-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
}
.metric-container {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    color: black !important;
}
.metric-container h1, 
.metric-container h3, 
.metric-container p {
    color: black !important;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar – Page Selection
# --------------------------------------------------
st.sidebar.markdown("## 🤖 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Single Prediction", "Batch Model Evaluation"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Single Prediction**: use pre‑trained models on one person.")
st.sidebar.markdown("📌 **Batch Evaluation**: upload CSV → retrain & evaluate all models.")

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
# PAGE 1 : SINGLE PREDICTION (your existing code, slightly adapted)
# --------------------------------------------------
def single_prediction_page():
    st.markdown("<h1 class='main-title'>📊 Single Obesity Prediction</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Model selection
    MODEL_DIR = "models"
    model_files = [
        f.replace(".pkl", "")
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".pkl") and f != "scaler.pkl"
    ]
    selected_model = st.sidebar.selectbox(
        "Select ML Model",
        ["-- Select Model --"] + model_files
    )

    # Load metrics
    results_df = load_performance()

    if selected_model != "-- Select Model --":
        # Display metrics (unchanged) ...
        if results_df is not None:
            row = results_df[results_df["Model"].str.lower() == selected_model.lower()]
            if not row.empty:
                acc = row['Accuracy'].values[0]
                auc = row['AUC'].values[0]
                prec = row['Precision'].values[0]
                rec = row['Recall'].values[0]
                f1 = row['F1 Score'].values[0]
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1: st.metric("Accuracy", f"{acc:.3f}")
                with col2: st.metric("AUC", f"{auc:.3f}")
                with col3: st.metric("Precision", f"{prec:.3f}")
                with col4: st.metric("Recall", f"{rec:.3f}")
                with col5: st.metric("F1 Score", f"{f1:.3f}")

        st.markdown("---")
        st.markdown("## 🔧 Enter Input Features")

        # Encoding maps (unchanged) ...
        gender_map = {"Male": 1, "Female": 0}
        yes_no_map = {"Yes": 1, "No": 0}
        caec_map = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
        calc_map = {"No": 0, "Sometimes": 1, "Frequently": 2}
        mtrans_map = {"Automobile": 0, "Bike": 1, "Motorbike": 2,
                      "Public Transportation": 3, "Walking": 4}

        # Input form (unchanged) ...
        with st.container():
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.number_input("Age", 10, 100, 21)
                height = st.number_input("Height (m)", 1.0, 2.5, 1.70)
                weight = st.number_input("Weight (kg)", 20.0, 200.0, 70.0)
            with col2:
                family = st.selectbox("Family History Overweight", ["Yes", "No"])
                favc = st.selectbox("High Calorie Food (FAVC)", ["Yes", "No"])
                fcvc = st.slider("Vegetable Intake (FCVC)", 1, 3, 2)
                ncp = st.slider("Meals per Day (NCP)", 1, 4, 3)
            with col3:
                caec = st.selectbox("Snacking (CAEC)", ["No", "Sometimes", "Frequently", "Always"])
                smoke = st.selectbox("Smoking", ["Yes", "No"])
                ch2o = st.slider("Water Intake (CH2O)", 1, 3, 2)
                faf = st.slider("Physical Activity (FAF)", 0, 3, 1)
                tue = st.slider("Technology Use (TUE)", 0, 3, 1)
                calc = st.selectbox("Alcohol Consumption (CALC)", ["No", "Sometimes", "Frequently"])
                mtrans = st.selectbox("Transport Mode (MTRANS)", list(mtrans_map.keys()))
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 Predict Obesity Level"):
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
                st.success(f"✅ **Predicted Obesity Category:** {prediction}")

# --------------------------------------------------
# PAGE 2 : BATCH MODEL EVALUATION (NEW)
# --------------------------------------------------
# def batch_evaluation_page():
#     st.markdown("<h1 class='main-title'>📁 Batch Model Evaluation</h1>", unsafe_allow_html=True)
#     st.markdown("---")
#     st.markdown("""
#     Upload a **CSV file** with the **same columns** as the original dataset.  
#     The six models will be **retrained on your uploaded data** (80/20 split) and evaluated.  
#     All metrics and classification reports are displayed below.
#     """)

#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         st.success(f"✅ File loaded: {uploaded_file.name} – shape {df.shape}")

#         with st.expander("📄 Preview of uploaded data"):
#             st.dataframe(df.head())

#         # Check required columns (simplified – you can expand)
#         required_cols = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#                          'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
#                          'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad']
#         missing = [col for col in required_cols if col not in df.columns]
#         if missing:
#             st.error(f"❌ Missing columns: {missing}")
#             return

#         st.markdown("---")
#         st.markdown("## 🔄 Running all 6 models ...")

#         # List of model functions (each returns (results_df, report_dict))
#         models = [
#             ("Logistic Regression", logistic_regression),
#             ("Decision Tree", decision_tree),
#             ("KNN", knn),
#             ("Naive Bayes", naive_bayes),
#             ("Random Forest", random_forest),
#             ("XGBoost", xgboost_model)
#         ]

#         all_results = []
#         all_reports = {}

#         progress_bar = st.progress(0)
#         status_text = st.empty()

#         for i, (name, func) in enumerate(models):
#             status_text.text(f"Training {name} ...")
#             try:
#                 result_row, report_dict = func(df)                result_row, report_dict = func(df)

#                 all_results.append(result_row)
#                 all_reports[name] = report_dict
#             except Exception as e:
#                 st.error(f"⚠️ Error in {name}: {str(e)}")
#             progress_bar.progress((i + 1) / len(models))

#         status_text.text("✅ All models completed!")
#         progress_bar.empty()

#         if all_results:
#             # Combine metrics
#             metrics_df = pd.concat(all_results, ignore_index=True)
#             metrics_df = metrics_df[['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']]

#             st.markdown("---")
#             st.markdown("## 📊 Performance Comparison")

#             # Color-coded table
#             st.dataframe(
#                 metrics_df.style
#                 .background_gradient(cmap='Blues', subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'])
#                 .format(precision=3)
#             )

#             # Download button
#             csv = metrics_df.to_csv(index=False).encode('utf-8')
#             st.download_button(
#                 label="📥 Download Metrics CSV",
#                 data=csv,
#                 file_name="batch_performance.csv",
#                 mime="text/csv"
#             )

#             # Classification Reports
#             st.markdown("---")
#             st.markdown("## 📋 Classification Reports")
#             tabs = st.tabs([name for name, _ in models])

#             for tab, (name, _) in zip(tabs, models):
#                 with tab:
#                     if name in all_reports:
#                         report_df = pd.DataFrame(all_reports[name]).transpose()
#                         st.dataframe(report_df.style.format(precision=3))
#                     else:
#                         st.warning("Report not available.")
#         else:
#             st.error("❌ No models could be evaluated successfully.")
def batch_evaluation_page():
    st.markdown("<h1 class='main-title'>📁 Batch Model Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    Upload a **CSV file** with the **same columns** as the original dataset.  
    The selected models will be **retrained on your uploaded data** (80/20 split) and evaluated.  
    All metrics and classification reports are displayed below.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File loaded: {uploaded_file.name} – shape {df.shape}")

        with st.expander("📄 Preview of uploaded data"):
            st.dataframe(df.head())

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
            compute_clicked = st.button("🚀 Compute")

        # Update session state
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

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (name, func) in enumerate(selected_models):
            status_text.text(f"Training {name} ...")
            try:
                result_row, report_dict = func(df)
                all_results.append(result_row)
                all_reports[name] = report_dict
            except Exception as e:
                st.error(f"⚠️ Error in {name}: {str(e)}")

            progress_bar.progress((i + 1) / len(selected_models))

        status_text.text("✅ All selected models completed!")
        progress_bar.empty()

        if all_results:
            metrics_df = pd.concat(all_results, ignore_index=True)
            metrics_df = metrics_df[
                ['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
            ]

            st.markdown("---")
            st.markdown("## 📊 Performance Comparison")

            st.dataframe(
                metrics_df.style
                .background_gradient(
                    cmap='Blues',
                    subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
                )
                .format(precision=3)
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
                    report_df = pd.DataFrame(all_reports[name]).transpose()
                    st.dataframe(report_df.style.format(precision=3))

        else:
            st.error("❌ No models could be evaluated successfully.")

# --------------------------------------------------
# Main router
# --------------------------------------------------
if page == "Single Prediction":
    single_prediction_page()
else:
    batch_evaluation_page()