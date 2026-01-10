import streamlit as st
import os
import pandas as pd
from streamlit_option_menu import option_menu

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="ML Model Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS Styling ----------
st.markdown("""
    <style>
    /* Hide Deploy Button */
    .stAppDeployButton {
        display: none;
    }
    
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .input-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* Divider styling */
    .divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Metric container */
    .metric-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    /* Success message */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e9ecef;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Dummy Prediction Function ----------
def predict(model_name, feature_dropdown, feature_text, feature_checkbox):
    """
    Dummy prediction logic.
    Replace this with actual model.predict() later.
    """
    return f"✅ Prediction successful using {model_name.replace('_', ' ').title()}"

# ---------- Sidebar ----------

st.set_page_config(
    page_title="My App",
    layout="wide",
    initial_sidebar_state="expanded"  # <-- IMPORTANT
)
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='color: #667eea;'>🤖 Model Dashboard</h2>
            <p style='color: #6c757d;'>Select a model to view metrics and make predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📂 Available Models")
    
    # Load Models
    MODEL_DIR = "models"
    model_files = [
        file.replace(".pkl", "")
        for file in os.listdir(MODEL_DIR)
        if file.endswith(".pkl")
    ]
    
    model_options = ["-- Select a model --"] + model_files
    
    selected_model = st.selectbox(
        "Choose Model",
        model_options,
        key="model_selector"
    )
    
    st.markdown("---")
    
    # Information panel
    with st.expander("ℹ️ How to Use", expanded=False):
        st.markdown("""
        1. **Select a model** from the dropdown
        2. **View performance metrics**
        3. **Input features** for prediction
        4. **Click Predict** to get results
        
        All models are trained and ready for inference.
        """)
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #6c757d; font-size: 0.8rem;'>
        ML Assignment 2<br>
        Powered by Streamlit
        </div>
    """, unsafe_allow_html=True)

# ---------- Main Content ----------

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 class='main-title'>📊 Machine Learning Dashboard</h1>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------- Load CSV ----------
df = pd.read_csv("model_result.csv")

# ---------- Main Content Area ----------
if selected_model != "-- Select a model --":
    
    model_data = df[
        df["Model_name"].str.lower()
        == selected_model.replace("_", " ").lower()
    ]
    
    # Model Header
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px;
                    margin-bottom: 2rem;
                    border-left: 4px solid #667eea;'>
            <h2 style='margin: 0; color: #495057;'>
                🚀 Selected Model: <span style='color: #667eea;'>{selected_model.replace('_', ' ').title()}</span>
            </h2>
        </div>
    """, unsafe_allow_html=True)
    
    # ---------- Display Metrics ----------
    if not model_data.empty:
        accuracy = model_data["Accuracy"].values[0]
        precision = model_data["Precision"].values[0]
        recall = model_data["Recall"].values[0]
        
        st.markdown("### 📈 Model Performance")
        
        # Create three columns for metrics with better styling
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class='metric-container'>
                    <div style='text-align: center;'>
                        <h3 style='color: #667eea; margin-bottom: 0.5rem;'>🎯 Accuracy</h3>
                        <h1 style='color: #28a745; margin: 0;'>{:.2%}</h1>
                        <p style='color: #6c757d; font-size: 0.9rem; margin-top: 0.5rem;'>
                        Correct predictions ratio
                        </p>
                    </div>
                </div>
            """.format(accuracy), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-container'>
                    <div style='text-align: center;'>
                        <h3 style='color: #667eea; margin-bottom: 0.5rem;'>⚖️ Precision</h3>
                        <h1 style='color: #17a2b8; margin: 0;'>{:.2%}</h1>
                        <p style='color: #6c757d; font-size: 0.9rem; margin-top: 0.5rem;'>
                        Positive prediction accuracy
                        </p>
                    </div>
                </div>
            """.format(precision), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-container'>
                    <div style='text-align: center;'>
                        <h3 style='color: #667eea; margin-bottom: 0.5rem;'>📊 Recall</h3>
                        <h1 style='color: #ffc107; margin: 0;'>{:.2%}</h1>
                        <p style='color: #6c757d; font-size: 0.9rem; margin-top: 0.5rem;'>
                        True positive rate
                        </p>
                    </div>
                </div>
            """.format(recall), unsafe_allow_html=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # ---------- Feature Inputs ----------
        st.markdown("### 🔧 Input Features for Prediction")
        
        # Create input card
        with st.container():
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                feature_dropdown = st.selectbox(
                    "📋 Feature 1 - Category",
                    ["Option A", "Option B", "Option C"],
                    help="Select a categorical feature value"
                )
                
                feature_checkbox = st.checkbox(
                    "✅ Enable Feature 3",
                    help="Toggle this feature on/off"
                )
            
            with col2:
                feature_text = st.text_input(
                    "🔢 Feature 2 - Numeric/Text",
                    placeholder="Enter value here...",
                    help="Input a numeric or text feature value"
                )
                
                # Add some spacing
                st.markdown("<br><br>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # ---------- Submit Button ----------
        st.markdown("### 🎯 Ready to Predict")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Run Prediction", use_container_width=True):
                with st.spinner("Making prediction..."):
                    # Simulate processing time
                    import time
                    time.sleep(0.5)
                    
                    result = predict(
                        selected_model,
                        feature_dropdown,
                        feature_text,
                        feature_checkbox
                    )
                    
                    # Display result in a nice container
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                                    padding: 1.5rem;
                                    border-radius: 10px;
                                    border-left: 4px solid #28a745;
                                    margin-top: 1rem;'>
                            <h3 style='color: #155724; margin: 0;'>
                                {result}
                            </h3>
                            <p style='color: #0c5460; margin-top: 0.5rem; margin-bottom: 0;'>
                                Model inference completed successfully
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ No metrics found for the selected model.")
        st.info("Please ensure the model is properly trained and metrics are recorded.")

else:
    # Welcome state
    st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem; background: #f8f9fa; border-radius: 15px;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>🤖</div>
            <h2 style='color: #495057;'>Welcome to the ML Dashboard</h2>
            <p style='color: #6c757d; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;'>
                Select a machine learning model from the sidebar to view its performance metrics 
                and make predictions using custom input features.
            </p>
            <div style='color: #667eea; font-size: 2rem; margin-top: 2rem;'>
                ↓
            </div>
            <p style='color: #6c757d; margin-top: 1rem;'>
                <strong>Start by choosing a model from the left sidebar</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)