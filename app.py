import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
from utils import preprocess_data, train_models, stack_models, plot_metrics

# ---------------------------
# 🎨 Page Config
# ---------------------------
st.set_page_config(page_title="AI Powered Online Fake Profile Detector", layout="wide")

st.markdown(
    """
    <style> 
    .stApp {
        background-image:url("https://img.freepik.com/free-vector/gradient-abstract-wireframe-background_23-2149009903.jpg");
        background-size: cover;
    }
    h1, h2, h3 { color: #4A148C; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 18px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True
)

# ---------------------------
# Session State
# ---------------------------
if "models" not in st.session_state: st.session_state.models = {}
if "df" not in st.session_state: st.session_state.df = None
if "X_train_struct" not in st.session_state: st.session_state.X_train_struct = None
if "X_test_struct" not in st.session_state: st.session_state.X_test_struct = None
if "X_train_text" not in st.session_state: st.session_state.X_train_text = None
if "X_test_text" not in st.session_state: st.session_state.X_test_text = None
if "y_train" not in st.session_state: st.session_state.y_train = None
if "y_test" not in st.session_state: st.session_state.y_test = None
if "tokenizer" not in st.session_state: st.session_state.tokenizer = None

# ---------------------------
# Sidebar Navigation (Styled)
# ---------------------------
st.markdown(
    """
    <style>
    /* Sidebar background gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ff6ec4 0%, #7873f5 100%);
        color: black;
    }
    
    /* Sidebar header style */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFD700;
        font-weight: bold;
        text-align: center;
    }
    
    /* Sidebar radio buttons styling */
    [data-testid="stSidebar"] .stRadio > div > label {
        font-size: 18px;
        font-weight: bold;
        background-color: #00008B;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 5px;
        color: black;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background-color: rgba(255, 255, 255, 0.3);
        color: #FFD700;
        transform: scale(1.05);
    }

    /* Scrollbar for sidebar */
    [data-testid="stSidebar"] ::-webkit-scrollbar {
        width: 8px;
    }
    [data-testid="stSidebar"] ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
        background: #FFD700;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation with emojis
# Sidebar Navigation using session state
if "section" not in st.session_state:
    st.session_state["section"] = "🏠 Landing Page"

section = st.sidebar.radio(
    "📍 Navigation",
    ["🏠 Home", "📂 Data & Config", "⚙️ Train Models", "📊 Evaluate", "🧑‍💻 Live Prediction", "💾 Save/Load"],
    index=["🏠 Landing Page", "📂 Data & Config", "⚙️ Train Models", "📊 Evaluate", "🧑‍💻 Live Prediction", "💾 Save/Load"].index(st.session_state["section"])
)


 
# ---------------------------
# Landing Page
# ---------------------------
if section == "🏠 Home":
    st.markdown("<h1 style='text-align: center;'>🕵️ AI Powered Online Fake Profile Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>CNN + LSTM + Random Forest + CatBoost (Stacked Ensemble)</p>", unsafe_allow_html=True)

    st.image("https://cdn-icons-png.flaticon.com/512/1077/1077012.png", width=120)

    st.markdown("## 🚀 About the Project")
    st.write("""
    Social media platforms are plagued with fake accounts.  
    This system uses **Machine Learning (CNN, LSTM, RF, CatBoost)** to **detect fake profiles** 
    with high accuracy.  
    """)

    st.markdown("## 🔄 Workflow")
    cols = st.columns(5)
    steps = ["📂 Upload Data", "⚙️ Train Models", "📊 Evaluate", "🧑‍💻 Predict", "💾 Save/Load"]
    for i, col in enumerate(cols): col.markdown(f"### {steps[i]}")

    st.markdown("## ✨ Features")
    st.success("✅ Upload dataset and preprocess automatically")
    st.success("✅ Train multiple models (RF, CatBoost, CNN, LSTM)")
    st.success("✅ Compare results with leaderboard")
    st.success("✅ Explain predictions with SHAP plots")
    st.success("✅ Real-time profile prediction")
    st.success("✅ Export reports (CSV + Excel)")

    st.info("👇press the  **Start** button to get started!")

       # 🚀 Start Button centered
    col1, col2, col3 = st.columns([2,1,2])  # middle column wider
    with col2:
        if st.button("🚀 Start"):
            st.session_state["section"] = "📂 Data & Config"



# ---------------------------
# Data & Config
# ---------------------------
elif section == "📂 Data & Config":
    st.header("📂 Upload Dataset")
    file = st.file_uploader("Upload CSV dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.write("Preview:", df.head())
        cols = df.columns.tolist()
        target_col = st.selectbox("Select target column (real/fake)", cols)
        text_col = st.selectbox("Select text column (bio/desc, optional)", ["None"] + cols)
        feature_cols = st.multiselect("Select numeric/categorical features", [c for c in cols if c != target_col])
        if st.button("Confirm Dataset"):
            (X_train_struct, X_test_struct, X_train_text, X_test_text,
             y_train, y_test), tokenizer = preprocess_data(
                df, target_col, text_col if text_col!="None" else None, feature_cols
            )
            st.session_state.X_train_struct, st.session_state.X_test_struct = X_train_struct, X_test_struct
            st.session_state.X_train_text, st.session_state.X_test_text = X_train_text, X_test_text
            st.session_state.y_train, st.session_state.y_test = y_train, y_test
            st.session_state.tokenizer = tokenizer
            st.success("Data configured and Ready for training!")
         
        

# ---------------------------
# Train Models
# ---------------------------
elif section == "⚙️ Train Models":
    if st.session_state.X_train_struct is None:
        st.warning("Please configure data first in 'Data & Config'")
    else:
        st.header("⚙️ Train Models")
        if st.button("Train CNN + LSTM + RF + CatBoost"):
            models = train_models(
                st.session_state.X_train_struct,
                st.session_state.y_train,
                st.session_state.X_train_text,
                st.session_state.X_test_text,
                st.session_state.tokenizer
            )
            st.session_state.models = models
            st.success("Models trained!")
        if st.session_state.models and st.button("Stack Models"):
            stacker = stack_models(
                {k: v for k, v in st.session_state.models.items() if k in ["rf", "catboost"]},
                st.session_state.X_train_struct,
                st.session_state.y_train
            )
            st.session_state.models["stack"] = stacker
            st.success("Stacked model created!")
       
    # ---------------------------
# Evaluate Models
# ---------------------------
# -----------------------
# Evaluation Page
# -----------------------
elif section == "📊 Evaluate":
    if not st.session_state.models:
        st.warning("Train models first.")
    else:
        st.header("📊 Evaluate Models")

        from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
        import plotly.graph_objects as go

        # -----------------------
        # 🌟 Summary Metrics (Top of Dashboard)
        # -----------------------
        st.subheader("📊 Model Performance Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        metrics = {}

        for name, model in st.session_state.models.items():
            try:
                if name in ["rf", "catboost", "stack"]:
                    preds = model.predict(st.session_state.X_test_struct)
                else:
                    preds = (model.predict(st.session_state.X_test_text) > 0.5).astype("int32")
                acc = accuracy_score(st.session_state.y_test, preds)
                metrics[name] = acc
            except Exception:
                metrics[name] = None

        if "rf" in metrics:
            col1.metric("🌲 RF", f"{metrics['rf']:.2%}" if metrics['rf'] else "N/A")
        if "catboost" in metrics:
            col2.metric("🐱 CatBoost", f"{metrics['catboost']:.2%}" if metrics['catboost'] else "N/A")
        if "cnn" in metrics:
            col3.metric("🧠 CNN", f"{metrics['cnn']:.2%}" if metrics['cnn'] else "N/A")
        if "lstm" in metrics:
            col4.metric("📉 LSTM", f"{metrics['lstm']:.2%}" if metrics['lstm'] else "N/A")
        if "stack" in metrics:
            col5.metric("🌀 Stacking", f"{metrics['stack']:.2%}" if metrics['stack'] else "N/A")

        st.markdown("---")  # divider

        # -----------------------
        # 📌 Detailed Evaluation for Each Model
        # -----------------------
        for name, model in st.session_state.models.items():
            st.subheader(name.upper())

            if name in ["rf", "catboost", "stack"]:
                y_pred = model.predict(st.session_state.X_test_struct)
                acc = accuracy_score(st.session_state.y_test, y_pred)
                st.write("Accuracy:", acc)
                st.text(classification_report(st.session_state.y_test, y_pred))

                fig = plot_metrics(st.session_state.y_test, y_pred, name)
                st.pyplot(fig)

                # 🔹 ROC Curve
                y_proba = model.predict_proba(st.session_state.X_test_struct)[:, 1]
                fpr, tpr, _ = roc_curve(st.session_state.y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC={roc_auc:.2f})"))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
                fig_roc.update_layout(title=f"ROC Curve - {name.upper()}",
                                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc)

                # 🔹 Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(st.session_state.y_test, y_proba)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))
                fig_pr.update_layout(title=f"Precision-Recall Curve - {name.upper()}",
                                     xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(fig_pr)

                # 🔹 Feature Importance (only for RF & CatBoost)
                if name in ["rf", "catboost"]:
                    importance = model.feature_importances_
                    feat_df = pd.DataFrame({"Feature": st.session_state.X_train_struct.columns,
                                            "Importance": importance})
                    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)
                    fig_imp = go.Figure([go.Bar(x=feat_df["Feature"], y=feat_df["Importance"], marker_color="indigo")])
                    fig_imp.update_layout(title=f"Top Features - {name.upper()}",
                                          xaxis_title="Features", yaxis_title="Importance")
                    st.plotly_chart(fig_imp)

            elif name in ["cnn", "lstm"]:
                X_test_text = st.session_state.X_test_text
                if X_test_text is not None:
                    y_proba = model.predict(X_test_text).ravel()
                    y_pred = (y_proba > 0.5).astype("int32")
                    acc = accuracy_score(st.session_state.y_test, y_pred)
                    st.write("Accuracy:", acc)
                    st.text(classification_report(st.session_state.y_test, y_pred))

                    fig = plot_metrics(st.session_state.y_test, y_pred, name)
                    st.pyplot(fig)

                    # 🔹 ROC Curve
                    fpr, tpr, _ = roc_curve(st.session_state.y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC={roc_auc:.2f})"))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                                 line=dict(dash="dash"), name="Random"))
                    fig_roc.update_layout(title=f"ROC Curve - {name.upper()}",
                                          xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                    st.plotly_chart(fig_roc)

                    # 🔹 Precision-Recall Curve
                    precision, recall, _ = precision_recall_curve(st.session_state.y_test, y_proba)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))
                    fig_pr.update_layout(title=f"Precision-Recall Curve - {name.upper()}",
                                         xaxis_title="Recall", yaxis_title="Precision")
                    st.plotly_chart(fig_pr)
                else:
                    st.warning(f"No text data available for {name.upper()}")

        # -----------------------
        # 📌 Final Comparison Table (with Download Option)
        # -----------------------
        comparison = []
        for name, model in st.session_state.models.items():
            try:
                if name in ["rf", "catboost", "stack"]:
                    preds = model.predict(st.session_state.X_test_struct)
                else:
                    preds = (model.predict(st.session_state.X_test_text) > 0.5).astype("int32")

                acc = accuracy_score(st.session_state.y_test, preds)
                prec = precision_score(st.session_state.y_test, preds, zero_division=0)
                rec = recall_score(st.session_state.y_test, preds, zero_division=0)
                f1 = f1_score(st.session_state.y_test, preds, zero_division=0)

                comparison.append({
                    "Model": name.upper(),
                    "Accuracy": round(acc, 3),
                    "Precision": round(prec, 3),
                    "Recall": round(rec, 3),
                    "F1-Score": round(f1, 3)
                })
            except Exception:
                pass

        if comparison:
            st.markdown("### 📊 Final Comparison Table")
            comp_df = pd.DataFrame(comparison)
            st.dataframe(comp_df, use_container_width=True)

            # 🔹 Download as CSV
            csv = comp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="model_comparison.csv",
                mime="text/csv"
            )

            # 🔹 Download as Excel
            import io
            from openpyxl import Workbook
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                comp_df.to_excel(writer, index=False, sheet_name="Results")

            st.download_button(
                label="📥 Download Results as Excel",
                data=output.getvalue(),
                file_name="model_comparison.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
  
            # -----------------------

 
# ---------------------------
# Live Prediction with SHAP
# ---------------------------
elif section == "🧑‍💻 Live Prediction":
    st.header("🧑‍💻 Live Prediction & Explainability")
    if st.session_state.X_train_struct is None:
        st.warning("Please configure and train models first.")
    else:
        # Take user inputs
        user_input = {}
        for col in st.session_state.X_train_struct.columns:
            val = st.number_input(f"Enter {col}", value=0.0)
            user_input[col] = val
        X_input_struct = pd.DataFrame([user_input])
        bio_text = st.text_input("Enter bio/description text")

        if st.button("Predict"):
            for name, model in st.session_state.models.items():
                st.subheader(name.upper())
                
                # ----------------------
                # Tree-based Models (RF, CatBoost, Stack)
                # ----------------------
                if name in ["rf", "catboost", "stack"]:
                    pred = model.predict(X_input_struct)[0]
                    st.write("Prediction:", "Fake" if pred == 1 else "Genuine")

                    
      


# ---------------------------
# Save/Load
# ---------------------------
elif section == "💾 Save/Load":
    st.header("💾 Save/Load Models")
    if st.session_state.models:
        if st.button("💾 Save Models"):
            joblib.dump(st.session_state.models, "models.pkl")
            st.success("✅ Models saved as models.pkl")
    uploaded = st.file_uploader("📂 Upload a saved model file (models.pkl)", type=["pkl"])
    if uploaded is not None:
        st.session_state.models = joblib.load(uploaded)
        st.success("✅ Models loaded successfully!")
        st.write("Loaded models:", list(st.session_state.models.keys()))
  
