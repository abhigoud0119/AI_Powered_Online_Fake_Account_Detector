import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from utils import train_models, stack_models, preprocess_data, plot_metrics


  # put your image in the same folder


# -----------------------
# 🎨 Extra: Custom Page Config + CSS
# -----------------------
st.set_page_config(page_title="Fake Profile Detector", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-image:url("https://img.freepik.com/free-vector/gradient-abstract-wireframe-background_23-2149009903.jpg");
        background-size: cover;
    }
    .main { background-color: #F0F8FF; }
    h1, h2, h3 { color: #4A148C; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 18px;
        font-size: 16px;
    }
    .css-1d391kg { background-color: #E3F2FD; } /* Sidebar background */
    </style>
    """, unsafe_allow_html=True
)

st.title("🔎 Online Fake Profile Detection")

st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Data & Config", "Train Models", "Evaluate", "Live Prediction", "Save/Load"])

if "models" not in st.session_state:
    st.session_state.models = {}
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None

if section == "Data & Config":
    st.header("Step 1: Upload Dataset")
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

            st.session_state.X_train_struct = X_train_struct
            st.session_state.X_test_struct = X_test_struct
            st.session_state.X_train_text = X_train_text
            st.session_state.X_test_text = X_test_text
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.tokenizer = tokenizer

            st.success("Data configured and split. Ready for training!")


elif section == "Train Models":
    if "X_train_struct" not in st.session_state or st.session_state.X_train_struct is None:
        st.warning("Please configure data first in 'Data & Config'")
    else:
        st.header("Step 2: Train Models")
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

elif section == "Evaluate":
    if not st.session_state.models:
        st.warning("Train models first.")
    else:
        st.header("Step 3: Evaluate Models")

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
                fig_roc.update_layout(title=f"ROC Curve - {name.upper()}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc)

                # 🔹 Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(st.session_state.y_test, y_proba)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))
                fig_pr.update_layout(title=f"Precision-Recall Curve - {name.upper()}", xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(fig_pr)

                # 🔹 Feature Importance (only for RF & CatBoost)
                if name in ["rf", "catboost"]:
                    importance = model.feature_importances_
                    feat_df = pd.DataFrame({"Feature": st.session_state.X_train_struct.columns, "Importance": importance})
                    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)
                    fig_imp = go.Figure([go.Bar(x=feat_df["Feature"], y=feat_df["Importance"], marker_color="indigo")])
                    fig_imp.update_layout(title=f"Top Features - {name.upper()}", xaxis_title="Features", yaxis_title="Importance")
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
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
                    fig_roc.update_layout(title=f"ROC Curve - {name.upper()}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                    st.plotly_chart(fig_roc)

                    # 🔹 Precision-Recall Curve
                    precision, recall, _ = precision_recall_curve(st.session_state.y_test, y_proba)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))
                    fig_pr.update_layout(title=f"Precision-Recall Curve - {name.upper()}", xaxis_title="Recall", yaxis_title="Precision")
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




elif section == "Live Prediction":
    st.header("Step 4: Live Prediction")

    if "X_train_struct" not in st.session_state or st.session_state.X_train_struct is None:
        st.warning("Please configure and train models first.")
    else:
        user_input = {}
        for col in st.session_state.X_train_struct.columns:
            val = st.number_input(f"Enter {col}", value=0.0)
            user_input[col] = val

        # Convert to DataFrame for structured models
        X_input_struct = pd.DataFrame([user_input])

        # Bio text input for CNN/LSTM
        bio_text = st.text_input("Enter bio/description text")

        if st.button("Predict"):
            for name, model in st.session_state.models.items():
                st.subheader(name.upper())

                if name in ["rf", "catboost", "stack"]:
                    pred = model.predict(X_input_struct)[0]
                    st.write("Prediction:", "Fake" if pred == 1 else "Genuine")

                elif name in ["cnn", "lstm"]:
                    if bio_text.strip():
                        tokenizer = st.session_state.tokenizer
                        from tensorflow.keras.preprocessing.sequence import pad_sequences
                        seq = tokenizer.texts_to_sequences([bio_text])
                        padded = pad_sequences(seq, maxlen=100)
                        pred = (model.predict(padded) > 0.5).astype("int32")[0][0]
                        st.write("Prediction:", "Fake" if pred == 1 else "Genuine")
                    else:
                        st.warning("Please enter bio text for CNN/LSTM prediction.")


elif section == "Save/Load":
    st.header("Step 5: Save/Load Models")

    # Save models
    if st.session_state.models:
        if st.button("💾 Save Models"):
            joblib.dump(st.session_state.models, "models.pkl")
            st.success("✅ Models saved as models.pkl")

    # Load models
    uploaded = st.file_uploader("📂 Upload a saved model file (models.pkl)", type=["pkl"])
    if uploaded is not None:
        st.session_state.models = joblib.load(uploaded)
        st.success("✅ Models loaded successfully!")

        # Show which models are available
        st.write("Loaded models:", list(st.session_state.models.keys()))
