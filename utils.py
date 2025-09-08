import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------
# Data Preprocessing
# -----------------------
def preprocess_data(df, target_col, text_col, feature_cols, max_words=20000, maxlen=150):
    y = df[target_col].astype(int)

    # Structured Features
    X_struct = df[feature_cols].copy()
    for col in X_struct.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X_struct[col] = le.fit_transform(X_struct[col].astype(str))

    # Text Features
    X_text, tokenizer = None, None
    if text_col:
        texts = df[text_col].astype(str).fillna("")
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        X_text = pad_sequences(sequences, maxlen=maxlen)

    # Split
    if X_text is not None:
        X_train_struct, X_test_struct, X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_struct, X_text, y, test_size=0.2, random_state=42
        )
        return (X_train_struct, X_test_struct, X_train_text, X_test_text, y_train, y_test), tokenizer
    else:
        X_train_struct, X_test_struct, y_train, y_test = train_test_split(
            X_struct, y, test_size=0.2, random_state=42
        )
        return (X_train_struct, X_test_struct, None, None, y_train, y_test), tokenizer


# -----------------------
# Deep Learning Models
# -----------------------
def build_cnn(vocab_size=20000, maxlen=150):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=maxlen),
        Conv1D(64, 5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_lstm(vocab_size=20000, maxlen=150):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=maxlen),
        LSTM(64),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# -----------------------
# ML Models
# -----------------------
def train_models(X_train_struct, y_train, X_train_text=None, X_test_text=None, tokenizer=None, maxlen=150):
    models = {}

    # -------------------
    # 1. Random Forest
    # -------------------
    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_struct, y_train)
    models["rf"] = rf

    # -------------------
    # 2. CatBoost
    # -------------------
    cat = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5,
        border_count=128,
        eval_metric="Accuracy",
        verbose=200
    )
    cat.fit(X_train_struct, y_train)
    models["catboost"] = cat

    # -------------------
    # 3. CNN
    # -------------------
    if X_train_text is not None and tokenizer is not None:
        vocab_size = min(len(tokenizer.word_index) + 1, 20000)
        cnn = build_cnn(vocab_size=vocab_size, maxlen=maxlen)
        cnn.fit(X_train_text, y_train, epochs=5, batch_size=64, verbose=1, validation_split=0.2)
        models["cnn"] = cnn

    # -------------------
    # 4. LSTM
    # -------------------
    if X_train_text is not None and tokenizer is not None:
        vocab_size = min(len(tokenizer.word_index) + 1, 20000)
        lstm = build_lstm(vocab_size=vocab_size, maxlen=maxlen)
        lstm.fit(X_train_text, y_train, epochs=4, batch_size=64, verbose=1, validation_split=0.2)
        models["lstm"] = lstm

    return models


# -----------------------
# Stacking
# -----------------------
def stack_models(models, X_train, y_train):
    estimators = [(k, v) for k, v in models.items() if k not in ["cnn", "lstm"]]
    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=CatBoostClassifier(verbose=0),
        passthrough=True,
        n_jobs=-1
    )
    stacker.fit(X_train, y_train)
    return stacker


# -----------------------
# Evaluation
# -----------------------
def plot_metrics(y_true, y_pred, name):
    cm = pd.crosstab(y_true, y_pred, rownames=["Actual"], colnames=["Predicted"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    return fig


def plot_roc_pr(y_true, y_proba, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    axes[0].plot([0, 1], [0, 1], "r--")
    axes[0].legend()
    axes[0].set_title(f"ROC Curve - {name}")

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[1].plot(recall, precision)
    axes[1].set_title(f"PR Curve - {name}")

    plt.tight_layout()
    return fig


# -----------------------
# SHAP Explainability
# -----------------------
def explain_model(model, X_sample, model_type="tree"):
    explainer, shap_values = None, None
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False)
    elif model_type == "deep":
        explainer = shap.DeepExplainer(model, X_sample[:50])
        shap_values = explainer.shap_values(X_sample[:50])
        shap.summary_plot(shap_values, X_sample[:50], show=False)
    return plt.gcf()


# -----------------------
# Save & Load Models
# -----------------------
def save_model(model, filename):
    if hasattr(model, "save"):  # Keras
        model.save(filename)
    else:
        joblib.dump(model, filename)

def load_model(filename):
    try:
        return joblib.load(filename)
    except:
        from tensorflow.keras.models import load_model
        return load_model(filename)


# -----------------------
# Report Export
# -----------------------
def export_report(metrics_df, filename="report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Fake Profile Detection Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Model Performance Summary:", styles["Heading2"]))

    for i, row in metrics_df.iterrows():
        text = f"{row['Model']}: Accuracy={row['Accuracy']:.2f}, Precision={row['Precision']:.2f}, Recall={row['Recall']:.2f}, F1={row['F1']:.2f}"
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 8))

    doc.build(elements)
    return filename
