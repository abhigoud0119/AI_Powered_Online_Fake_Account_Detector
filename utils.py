import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

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
    n_estimators=800,        # number of trees (500–1000)
    max_depth=15,            # control tree depth (try 10–20)
    min_samples_split=5,     # min samples to split node
    class_weight="balanced", # handles class imbalance
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
    # 3. CNN (if text provided)
    # -------------------
    if X_train_text is not None and tokenizer is not None:
        vocab_size = min(len(tokenizer.word_index) + 1, 20000)
        cnn = build_cnn(vocab_size=vocab_size, maxlen=maxlen)
        cnn.fit(X_train_text, y_train, epochs=5, batch_size=64, verbose=1, validation_split=0.2)
        models["cnn"] = cnn

    # -------------------
    # 4. LSTM (if text provided)
    # -------------------
    if X_train_text is not None and tokenizer is not None:
        vocab_size = min(len(tokenizer.word_index) + 1, 20000)
        lstm = build_lstm(vocab_size=vocab_size, maxlen=maxlen)
        lstm.fit(X_train_text, y_train, epochs=4, batch_size=64, verbose=1, validation_split=0.2)
        models["lstm"] = lstm

    return models



def stack_models(models, X_train, y_train):
    estimators = [(k, v) for k, v in models.items()]
    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=CatBoostClassifier(),
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
