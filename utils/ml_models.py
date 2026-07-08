"""
ml_models.py
Trains and compares Random Forest, Decision Tree, and Logistic Regression
to predict expense Category. Best model (by weighted F1) saved with Joblib.
"""
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_category_model.joblib")


def _build_pipeline(model):
    pre = ColumnTransformer([
        ("text", TfidfVectorizer(max_features=300, ngram_range=(1, 2)), "Description"),
        ("num",  StandardScaler(), ["AbsAmount"]),
    ])
    return Pipeline([("prep", pre), ("clf", model)])


def train_and_compare(df: pd.DataFrame):
    data = df[["Description", "AbsAmount", "Category"]].dropna()
    vc   = data["Category"].value_counts()
    data = data[data["Category"].isin(vc[vc >= 2].index)]

    if data["Category"].nunique() < 2 or len(data) < 20:
        return None, "Not enough labelled data (need ≥ 20 rows, ≥ 2 categories)."

    X = data[["Description", "AbsAmount"]]
    y = data["Category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    candidates = {
        "Random Forest":      RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
        "Decision Tree":      DecisionTreeClassifier(max_depth=10, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    results = {}
    best_name, best_f1, best_pipe = None, -1, None

    for name, model in candidates.items():
        pipe  = _build_pipeline(model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        results[name] = {
            "accuracy":         round(accuracy_score(y_test, preds), 4),
            "precision":        round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
            "recall":           round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
            "f1":               round(f1, 4),
            "confusion_matrix": confusion_matrix(y_test, preds, labels=sorted(y.unique())),
            "labels":           sorted(y.unique()),
        }
        if f1 > best_f1:
            best_f1, best_name, best_pipe = f1, name, pipe

    feature_importance = None
    if best_name in ("Random Forest", "Decision Tree"):
        try:
            tfidf_feats = best_pipe.named_steps["prep"] \
                .named_transformers_["text"].get_feature_names_out()
            all_feats = list(tfidf_feats) + ["AbsAmount"]
            imps = best_pipe.named_steps["clf"].feature_importances_
            feature_importance = sorted(
                zip(all_feats, imps), key=lambda x: x[1], reverse=True
            )[:15]
        except Exception:
            pass

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)

    return {
        "results":            results,
        "best_model":         best_name,
        "best_f1":            round(best_f1, 4),
        "feature_importance": feature_importance,
        "model_path":         MODEL_PATH,
    }, None


def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


def predict_categories(pipeline, df: pd.DataFrame):
    return pipeline.predict(df[["Description", "AbsAmount"]])
