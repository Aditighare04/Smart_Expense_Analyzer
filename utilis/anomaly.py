"""
anomaly.py
Flags unusual transactions using Isolation Forest.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    expense_df = df[df["IsExpense"]].copy()
    if len(expense_df) < 10:
        expense_df["Anomaly"] = False
        return expense_df

    le = LabelEncoder()
    expense_df["_cat"] = le.fit_transform(expense_df["Category"].astype(str))
    expense_df["_day"] = expense_df["Date"].dt.weekday

    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    preds = model.fit_predict(expense_df[["AbsAmount", "_cat", "_day"]])
    expense_df["Anomaly"] = preds == -1

    return expense_df.drop(columns=["_cat", "_day"])
