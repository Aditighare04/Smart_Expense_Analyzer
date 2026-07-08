"""
forecasting.py
Lightweight next-month expense forecasting.
Primary  : Linear Regression on monthly totals (trend-based).
Fallback : Moving average (if < 3 months of history).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def monthly_expense_series(df: pd.DataFrame) -> pd.DataFrame:
    exp = df[df["IsExpense"]].copy()
    exp["MonthPeriod"] = exp["Date"].dt.to_period("M")
    monthly = (
        exp.groupby("MonthPeriod")["AbsAmount"].sum()
        .reset_index()
    )
    monthly["MonthPeriod"] = monthly["MonthPeriod"].astype(str)
    monthly = monthly.sort_values("MonthPeriod").reset_index(drop=True)
    monthly["t"] = np.arange(len(monthly))
    return monthly


def forecast_next_month(df: pd.DataFrame):
    monthly = monthly_expense_series(df)

    if monthly.empty:
        return monthly, None, "none", None

    last_period = pd.Period(monthly["MonthPeriod"].iloc[-1], freq="M")
    next_label  = str(last_period + 1)

    if len(monthly) < 3:
        forecast_val = monthly["AbsAmount"].mean()
        method = "Moving Average (limited history)"
    else:
        lr = LinearRegression()
        lr.fit(monthly[["t"]], monthly["AbsAmount"])
        forecast_val = max(float(lr.predict([[monthly["t"].iloc[-1] + 1]])[0]), 0)
        method = "Linear Regression (trend-based)"

    return monthly, round(forecast_val, 2), method, next_label
