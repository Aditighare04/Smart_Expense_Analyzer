"""
data_processing.py
Cleans, parses, and feature-engineers any uploaded transaction data,
regardless of whether it came from CSV, Excel, or PDF extraction.
"""
import pandas as pd
import numpy as np

REQUIRED_COLS_CANDIDATES = {
    "date":        ["date", "transaction date", "txn date", "value date"],
    "description": ["description", "merchant", "narration", "details",
                    "particulars", "remarks"],
    "amount":      ["amount", "amt", "value"],
    "type":        ["type", "transaction type", "debit/credit", "dr/cr"],
    "category":    ["category", "label"],
}


def _match_column(columns, candidates):
    lower_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    for std_name, cands in REQUIRED_COLS_CANDIDATES.items():
        match = _match_column(df.columns, cands)
        if match:
            label = "Type" if std_name == "type" else std_name.capitalize()
            rename[match] = label
    return df.rename(columns=rename)


def clean_data(df: pd.DataFrame):
    df = standardize_columns(df)

    if "Date" not in df.columns:
        raise ValueError("Could not find a Date column. Please check your file headers.")
    if "Amount" not in df.columns:
        raise ValueError("Could not find an Amount column. Please check your file headers.")
    if "Description" not in df.columns:
        df["Description"] = "Unknown"

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    n_before = len(df)
    df = df.dropna(subset=["Date"])

    # Clean amount
    df["Amount"] = (
        df["Amount"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip()
    )
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])

    # Normalise sign using Type column if amounts are all positive
    if "Type" in df.columns and (df["Amount"] >= 0).all():
        t = df["Type"].astype(str).str.lower().str.strip()
        df.loc[t.isin(["debit", "expense", "dr", "withdrawal"]), "Amount"] = \
            -df.loc[t.isin(["debit", "expense", "dr", "withdrawal"]), "Amount"].abs()
        df.loc[t.isin(["credit", "income", "cr", "deposit"]), "Amount"] = \
            df.loc[t.isin(["credit", "income", "cr", "deposit"]), "Amount"].abs()

    if "Type" not in df.columns:
        df["Type"] = np.where(df["Amount"] >= 0, "Credit", "Debit")

    df["Description"] = df["Description"].fillna("Unknown").astype(str).str.strip()
    df = df.drop_duplicates(subset=["Date", "Description", "Amount"])

    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b %Y")
    df["Weekday"]   = df["Date"].dt.day_name()
    df["IsExpense"] = df["Amount"] < 0
    df["AbsAmount"] = df["Amount"].abs()

    df = df.sort_values("Date").reset_index(drop=True)
    meta = {"rows_dropped": max(0, n_before - len(df))}
    return df, meta


def get_summary_stats(df: pd.DataFrame) -> dict:
    exp = df[df["IsExpense"]]
    inc = df[~df["IsExpense"]]
    total_expense = exp["AbsAmount"].sum()
    total_income  = inc["AbsAmount"].sum()
    savings       = total_income - total_expense
    savings_rate  = (savings / total_income * 100) if total_income > 0 else 0
    return {
        "total_income":      round(total_income, 2),
        "total_expense":     round(total_expense, 2),
        "savings":           round(savings, 2),
        "savings_rate":      round(savings_rate, 2),
        "transaction_count": len(df),
        "avg_transaction":   round(exp["AbsAmount"].mean(), 2) if len(exp) else 0,
    }
