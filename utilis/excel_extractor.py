"""
excel_extractor.py
Extracts transactions from Excel workbooks (.xlsx / .xls).

Handles:
  • Single-sheet workbooks  — reads the first sheet directly.
  • Multi-sheet workbooks   — finds the sheet whose headers best match the
    expected schema (Date, Description, Amount) and uses that one.
  • Header row not at row 0 — scans the first 15 rows for the header row
    so the app still works for statements with title/logo rows at the top.

Returns a DataFrame with columns: Date, Description, Amount, (Type)
ready to pass into clean_data().
"""
import io
import pandas as pd

MATCH_COLS = ["date", "amount", "description", "narration", "debit", "credit"]


def _header_score(row) -> int:
    """How many expected column keywords appear in this row."""
    text = " ".join(str(v).lower() for v in row if v is not None)
    return sum(1 for kw in MATCH_COLS if kw in text)


def _find_header_row(df_raw: pd.DataFrame):
    """Scan first 15 rows, return (header_row_idx, df_with_correct_header)."""
    for i in range(min(15, len(df_raw))):
        if _header_score(df_raw.iloc[i].values) >= 2:
            new_df = df_raw.iloc[i + 1:].copy()
            new_df.columns = df_raw.iloc[i].values
            new_df = new_df.reset_index(drop=True)
            return new_df
    return df_raw  # header already at row 0


def _best_sheet(xl: pd.ExcelFile) -> pd.DataFrame:
    """Return the sheet most likely to contain transaction data."""
    best_score, best_df = -1, None
    for name in xl.sheet_names:
        raw = xl.parse(name, header=None)
        if raw.empty:
            continue
        score = sum(_header_score(raw.iloc[i].values)
                    for i in range(min(5, len(raw))))
        if score > best_score:
            best_score = score
            best_df = raw
    return best_df if best_df is not None else xl.parse(xl.sheet_names[0], header=None)


def extract_transactions_from_excel(file) -> pd.DataFrame:
    """
    Accepts a file path (str) or any file-like / Streamlit UploadedFile object.
    Returns DataFrame with columns: Date, Description, Amount, Type (if found).
    """
    if hasattr(file, "read"):
        raw_bytes = file.read()
        src = io.BytesIO(raw_bytes)
    else:
        src = file

    xl = pd.ExcelFile(src)
    raw = _best_sheet(xl)
    df  = _find_header_row(raw)

    # Drop completely empty rows/cols
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    if df.empty:
        raise ValueError(
            "The Excel file appears empty or no transaction table was found.\n"
            "Make sure the sheet contains Date and Amount columns."
        )

    return df
