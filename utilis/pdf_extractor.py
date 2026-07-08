"""
pdf_extractor.py
Extracts transactions from bank/credit-card statement PDFs.

Strategy:
  1. Table extraction via pdfplumber  (covers most grid-based statements)
  2. Regex fallback on raw text lines (for non-grid / plain-text PDFs)

Returns a DataFrame with columns: Date, Description, Amount, Type
ready for clean_data().

Note: scanned / image-only PDFs are not supported (no OCR) — user should
export as digital PDF from net-banking.
"""
import io
import re
import pandas as pd
import pdfplumber

DATE_HEADERS    = ["date", "txn date", "transaction date", "value date"]
DESC_HEADERS    = ["description", "narration", "particulars", "details",
                   "merchant", "remarks"]
AMOUNT_HEADERS  = ["amount", "debit", "credit", "withdrawal", "deposit", "amt"]

DATE_RE   = r"(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})"
AMOUNT_RE = r"[-+]?₹?\s?\d[\d,]*\.?\d{0,2}"


# ── Table-based extraction ────────────────────────────────────────────────────

def _map_row(headers, row):
    rec = {}
    for h, v in zip(headers, row):
        if not h or not v:
            continue
        hl = str(h).strip().lower()
        vl = str(v).strip()
        if any(k in hl for k in DATE_HEADERS) and "date" not in rec:
            rec["date"] = vl
        elif any(k in hl for k in DESC_HEADERS) and "description" not in rec:
            rec["description"] = vl
        elif "debit" in hl or "withdrawal" in hl:
            if vl and vl not in ("", "0", "0.00", "-"):
                rec["amount"] = vl
                rec["type"]   = "Debit"
        elif "credit" in hl or "deposit" in hl:
            if vl and vl not in ("", "0", "0.00", "-"):
                rec["amount"] = vl
                rec["type"]   = "Credit"
        elif any(k in hl for k in AMOUNT_HEADERS) and "amount" not in rec:
            rec["amount"] = vl
    return rec


def _table_extract(pdf) -> pd.DataFrame:
    rows = []
    for page in pdf.pages:
        for table in (page.extract_tables() or []):
            if not table or len(table) < 2:
                continue
            headers = table[0]
            for row in table[1:]:
                r = _map_row(headers, row)
                if r.get("date") and r.get("amount"):
                    rows.append(r)
    return pd.DataFrame(rows)


# ── Text / Regex fallback ────────────────────────────────────────────────────

_LINE_RE = re.compile(
    rf"(?P<date>{DATE_RE})\s+"
    rf"(?P<description>.+?)\s+"
    rf"(?P<amount>{AMOUNT_RE})"
    rf"\s*(?P<sign>CR|DR|credit|debit)?",
    re.IGNORECASE,
)


def _text_extract(pdf) -> pd.DataFrame:
    rows = []
    for page in pdf.pages:
        text = page.extract_text() or ""
        for line in text.split("\n"):
            m = _LINE_RE.search(line.strip())
            if m:
                sign = (m.group("sign") or "").upper()
                rows.append({
                    "date":        m.group("date"),
                    "description": m.group("description").strip(),
                    "amount":      m.group("amount").replace(",", "").replace("₹", ""),
                    "type":        ("Credit" if "CR" in sign else
                                   "Debit"  if "DR" in sign else None),
                })
    return pd.DataFrame(rows)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_transactions_from_pdf(file) -> pd.DataFrame:
    """
    Accepts a file path (str) or any file-like / Streamlit UploadedFile object.
    Returns DataFrame with columns: Date, Description, Amount, Type.
    """
    if hasattr(file, "read"):
        raw = file.read()
        src = io.BytesIO(raw)
    else:
        src = file

    with pdfplumber.open(src) as pdf:
        df = _table_extract(pdf)
        if df.empty:
            if hasattr(src, "seek"):
                src.seek(0)
            with pdfplumber.open(src if not isinstance(src, str) else src) as pdf2:
                df = _text_extract(pdf2)

    if df.empty:
        raise ValueError(
            "No transactions found in this PDF.\n"
            "• Make sure it is a digital PDF (not a scanned image).\n"
            "• Try downloading the statement directly from your bank's portal."
        )

    df = df.rename(columns={
        "date": "Date", "description": "Description",
        "amount": "Amount", "type": "Type",
    })
    if "Description" not in df.columns:
        df["Description"] = "Unknown"
    if "Type" not in df.columns:
        df["Type"] = None

    return df[["Date", "Description", "Amount", "Type"]].copy()
