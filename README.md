# 💰 Smart Expense Analyzer

A complete, **100% free**, end-to-end Data Science web application for personal expense analysis.

**Supports CSV · Excel (.xlsx/.xls) · PDF bank statements**

---

## 🚀 Features
- Upload CSV, Excel, or PDF bank statements — auto-detected
- Automatic data cleaning, duplicate removal, date parsing
- Rule-based + ML expense categorization (Random Forest / Decision Tree / Logistic Regression)
- Interactive dashboard: category pie, monthly trend, income vs expense, top merchants, day-of-week
- Next-month expense forecasting (Linear Regression, no Prophet needed)
- Anomaly detection with Isolation Forest
- Rule-based budget recommendations with savings-rate feedback
- Download processed data as CSV or Excel

---

## 🛠 Tech Stack
Python · Streamlit · Pandas · NumPy · Scikit-learn · Plotly · Joblib · pdfplumber · openpyxl

---

## ⚙️ Setup (Windows 11 + VS Code)

```bash
# 1. Clone
git clone https://github.com/<your-username>/smart-expense-analyzer.git
cd smart-expense-analyzer

# 2. Create virtual environment
python -m venv venv

# 3. Activate (PowerShell — run this FIRST if you get a security error)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Generate sample data (optional)
python generate_sample_data.py

# 6. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📂 Supported File Formats

| Format | Extensions | Notes |
|---|---|---|
| CSV | `.csv` | Standard bank export. Needs Date + Amount columns. |
| Excel | `.xlsx`, `.xls` | Multi-sheet supported. Header row auto-detected. |
| PDF | `.pdf` | Digital statements (not scanned/image PDFs). |

---

## 📋 Expected Column Names (auto-detected, case-insensitive)

| Field | Accepted names |
|---|---|
| Date | `Date`, `Transaction Date`, `Txn Date`, `Value Date` |
| Amount | `Amount`, `Amt`, `Debit`, `Credit`, `Value` |
| Description | `Description`, `Merchant`, `Narration`, `Particulars` |
| Type | `Type`, `Dr/Cr` *(optional)* |
| Category | `Category`, `Label` *(optional — predicted if missing)* |

---

## 📁 Project Structure

```
smart-expense-analyzer/
├── app.py                        # Main Streamlit application
├── requirements.txt
├── generate_sample_data.py       # Generate demo dataset
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml               # Theme + server settings
├── utils/
│   ├── __init__.py
│   ├── data_processing.py        # Cleaning & feature engineering
│   ├── categorization.py         # Rule-based keyword categorization
│   ├── pdf_extractor.py          # PDF → DataFrame (pdfplumber)
│   ├── excel_extractor.py        # Excel → DataFrame (openpyxl/xlrd)
│   ├── ml_models.py              # Train/compare/save classifiers
│   ├── forecasting.py            # Next-month forecasting
│   ├── anomaly.py                # Isolation Forest anomaly detection
│   └── budget.py                 # Rule-based budget recommendations
├── sample_data/
│   └── sample_transactions.csv
└── models/
    └── .gitkeep                  # Saved .joblib models go here
```

---

## 🚢 Free Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub
2. Go to https://share.streamlit.io → **New app**
3. Select your repo → branch `main` → file `app.py`
4. Click **Deploy** — free hosting, shareable link

---

## 📄 License
MIT License — free for academic and portfolio use.
