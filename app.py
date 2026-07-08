"""
app.py — Smart Expense Analyzer
Supports: CSV (.csv) · Excel (.xlsx / .xls) · PDF bank statements (.pdf)

Run: streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_processing   import clean_data, get_summary_stats
from utils.categorization    import apply_rule_based_categories
from utils.ml_models         import train_and_compare, load_model, predict_categories
from utils.forecasting       import forecast_next_month
from utils.anomaly           import detect_anomalies
from utils.budget            import generate_budget_suggestions
from utils.pdf_extractor     import extract_transactions_from_pdf
from utils.excel_extractor   import extract_transactions_from_excel

# ─────────────────────────────────────────────── page config
st.set_page_config(
    page_title="Smart Expense Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { padding-top: 0.8rem; }
div[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }
div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #666; }
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; padding: 8px 14px; }
.upload-box { border: 2px dashed #2E7D32; border-radius: 10px;
              padding: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

if "df" not in st.session_state:
    st.session_state.df = None

# ─────────────────────────────────────────────── helpers
FILE_ICONS = {".csv": "📄", ".xlsx": "📊", ".xls": "📊", ".pdf": "📑"}

def load_file(uploaded_file) -> pd.DataFrame:
    """Route uploaded file to the correct extractor and return a raw DataFrame."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".csv":
        return pd.read_csv(uploaded_file)
    elif ext in (".xlsx", ".xls"):
        return extract_transactions_from_excel(uploaded_file)
    elif ext == ".pdf":
        return extract_transactions_from_pdf(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ─────────────────────────────────────────────── sidebar
st.sidebar.title("💰 Smart Expense Analyzer")
st.sidebar.caption("Upload • Analyze • Forecast — 100 % free & offline")

# ── File Upload Section ──
st.sidebar.subheader("1. Load Data")

st.sidebar.markdown("""
**Supported formats**
| Icon | Format | Notes |
|---|---|---|
| 📄 | CSV | Any bank export |
| 📊 | Excel | .xlsx / .xls |
| 📑 | PDF | Digital statements |
""")

upload = st.sidebar.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls", "pdf"],
    help="Upload your bank statement in CSV, Excel, or PDF format.",
)
use_sample = st.sidebar.button("📂 Use Sample Data (CSV)")

if upload is not None:
    icon = FILE_ICONS.get(os.path.splitext(upload.name)[1].lower(), "📁")
    with st.sidebar:
        with st.spinner(f"{icon} Reading {upload.name}…"):
            try:
                raw_df  = load_file(upload)
                cleaned, meta = clean_data(raw_df)
                cleaned = apply_rule_based_categories(cleaned)
                st.session_state.df = cleaned
                st.sidebar.success(
                    f"{icon} Loaded **{len(cleaned)}** transactions "
                    f"from `{upload.name}`"
                )
                if meta["rows_dropped"] > 0:
                    st.sidebar.info(
                        f"ℹ️ {meta['rows_dropped']} rows skipped "
                        "(invalid date or amount)."
                    )
            except Exception as e:
                st.sidebar.error(f"❌ Could not process file:\n\n{e}")

if use_sample:
    sample_path = os.path.join("sample_data", "sample_transactions.csv")
    raw_df  = pd.read_csv(sample_path)
    cleaned, _ = clean_data(raw_df)
    cleaned = apply_rule_based_categories(cleaned)
    st.session_state.df = cleaned
    st.sidebar.success(f"📄 Sample data loaded — {len(cleaned)} transactions.")

st.sidebar.markdown("---")

# ── Navigation ──
page = st.sidebar.radio(
    "2. Navigate",
    ["📊 Dashboard",
     "🤖 ML Categorization",
     "📈 Forecasting",
     "🚨 Anomaly Detection",
     "💡 Budget Advisor",
     "📁 Data & Export"],
)

df = st.session_state.df

# ─────────────────────────────────────────────── welcome screen
if df is None:
    st.title("💰 Smart Expense Analyzer")
    st.markdown("### Upload a file or use the sample dataset to get started.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **📄 CSV**
        Standard spreadsheet export from any bank.
        Needs: `Date`, `Amount` columns.
        """)
    with c2:
        st.markdown("""
        **📊 Excel (.xlsx / .xls)**
        Multi-sheet workbooks supported.
        Header row is detected automatically.
        """)
    with c3:
        st.markdown("""
        **📑 PDF**
        Digital bank/credit-card statements.
        Table & text layouts both supported.
        """)

    st.info("👈 Use the sidebar to upload your file or click **'Use Sample Data'**.")
    st.markdown("""
    #### Minimum required columns (any of these names will be auto-detected)
    | Column | Accepted header names |
    |---|---|
    | Date | `Date`, `Transaction Date`, `Txn Date`, `Value Date` |
    | Amount | `Amount`, `Amt`, `Value`, `Debit`, `Credit` |
    | Description | `Description`, `Merchant`, `Narration`, `Particulars` |
    | Type | `Type`, `Dr/Cr` — *optional, auto-derived from sign if absent* |
    """)
    st.stop()

# ─────────────────────────────────────────────── shared filters
st.sidebar.markdown("---")
st.sidebar.subheader("3. Filters")
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date])
all_cats   = sorted(df["Category"].unique())
sel_cats   = st.sidebar.multiselect("Categories", all_cats, default=all_cats)

if len(date_range) == 2:
    mask = (
        (df["Date"] >= pd.Timestamp(date_range[0])) &
        (df["Date"] <= pd.Timestamp(date_range[1]))
    )
    df_f = df[mask]
else:
    df_f = df.copy()
df_f = df_f[df_f["Category"].isin(sel_cats)]

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · Scikit-learn · Plotly")

# ══════════════════════════════════════════════════════ DASHBOARD
if page == "📊 Dashboard":
    st.title("📊 Expense Dashboard")
    stats = get_summary_stats(df_f)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💵 Total Income",    f"₹{stats['total_income']:,.0f}")
    c2.metric("💸 Total Expense",   f"₹{stats['total_expense']:,.0f}")
    c3.metric("🏦 Savings",         f"₹{stats['savings']:,.0f}")
    c4.metric("📈 Savings Rate",    f"{stats['savings_rate']}%")
    c5.metric("🔢 Transactions",    stats["transaction_count"])

    st.markdown("---")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.subheader("🍕 Spending by Category")
        cat_data = (
            df_f[df_f["IsExpense"]]
            .groupby("Category")["AbsAmount"].sum().reset_index()
        )
        fig = px.pie(cat_data, names="Category", values="AbsAmount", hole=0.42)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.subheader("📅 Monthly Expense Trend")
        monthly = df_f[df_f["IsExpense"]].copy()
        monthly["MonthPeriod"] = monthly["Date"].dt.to_period("M").astype(str)
        monthly = monthly.groupby("MonthPeriod")["AbsAmount"].sum().reset_index()
        fig2 = px.line(monthly, x="MonthPeriod", y="AbsAmount", markers=True,
                       labels={"AbsAmount": "Expense (₹)", "MonthPeriod": "Month"})
        st.plotly_chart(fig2, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("🏪 Top 10 Merchants")
        merch = (
            df_f[df_f["IsExpense"]].groupby("Description")["AbsAmount"].sum()
            .reset_index().sort_values("AbsAmount", ascending=False).head(10)
        )
        fig3 = px.bar(merch, x="AbsAmount", y="Description", orientation="h",
                      labels={"AbsAmount": "Amount (₹)", "Description": ""})
        fig3.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        st.subheader("💹 Income vs Expense by Month")
        iv = df_f.copy()
        iv["MonthPeriod"] = iv["Date"].dt.to_period("M").astype(str)
        iv["Flow"] = np.where(iv["IsExpense"], "Expense", "Income")
        iv = iv.groupby(["MonthPeriod", "Flow"])["AbsAmount"].sum().reset_index()
        fig4 = px.bar(iv, x="MonthPeriod", y="AbsAmount", color="Flow",
                      barmode="group",
                      color_discrete_map={"Income": "#2E7D32", "Expense": "#C62828"},
                      labels={"AbsAmount": "Amount (₹)", "MonthPeriod": "Month"})
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("📆 Spending by Day of the Week")
    dow = (
        df_f[df_f["IsExpense"]].groupby("Weekday")["AbsAmount"].sum()
        .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        .reset_index()
    )
    fig5 = px.bar(dow, x="Weekday", y="AbsAmount",
                  labels={"AbsAmount": "Total Spent (₹)"})
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════ ML
elif page == "🤖 ML Categorization":
    st.title("🤖 ML Expense Category Prediction")
    st.markdown(
        "Compares **Random Forest**, **Decision Tree**, and **Logistic Regression** "
        "on your data, picks the best, and saves it with Joblib."
    )

    if st.button("🚀 Train & Compare All Models", type="primary"):
        with st.spinner("Training models…"):
            summary, err = train_and_compare(df)
        if err:
            st.warning(err)
        else:
            st.success(
                f"✅ Best model: **{summary['best_model']}** "
                f"(Weighted F1 = {summary['best_f1']})"
            )

            st.subheader("Model Comparison")
            rows = [
                {"Model": n, "Accuracy": v["accuracy"], "Precision": v["precision"],
                 "Recall": v["recall"], "F1 Score": v["f1"]}
                for n, v in summary["results"].items()
            ]
            comp_df = pd.DataFrame(rows)
            st.dataframe(comp_df.style.highlight_max(
                subset=["Accuracy","Precision","Recall","F1 Score"],
                color="#c8e6c9"), use_container_width=True)

            fig = px.bar(
                comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                x="Model", y="Score", color="Metric", barmode="group",
                title="Accuracy / Precision / Recall / F1 per Model",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"Confusion Matrix — {summary['best_model']}")
            res = summary["results"][summary["best_model"]]
            cm_df = pd.DataFrame(res["confusion_matrix"],
                                 index=res["labels"], columns=res["labels"])
            fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                               labels={"x":"Predicted","y":"Actual","color":"Count"})
            st.plotly_chart(fig_cm, use_container_width=True)

            if summary["feature_importance"]:
                st.subheader("Top Feature Importances")
                fi_df = pd.DataFrame(summary["feature_importance"],
                                     columns=["Feature","Importance"])
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h")
                fig_fi.update_layout(yaxis={"categoryorder":"total ascending"})
                st.plotly_chart(fig_fi, use_container_width=True)

            st.info(f"💾 Model saved → `{summary['model_path']}`")

    st.markdown("---")
    st.subheader("🔮 Predict Category for a Single Transaction")
    model = load_model()
    if model is None:
        st.info("Train a model first using the button above.")
    else:
        col1, col2 = st.columns(2)
        desc = col1.text_input("Transaction description", "Swiggy order")
        amt  = col2.number_input("Amount (₹)", min_value=0.0, value=350.0)
        if st.button("Predict"):
            test_df = pd.DataFrame({"Description":[desc], "AbsAmount":[amt]})
            pred    = predict_categories(model, test_df)[0]
            st.success(f"Predicted category: **{pred}**")

# ══════════════════════════════════════════════════════ FORECASTING
elif page == "📈 Forecasting":
    st.title("📈 Next-Month Expense Forecast")
    monthly, fval, method, next_label = forecast_next_month(df_f)

    if fval is None:
        st.warning("Not enough data to forecast. Upload more transactions.")
    else:
        c1, c2 = st.columns(2)
        c1.metric(f"Predicted Expense — {next_label}", f"₹{fval:,.0f}")
        c2.metric("Method", method)

        actual   = monthly.assign(Type="Actual")
        forecast = pd.DataFrame({
            "MonthPeriod": [next_label],
            "AbsAmount":   [fval],
            "t":           [monthly["t"].iloc[-1] + 1],
            "Type":        ["Forecast"],
        })
        full = pd.concat([actual, forecast], ignore_index=True)
        fig  = px.line(full, x="MonthPeriod", y="AbsAmount", color="Type",
                       markers=True, title="Monthly Expense History + Forecast",
                       color_discrete_map={"Actual":"#1565C0","Forecast":"#E65100"},
                       labels={"AbsAmount":"Amount (₹)","MonthPeriod":"Month"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Uses lightweight Linear Regression on monthly totals (no Prophet required).")

# ══════════════════════════════════════════════════════ ANOMALY
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Unusual Spending Detection")
    st.markdown("Uses **Isolation Forest** to flag statistically unusual transactions.")

    contamination = st.slider("Sensitivity — expected anomaly %",
                              min_value=1, max_value=15, value=5, step=1) / 100
    result_df = detect_anomalies(df_f, contamination=contamination)
    anomalies = result_df[result_df["Anomaly"]]

    c1, c2 = st.columns(2)
    c1.metric("Flagged Transactions", len(anomalies))
    c2.metric("Total Expense Transactions", len(result_df))

    fig = px.scatter(
        result_df, x="Date", y="AbsAmount", color="Anomaly",
        color_discrete_map={True:"#D32F2F", False:"#1976D2"},
        hover_data=["Description","Category"],
        title="All Expense Transactions (🔴 = flagged as unusual)",
        labels={"AbsAmount":"Amount (₹)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    if not anomalies.empty:
        st.subheader("📋 Flagged Transactions")
        st.dataframe(
            anomalies[["Date","Description","Category","AbsAmount"]]
            .sort_values("AbsAmount", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.success("No unusual transactions detected at this sensitivity level.")

# ══════════════════════════════════════════════════════ BUDGET
elif page == "💡 Budget Advisor":
    st.title("💡 Budget Recommendations")
    st.markdown("Transparent rule-based engine — no AI/LLM, fully explainable.")

    suggestions, by_cat = generate_budget_suggestions(df_f)

    if not by_cat.empty:
        fig = px.bar(
            by_cat, x="Category", y="Percent", text="Percent",
            title="Spending Share by Category (%)",
            labels={"Percent":"Share (%)"},
            color="Percent", color_continuous_scale="RdYlGn_r",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("💬 Suggestions")
    for s in suggestions:
        st.markdown(f"- {s}")

# ══════════════════════════════════════════════════════ DATA & EXPORT
elif page == "📁 Data & Export":
    st.title("📁 Cleaned Data & Export")

    stats = get_summary_stats(df_f)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (filtered)", len(df_f))
    c2.metric("Total rows in session", len(df))
    c3.metric("Categories", df_f["Category"].nunique())

    st.dataframe(df_f, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        csv_bytes = df_f.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download as CSV",
            data=csv_bytes,
            file_name="processed_expenses.csv",
            mime="text/csv",
        )
    with col2:
        import io as _io
        excel_buf = _io.BytesIO()
        df_f.to_excel(excel_buf, index=False, engine="openpyxl")
        st.download_button(
            "⬇️ Download as Excel",
            data=excel_buf.getvalue(),
            file_name="processed_expenses.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
