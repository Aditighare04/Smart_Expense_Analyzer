"""
budget.py
Rule-based budgeting recommendations — no AI, fully explainable.
"""
import pandas as pd

THRESHOLDS = {
    "Food": 0.30, "Shopping": 0.20, "Entertainment": 0.10,
    "Travel": 0.15, "Bills": 0.15, "Health": 0.10, "Other": 0.10,
}

ADVICE = {
    "Food":          "Consider reducing food delivery / dining-out expenses.",
    "Shopping":      "Try setting a monthly shopping cap or wish-list rule.",
    "Entertainment": "Review recurring subscriptions — cancel unused ones.",
    "Travel":        "Consider carpooling, public transport, or advance booking.",
    "Bills":         "Check for better recharge / data / utility plans.",
    "Health":        "Review recurring medical expenses and health plan coverage.",
    "Other":         "Track miscellaneous expenses more closely to find patterns.",
}


def generate_budget_suggestions(df: pd.DataFrame):
    expense_df = df[df["IsExpense"]]
    total_expense = expense_df["AbsAmount"].sum()
    if total_expense == 0:
        return [], pd.DataFrame()

    by_cat = (
        expense_df.groupby("Category")["AbsAmount"].sum().reset_index()
        .rename(columns={"AbsAmount": "Total"})
    )
    by_cat["Percent"] = (by_cat["Total"] / total_expense * 100).round(2)
    by_cat = by_cat.sort_values("Percent", ascending=False)

    suggestions = []
    for _, row in by_cat.iterrows():
        cat, pct = row["Category"], row["Percent"] / 100
        thresh = THRESHOLDS.get(cat)
        if thresh and pct > thresh:
            tip = ADVICE.get(cat, f"Review {cat} spending.")
            suggestions.append(
                f"⚠️ **{cat}** — {row['Percent']}% of total expense. {tip}"
            )

    if not suggestions:
        suggestions.append("✅ Great job! Your spending is balanced across all categories.")

    income = df[~df["IsExpense"]]["AbsAmount"].sum()
    if income > 0:
        sr = (income - total_expense) / income * 100
        if sr < 10:
            suggestions.append(
                f"💡 Your savings rate is only **{sr:.1f}%**. "
                "Aim for at least 20% to build a healthy financial cushion."
            )
        elif sr >= 20:
            suggestions.append(
                f"🎉 Excellent savings rate of **{sr:.1f}%** — keep it up!"
            )

    return suggestions, by_cat
