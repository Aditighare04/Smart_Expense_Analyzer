"""
generate_sample_data.py
Generates realistic synthetic transactions for demo/testing.
Run: python generate_sample_data.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)

CATEGORIES = {
    "Food":          ["Swiggy", "Zomato", "Dominos", "McDonalds", "Starbucks", "BigBasket"],
    "Travel":        ["Uber", "Ola", "IRCTC", "Indigo Airlines", "Petrol Pump", "Rapido"],
    "Shopping":      ["Amazon", "Flipkart", "Myntra", "Ajio", "Reliance Digital"],
    "Bills":         ["Airtel Postpaid", "Jio Recharge", "Electricity Board", "Broadband Bill"],
    "Entertainment": ["Netflix", "Spotify", "BookMyShow", "Hotstar"],
    "Health":        ["Apollo Pharmacy", "Practo", "Gym Membership", "1mg"],
    "Education":     ["Udemy", "Coursera", "Bookstore", "Tuition Fees"],
    "Rent":          ["House Rent"],
    "Income":        ["Salary Credit", "Freelance Payment", "Interest Credit"],
    "Other":         ["ATM Withdrawal", "Miscellaneous", "Gift"],
}

RANGES = {
    "Food": (100,1200), "Travel": (50,3000), "Shopping": (300,6000),
    "Bills": (200,3000), "Entertainment": (99,800), "Health": (150,5000),
    "Education": (300,4000), "Rent": (9000,13000), "Income": (20000,50000),
    "Other": (50,2000),
}

WEIGHTS = {
    "Food":0.22,"Travel":0.14,"Shopping":0.16,"Bills":0.10,
    "Entertainment":0.08,"Health":0.06,"Education":0.05,
    "Rent":0.03,"Income":0.07,"Other":0.09,
}


def random_date(start, end):
    return start + pd.Timedelta(days=np.random.randint(0, (end - start).days))


def generate(n_rows=900, months=10, start="2024-09-01"):
    s, e = pd.Timestamp(start), pd.Timestamp(start) + pd.DateOffset(months=months)
    cats  = list(CATEGORIES.keys())
    p     = np.array([WEIGHTS[c] for c in cats]); p /= p.sum()
    rows  = []

    for _ in range(n_rows):
        cat     = np.random.choice(cats, p=p)
        merch   = np.random.choice(CATEGORIES[cat])
        lo, hi  = RANGES[cat]
        amount  = round(np.random.uniform(lo, hi), 2)
        date    = random_date(s, e)
        income  = cat == "Income"
        rows.append({
            "Date":        date,
            "Description": merch,
            "Amount":      amount if income else -amount,
            "Type":        "Credit" if income else "Debit",
            "Category":    cat,
        })

    # Inject anomalies
    for _ in range(8):
        cat   = np.random.choice(["Shopping","Health","Travel"])
        rows.append({
            "Date":        random_date(s, e),
            "Description": np.random.choice(CATEGORIES[cat]) + " (Large)",
            "Amount":      -round(np.random.uniform(18000, 45000), 2),
            "Type":        "Debit",
            "Category":    cat,
        })

    # Monthly salary + rent
    for m in range(months):
        rows += [
            {"Date": s+pd.DateOffset(months=m,days=1), "Description":"Salary Credit",
             "Amount": round(np.random.uniform(35000,42000),2), "Type":"Credit", "Category":"Income"},
            {"Date": s+pd.DateOffset(months=m,days=2), "Description":"House Rent",
             "Amount": -round(np.random.uniform(9000,12000),2), "Type":"Debit", "Category":"Rent"},
        ]

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("sample_data", exist_ok=True)
    df = generate()
    df.to_csv("sample_data/sample_transactions.csv", index=False)
    print(f"✅ Generated {len(df)} rows → sample_data/sample_transactions.csv")
