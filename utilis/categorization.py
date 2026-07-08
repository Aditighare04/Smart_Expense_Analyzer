"""
categorization.py
Rule-based keyword categorization used both as instant labelling and as the
labelling function for ML training data.
"""
import pandas as pd

KEYWORD_RULES = {
    "Food":          ["swiggy", "zomato", "domino", "mcdonald", "restaurant",
                      "starbucks", "grocery", "cafe", "food", "kfc", "burger",
                      "pizza", "dine", "blinkit", "instamart", "bigbasket"],
    "Travel":        ["uber", "ola", "irctc", "indigo", "airlines", "petrol",
                      "fuel", "metro", "flight", "train", "taxi", "fare",
                      "diesel", "rapido", "redbus", "makemytrip"],
    "Shopping":      ["amazon", "flipkart", "myntra", "mall", "ajio",
                      "shopping", "store", "purchase", "mart", "meesho",
                      "nykaa", "reliance digital"],
    "Bills":         ["electricity", "water department", "airtel", "jio",
                      "broadband", "bill", "recharge", "postpaid", "dth",
                      "internet", "bsnl", "vi ", "vodafone"],
    "Entertainment": ["netflix", "spotify", "bookmyshow", "hotstar", "gaming",
                      "movie", "cinema", "prime video", "zee5", "youtube"],
    "Health":        ["pharmacy", "practo", "gym", "hospital", "clinic",
                      "medical", "doctor", "medicine", "apollo", "1mg",
                      "netmeds", "cult.fit"],
    "Education":     ["udemy", "coursera", "bookstore", "tuition", "course",
                      "fees", "school", "college", "byju", "unacademy"],
    "Rent":          ["house rent", "rent"],
    "Income":        ["salary", "freelance", "interest credit", "refund",
                      "credited", "bonus", "cashback"],
    "Other":         ["atm", "miscellaneous", "gift", "donation",
                      "withdrawal", "transfer"],
}


def rule_based_category(description: str) -> str:
    text = str(description).lower()
    for category, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text:
                return category
    return "Other"


def apply_rule_based_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    predicted = df["Description"].apply(rule_based_category)
    if "Category" in df.columns:
        df["Category"] = df["Category"].fillna(predicted)
    else:
        df["Category"] = predicted
    df.loc[(~df["IsExpense"]) & (df["Category"] == "Other"), "Category"] = "Income"
    return df
