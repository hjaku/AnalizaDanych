import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Wczytanie danych
df = pd.read_excel("Data.xlsx")

# Usunięcie kolumny ID
df = df.drop(columns=["Engagement_ID"])

# Zmienne X i y
X = df.drop(columns=["Policy_violation_occured (0/1)"])
y = df["Policy_violation_occured (0/1)"]

# 2. Kolumny numeryczne i kategoryczne

num_cols = ["Hours_Charged", "Number_of_personnel_involved"]
cat_cols = ["EP_Rank", "Additional_Approval_Required"]

# One-Hot Encoding
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# 3. Modele bazowe (bez ulepszeń!)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

# 4. Podział danych

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 5. Trenowanie i ocena modeli

results = []

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Metryki
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision (class 1)": prec,
        "Recall (class 1)": rec,
        "F1-score (class 1)": f1
    })

    print(f"\n Macierz konfuzji: {name}")
    print(confusion_matrix(y_test, y_pred))

# 6. Tabela wyników

results_df = pd.DataFrame(results)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print("\n Porównanie modeli bazowych \n")
print(results_df.to_string(index=False))
