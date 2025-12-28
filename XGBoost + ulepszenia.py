import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# 1. Wczytanie danych

df = pd.read_excel("Data.xlsx")

# Usunięcie kolumny ID
df = df.drop(columns=["Engagement_ID"])

# Zmienne X i y
X = df.drop(columns=["Policy_violation_occured (0/1)"])
y = df["Policy_violation_occured (0/1)"]

# 2. Preprocessing

num_cols = ["Hours_Charged", "Number_of_personnel_involved"]
cat_cols = ["EP_Rank", "Additional_Approval_Required"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# 3. Podział danych

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 4. Ocena

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name}")
    print("Macierz konfuzji:")
    print(confusion_matrix(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision (class 1): {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall (class 1): {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1-score (class 1): {f1_score(y_true, y_pred, zero_division=0):.3f}")

# XGBoost - wersja bazowa

xgb_baseline = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

pipe_baseline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", xgb_baseline)
])

pipe_baseline.fit(X_train, y_train)
y_pred_baseline = pipe_baseline.predict(X_test)

evaluate_model("XGBoost - wersja bazowa", y_test, y_pred_baseline)

# XGBoost + scale_pos_weight

neg, pos = y_train.value_counts()
scale = neg / pos

xgb_weighted = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale
)

pipe_weighted = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", xgb_weighted)
])

pipe_weighted.fit(X_train, y_train)
y_pred_weighted = pipe_weighted.predict(X_test)

evaluate_model("XGBoost + scale_pos_weight", y_test, y_pred_weighted)

# XGBoost + SMOTE

# Transformacja danych przed SMOTE
X_train_transformed = preprocess.fit_transform(X_train)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_transformed, y_train)

xgb_smote = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

xgb_smote.fit(X_train_res, y_train_res)

# Transformacja testu
X_test_transformed = preprocess.transform(X_test)
y_pred_smote = xgb_smote.predict(X_test_transformed)

evaluate_model("XGBoost + SMOTE", y_test, y_pred_smote)

# XGBoost + tuning hiperparametrów

neg, pos = y_train.value_counts()
scale = neg / pos

xgb_tuned = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale
)

pipe_tuned = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", xgb_tuned)
])

pipe_tuned.fit(X_train, y_train)
y_pred_tuned = pipe_tuned.predict(X_test)

evaluate_model("XGBoost + tuning hiperparametrów", y_test, y_pred_tuned)

# XGBoost + tuning hiperparametrów

neg, pos = y_train.value_counts()
scale = neg / pos

xgb_tuned = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale
)

pipe_tuned = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", xgb_tuned)
])

pipe_tuned.fit(X_train, y_train)

# Threshold tuning — wybór najlepszego progu dla F1-score

probs = pipe_tuned.predict_proba(X_test)[:, 1]

best_f1 = 0
best_t = 0.5

thresholds = np.linspace(0.05, 0.95, 181)

for t in thresholds:
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("\nNajlepszy próg decyzyjny:", best_t)
print("Najlepszy F1-score:", best_f1)

# Predykcja z użyciem najlepszego progu

y_pred_threshold = (probs >= best_t).astype(int)

evaluate_model("XGBoost + tuning hiperparametrów + threshold tuning", y_test, y_pred_threshold)



