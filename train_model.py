import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef


# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("breast-cancer.csv")

# Drop ID column
df = df.drop(columns=["id"])

# Convert diagnosis to numeric
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Define features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# -----------------------------
# 2. Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 4. Evaluation Function
# -----------------------------
def evaluate(model, X_tr, X_te):
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = 0

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": auc,
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

# -----------------------------
# 5. Train Models
# -----------------------------
results = {}

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    if name in ["Logistic Regression", "KNN"]:
        results[name] = evaluate(model, X_train_scaled, X_test_scaled)
    else:
        results[name] = evaluate(model, X_train, X_test)

# -----------------------------
# 6. Show Results
# -----------------------------
results_df = pd.DataFrame(results).T
print(results_df)

# -----------------------------
# 7. Save Models
# -----------------------------
os.makedirs("model", exist_ok=True)

for name, model in models.items():
    joblib.dump(model, f"model/{name.replace(' ', '_')}.pkl")

joblib.dump(scaler, "model/scaler.pkl")

print("\nModels saved successfully.")
