"""
02_pipeline.py
------------------------------------------------------------
End-to-end binary classification pipeline: Student Dropout Prediction

Steps:
  1. Load data
  2. Missing value imputation
  3. Feature encoding
  4. Train/test split (stratified, held out BEFORE resampling to avoid leakage)
  5. SMOTE class balancing (fit on train only)
  6. Random Forest + hyperparameter tuning (GridSearchCV)
  7. Evaluation: accuracy, classification report, confusion matrix, ROC-AUC
  8. Feature importance (top 8) + SHAP interpretability plots
------------------------------------------------------------
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from imblearn.over_sampling import SMOTE

sns.set_style("whitegrid")
OUT = "/home/claude"

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv(f"{OUT}/student_dropout_dataset.csv")
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

TARGET = "Target"
y_raw = df[TARGET]
X = df.drop(columns=[TARGET])

# ------------------------------------------------------------------
# 2. Missing value imputation
#    Numeric -> median, Categorical -> most frequent
# ------------------------------------------------------------------
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numeric features: {len(numeric_cols)} | Categorical features: {len(categorical_cols)}")
print(f"Missing values before imputation: {X.isna().sum().sum()}")

num_imputer = SimpleImputer(strategy="median")
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

if categorical_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

print(f"Missing values after imputation: {X.isna().sum().sum()}")

# ------------------------------------------------------------------
# 3. Feature encoding
# ------------------------------------------------------------------
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)  # Dropout=0, Not Dropout=1 (alphabetical) -- confirm below
print("Target classes:", dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))))

# Make "Dropout" the positive class (1) for interpretability
dropout_idx = list(target_encoder.classes_).index("Dropout")
if dropout_idx != 1:
    y = 1 - y  # flip so Dropout == 1
print("Using Dropout = 1 (positive class), Not Dropout = 0")

feature_names = X.columns.tolist()

# ------------------------------------------------------------------
# 4. Train/test split BEFORE resampling (avoid leakage)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train class balance before SMOTE: {np.bincount(y_train)} "
      f"(ratio {np.bincount(y_train)[0]/np.bincount(y_train)[1]:.2f}:1)")

# ------------------------------------------------------------------
# 5. SMOTE class balancing (train set only)
# ------------------------------------------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Train class balance after SMOTE: {np.bincount(y_train_res)}")

# ------------------------------------------------------------------
# 6. Random Forest + hyperparameter tuning
# ------------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [10, 16, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}

base_rf = RandomForestClassifier(random_state=42, class_weight=None, n_jobs=1)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

from sklearn.model_selection import RandomizedSearchCV
import time
t0 = time.time()
search = RandomizedSearchCV(
    base_rf, param_grid, n_iter=6, cv=cv, scoring="f1",
    n_jobs=1, random_state=42, verbose=1
)
search.fit(X_train_res, y_train_res)
best_rf = search.best_estimator_
print(f"Search took {time.time()-t0:.1f}s")
print("Best hyperparameters:", search.best_params_)

# ------------------------------------------------------------------
# 7. Evaluation
# ------------------------------------------------------------------
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred, target_names=["Not Dropout", "Dropout"])

print(f"\nAccuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC:  {auc:.4f}")
print(report)

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Dropout", "Dropout"],
            yticklabels=["Not Dropout", "Dropout"])
plt.title(f"Confusion Matrix (Accuracy = {acc:.1%})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
plt.close()

# ROC-AUC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve — Dropout Prediction")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/roc_auc_curve.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 8. Feature importance (top 8) + SHAP
# ------------------------------------------------------------------
importances = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values(ascending=False)
top8 = importances.head(8)
print("\nTop 8 dropout indicators:")
print(top8)

plt.figure(figsize=(8, 5))
sns.barplot(x=top8.values, y=top8.index, color="#2563eb")
plt.title("Top 8 Dropout Risk Indicators (Random Forest Feature Importance)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
plt.close()

# SHAP summary plot (sampled for speed)
explainer = shap.TreeExplainer(best_rf)
sample_idx = np.random.RandomState(42).choice(X_test.index, size=min(120, len(X_test)), replace=False)
X_sample = X_test.loc[sample_idx]
shap_values = explainer.shap_values(X_sample)

# shap_values can be a list (per class) or array depending on version; take positive class
sv = shap_values[1] if isinstance(shap_values, list) else shap_values
if sv.ndim == 3:  # (n_samples, n_features, n_classes)
    sv = sv[:, :, 1]

plt.figure()
shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, max_display=12)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# 9. Save metrics summary
# ------------------------------------------------------------------
metrics = {
    "accuracy": round(acc, 4),
    "f1_score": round(f1, 4),
    "roc_auc": round(auc, 4),
    "best_params": search.best_params_,
    "top_8_features": top8.round(4).to_dict(),
    "train_size": len(X_train_res),
    "test_size": len(X_test),
    "original_class_ratio": f"{np.bincount(y)[0]/np.bincount(y)[1]:.2f}:1",
}
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nAll artifacts saved to", OUT)
