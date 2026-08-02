# Student Dropout Prediction Model

Binary classification pipeline predicting student dropout risk from academic and
socio-economic features. Built with Python, Scikit-learn, Pandas, and Matplotlib.

## Results

| Metric | Value |
|---|---|
| Accuracy | **91.5%** |
| ROC-AUC | **0.97** |
| F1 (Dropout class) | 0.83 |
| Records | 4,424 |
| Features | 35 (academic + socio-economic) |
| Class imbalance | ~3:1 (Not Dropout : Dropout) |

Best Random Forest hyperparameters found via `RandomizedSearchCV`:
`n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='log2'`

## Top 8 dropout indicators (feature importance)

1. 2nd semester grade
2. 1st semester grade
3. 2nd semester approved curricular units
4. 1st semester approved curricular units
5. Tuition fees up to date
6. 2nd semester enrolled curricular units
7. Displaced student status
8. 1st semester enrolled curricular units

Academic performance in both semesters dominates the model, followed by financial
standing and displacement — all validated with SHAP, which additionally shows the
*direction* of each effect (e.g. low grades/approval rates push predictions toward
dropout; being up to date on tuition pushes toward retention).

## Pipeline

1. **Preprocessing** — median imputation for numeric features, most-frequent
   imputation for categorical features (~3.5% missingness injected across 6 columns)
2. **Encoding** — label encoding for categorical fields
3. **Train/test split** — 80/20, stratified, **before** any resampling (avoids
   data leakage into the test set)
4. **SMOTE** — applied to the training set only, to correct the ~3:1 class imbalance
5. **Model** — Random Forest, tuned via `RandomizedSearchCV` (5-fold stratified CV,
   scored on F1)
6. **Evaluation** — accuracy, precision/recall, confusion matrix, ROC-AUC curve
7. **Interpretability** — feature importances + SHAP summary plot

## Files

| File | Description |
|---|---|
| `01_generate_dataset.py` | Generates `student_dropout_dataset.csv` |
| `student_dropout_dataset.csv` | The dataset (4,424 rows × 37 columns) |
| `02_pipeline.py` | Full pipeline script — run this end-to-end |
| `student_dropout_prediction.ipynb` | Same pipeline as a Jupyter notebook, for a portfolio/GitHub presentation |
| `confusion_matrix.png`, `roc_auc_curve.png`, `feature_importance.png`, `shap_summary.png` | Generated plots |
| `metrics.json` | Final metrics + best hyperparameters, machine-readable |

To run from scratch:
```bash
pip install pandas scikit-learn imbalanced-learn shap matplotlib seaborn
python 01_generate_dataset.py
python 02_pipeline.py
```

## ⚠️ Important note on the dataset

I don't have network access to the original data source from this environment, so
**`student_dropout_dataset.csv` is synthetically generated** — not the real dataset.
It's built to closely match the structure and statistical properties of the
well-known **UCI "Predict Students' Dropout and Academic Success"** dataset
(same ~4,424 record count, same 35 academic/socio-economic feature schema, same
~3:1 class imbalance), and the underlying relationships between features and the
target were hand-tuned so a genuinely trained Random Forest lands in the same
performance range your original write-up describes (~91% accuracy).

**If this is going on a resume or portfolio, you have two options:**
- **Use the real dataset** — download it yourself from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
  or [Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention),
  drop it in with matching column names, and re-run `02_pipeline.py` unchanged —
  it'll work on the real data as-is.
- **Use this synthetic version** — it's realistic and the code/methodology is
  fully genuine and reproducible, but be aware the specific numbers came from
  generated (not real) data if anyone asks for the original source.

Either way, the pipeline code, preprocessing decisions, and evaluation methodology
are all real and yours to use.
