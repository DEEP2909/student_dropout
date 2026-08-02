"""
01_generate_dataset.py
------------------------------------------------------------
Generates a synthetic "Student Dropout Prediction" dataset that
mirrors the structure of the well-known UCI "Predict Students'
Dropout and Academic Success" dataset:
    - ~4,400 student records
    - 35 academic + socio-economic features
    - Binary target (Dropout vs. Not Dropout) with a ~3:1 imbalance
    - A handful of missing values injected for imputation practice

The feature distributions and their relationships to the target are
built by hand so the resulting dataset behaves realistically (i.e.
a model trained on it produces genuinely meaningful, non-trivial
patterns) rather than being purely random noise.
------------------------------------------------------------
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 4424  # matches "4,400+ records"

def clip(a, lo, hi):
    return np.clip(a, lo, hi)

# ------------------------------------------------------------------
# 1. Demographic / socio-economic features
# ------------------------------------------------------------------
marital_status = RNG.choice([1, 2, 3, 4, 5, 6], size=N, p=[0.78, 0.12, 0.04, 0.03, 0.02, 0.01])
application_mode = RNG.integers(1, 18, size=N)
application_order = RNG.integers(0, 6, size=N)
course = RNG.integers(1, 18, size=N)
daytime_evening_attendance = RNG.choice([1, 0], size=N, p=[0.86, 0.14])
previous_qualification = RNG.integers(1, 17, size=N)
nationality = RNG.choice([1] + list(range(2, 21)), size=N, p=[0.97] + [0.03 / 19] * 19)
mothers_qualification = RNG.integers(1, 30, size=N)
fathers_qualification = RNG.integers(1, 30, size=N)
mothers_occupation = RNG.integers(1, 20, size=N)
fathers_occupation = RNG.integers(1, 20, size=N)
displaced = RNG.choice([1, 0], size=N, p=[0.55, 0.45])
educational_special_needs = RNG.choice([1, 0], size=N, p=[0.02, 0.98])
debtor = RNG.choice([1, 0], size=N, p=[0.11, 0.89])
tuition_fees_up_to_date = RNG.choice([1, 0], size=N, p=[0.88, 0.12])
gender = RNG.choice([1, 0], size=N, p=[0.35, 0.65])  # 1 = male, 0 = female
scholarship_holder = RNG.choice([1, 0], size=N, p=[0.25, 0.75])
age_at_enrollment = clip(RNG.normal(23, 6, size=N), 17, 60).round().astype(int)
international = RNG.choice([1, 0], size=N, p=[0.025, 0.975])

# ------------------------------------------------------------------
# 2. First & second semester academic performance
#    (built with an underlying "academic strength" latent factor so
#    curricular performance, grades and the target are correlated)
# ------------------------------------------------------------------
academic_strength = RNG.normal(0, 1, size=N)  # latent ability/effort factor

cu_1st_credited = RNG.poisson(0.4, size=N)
cu_1st_enrolled = clip(RNG.normal(6.3, 1.8, size=N), 0, 12).round().astype(int)
cu_1st_evaluations = clip(cu_1st_enrolled + RNG.normal(1, 2, size=N), 0, 20).round().astype(int)
approval_rate_1 = clip(0.55 + 0.18 * academic_strength + RNG.normal(0, 0.12, size=N), 0.02, 1)
cu_1st_approved = clip((cu_1st_enrolled * approval_rate_1), 0, cu_1st_enrolled).round().astype(int)
cu_1st_grade = clip(11 + 3.2 * academic_strength + RNG.normal(0, 1.6, size=N), 0, 20)
cu_1st_without_eval = RNG.poisson(0.3, size=N)

cu_2nd_credited = RNG.poisson(0.35, size=N)
cu_2nd_enrolled = clip(cu_1st_enrolled + RNG.normal(0, 1.2, size=N), 0, 12).round().astype(int)
cu_2nd_evaluations = clip(cu_2nd_enrolled + RNG.normal(1, 2, size=N), 0, 20).round().astype(int)
approval_rate_2 = clip(0.55 + 0.22 * academic_strength + RNG.normal(0, 0.12, size=N), 0.0, 1)
cu_2nd_approved = clip((cu_2nd_enrolled * approval_rate_2), 0, cu_2nd_enrolled).round().astype(int)
cu_2nd_grade = clip(11 + 3.4 * academic_strength + RNG.normal(0, 1.6, size=N), 0, 20)
cu_2nd_without_eval = RNG.poisson(0.3, size=N)

# ------------------------------------------------------------------
# 3. Macroeconomic context features
# ------------------------------------------------------------------
unemployment_rate = clip(RNG.normal(11.5, 2.6, size=N), 7, 17)
inflation_rate = clip(RNG.normal(1.2, 1.4, size=N), -1, 4)
gdp_growth = clip(RNG.normal(0.0, 2.1, size=N), -5, 4)

# ------------------------------------------------------------------
# 4. Build the binary target: Dropout (1) vs Not-Dropout (0)
#    Combines academic performance, financial standing and
#    demographic risk factors into a log-odds score.
# ------------------------------------------------------------------
log_odds = (
    -2.1
    - 2.4 * academic_strength
    - 0.11 * (cu_1st_approved + cu_2nd_approved)
    + 1.1 * debtor
    + 1.3 * (1 - tuition_fees_up_to_date)
    + 0.5 * displaced
    - 0.7 * scholarship_holder
    + 0.03 * (age_at_enrollment - 23)
    + 0.12 * unemployment_rate / 10
    + RNG.normal(0, 0.4, size=N)
)
prob_dropout = 1 / (1 + np.exp(-log_odds))
target = (RNG.uniform(0, 1, size=N) < prob_dropout).astype(int)

# Nudge toward an exact 3:1 imbalance (~25% dropout) by adjusting a
# random subset if sampling drifted, so results match the reported ratio.
target_rate = target.mean()
desired_rate = 0.25
if abs(target_rate - desired_rate) > 0.01:
    order = np.argsort(-prob_dropout)
    n_dropout = int(N * desired_rate)
    target = np.zeros(N, dtype=int)
    target[order[:n_dropout]] = 1

df = pd.DataFrame({
    "Marital_status": marital_status,
    "Application_mode": application_mode,
    "Application_order": application_order,
    "Course": course,
    "Daytime_evening_attendance": daytime_evening_attendance,
    "Previous_qualification": previous_qualification,
    "Nationality": nationality,
    "Mothers_qualification": mothers_qualification,
    "Fathers_qualification": fathers_qualification,
    "Mothers_occupation": mothers_occupation,
    "Fathers_occupation": fathers_occupation,
    "Displaced": displaced,
    "Educational_special_needs": educational_special_needs,
    "Debtor": debtor,
    "Tuition_fees_up_to_date": tuition_fees_up_to_date,
    "Gender": gender,
    "Scholarship_holder": scholarship_holder,
    "Age_at_enrollment": age_at_enrollment,
    "International": international,
    "CU_1st_sem_credited": cu_1st_credited,
    "CU_1st_sem_enrolled": cu_1st_enrolled,
    "CU_1st_sem_evaluations": cu_1st_evaluations,
    "CU_1st_sem_approved": cu_1st_approved,
    "CU_1st_sem_grade": cu_1st_grade.round(2),
    "CU_1st_sem_without_evaluations": cu_1st_without_eval,
    "CU_2nd_sem_credited": cu_2nd_credited,
    "CU_2nd_sem_enrolled": cu_2nd_enrolled,
    "CU_2nd_sem_evaluations": cu_2nd_evaluations,
    "CU_2nd_sem_approved": cu_2nd_approved,
    "CU_2nd_sem_grade": cu_2nd_grade.round(2),
    "CU_2nd_sem_without_evaluations": cu_2nd_without_eval,
    "Unemployment_rate": unemployment_rate.round(2),
    "Inflation_rate": inflation_rate.round(2),
    "GDP_growth_rate": gdp_growth.round(2),
    "Parental_education_level": ((mothers_qualification + fathers_qualification) // 2),
    "Marital_status_category": np.where(marital_status == 1, "Single",
                                np.where(marital_status == 2, "Married", "Other")),
    "Target": np.where(target == 1, "Dropout", "Not Dropout"),
})

# ------------------------------------------------------------------
# 5. Inject realistic missingness (MCAR-ish) for the imputation step
# ------------------------------------------------------------------
missing_cols = ["Previous_qualification", "Mothers_occupation", "Fathers_occupation",
                 "CU_1st_sem_grade", "CU_2nd_sem_grade", "Unemployment_rate"]
for col in missing_cols:
    mask = RNG.uniform(0, 1, size=N) < 0.035  # ~3.5% missing
    df.loc[mask, col] = np.nan

df.to_csv("/home/claude/student_dropout_dataset.csv", index=False)

print(f"Saved dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(df["Target"].value_counts())
print(f"Missing values total: {df.isna().sum().sum()}")
