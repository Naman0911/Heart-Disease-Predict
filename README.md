# Heart Disease Prediction

**Project:** Heart Disease Prediction using Machine Learning (Logistic Regression)

**Repository:** Naman0911/Heart-Disease-Predict

---

## Project Overview
This repository contains a reproducible workflow to explore, preprocess, and model the UCI Heart Disease dataset for binary classification (presence vs absence of heart disease). The goal is to build an interpretable logistic regression baseline, perform EDA and feature engineering, and tune the model using cross-validation and hyperparameter search.

The work is appropriate for learning end-to-end machine learning best practices: EDA, cleaning, imputation, encoding, scaling, training, evaluation and interpretation.

---

## Dataset
- **Source:** UCI Machine Learning Repository / Kaggle copies of the UCI Heart Disease dataset.
- **Target:** `num` (0 = No Disease, 1 = Disease). Some original UCI variants use 0–4; this project converts 1–4 → 1 to make it binary.
- **Shape used in this project:** ~920 rows and 11 features.

**Important features:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.

---

## Repository Structure
```
Heart-Disease-Predict/
├─ data/
├─ notebooks/
│   ├─ 01_EDA.ipynb
│   ├─ 02_Preprocessing.ipynb
│   └─ 03_Modeling_and_Tuning.ipynb
├─ src/
│   ├─ preprocess.py
│   └─ modeling.py
├─ models/
├─ requirements.txt
└─ README.md
```

---

## EDA Summary
- Inspected datatypes, missingness and distributions.
- Numerical features treated with appropriate imputation:
  - Median: chol, trestbps, oldpeak
  - Mean: thalach
- Categorical features: mode imputation; slope dropped due to excessive missingness.
- Heart disease increases sharply after ~45–50 years.

---

## Preprocessing Decisions
- Missing values handled per column.
- Encoding: label-mapping or one-hot.
- Scaling: StandardScaler after split.
- Split: test_size=0.25, stratify=y, random_state=42.
- Feature selection based on correlation + model coefficients.

---

## Modeling & Results
- Model: Logistic Regression
- Validation: Stratified K-Fold + GridSearchCV / RandomizedSearchCV
- Baseline accuracy: ~81.25%
- Tuned accuracy: ~82%
- Confusion matrix example:
```
[[ 79  27]
 [ 20 104]]
```

---

## Reproduce Steps
1. Clone repository.
2. Download dataset.
3. Install dependencies.
4. Run notebooks in order.

---

## Requirements
See `requirements.txt`.

---

## Author
**Naman Upadhyay**
