## Heart Disease Prediction

- Project: Heart Disease Prediction using Machine Learning (Logistic Regression)

Repository: Naman0911/Heart-Disease-Predict

Project Overview

This repository contains a reproducible workflow to explore, preprocess, and model the UCI Heart Disease dataset for binary classification (presence vs absence of heart disease). 
The goal is to build an interpretable logistic regression baseline, perform EDA and feature engineering, and tune the model using cross-validation and hyperparameter search.
The work is appropriate for learning end-to-end machine learning best practices: EDA, cleaning, imputation, encoding, scaling, training, evaluation and interpretation.

Dataset
Source: UCI Machine Learning Repository / Kaggle copies of the UCI Heart Disease dataset.
Target: num (0 = No Disease, 1 = Disease). Some original UCI variants use 0–4; this project converts 1–4 → 1 to make it binary.
Shape used in this project: ~920 rows and 11 features (after any merges/cleaning used here).
Important features (examples): age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.

Repository Structure
Heart-Disease-Predict/
├─ data/                # raw and processed CSVs (not committed for privacy/size)
├─ notebooks/           # EDA and modeling notebooks (Jupyter)
│   ├─ 01_EDA.ipynb
│   ├─ 02_Preprocessing.ipynb
│   └─ 03_Modeling_and_Tuning.ipynb
├─ src/                 # optional: python modules for preprocessing / modeling
│   ├─ preprocess.py
│   └─ modeling.py
├─ models/               # saved model artifacts (pickle) -- .gitignored
├─ requirements.txt
└─ README.md             # this file
EDA Summary (high level)

Checked datatypes, missingness, and basic distributions for each column.
Numerical columns such as chol, trestbps, thalach, oldpeak were inspected for skewness and outliers.
chol was found to be approximately normal but contained outliers; median imputation was used due to outliers.
trestbps had missing values and skew; median imputation was chosen.
thalach (max heart rate) was approximately normal with only a few outliers → mean imputation used.
oldpeak is heavily right-skewed (many zeros) → median imputation used.
Categorical columns (fbs, restecg, cp, exang, slope, thal, ca) were inspected for distribution and missingness.
fbs and other binary/categorical columns used mode imputation when needed.
slope had ~34% missing values and was dropped in the final pipeline (alternative: mark as "unknown").
Grouped analysis showed heart disease rates increase notably after ~45–50 years, with the highest risk in 50–70 age groups.
Preprocessing Decisions

Missing value handling: chosen per-column (median for skewed numerics, mean for normally distributed numerics with few outliers, mode for categoricals). For features with excessive missingness (e.g., slope ~34%) the column was dropped.
Encoding: categorical variables were label-mapped or one-hot encoded depending on the model/pipeline.
Scaling: StandardScaler applied to numeric features after train/test split (no leakage). Pipelines were used to ensure transformations are fit on training data only.
Train/test split: test_size=0.25, stratify=y, random_state=42 (consistent reproducibility).


Feature selection: low-correlation and low-importance features (e.g., fbs, restecg, trestbps in this dataset) were optionally removed after model-based importance checks.

Modeling & Results
Model: Logistic Regression (interpretable baseline)
Scaling: StandardScaler (fit on training set only)
Validation: Stratified K-Fold cross-validation and GridSearchCV / RandomizedSearchCV for hyperparameter tuning

Representative results from this project:

Baseline accuracy: ~0.81.25 (81.25%) on the held-out test set.

Confusion matrix (example):

[[ 79  27]
 [ 20 104]]

After hyperparameter tuning (GridSearchCV / RandomizedSearchCV) the model reached ~82% accuracy.
These results are consistent with expected performance for a well-tuned logistic regression on the UCI heart disease dataset.

How to Reproduce (high level)
Clone the repo.
Prepare the dataset (download UCI heart disease CSV, convert the target to binary if required).
Install dependencies from requirements.txt.
Run the notebooks in order: EDA → Preprocessing → Modeling.
Use the provided pipeline in src/ to train and evaluate a final model; save the best model into models/.
Tips & Next Steps
Try alternative models (Random Forest, XGBoost, CatBoost) to push accuracy toward 85–90%.
Add feature interactions and polynomial features where medically appropriate.
Consider calibration for predicted probabilities if this will be used in clinical settings.
Add a missingness indicator feature for columns that had imputation applied.

Requirements
Minimal requirements (example):
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
(Exact packages are listed in requirements.txt.)

License & Contact
This project is provided for educational purposes. Check license in the repository (e.g., MIT) before reuse.
Author / Contact: Naman Upadhyay
