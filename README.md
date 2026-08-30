# Credit Card Behaviour Score — Predicting Default Probability

Developing a risk management framework for credit card defaults, built for **IDFC FIRST Bank** as part of the **Convolve** case-study challenge.

## Overview

Accurate credit risk assessment is central to managing defaults and protecting profitability in the financial services sector. This project builds a **Behaviour Score** — a predictive model that estimates the probability of a customer defaulting on their credit card — to support portfolio risk management and credit decisioning at IDFC FIRST Bank.

The pipeline covers the full lifecycle of a tabular ML problem on a severely imbalanced dataset: exploratory analysis, missing-value imputation, feature engineering and selection, class-imbalance handling, model training, ensembling, and threshold optimisation.

## Dataset

| | Rows | Description |
|---|---|---|
| Development data | 96,806 | Labelled records used for training/validation, target column `bad_flag` (1 = default, 0 = no default) |
| Validation data | 41,792 | Unlabelled records for which default probabilities are predicted |

Features fall into four groups:
- **On-us attributes** (`onus_attribute_*`) — credit limit and account-level attributes
- **Transaction-level attributes** (`transaction_attribute_*`) — transaction counts and rupee value by merchant type
- **Bureau tradeline attributes** (`bureau_*`) — credit history, product holdings, historical delinquencies
- **Bureau enquiry attributes** (`bureau_enquiry_*`) — recent credit enquiries (e.g. personal-loan enquiries in the last 3 months)



## Methodology

**1. Exploratory Data Analysis** — Distribution plots, correlation heatmaps, and joint plots across on-us, bureau, and transaction attributes to identify signal and multicollinearity (e.g. a strong positive relationship between `onus_attribute_17`, `onus_attribute_23`, and default risk).

**2. Missing Value Imputation** — Columns with more than 50% missing values were dropped. Remaining gaps were imputed and compared using:
- **MCMC** (Markov Chain Monte Carlo, via Bayesian Ridge–style sampling) — chosen as the final method for its ability to capture complex inter-feature dependencies
- **MICE** (Multiple Imputation by Chained Equations) — faster, but less effective on complex multivariate relationships

**3. Feature Engineering & Selection**
- Automated interaction/transform discovery with **AutoFE**
- Statistical ranking with the **ANOVA F-test** and **Recursive Feature Elimination (RFE)**
- Dimensionality reduction with **PCA**, retaining components covering ≥90% variance
- Final feature set narrowed to the most informative ~282 engineered features

**4. Handling Class Imbalance** — The default class is a small minority of the dataset. Several strategies were evaluated (random undersampling, Tomek Links, NearMiss, SMOTE, ADASYN); a combination of **undersampling + SMOTE oversampling** on the training split gave the best results.

**5. Threshold Optimisation — Logit Shift & Youden's J** — Rather than using a default 0.5 cutoff, the decision threshold was shifted using **Youden's J statistic** (`J = Sensitivity + Specificity − 1`) on the ROC curve to better balance detection of the minority (default) class against overall accuracy.

**6. Model Training** — Individual models trained and compared:
- Logistic Regression, Decision Tree, Random Forest, SVM
- XGBoost, LightGBM, CatBoost (each hyperparameter-tuned with **Optuna**)
- A genetic-algorithm-based feature/hyperparameter search (via **DEAP**)

**7. Ensembling** — A **soft Voting Classifier** combining 15 tuned base models (5 each of XGBoost, CatBoost, and LightGBM) was selected as the final model, validated with 100-fold stratified cross-validation. A Stacking Classifier was also evaluated as an alternative.

## Results

| Model | Recall | Accuracy | Precision | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| XGBoost | 0.11 | 0.97 | 0.07 | 0.09 | 0.7379 |
| Random Forest | 0.04 | 0.98 | 0.10 | 0.05 | 0.7670 |
| Logistic Regression | 0.15 | 0.95 | 0.06 | 0.09 | 0.7402 |
| Decision Tree | 0.18 | 0.95 | 0.07 | 0.10 | 0.7543 |
| SVM | 0.16 | 0.95 | 0.07 | 0.09 | 0.7409 |
| Voting Classifier | 0.20 | 0.96 | 0.08 | 0.12 | 0.7786 |
| Voting Classifier (Grid Search) | 0.22 | 0.96 | 0.09 | 0.13 | 0.7805 |
| XGBoost + Genetic Algorithm | 0.21 | 0.96 | 0.09 | 0.12 | 0.7813 |
| **Voting Classifier (Grid Search) + Logit Shift** | **0.23** | **0.96** | **0.10** | **0.14** | **0.7832** |

The **Voting Classifier (Grid Search) + Logit Shift** was selected as the best model based on ROC AUC — the most relevant metric given the severe class imbalance and the priority on distinguishing defaulters from non-defaulters over raw accuracy.

Full metrics, confusion matrices, and discussion are in [`Final_report_file.pdf`](./Final_report_file.pdf).

## Future Work

- Integrate **cost-sensitive learning** alongside logit shift to directly penalise minority-class misclassification
- Explore **genetic-algorithm-driven** hybrid feature selection combining filter and embedded methods
- Try non-linear dimensionality reduction (**t-SNE**, **UMAP**, autoencoders) to surface patterns PCA may miss
- Add **SHAP** / **LIME** explainability to interpret feature contributions and support model governance

