# Project Flow — Credit Card Behavioural Risk Scoring

This file gives a compact view of the modelling pipeline used in the final notebook.

```mermaid
flowchart TD

A[Raw Data] --> B[Schema + Integrity Checks]
B --> C[Stratified 70/15/15 Split]

C --> D[Full-Train EDA]
D --> D1[Target Imbalance]
D --> D2[Missingness Patterns]
D --> D3[Feature Families]
D --> D4[Distributions and Outliers]

D --> E[Feature Discovery]
E --> E1[ANOVA]
E --> E2[Mutual Information]
E --> E3[Univariate ROC-AUC]
E --> E4[KS Statistic]

E1 --> F[Candidate Pool]
E2 --> F
E3 --> F
E4 --> F

F --> G[WoE / IV + Quantile Profiles + Correlation + Drift]

G --> H[Representation Comparison]
H --> H1[Raw / Selected]
H --> H2[PCA]

H1 --> I[Model Screening]
H2 --> I

I --> I1[Logistic]
I --> I2[XGBoost]
I --> I3[LightGBM]
I --> I4[CatBoost]

I --> J[MI Top-200 Selected]

J --> K[Imbalance Experiments]

K --> K1[ROS / RUS]
K --> K2[SMOTE / Borderline-SMOTE / ADASYN]
K --> K3[Balanced RF / EasyEnsemble / RUSBoost]
K --> K4[Class Weights]
K --> K5[RUS + Sample Weights]

K --> L[Keep Only Stable Improvements]

L --> M[Merge Train + Tuning]
M --> N[5-Fold OOF]

N --> N1[XGB weight 1]
N --> N2[XGB weight 3]
N --> N3[CatBoost]

N1 --> O[OOF Weight Search]
N2 --> O
N3 --> O

O --> P["0.10 XGB1 + 0.25 XGB3 + 0.65 CatBoost"]

P --> Q[OOF Threshold Search]
Q --> R[Locked Holdout]

R --> S[ROC-AUC 0.8574]
R --> T[PR-AUC 0.1112]
R --> U[Precision 11.6% / Recall 32.0% / F1 17.1%]
R --> V[Capture / Lift / Deciles]

S --> W[Refit on All Labelled Data]
T --> W
U --> W
V --> W

W --> X[Score 41,792 Accounts]
X --> Y[Final_output_Updated.csv]
```

## Final Decision Logic

```text
PCA
└─ tested → weaker than supervised/raw paths → REJECTED

MI top-200
└─ strong nonlinear boosted-model path → KEPT

Aggressive resampling
└─ often reduced PR-AUC / precision → REJECTED

Moderate class weighting
└─ improved minority sensitivity without destroying ranking → KEPT

RUS + sample weights
├─ promising single tuning split
└─ weaker 5-fold OOF performance → REJECTED

Final model
└─ OOF-selected weighted ensemble
   ├─ 10% XGBoost, weight 1
   ├─ 25% XGBoost, weight 3
   └─ 65% CatBoost
```
