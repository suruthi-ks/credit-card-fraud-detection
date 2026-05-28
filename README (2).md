# 🔐 Credit Card Fraud Detection — Machine Learning

> End-to-end fraud detection pipeline: exploratory data analysis, class imbalance handling (SMOTE vs class weights), and Random Forest classification on 284,807 real credit card transactions.

---

## Project Overview

This project builds a **binary fraud detection classifier** on a real-world dataset of 284,807 European credit card transactions recorded over 48 hours, following the **CRISP-DM** framework end-to-end.

The core challenge is extreme class imbalance — only **0.17% of transactions are fraudulent**. A naive classifier that labels everything as legitimate scores 99.83% accuracy while detecting zero fraud, making standard accuracy useless as a metric.

Two imbalance-handling strategies are trained and compared on the same test set:

**Module:** IBM3201 – Data Mining and Predictive Analytics | INTI International University (Jan 2026)

---

## Key Results

| Metric | Model A — SMOTE | Model B — Class Weights |
|---|---|---|
| Fraud Precision | 0.92 | **0.99** |
| Fraud Recall | **0.76** (72/95 detected) | 0.73 (69/95 detected) |
| Fraud F1-Score | 0.83 | **0.84** |
| ROC-AUC | **0.9436** | 0.9141 |
| False Positives | 6 | **1** |

> **Model A (SMOTE) is recommended** for production: higher recall and AUC give banks more flexibility to tune the classification threshold to their risk appetite. Model B is preferable where minimising false alarms and protecting customer experience is the priority.

---

## Repository Structure

```
credit-card-fraud-detection/
│
├── notebooks/
│   └── credit_card_fraud_detection.ipynb   # Full pipeline notebook
│
├── data/                                    # Place downloaded CSVs here
│   └── (see Dataset section below)
│
├── outputs/
│   └── charts/
│       ├── class_distribution.png
│       ├── amount_distribution.png
│       ├── time_distribution.png
│       ├── pca_boxplots.png
│       ├── correlation_heatmap.png
│       ├── scatter_amount_v14.png
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       └── feature_importance.png
│
├── reports/
│   └── IBM3201_Lab_Report.pdf              # Full written report (CRISP-DM)
│
└── README.md
```

---

## Dataset

**Source:** Provided as course material — IBM3201, INTI International University (January 2026 session).

📁 **[Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1cO0ckFPA0BIxY3Gx7fJ54WS-EymvnmNM?usp=sharing)**

Download both CSV files and place them inside the `data/` folder before running the notebook:
- `creditcard_transactions_part1.csv`
- `creditcard_transactions_part2.csv`

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| After deduplication | 283,726 |
| Features | 30 (Time, V1–V28 PCA components, Amount) |
| Target | `Class` (0 = Legitimate, 1 = Fraud) |
| Fraud rate | 0.1727% (492 cases) |

> V1–V28 are anonymised PCA components to protect cardholder privacy. Only `Time` and `Amount` retain their original meaning.

---

## What's Inside the Notebook

The notebook is organised into 8 self-contained sections:

### Section 1 — Setup
Installs `imbalanced-learn` and imports all required libraries. Sets a global `RANDOM_STATE = 42` for reproducibility.

### Section 2 — Data Loading
Loads and concatenates both CSV files into a single DataFrame of 284,807 records and 31 columns.

### Section 3 — Data Quality Check
- Zero missing values across all 31 columns
- 1,081 duplicate rows identified and removed (likely from overlapping recording windows between the two files)
- Outliers in PCA features retained — they represent genuine extreme transaction behaviour, which is exactly what fraud detection must learn to recognise

### Section 4 — Exploratory Data Analysis
Six visualisations with business interpretations:

- **Class Distribution** — visual proof of the 578:1 imbalance and why accuracy is misleading
- **Transaction Amount Distribution** — fraud spikes heavily in the $0–$100 range by volume
- **Transaction Time Distribution** — legitimate transactions show a human bimodal rhythm (two daily peaks); fraud spikes erratically at all hours
- **PCA Feature Boxplots (V14, V17, V12)** — near-perfect distributional separation between classes; fraud medians sit at −6, −5, −4 respectively vs ~0 for legitimate
- **Correlation Heatmap** — V17 (−0.33), V14 (−0.30), V12 (−0.26) are strongest correlates of fraud; Amount has near-zero correlation (0.01)
- **Scatter Plot (Amount vs V14)** — legitimate transactions cluster at V14 ≈ 0; all fraud clusters at V14 between −5 and −20

**Cross-tabulation finding:** Fraud rate increases proportionally with transaction amount — highest at 0.42% in the $500–$1,000 bucket, nearly 3× the rate in the $0–$50 bucket.

### Section 5 — Data Preprocessing
- `StandardScaler` applied to `Amount` and `Time` only (V1–V28 already PCA-standardised)
- Scaler fitted on training data only to prevent data leakage
- Stratified 80/20 train-test split — test set contains 95 confirmed fraud cases

### Section 6 — Model Training

**Strategy 1 — SMOTE:** Generates synthetic fraud samples in feature space, balancing the training set to 50:50 (226,602 each class). Applied to training data only.

**Strategy 2 — Class Weights:** Sets `class_weight='balanced'`, penalising fraud misclassification ~599× more heavily than legitimate. No data modification.

Both use **Random Forest (100 trees)** — chosen for its ability to capture non-linear patterns in high-dimensional PCA space and its built-in feature importance scores.

### Section 7 — Evaluation
Confusion matrices, ROC curves, and feature importance rankings for both models side by side.

**Top feature importances (Model A — SMOTE):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | V14 | 0.2280 |
| 2 | V10 | 0.1133 |
| 3 | V17 | 0.1041 |
| 4 | V4  | 0.0885 |
| 5 | V12 | 0.0788 |

V14 alone drives ~22.8% of all model decisions. V14 + V10 + V17 together account for ~44.5% of total feature importance.

### Section 8 — Conclusion
Summary of findings, model recommendation, limitations, and future work directions.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, cross-tabulation |
| `numpy` | Numerical operations |
| `matplotlib` | Core chart creation |
| `seaborn` | Correlation heatmap and styling |
| `scikit-learn` | StandardScaler, train-test split, Random Forest, evaluation metrics |
| `imbalanced-learn` | SMOTE oversampling |

---

## How to Run

### Option A — Google Colab (recommended)
1. Upload `notebooks/credit_card_fraud_detection.ipynb` to [Google Colab](https://colab.research.google.com)
2. Download the CSVs from the Drive link above and upload them when prompted
3. Run all cells in order (Runtime → Run all)

### Option B — Local
```bash
git clone https://github.com/suruthi-ks/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
jupyter notebook notebooks/credit_card_fraud_detection.ipynb
```
> Place both CSV files in the `data/` folder before running. The notebook saves charts to `outputs/charts/` automatically.

---

## License

This project was developed for academic purposes as part of IBM3201 – Data Mining and Predictive Analytics.
