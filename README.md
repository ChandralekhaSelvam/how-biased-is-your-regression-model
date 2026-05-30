# How Biased Is Your Regression Model?

> **A Bias-Aware Framework for Telecom Churn Prediction: Unifying Omitted Variable Bias, Fairness Metrics, and MLOps Governance**

This repository contains the code, data pipeline, and supporting materials for the paper *"How Biased Is Your Regression Model"* by **Chandra Lekha S**.

The paper formalizes the connection between classical omitted variable bias (OVB) and modern algorithmic fairness in the context of telecom customer churn prediction, and proposes a production-ready Bias-Aware MLOps Architecture for deploying equitable predictive models at scale.

---

## 📋 Table of Contents

- [Motivation](#motivation)
- [Key Contributions](#key-contributions)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Data](#data)
- [Reproducing the Paper's Results](#reproducing-the-papers-results)
- [Key Components](#key-components)
- [Results Summary](#results-summary)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Motivation

Telecommunications operators rely on regression models for high-stakes decisions: churn intervention, network capacity planning, and customer lifetime value estimation. These models often achieve impressive global accuracy (90%+) while quietly misallocating capital across customer segments.

A model with 95% overall accuracy can still systematically underpredict churn for rural prepaid customers by 4%, missing tens of thousands of at-risk subscribers and misdirecting retention budgets. We call this **regression bias**, and its most insidious form is **omitted variable bias (OVB)** — when a relevant predictor is missing from the model, its effect gets absorbed by the included features, distorting their coefficients and producing segment-level disparities that aggregate metrics cannot detect.

This project provides:

1. A mathematical framework connecting OVB to ML fairness
2. An empirical demonstration on the IBM Telco Customer Churn dataset
3. A custom **Fair Logistic Regression** implementation with tunable fairness penalty
4. A reusable segment-level disparity metric `D_segment_disparity`
5. A blueprint for embedding bias detection into production MLOps pipelines

---

## Key Contributions

| # | Contribution | Where in Repo |
|---|---|---|
| 1 | **Theoretical**: Formal derivation of OVB as the structural source of segment disparity | `paper/` and `docs/methodology.md` |
| 2 | **Empirical**: 20-model comparison showing accuracy can coexist with 15% FPR disparity | `notebooks/00_full_pipeline.ipynb` |
| 3 | **Algorithmic**: Fair Logistic Regression with closed-form `D_segment_disparity` penalty | `src/fair_regression.py` |
| 4 | **Architectural**: 5-layer Bias-Aware MLOps Architecture for production deployment | `docs/architecture.md` |
| 5 | **Governance**: Mapping to GSMA and TM Forum frameworks | Paper Section 11 |

---

## Repository Structure

```
how-biased-is-your-regression-model/
├── README.md                       This file
├── LICENSE                         MIT
├── requirements.txt                Python dependencies
├── .gitignore
│
├── paper/
│   ├── HowBiasedIsYourRegressionModel.pdf
│   └── figures_source/             Original figure files
│
├── notebooks/                      Jupyter / Colab notebooks
│   ├── 00_full_pipeline.ipynb      End-to-end reproduction
│   ├── 01_data_preparation.ipynb   Merging 6 Telco files
│   ├── 02_baseline_models.ipynb    Cross-validation on 20 models
│   ├── 03_ovb_demonstration.ipynb  OVB illustration with feature dropping
│   ├── 04_fair_regression.ipynb    Fair Logistic with λ sweep
│   └── 05_segment_analysis.ipynb   Contract / gender / senior breakdowns
│
├── src/                            Reusable Python modules
│   ├── __init__.py
│   ├── data_loader.py              Loads + merges the 6 IBM Telco files
│   ├── preprocessing.py            Encoding, scaling, leak removal
│   ├── fair_regression.py          FairLogisticRegression class
│   ├── metrics.py                  segment_disparity(), mean_error()
│   └── visualizations.py           Paper figures 1–5
│
├── data/
│   ├── README.md                   Where to download the IBM Telco dataset
│   └── raw/                        Put the Excel files here (gitignored)
│
├── outputs/
│   ├── figures/                    PNG/SVG versions of paper figures
│   └── tables/                     CSV exports of result tables
│
└── docs/
    ├── architecture.md             5-layer MLOps architecture
    ├── methodology.md              Extended methodology
    └── citation.bib                BibTeX entry
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/how-biased-is-your-regression-model.git
cd how-biased-is-your-regression-model

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
# .venv\Scripts\activate            # Windows

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
matplotlib>=3.7
seaborn>=0.13
scipy>=1.11
openpyxl>=3.1
jupyter>=1.0
```

### Running on Google Colab

When running in Colab, upload the data files to `/content/Data/` or mount Google Drive.

---

## Data

This project uses the **IBM Telco Customer Churn dataset**, which is publicly available but not redistributed here due to licensing.

### Download

1. Download from IBM Cognos Analytics samples or the [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Place the following Excel files in `data/raw/`:
   - `Telco_customer_churn.xlsx` (main)
   - `Telco_customer_churn_population.xlsx`
   - `Telco_customer_churn_demographics.xlsx`
   - `Telco_customer_churn_location.xlsx`
   - `Telco_customer_churn_services.xlsx`
   - `Telco_customer_churn_status.xlsx`

### Dataset Statistics

- **7,043 customer records**
- **21 base features** + augmented features from merging
- **Target:** `Churn Value` (binary)
- **Churn rate:** ~26.5%
- **Synthetic augmentation:** `NetworkStrain` variable simulating omitted network-congestion data (described in Section 9.1 of the paper)

---

## Reproducing the Paper's Results

### Quick Run

```bash
jupyter notebook notebooks/00_full_pipeline.ipynb
```

Run all cells. Expected runtime: ~5–8 minutes on a standard laptop, ~2 minutes on Colab with GPU.

### Step-by-Step

If you prefer to walk through each stage:

| Notebook | Paper Section | What It Produces |
|---|---|---|
| `01_data_preparation.ipynb` | §9.1 | Merged dataset with NetworkStrain |
| `02_baseline_models.ipynb` | §9.2 | Figure 1 (20-model comparison) |
| `03_ovb_demonstration.ipynb` | §4, §8.3 | Table 3 (coefficient inflation) |
| `04_fair_regression.ipynb` | §9.3 | Figure 2 (Pareto frontier) |
| `05_segment_analysis.ipynb` | §9.4, §9.5 | Figures 3, 4, 5 |

### Using the Modules Directly

```python
from src.data_loader import load_and_merge_telco
from src.preprocessing import prepare_features
from src.fair_regression import FairLogisticRegression
from src.metrics import segment_disparity

# Load and prepare
df = load_and_merge_telco(data_dir="data/raw/")
X, y, segments = prepare_features(df, segment_col="Contract")

# Train fair model
model = FairLogisticRegression(
    lambda_complexity=0.1,
    lambda_fairness=2.0,
    segments=segments
)
model.fit(X, y)

# Evaluate
disparity = segment_disparity(model, X, y, segments)
print(f"Segment disparity: {disparity:.4f}")
```

---

## Key Components

### `FairLogisticRegression` (`src/fair_regression.py`)

A scikit-learn compatible classifier that augments standard logistic regression's cross-entropy loss with a segment disparity penalty:

```
L_total = L_prediction + λ_complexity · ||β||² + λ_fairness · D_segment_disparity
```

where `D_segment_disparity` is the maximum squared difference in mean residuals between any two segments:

```
D_segment_disparity = max_(s,s' ∈ S) ( MeanError_s − MeanError_(s') )²
```

The optimizer uses `scipy.optimize.minimize` with the L-BFGS-B method. The class implements `fit`, `predict`, and `predict_proba`, making it usable in any sklearn pipeline.

### Segment Metrics (`src/metrics.py`)

- `segment_disparity(model, X, y, segments)` — maximum absolute mean-error difference across segments
- `mean_error_by_segment(y_true, y_pred, segments)` — returns dict of segment → mean error
- `equalized_odds_difference(y_true, y_pred, segments)` — EO_diff as defined in Section 5.2.2
- `demographic_parity_difference(y_pred, segments)` — DP_diff as defined in Section 5.2.1

### Visualization Helpers (`src/visualizations.py`)

Reproduces every figure in the paper. Each function returns a `matplotlib.Figure` for saving or inline display:

- `plot_accuracy_vs_disparity(results_df)` → Figure 1
- `plot_fairness_accuracy_tradeoff(fair_df)` → Figure 2
- `plot_segment_breakdown(df_analysis)` → Figure 3
- `plot_accuracy_by_gender(df_analysis)` → Figure 4
- `plot_roc_by_contract(df_analysis)` → Figure 5

---

## Results Summary

| Model | Accuracy | Segment Disparity (Contract) |
|---|---|---|
| Gradient Boosting | 0.886 | 0.142 |
| XGBoost | 0.881 | 0.135 |
| Random Forest | 0.874 | 0.151 |
| Logistic Regression | 0.809 | 0.161 |
| **Fair Logistic (λ=2.0)** | **0.813** | **0.136** |
| Bernoulli NB | 0.731 | 0.201 |

**Key finding:** The fairness-penalized logistic regression reduces segment disparity by 15% while *improving* accuracy by 0.4 percentage points — a strict Pareto improvement that contradicts the common assumption that fairness must cost accuracy.

For the OVB illustration (Table 3 in the paper):

| Model | β̂_tenure (true=0.50) | Max Segment Disparity ($) |
|---|---|---|
| OLS (biased, NetworkStrain omitted) | 1.70 | $5.50 |
| Fair Ridge (λ_f=0.5) | 0.65 | $0.95 |

---

## Limitations

The paper acknowledges and this code reflects the following limitations:

- **Single dataset**: Results may differ on proprietary operator data with real network telemetry.
- **Synthetic NetworkStrain**: The omitted variable is artificially constructed; real omitted variables (tower congestion logs, handset quality) may have different correlation structures.
- **Binary protected attributes**: Gender and contract type are treated as binary; intersectional fairness (e.g., senior × female × rural) is not explored.
- **Single-period evaluation**: No temporal validation; concept drift could re-introduce bias post-deployment.
- **Modest disparity reduction at scale**: λ=2.0 reduces disparity from 0.161 to 0.136; more aggressive debiasing may be required for production deployment in regulated jurisdictions.

---

## License

This code is released under the **MIT License** — see [`LICENSE`](LICENSE) for details.

The IBM Telco Customer Churn dataset has its own license; please respect it when downloading.

---

## Contact

**Chandra Lekha S**
Email: lekhaselvam92@gmail.com

For questions, issues, or collaboration inquiries, please [open an issue](../../issues) on this repository.

---

## Acknowledgments

The author thanks IBM for making the Telco Customer Churn dataset publicly available through the IBM Data Asset Exchange.
