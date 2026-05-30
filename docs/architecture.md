# Bias-Aware MLOps Architecture

This document expands Section 10 of the paper with implementation
guidance for operationalizing the five-layer architecture.

```
┌────────────────────────────────────────────────────────────┐
│  1. DATA LAYER                                             │
│  Feature Store & Profiler                                  │
│  → audits raw ingestion, detects representation imbalances │
│  → applies automated reweighting / slice analysis          │
└────────────────────────────────────────────────────────────┘
              ▼
┌────────────────────────────────────────────────────────────┐
│  2. TRAINING LAYER                                         │
│  Fairness-Penalized Pipeline                               │
│  → trains models with λ_fairness · D_segment_disparity     │
│  → evaluates Lasso, Ridge, Gradient Boosting variants      │
└────────────────────────────────────────────────────────────┘
              ▼
┌────────────────────────────────────────────────────────────┐
│  3. VALIDATION LAYER                                       │
│  Automated Gating Mechanisms                               │
│  → checks Mean Error, EO_diff against thresholds           │
│  → blocks biased models from production                    │
└────────────────────────────────────────────────────────────┘
              ▼
┌────────────────────────────────────────────────────────────┐
│  4. DEPLOYMENT LAYER                                       │
│  Model Registry & Serving                                  │
│  → version-controlled rollout via shadow / canary releases │
└────────────────────────────────────────────────────────────┘
              ▼
┌────────────────────────────────────────────────────────────┐
│  5. MONITORING LAYER                                       │
│  Continuous Observability                                  │
│  → monitors data drift and concept drift                   │
│  → triggers automated retraining cycles                    │
└────────────────────────────────────────────────────────────┘
```

## Mapping each layer to the code in this repo

| Layer | Module / File |
|---|---|
| 1. Data Layer | `src/data_loader.py`, `src/preprocessing.py` |
| 2. Training Layer | `src/fair_regression.py`, notebook `04_fair_regression.ipynb` |
| 3. Validation Layer | `src/metrics.py` (the gating thresholds use these) |
| 4. Deployment Layer | Out of scope for this paper — operator MLOps platform |
| 5. Monitoring Layer | Out of scope — usually a separate drift-detection service |

## Suggested gating thresholds

For production deployment, the paper recommends the following gates in
the Validation Layer:

- **MeanError per segment:** absolute value ≤ 0.05 for the largest 80%
  of segments by volume
- **EO_diff (Equalized Odds Difference):** ≤ 0.10
- **D_segment_disparity:** ≤ 0.15

These are starting points. Operators should calibrate based on
historical model performance and regulatory requirements.

## Governance alignment

The architecture aligns with:

- **GSMA Responsible AI Maturity Roadmap** — covers Vision, Operating
  Model, Tools, and Change Management.
- **TM Forum AI Governance Toolkit** — provides metadata management
  and data lineage standards.

See paper Section 11 for the full governance discussion.
