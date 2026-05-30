# Data Directory

The IBM Telco Customer Churn dataset used in this project is **not committed to this repository** due to licensing. You must download it separately and place the files in `data/raw/`.

## Required Files

Place the following 6 Excel files in `data/raw/`:

| File | Description | Approx. Size |
|---|---|---|
| `Telco_customer_churn.xlsx` | Main customer dataset with churn labels | ~1.5 MB |
| `Telco_customer_churn_population.xlsx` | Zip-code-level population data | ~50 KB |
| `Telco_customer_churn_demographics.xlsx` | Customer demographics (gender, senior status) | ~500 KB |
| `Telco_customer_churn_location.xlsx` | Geographic info (lat/long, city, state) | ~700 KB |
| `Telco_customer_churn_services.xlsx` | Service subscriptions and usage | ~1.2 MB |
| `Telco_customer_churn_status.xlsx` | Churn category, satisfaction score, CLTV | ~600 KB |

## Where to Download

### Option 1: IBM Cognos Sample Data Sets
Available through the IBM Cognos Analytics samples portal.

### Option 2: Kaggle Mirror (most accessible)
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Note: The Kaggle mirror packages the data slightly differently — you may need to download the full IBM split from the IBM Data Asset Exchange to get all 6 files.

### Option 3: IBM Data Asset Exchange
[https://developer.ibm.com/exchanges/data/all/telco-customer-churn/](https://developer.ibm.com/exchanges/data/all/telco-customer-churn/)

## Dataset Schema

After all 6 files are merged via the pipeline in `notebooks/01_data_preparation.ipynb` or `src/data_loader.py`:

- **Rows:** 7,043 customer records
- **Features:** ~25–30 after merge (depends on which leak columns are removed)
- **Target:** `Churn Value` (binary 0/1)
- **Class balance:** ~26.5% churners

## Synthetic Augmentation

The pipeline also generates a synthetic `NetworkStrain` variable to demonstrate omitted variable bias. This is computed deterministically from postal-code density and contract type, with a fixed random seed for reproducibility. See `src/preprocessing.py` for the exact generation formula (matching Section 9.1 of the paper).

## Leak Columns

Several columns in the raw data are post-hoc to the churn event and would cause data leakage if used as features. The pipeline automatically removes:

- `Churn Score`
- `CLTV` (Customer Lifetime Value computed using churn)
- `Churn Category`
- `Customer Status`
- `Satisfaction Score`
- `Total Revenue`, `Total Refunds`
- `Churn Reason`, `Churn Label`

See `src/preprocessing.py` for the complete list.

## Licensing

The IBM Telco Customer Churn dataset is provided by IBM under their sample data terms. Please review the license at your download source before redistributing.
