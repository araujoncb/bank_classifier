# bank-classifier

Machine learning pipeline for classifying bank transactions into a chart of accounts.

The project trains scikit-learn models using historical records and then classifies new statement rows from CSV/PDF inputs. It can also read/write transaction data from Microsoft Dataverse.

## Highlights

- Multi-target classification workflow (`ID_PlanoDeConta`, `Classification`, `Company_Name`, `Description2`)
- Dataverse integration via Azure AD app credentials
- Confidence-based review flow (`Needs Review` for low-confidence predictions)
- Reproducible synthetic sample data included for safe demos

## Project Structure

- `src/config.py`: environment configuration
- `src/parser.py`: parses transaction files
- `src/features.py`: feature preparation
- `src/model.py`: model pipeline training/inference helpers
- `src/dataverse.py`: Dataverse auth and API access
- `train.py`: model training entrypoint
- `classify.py`: classification entrypoint
- `data/raw/sample_transactions_sanitized.csv`: synthetic input example
- `data/processed/sample_classified_sanitized.csv`: synthetic output example

## Requirements

- Python 3.10+
- Access to Dataverse only if using train/classify with remote data

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

1. Copy the template:

```bash
cp .env.example .env
```

2. Fill your own Azure/Dataverse values in `.env`.

Required variables include:

- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `DATAVERSE_URL`
- `DV_*` table names and `*_COL` mappings

## Usage

### 1) Train models

```bash
python train.py
```

This writes local model artifacts to `models/`.

### 2) Classify a CSV file

```bash
python classify.py --input data/raw/sample_transactions_sanitized.csv --dry-run
```

### 3) Classify and write back to Dataverse

```bash
python classify.py --input data/raw/sample_transactions_sanitized.csv
```

## Confidence Workflow

Rows below your confidence threshold are flagged for manual review.

- `confidence_score` is generated per row.
- rows below threshold are tagged as `Needs Review`.

## Limitations

- Model quality depends on historical label quality and class balance.
- New vendors/descriptions may require periodic retraining.
- Chart-of-accounts mapping is schema-dependent.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
