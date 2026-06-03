# bank-classifier

Machine learning pipeline for classifying bank transactions into a chart of accounts.

The project trains scikit-learn models using historical records and then classifies new statement rows from CSV/PDF inputs. It can also read/write transaction data from Microsoft Dataverse.

## Highlights

- Multi-target classification workflow (`ID_PlanoDeConta`, `Classification`, `Company_Name`, `Description2`)
- Dataverse integration via Azure AD app credentials
- Confidence-based review flow (`Needs Review` for low-confidence predictions)
- Reproducible synthetic sample data included for safe demos

## Security and Privacy

This repository is prepared for public portfolio use.

- Never commit `.env` with real credentials.
- Keep all real financial statements out of version control.
- Use only synthetic demo CSV files in `data/raw` and `data/processed`.
- Treat any previously exposed credentials as compromised and rotate them.

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

## GitHub Publishing Checklist

Use this checklist before making the repository public:

1. Rotate Azure credentials if they were ever stored in local tracked files.
2. Confirm `.env` is ignored and not staged.
3. Confirm only synthetic sample data exists under `data/raw` and `data/processed`.
4. Confirm `models/` contains no trained artifacts unless intentionally sharing synthetic/demo models.
5. Run a secret scan over tracked files.

## Suggested GitHub Setup

After local commit:

```bash
git init
git add .
git commit -m "Prepare bank-classifier for public portfolio"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

Then set repository metadata on GitHub:

- Description: "Bank transaction classifier with Dataverse integration"
- Topics: `python`, `scikit-learn`, `dataverse`, `ml`, `classification`

## Limitations

- Model quality depends on historical label quality and class balance.
- New vendors/descriptions may require periodic retraining.
- Chart-of-accounts mapping is schema-dependent.

## License

Add a license file before publishing (MIT is a common portfolio choice).
