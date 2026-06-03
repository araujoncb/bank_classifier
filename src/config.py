import os
from dotenv import load_dotenv

load_dotenv()

def _req(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val

# Azure / Dataverse connection
AZURE_TENANT_ID     = _req("AZURE_TENANT_ID")
AZURE_CLIENT_ID     = _req("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = _req("AZURE_CLIENT_SECRET")
DATAVERSE_URL       = _req("DATAVERSE_URL").rstrip("/")

# Table names
DV_HISTORY_TABLE = _req("DV_HISTORY_TABLE")
DV_COA_TABLE     = _req("DV_COA_TABLE")
DV_OUTPUT_TABLE  = _req("DV_OUTPUT_TABLE")
HISTORY_PK_COL   = _req("HISTORY_PK_COL")

# Transaction column logical names
DATE_COL    = _req("DATE_COL")
DESC_COL    = _req("DESC_COL")
AMOUNT_COL  = _req("AMOUNT_COL")
INOUT_COL   = _req("INOUT_COL")
COMPANY_COL = _req("COMPANY_COL")
CLASS_COL   = _req("CLASS_COL")
DESC2_COL   = _req("DESC2_COL")
COA_ID_COL  = _req("COA_ID_COL")
SOURCE_COL  = _req("SOURCE_COL")
HISTORY_ORDER_COL = os.getenv("HISTORY_ORDER_COL", DATE_COL)

# CoA column logical names
COA_ID3_COL   = _req("COA_ID3_COL")
COA_NAME3_COL = _req("COA_NAME3_COL")
COA_ID2_COL   = _req("COA_ID2_COL")
COA_NAME2_COL = _req("COA_NAME2_COL")
COA_ID1_COL   = _req("COA_ID1_COL")
COA_NAME1_COL = _req("COA_NAME1_COL")

CURRENCY_COL         = os.getenv("CURRENCY_COL", "cr1a0_currency")

CLASSIFIED_BY_COL    = _req("CLASSIFIED_BY_COL")
SOURCE_FILE_COL      = _req("SOURCE_FILE_COL")
CLASSIFIED_AT_COL    = _req("CLASSIFIED_AT_COL")
CONFIDENCE_SCORE_COL = _req("CONFIDENCE_SCORE_COL")

CONFIDENCE_THRESHOLD     = float(_req("CONFIDENCE_THRESHOLD"))
AI_TRAINING_LAG_MONTHS   = int(os.getenv("AI_TRAINING_LAG_MONTHS", "2"))

# Per-target confidence controls (override in .env as needed)
VERY_LOW_CONFIDENCE_THRESHOLD = float(os.getenv("VERY_LOW_CONFIDENCE_THRESHOLD", "0.20"))
CORE_CONFIDENCE_THRESHOLD = float(os.getenv("CORE_CONFIDENCE_THRESHOLD", str(CONFIDENCE_THRESHOLD)))
ID_PLANO_DE_CONTA_THRESHOLD = float(os.getenv("ID_PLANO_DE_CONTA_THRESHOLD", str(CORE_CONFIDENCE_THRESHOLD)))
CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", str(CORE_CONFIDENCE_THRESHOLD)))
COMPANY_NAME_THRESHOLD = float(os.getenv("COMPANY_NAME_THRESHOLD", "0.35"))
DESCRIPTION2_THRESHOLD = float(os.getenv("DESCRIPTION2_THRESHOLD", "0.35"))

# Rare labels in high-cardinality targets are collapsed to this label during training.
RARE_LABEL_MIN_COUNT = int(os.getenv("RARE_LABEL_MIN_COUNT", "3"))

# Paths
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATA_OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# The four targets the model predicts (Dataverse column → friendly key)
TARGETS: dict[str, str] = {
    COA_ID_COL:  "ID_PlanoDeConta",
    CLASS_COL:   "Classification",
    COMPANY_COL: "Company_Name",
    DESC2_COL:   "Description2",
}
