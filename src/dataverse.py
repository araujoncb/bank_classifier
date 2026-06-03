import requests
import msal
import pandas as pd

from src import config

_token_cache: dict = {}


def _get_token() -> str:
    scope = f"{config.DATAVERSE_URL}/.default"
    if _token_cache.get("scope") == scope and _token_cache.get("token"):
        return _token_cache["token"]

    app = msal.ConfidentialClientApplication(
        client_id=config.AZURE_CLIENT_ID,
        client_credential=config.AZURE_CLIENT_SECRET,
        authority=f"https://login.microsoftonline.com/{config.AZURE_TENANT_ID}",
    )
    result = app.acquire_token_for_client(scopes=[scope])
    if "access_token" not in result:
        raise RuntimeError(f"Dataverse auth failed: {result.get('error_description')}")

    _token_cache["scope"] = scope
    _token_cache["token"] = result["access_token"]
    return result["access_token"]


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Prefer": "odata.include-annotations=OData.Community.Display.V1.FormattedValue",
    }


def _get_all(url: str) -> list[dict]:
    rows = []
    while url:
        r = requests.get(url, headers=_headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        rows.extend(data.get("value", []))
        url = data.get("@odata.nextLink")
    return rows


def _history_cols() -> list[str]:
    return [
        config.HISTORY_PK_COL,
        config.DATE_COL,
        config.DESC_COL,
        config.AMOUNT_COL,
        config.INOUT_COL,
        config.SOURCE_COL,
        config.COMPANY_COL,
        config.CLASS_COL,
        config.DESC2_COL,
        config.COA_ID_COL,
        config.CURRENCY_COL,
    ]


def _history_rename_map() -> dict[str, str]:
    return {
        config.HISTORY_PK_COL: "history_row_id",
        config.DATE_COL: "Date",
        config.DESC_COL: "Description",
        config.AMOUNT_COL: "Amount",
        config.INOUT_COL: "InOut",
        config.SOURCE_COL: "Source",
        config.COMPANY_COL: "Company_Name",
        config.CLASS_COL: "Classification",
        config.DESC2_COL: "Description2",
        config.COA_ID_COL: "ID_PlanoDeConta",
        config.CURRENCY_COL: "Currency",
    }


def fetch_history() -> pd.DataFrame:
    cols = _history_cols()
    select = ",".join(cols)
    url = f"{config.DATAVERSE_URL}/api/data/v9.2/{config.DV_HISTORY_TABLE}?$select={select}"
    rows = _get_all(url)
    df = pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)
    return df.rename(columns=_history_rename_map())


def fetch_unclassified_rows(limit: int | None = None) -> pd.DataFrame:
    cols = _history_cols()
    select = ",".join(cols)
    filters = [
        f"({config.COA_ID_COL} eq null or {config.COA_ID_COL} eq '')",
        f"({config.CLASS_COL} eq null or {config.CLASS_COL} eq '')",
        f"({config.COMPANY_COL} eq null or {config.COMPANY_COL} eq '')",
        f"({config.DESC2_COL} eq null or {config.DESC2_COL} eq '')",
    ]
    odata_filter = " or ".join(filters)
    order_by = f"{config.HISTORY_ORDER_COL} asc,{config.HISTORY_PK_COL} asc"
    params = {
        "$select": select,
        "$filter": odata_filter,
        "$orderby": order_by,
    }
    if limit:
        params["$top"] = str(int(limit))

    base_url = f"{config.DATAVERSE_URL}/api/data/v9.2/{config.DV_HISTORY_TABLE}"
    r = requests.get(base_url, headers=_headers(), params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = data.get("value", [])

    if not limit:
        next_url = data.get("@odata.nextLink")
        while next_url:
            r2 = requests.get(next_url, headers=_headers(), timeout=30)
            r2.raise_for_status()
            page = r2.json()
            rows.extend(page.get("value", []))
            next_url = page.get("@odata.nextLink")

    df = pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)
    return df.rename(columns=_history_rename_map())


def fetch_chart_of_accounts() -> pd.DataFrame:
    cols = [
        config.COA_ID1_COL, config.COA_NAME1_COL,
        config.COA_ID2_COL, config.COA_NAME2_COL,
        config.COA_ID3_COL, config.COA_NAME3_COL,
    ]
    select = ",".join(cols)
    url = f"{config.DATAVERSE_URL}/api/data/v9.2/{config.DV_COA_TABLE}?$select={select}"
    rows = _get_all(url)
    df = pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)
    df = df.rename(columns={
        config.COA_ID1_COL:   "ID_1",
        config.COA_NAME1_COL: "Nivel_1",
        config.COA_ID2_COL:   "ID_2",
        config.COA_NAME2_COL: "Nivel_2",
        config.COA_ID3_COL:   "ID_3",
        config.COA_NAME3_COL: "Nivel_3",
    })
    return df


def push_classified(df: pd.DataFrame) -> None:
    url = f"{config.DATAVERSE_URL}/api/data/v9.2/{config.DV_OUTPUT_TABLE}"
    hdrs = _headers()
    for _, row in df.iterrows():
        payload = {
            config.DATE_COL: str(row.get("Date", "")),
            config.DESC_COL: row.get("Description", ""),
            config.AMOUNT_COL: row.get("Amount"),
            config.INOUT_COL: row.get("InOut", ""),
            config.COMPANY_COL: row.get("Company_Name", ""),
            config.CLASS_COL: row.get("Classification", ""),
            config.DESC2_COL: row.get("Description2", ""),
            config.COA_ID_COL: row.get("ID_PlanoDeConta", ""),
            config.CURRENCY_COL: row.get("Currency", ""),
            config.CLASSIFIED_BY_COL: row.get("classified_by", ""),
            config.SOURCE_FILE_COL: row.get("source_file", ""),
            config.CLASSIFIED_AT_COL: row.get("classified_at", ""),
            config.CONFIDENCE_SCORE_COL: row.get("confidence_score"),
        }
        r = requests.post(url, json=payload, headers=hdrs, timeout=30)
        r.raise_for_status()


def patch_classified_rows(df: pd.DataFrame) -> None:
    if "history_row_id" not in df.columns:
        raise ValueError("history_row_id is required to patch Dataverse rows")

    hdrs = _headers()
    patched = 0
    for _, row in df.iterrows():
        row_id = str(row.get("history_row_id", "")).strip()
        if not row_id:
            continue

        payload = {
            config.COMPANY_COL: row.get("Company_Name", ""),
            config.CLASS_COL: row.get("Classification", ""),
            config.DESC2_COL: row.get("Description2", ""),
            config.COA_ID_COL: row.get("ID_PlanoDeConta", ""),
            config.CLASSIFIED_BY_COL: row.get("classified_by", ""),
            config.SOURCE_FILE_COL: row.get("source_file", ""),
            config.CLASSIFIED_AT_COL: row.get("classified_at", ""),
            config.CONFIDENCE_SCORE_COL: row.get("confidence_score"),
        }

        url = f"{config.DATAVERSE_URL}/api/data/v9.2/{config.DV_HISTORY_TABLE}({row_id})"
        r = requests.patch(url, json=payload, headers=hdrs, timeout=30)
        r.raise_for_status()
        patched += 1

    print(f"  Patched {patched} source rows")
