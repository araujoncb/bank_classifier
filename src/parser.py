import os
import pandas as pd
import pdfplumber

_REQUIRED_COLS = ["Date", "Description", "Amount"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    col_map: dict[str, str] = {}
    for col in df.columns:
        lower = col.lower()
        if "date" in lower:
            col_map[col] = "Date"
        elif "description" in lower and "2" not in lower:
            col_map[col] = "Description"
        elif "amount" in lower or "value" in lower or "valor" in lower:
            col_map[col] = "Amount"
        elif "in" in lower and "out" in lower:
            col_map[col] = "InOut"
        elif lower in ("d/c", "type", "tipo"):
            col_map[col] = "InOut"
    df = df.rename(columns=col_map)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Could not map columns {missing} from: {list(df.columns)}")
    keep = _REQUIRED_COLS + (["InOut"] if "InOut" in df.columns else [])
    return df[keep].copy()


def _parse_csv(filepath: str) -> pd.DataFrame:
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(filepath, sep=sep, dtype=str)
            if df.shape[1] > 1:
                return _normalize(df)
        except Exception:
            continue
    raise ValueError(f"Could not parse CSV: {filepath}")


def _parse_pdf(filepath: str) -> pd.DataFrame:
    frames = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                header = [str(c).strip() if c else "" for c in table[0]]
                rows = [[str(c).strip() if c else "" for c in r] for r in table[1:]]
                frames.append(pd.DataFrame(rows, columns=header))
    if not frames:
        raise ValueError(f"No tables found in PDF: {filepath}")
    df = pd.concat(frames, ignore_index=True)
    return _normalize(df)


def parse(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        df = _parse_pdf(filepath)
    elif ext in (".csv", ".txt"):
        df = _parse_csv(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    df["Amount"] = pd.to_numeric(df["Amount"].str.replace(",", "."), errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "Amount"])
    df = df.reset_index(drop=True)
    if "InOut" not in df.columns:
        df["InOut"] = df["Amount"].apply(lambda x: "In" if x >= 0 else "Out")
    return df
