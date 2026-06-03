import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from src import config
from src import dataverse, model
from src.parser import parse


def _fix_mojibake_text(value):
    if not isinstance(value, str):
        return value
    if not any(marker in value for marker in ("Ã", "Â", "â€", "â€™", "â€œ", "â€")):
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def _repair_output_text(df: pd.DataFrame) -> None:
    for col in ("Company_Name", "Description2"):
        if col in df.columns:
            df[col] = df[col].map(_fix_mojibake_text)


def _predict_with_confidence(pipeline, X: pd.DataFrame) -> tuple[list, list[float]]:
    labels = pipeline.predict(X)
    proba = pipeline.predict_proba(X)
    confidence = proba.max(axis=1).tolist()
    return list(labels), confidence


def _classify_dataframe(
    df: pd.DataFrame,
    pipelines: dict,
    coa_set: set,
    source_label: str,
) -> pd.DataFrame:
    missing = [col for col in model.FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    source_missing = df["Source"].isna() | (df["Source"].astype(str).str.strip() == "")
    if source_missing.any():
        raise ValueError("Source has blank/null values; populate SOURCE_COL during transformation before classification")

    for target, pipeline in pipelines.items():
        preds, confs = _predict_with_confidence(pipeline, df[model.FEATURES])
        df[target] = preds
        df[f"{target}_confidence"] = confs

    # Repair common mojibake patterns before exporting and write-back.
    _repair_output_text(df)

    core_targets = [t for t in ("ID_PlanoDeConta", "Classification") if t in pipelines]
    optional_targets = [t for t in ("Company_Name", "Description2") if t in pipelines]

    conf_thresholds = {
        "ID_PlanoDeConta": config.ID_PLANO_DE_CONTA_THRESHOLD,
        "Classification": config.CLASSIFICATION_THRESHOLD,
        "Company_Name": config.COMPANY_NAME_THRESHOLD,
        "Description2": config.DESCRIPTION2_THRESHOLD,
    }

    core_conf_cols = [f"{t}_confidence" for t in core_targets]
    if core_conf_cols:
        df["confidence_score"] = df[core_conf_cols].min(axis=1)
    else:
        any_conf_cols = [f"{t}_confidence" for t in pipelines]
        df["confidence_score"] = df[any_conf_cols].min(axis=1)

    # Flag ID_PlanoDeConta predictions outside CoA as needing review
    unknown_coa = pd.Series(False, index=df.index)
    if "ID_PlanoDeConta" in df.columns:
        unknown_coa = ~df["ID_PlanoDeConta"].isin(coa_set)
        if unknown_coa.any():
            print(f"  WARNING: {unknown_coa.sum()} rows predicted with unknown CoA ID")

    # Core targets decide AI vs Needs Review, optional targets are blanked when low confidence.
    row_very_low_core = pd.Series(False, index=df.index)
    row_low_core = pd.Series(False, index=df.index)

    for target in core_targets:
        conf_col = f"{target}_confidence"
        row_very_low_core |= df[conf_col] < config.VERY_LOW_CONFIDENCE_THRESHOLD
        row_low_core |= df[conf_col] < conf_thresholds[target]

    prediction_cols = [t for t in model.TARGETS if t in df.columns]
    if prediction_cols and row_very_low_core.any():
        df.loc[row_very_low_core, prediction_cols] = ""
        print(
            f"  {row_very_low_core.sum()} rows below core very-low threshold "
            "- all predictions cleared for manual classification"
        )

    for target in optional_targets:
        conf_col = f"{target}_confidence"
        optional_low = (df[conf_col] < conf_thresholds[target]) & ~row_very_low_core
        if optional_low.any():
            df.loc[optional_low, target] = ""
            print(
                f"  {optional_low.sum()} rows below {target} threshold "
                f"({conf_thresholds[target]:.2f}) - {target} cleared"
            )

    df["classified_by"] = "AI"
    df.loc[unknown_coa | row_low_core, "classified_by"] = "Needs Review"

    conf_cols = [f"{t}_confidence" for t in pipelines]
    df.drop(columns=conf_cols, inplace=True)

    df["source_file"] = source_label
    df["classified_at"] = datetime.now(timezone.utc).isoformat()
    return df


def _persist_local_output(df: pd.DataFrame) -> str:
    os.makedirs(config.DATA_OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(config.DATA_OUT_DIR, f"classified_{ts}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["dataverse", "file"],
        default="dataverse",
        help="Input source: dataverse (default) or file",
    )
    parser.add_argument("--input", help="Path to CSV or PDF statement (required when --source file)")
    parser.add_argument("--currency", choices=["BRL", "USD"], help="Currency for file mode")
    parser.add_argument("--limit", type=int, help="Top N unclassified Dataverse rows to classify")
    parser.add_argument("--dry-run", action="store_true", help="Skip Dataverse write-back")
    args = parser.parse_args()

    if args.source == "file":
        if not args.input:
            parser.error("--input is required when --source file")
        if not args.currency:
            parser.error("--currency is required when --source file")

    print(f"Loading models from {config.MODELS_DIR} ...")
    pipelines, coa_labels = model.load(config.MODELS_DIR)
    coa_set = set(coa_labels)

    if args.source == "file":
        print(f"Parsing {args.input} ...")
        df = parse(args.input)
        print(f"  {len(df)} transactions found")
        df["Currency"] = args.currency
        if "Source" not in df.columns:
            df["Source"] = os.path.basename(args.input)
        source_label = os.path.basename(args.input)
    else:
        print("Fetching unclassified rows from Dataverse ...")
        df = dataverse.fetch_unclassified_rows(limit=args.limit)
        print(f"  {len(df)} rows fetched")
        if df.empty:
            print("No unclassified rows found. Nothing to do.")
            return
        source_label = "dataverse"

    df = _classify_dataframe(df, pipelines, coa_set, source_label=source_label)

    _persist_local_output(df)
    needs_review = (df["classified_by"] == "Needs Review").sum()
    print(f"  AI: {len(df) - needs_review}  |  Needs Review: {needs_review} / {len(df)} rows")

    if args.dry_run:
        print("Dry run - skipping Dataverse write-back.")
        return

    if args.source == "file":
        print("Pushing to Dataverse output table...")
        dataverse.push_classified(df)
    else:
        print("Patching Dataverse source rows...")
        dataverse.patch_classified_rows(df)
    print("Done.")


if __name__ == "__main__":
    main()
