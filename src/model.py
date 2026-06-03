import os
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.features import build_transformer
from src import config

TARGETS = ["ID_PlanoDeConta", "Classification", "Company_Name", "Description2"]
FEATURES = ["Description", "Amount", "InOut", "Date", "Currency", "Source"]
RARE_LABEL_TARGETS = {"Company_Name", "Description2"}


def _collapse_rare_labels(subset: pd.DataFrame, target: str) -> tuple[pd.DataFrame, int, int]:
    if target not in RARE_LABEL_TARGETS:
        return subset, 0, 0
    min_count = max(1, config.RARE_LABEL_MIN_COUNT)
    if min_count <= 1:
        return subset, 0, 0

    counts = subset[target].value_counts()
    rare_labels = counts[counts < min_count].index
    if len(rare_labels) == 0:
        return subset, 0, 0

    out = subset.copy()
    rare_mask = out[target].isin(rare_labels)
    rows_collapsed = int(rare_mask.sum())
    labels_collapsed = int(len(rare_labels))
    out.loc[rare_mask, target] = "NA"
    return out, rows_collapsed, labels_collapsed


def train(df: pd.DataFrame) -> dict[str, Pipeline]:
    pipelines = {}
    for target in TARGETS:
        if target not in df.columns or df[target].isna().all():
            print(f"  [skip] {target}: no data")
            continue
        mask = df[target].notna() & (df[target] != "")
        subset = df[mask]
        if len(subset) < 10:
            print(f"  [skip] {target}: too few samples ({len(subset)})")
            continue

        subset, rows_collapsed, labels_collapsed = _collapse_rare_labels(subset, target)
        if rows_collapsed:
            print(
                f"  [prep] {target}: collapsed {labels_collapsed} rare labels "
                f"({rows_collapsed} rows) into 'NA'"
            )

        pipeline = Pipeline([
            ("features", build_transformer()),
            ("clf", LogisticRegression(max_iter=10000, C=1.0, class_weight="balanced", solver="saga", tol=1e-3)),
        ])
        pipeline.fit(subset[FEATURES], subset[target])
        pipelines[target] = pipeline
        print(f"  [ok]   {target}: trained on {len(subset)} rows")
    return pipelines


def evaluate(pipelines: dict[str, Pipeline], df_test: pd.DataFrame) -> None:
    for target, pipeline in pipelines.items():
        if target not in df_test.columns:
            continue
        mask = df_test[target].notna() & (df_test[target] != "")
        subset = df_test[mask]
        if subset.empty:
            continue
        y_pred = pipeline.predict(subset[FEATURES])
        acc = accuracy_score(subset[target], y_pred)
        print(f"\n{'='*50}")
        print(f"  {target}  —  accuracy: {acc:.3f}")
        print(classification_report(subset[target], y_pred, zero_division=0))


def save(pipelines: dict[str, Pipeline], models_dir: str, coa_labels: list[str]) -> None:
    os.makedirs(models_dir, exist_ok=True)
    for target, pipeline in pipelines.items():
        path = os.path.join(models_dir, f"{target}.joblib")
        joblib.dump(pipeline, path)
    labels_path = os.path.join(models_dir, "coa_labels.json")
    with open(labels_path, "w") as f:
        json.dump(sorted(coa_labels), f, indent=2)
    print(f"\nModels saved to {models_dir}/")


def load(models_dir: str) -> tuple[dict[str, Pipeline], list[str]]:
    pipelines = {}
    for target in TARGETS:
        path = os.path.join(models_dir, f"{target}.joblib")
        if os.path.exists(path):
            pipelines[target] = joblib.load(path)
    labels_path = os.path.join(models_dir, "coa_labels.json")
    with open(labels_path) as f:
        coa_labels = json.load(f)
    return pipelines, coa_labels


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(df, test_size=0.2, stratify=df["ID_PlanoDeConta"], random_state=42)
    except ValueError:
        return train_test_split(df, test_size=0.2, random_state=42)
