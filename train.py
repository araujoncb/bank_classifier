import sys
from src import config
from src import dataverse, model


def main():
    print("Fetching chart of accounts...")
    coa_df = dataverse.fetch_chart_of_accounts()
    valid_coa_ids = set(coa_df["ID_3"].dropna().unique())
    print(f"  {len(valid_coa_ids)} valid ID_3 values found")

    print("\nFetching transaction history...")
    df = dataverse.fetch_history()
    print(f"  {len(df)} rows fetched")

    before = len(df)
    df = df.dropna(subset=["ID_PlanoDeConta"])
    df = df[df["ID_PlanoDeConta"] != ""]
    print(f"  {before - len(df)} rows dropped (missing ID_PlanoDeConta)")

    invalid = df[~df["ID_PlanoDeConta"].isin(valid_coa_ids)]["ID_PlanoDeConta"].unique()
    if len(invalid):
        print(f"  WARNING: {len(invalid)} unknown CoA IDs in history: {list(invalid)[:10]}")

    if len(df) < 20:
        print("Not enough training data (need at least 20 rows). Aborting.")
        sys.exit(1)

    print("\nSplitting train/test (80/20)...")
    df_train, df_test = model.split(df)
    print(f"  train={len(df_train)}  test={len(df_test)}")

    print("\nTraining models...")
    pipelines = model.train(df_train)

    print("\nEvaluating on test set...")
    model.evaluate(pipelines, df_test)

    print("\nSaving models...")
    model.save(pipelines, config.MODELS_DIR, list(valid_coa_ids))
    print("Done.")


if __name__ == "__main__":
    main()
