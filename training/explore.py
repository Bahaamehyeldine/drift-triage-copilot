from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/bank-additional-full.csv")


def main() -> None:
    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(DATASET_PATH, sep=";")

    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)

    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}):")
    print(df.columns.tolist())

    # ------------------------------------------------------------------
    # Target distribution
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TARGET DISTRIBUTION")
    print("=" * 80)

    target_counts = df["y"].value_counts().sort_index()
    target_percent = df["y"].value_counts(normalize=True).sort_index() * 100

    summary = pd.DataFrame(
        {
            "count": target_counts,
            "percentage": target_percent.round(2),
        }
    )

    print(summary)

    positive_rate = (df["y"] == "yes").mean() * 100
    print(f"\nPositive class rate (y = 'yes'): {positive_rate:.2f}%")

    # ------------------------------------------------------------------
    # Duration (known target leakage)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DURATION FEATURE")
    print("=" * 80)

    print("Exists:", "duration" in df.columns)

    if "duration" in df.columns:
        print(df["duration"].describe())

        print(
            "\nNOTE: 'duration' is a known target leakage feature because "
            "it is only available after the marketing call has completed. "
            "It must be excluded from any production training pipeline."
        )

    # ------------------------------------------------------------------
    # pdays exploration
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PDAYS ANALYSIS")
    print("=" * 80)

    sentinel_count = (df["pdays"] == 999).sum()
    real_count = (df["pdays"] != 999).sum()

    print(f"Rows with pdays == 999 : {sentinel_count:,}")
    print(f"Rows with real pdays   : {real_count:,}")

    print("\nTop pdays values:")
    print(df["pdays"].value_counts().head(10))

    # ------------------------------------------------------------------
    # Unknown categorical values
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("'UNKNOWN' VALUE ANALYSIS")
    print("=" * 80)

    categorical_columns = df.select_dtypes(include="object").columns

    unknown_summary = []

    for column in categorical_columns:
        count = (df[column] == "unknown").sum()

        if count > 0:
            unknown_summary.append(
                {
                    "column": column,
                    "unknown_count": count,
                    "percentage": round(count / len(df) * 100, 2),
                }
            )

    unknown_df = (
        pd.DataFrame(unknown_summary)
        .sort_values("unknown_count", ascending=False)
        .reset_index(drop=True)
    )

    if unknown_df.empty:
        print("No categorical columns contain 'unknown'.")
    else:
        print(unknown_df)

        largest = unknown_df.iloc[0]

        print("\nLargest source of 'unknown' values:")
        print(f"  Column     : {largest['column']}")
        print(f"  Count      : {largest['unknown_count']:,}")
        print(f"  Percentage : {largest['percentage']}%")

    # ------------------------------------------------------------------
    # True missing values
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MISSING VALUES")
    print("=" * 80)

    missing = df.isna().sum()

    if missing.sum() == 0:
        print("No true NaN values detected.")
    else:
        print(missing[missing > 0])


if __name__ == "__main__":
    main()