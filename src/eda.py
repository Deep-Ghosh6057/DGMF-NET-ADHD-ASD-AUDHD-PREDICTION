"""
===============================================================================
DGMF-Net : Exploratory Data Analysis (EDA)
===============================================================================

"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGURE_DIR = PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def dataset_overview(df):
    """
    Dataset overview
    """

    print("=" * 90)
    print("SECTION 6 : EXPLORATORY DATA ANALYSIS")
    print("=" * 90)

    print("\nDataset Shape")
    print("-" * 90)
    print(df.shape)

    print("\nNumber of Features :", df.shape[1])
    print("Number of Samples :", df.shape[0])

    numeric = df.select_dtypes(include="number").columns

    categorical = df.select_dtypes(exclude="number").columns

    print("\nNumeric Features :", len(numeric))
    print("Categorical Features :", len(categorical))

    print("\nMemory Usage")

    memory = df.memory_usage(deep=True).sum() / (1024**2)

    print(f"{memory:.2f} MB")

    print("\nTop 20 Columns")

    print(df.columns[:20].tolist())

    print("=" * 90)

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric": len(numeric),
        "categorical": len(categorical)
    }





import matplotlib.pyplot as plt


def missing_value_analysis(df):
    """
    Missing value analysis.
    """

    print("\n" + "=" * 90)
    print("MISSING VALUE ANALYSIS")
    print("=" * 90)

    # ----------------------------------------------------------
    # Missing Value Count
    # ----------------------------------------------------------

    missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(ascending=False)

    missing_percent = (missing / len(df) * 100).round(2)

    missing_table = pd.DataFrame({
        "Missing Values": missing,
        "Percentage (%)": missing_percent
    })

    # ----------------------------------------------------------
    # Save Table
    # ----------------------------------------------------------

    table_dir = PROJECT_ROOT / "outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    table_path = table_dir / "missing_values.csv"

    missing_table.to_csv(table_path)

    print(f"\nMissing Value Table Saved:")
    print(table_path)

    # ----------------------------------------------------------
    # Display Top 20
    # ----------------------------------------------------------

    print("\nTop 20 Missing Columns")

    print(missing_table.head(20))

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------

    if len(missing_table) > 0:

        plt.figure(figsize=(12,6))

        missing_table.head(20)["Missing Values"].plot(kind="bar")

        plt.title("Top 20 Missing Value Columns")

        plt.ylabel("Missing Count")

        plt.tight_layout()

        figure_path = FIGURE_DIR / "missing_values.png"

        plt.savefig(figure_path, dpi=300)

        plt.show()

        print("\nFigure Saved:")
        print(figure_path)

    else:

        print("\nNo Missing Values Found.")

    return missing_table


import matplotlib.pyplot as plt


def plot_numeric_distribution(df, column_name):
    """
    Generic Numeric Feature Distribution Analysis
    """

    print("\n" + "=" * 90)
    print(f"{column_name} ANALYSIS")
    print("=" * 90)

    if column_name not in df.columns:
        print(f"{column_name} column not found.")
        return

    data = df[column_name]

    # ----------------------------------------------------------
    # Summary Statistics
    # ----------------------------------------------------------

    print("\nSummary Statistics")
    print("-" * 90)
    print(data.describe())

    print("\nMissing Values")
    print(data.isnull().sum())

    # ----------------------------------------------------------
    # Histogram
    # ----------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(data.dropna(), bins=30)

    plt.title(f"{column_name} Distribution")

    plt.xlabel(column_name)

    plt.ylabel("Frequency")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    histogram_path = FIGURE_DIR / f"{column_name.lower()}_distribution.png"

    plt.savefig(histogram_path, dpi=300)

    plt.show()

    print("\nHistogram Saved:")
    print(histogram_path)

    # ----------------------------------------------------------
    # Boxplot
    # ----------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.boxplot(data.dropna(), vert=True)

    plt.title(f"{column_name} Boxplot")

    plt.ylabel(column_name)

    plt.tight_layout()

    boxplot_path = FIGURE_DIR / f"{column_name.lower()}_boxplot.png"

    plt.savefig(boxplot_path, dpi=300)

    plt.show()

    print("\nBoxplot Saved:")
    print(boxplot_path)


def numerical_summary(df):
    """
    Generate summary statistics for all numerical features.
    """

    print("\n" + "=" * 90)
    print("NUMERICAL FEATURE SUMMARY")
    print("=" * 90)

    numeric_df = df.select_dtypes(include="number")

    summary = pd.DataFrame({
        "Feature": numeric_df.columns,
        "Data Type": numeric_df.dtypes.astype(str).values,
        "Missing Values": numeric_df.isnull().sum().values,
        "Missing (%)": (
            numeric_df.isnull().sum() / len(df) * 100
        ).round(2).values,
        "Mean": numeric_df.mean().values,
        "Median": numeric_df.median().values,
        "Std": numeric_df.std().values,
        "Min": numeric_df.min().values,
        "Max": numeric_df.max().values,
        "Unique Values": numeric_df.nunique().values
    })

    summary = summary.sort_values(
        by="Missing (%)",
        ascending=False
    )

    table_dir = PROJECT_ROOT / "outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    save_path = table_dir / "numerical_summary.csv"

    summary.to_csv(save_path, index=False)

    print(f"\nSummary saved to:\n{save_path}")

    print("\nTop 20 Features")

    print(summary.head(20))

    return summary    