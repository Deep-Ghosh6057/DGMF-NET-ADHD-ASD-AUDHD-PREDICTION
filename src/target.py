import pandas as pd
from pathlib import Path

from src.config import PROJECT_ROOT


def create_target_dataset(df):
    """
    Create four-class target for ADHD / ASD prediction.

    Classes
    -------
    0 : Healthy
    1 : ADHD
    2 : ASD
    3 : AuDHD
    """

    print("=" * 90)
    print("SECTION 7 : TARGET CREATION")
    print("=" * 90)

    target_df = df.copy()

    # --------------------------------------------------------
    # Keep only valid records
    # --------------------------------------------------------

    valid_codes = [1, 2, 3]

    target_df = target_df[
        target_df["ADHDind_2324"].isin(valid_codes)
        &
        target_df["AutismInd_2324"].isin(valid_codes)
    ].copy()

    print(f"Valid Samples : {len(target_df):,}")

    # --------------------------------------------------------
    # Create Target
    # --------------------------------------------------------

    def assign_target(row):

        adhd = row["ADHDind_2324"] == 3
        autism = row["AutismInd_2324"] == 3

        if (not adhd) and (not autism):
            return 0

        elif adhd and (not autism):
            return 1

        elif (not adhd) and autism:
            return 2

        else:
            return 3

    target_df["TARGET"] = target_df.apply(assign_target, axis=1)

    class_names = {
        0: "Healthy",
        1: "ADHD",
        2: "ASD",
        3: "AuDHD"
    }

    target_df["TARGET_NAME"] = target_df["TARGET"].map(class_names)

    print("\nTarget Distribution")
    print("-" * 90)

    print(target_df["TARGET_NAME"].value_counts())

    # --------------------------------------------------------
    # Save Dataset
    # --------------------------------------------------------

    save_path = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "NSCH_Target_Dataset.csv"
    )

    target_df.to_csv(save_path, index=False)

    print("\nDataset Saved")

    print(save_path)

    return target_df



