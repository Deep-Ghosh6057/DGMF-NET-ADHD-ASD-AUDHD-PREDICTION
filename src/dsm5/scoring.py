import numpy as np
import pandas as pd

SPECIAL_MISSING = [90, 95, 96, 99]


def compute_dsm5_scores(dsm5_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute DSM-5 domain scores from mapped features.
    """

    df = dsm5_df.copy()

    # Replace NSCH special missing codes with NaN
    df = df.replace(SPECIAL_MISSING, np.nan)

    scores = pd.DataFrame(index=df.index)

    scores["ADHD_Inattention_Score"] = df[
        ["ADHDind_2324", "ADHDSevInd_2324", "SchoolReadiness_2324"]
    ].mean(axis=1, skipna=True)

    scores["ADHD_Hyperactivity_Score"] = df[
        ["MEDB10ScrQ5_2324", "ADHDBehTreat_2324"]
    ].mean(axis=1, skipna=True)

    scores["ASD_SocialCommunication_Score"] = df[
        ["MakeFriend_2324", "bullied_2324"]
    ].mean(axis=1, skipna=True)

    scores["ASD_RestrictedRepetitive_Score"] = df[
        ["AutismInd_2324", "ASDSevInd_2324"]
    ].mean(axis=1, skipna=True)

    # Fill missing values using the median of each score
    scores = scores.fillna(scores.median())

    return scores


