import pandas as pd

DSM5_MAPPING = {
    "ADHD_Inattention": [
        "ADHDind_2324",
        "ADHDSevInd_2324",
        "SchoolReadiness_2324"
    ],

    "ADHD_Hyperactivity": [
        "MEDB10ScrQ5_2324",
        "ADHDBehTreat_2324"
    ],

    "ASD_SocialCommunication": [
        "MakeFriend_2324",
        "bullied_2324"
    ],

    "ASD_RestrictedRepetitive": [
        "AutismInd_2324",
        "ASDSevInd_2324"
    ]
}


def map_dsm5_domains(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract DSM-5 domain features from the NSCH dataset.
    """

    selected_columns = []

    for features in DSM5_MAPPING.values():
        selected_columns.extend(features)

    selected_columns = list(dict.fromkeys(selected_columns))

    missing = [c for c in selected_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing DSM-5 variables: {missing}"
        )

    return df[selected_columns].copy()

