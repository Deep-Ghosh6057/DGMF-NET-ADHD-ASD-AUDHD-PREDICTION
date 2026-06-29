import pandas as pd


def fuse_features(dsm5_df, embedding_df):
    """
    Concatenate DSM-5 features and graph embeddings.
    """

    fused_df = pd.concat(
        [
            dsm5_df.reset_index(drop=True),
            embedding_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return fused_df