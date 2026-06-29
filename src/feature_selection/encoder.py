from sklearn.preprocessing import LabelEncoder


def encode_target(y):
    """
    Encode class labels into integers.

    Returns
    -------
    y_encoded : ndarray
    encoder : LabelEncoder
    """

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    return y_encoded, encoder


def decode_target(y_encoded, encoder):
    """
    Convert integer labels back to class names.
    """

    return encoder.inverse_transform(y_encoded)