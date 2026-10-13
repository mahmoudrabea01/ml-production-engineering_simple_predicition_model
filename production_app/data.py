import logging as log

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

logger = log.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Breast Cancer dataset and separate features from target."""

    logger.info("Loading Breast Cancer dataset")

    dataset = load_breast_cancer(as_frame=True)
    dataframe = dataset.frame

    X = dataframe.drop(columns=["target"])
    y = dataframe["target"]

    logger.info(
        "Dataset loaded: %s rows, %s features",
        X.shape[0],
        X.shape[1],
    )

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split features and target into training and testing sets."""
    logger.info("Splitting dataset into training and testing sets")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Split complete: %s training rows, %s testing rows",
        X_train.shape[0],
        X_test.shape[0],
    )

    return X_train, X_test, y_train, y_test
