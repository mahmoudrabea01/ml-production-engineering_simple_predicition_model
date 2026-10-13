import logging as log

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from production_app.preprocessing import build_preprocessor

logger = log.getLogger(__name__)


def build_model() -> Pipeline:
    """Create the complete preprocessing and model pipeline."""

    return Pipeline(
        steps=[
            ("scaler", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the pipeline using training data."""

    logger.info("Starting model training")

    model = build_model()
    model.fit(X_train, y_train)

    logger.info("Model training completed")

    return model
