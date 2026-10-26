import logging as log

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from production_app.feature_selection import select_features
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
    random_state: int = 48,
) -> tuple[Pipeline, list[str]]:
    """Select features and train the final classification model."""

    logger.info("Starting feature selection")

    selected_features = select_features(
        X_train,
        y_train,
        random_state=random_state,
    )

    # Train only on selected features.
    X_train_scaled = X_train.loc[:, selected_features]

    logger.info(
        "Training model using %d selected features",
        len(selected_features),
    )

    model = build_model()
    model.fit(X_train_scaled, y_train)

    logger.info("Model training completed")

    return model, selected_features
