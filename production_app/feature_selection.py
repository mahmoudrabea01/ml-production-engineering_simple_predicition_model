import logging as log

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from production_app.preprocessing import build_preprocessor

logger = log.getLogger(__name__)


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 48,
) -> list[str]:
    """Select features using L1-regularized Logistic Regression."""

    # Scale the training data before L1 feature selection.
    scaler = build_preprocessor()
    X_train_scaled = scaler.fit_transform(X_train)

    # L1 regularization can shrink coefficients to exactly zero.
    selector = SelectFromModel(
        estimator=LogisticRegression(
            l1_ratio=1.0,
            solver="liblinear",
            C=0.1,
            max_iter=1000,
            random_state=random_state,
        ),
        threshold=1e-8,
    )

    selector.fit(X_train_scaled, y_train)

    # Preserve the original DataFrame column order.
    selected_features = X_train.columns[selector.get_support()].tolist()

    if not selected_features:
        raise ValueError("Feature selection produced no features.")

    logger.info(
        "Selected %d out of %d features",
        len(selected_features),
        X_train.shape[1],
    )
    logger.info("Selected features: %s", selected_features)

    return selected_features
