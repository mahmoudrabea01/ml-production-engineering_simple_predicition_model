import logging as log
from pathlib import Path

import joblib as jl
import pandas as pd
from sklearn.pipeline import Pipeline

logger = log.getLogger(__name__)


def save_model(model: Pipeline, model_path: Path) -> None:
    """Save a fitted model pipeline to disk."""

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        jl.dump(model, model_path)
    except OSError:

        logger.exception("Could not save model to %s", model_path)

        raise

    logger.info("Model saved to %s", model_path)


def load_model(model_path: Path) -> Pipeline:
    """Load a previously saved model pipeline."""

    try:
        model = jl.load(model_path)

    except (OSError, ValueError, EOFError):
        logger.exception("Could not load model from %s", model_path)

        raise

    if not isinstance(model, Pipeline):
        raise TypeError("The saved object is not a scikit-learn Pipeline.")

    logger.info("Model loaded from %s", model_path)

    return model


def predict(model: Pipeline, features: pd.DataFrame) -> list:
    """Generate predictions for input features."""

    predictions = model.predict(features)

    return predictions.tolist()
