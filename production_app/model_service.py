import logging as log
from pathlib import Path

import joblib as jl
import pandas as pd
from sklearn.pipeline import Pipeline

logger = log.getLogger(__name__)


def save_model(model: Pipeline, model_path: Path, selected_features: list[str]) -> None:
    """Save the fitted pipeline and selected feature names."""

    artifact = {
        "model": model,
        "selected_features": selected_features,
    }

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        jl.dump(artifact, model_path)

    except OSError:
        logger.exception("Could not save model to %s", model_path)
        raise

    logger.info("Model saved to %s", model_path)


def load_model(
    model_path: Path,
) -> tuple[Pipeline, list[str]]:
    """Load the fitted pipeline and selected feature names."""

    try:
        artifact = jl.load(model_path)
        print("DEBUG - Model path:", model_path.resolve())
        print("DEBUG - Artifact type:", type(artifact))
        print(
            "DEBUG - Artifact keys:",
            list(artifact.keys()) if isinstance(artifact, dict) else "Not a dictionary",
        )

    except (OSError, ValueError, EOFError):
        logger.exception("Could not load model from %s", model_path)
        raise

    # if not isinstance(artifact, Pipeline):
    #     raise TypeError("The saved artifact must be a dictionary.")

    if not isinstance(artifact, dict):
        raise TypeError("The saved artifact must be a dictionary.")

    model = artifact.get("model")
    selected_features = artifact.get("selected_features")

    if not isinstance(model, Pipeline):
        raise TypeError("The saved model is not a scikit-learn Pipeline.")

    if (
        not isinstance(selected_features, list)
        or not selected_features
        or not all(isinstance(feature, str) for feature in selected_features)
        or len(set(selected_features)) != len(selected_features)
    ):
        raise ValueError("The saved feature list is invalid.")

    if model.n_features_in_ != len(selected_features):
        raise ValueError("The saved feature list does not match the trained model.")

    logger.info("Model loaded from %s", model_path)
    logger.info("Loaded %d selected features", len(selected_features))

    return model, selected_features


def predict(model: Pipeline, features: pd.DataFrame) -> list:
    """Generate predictions using the fitted pipeline."""

    predictions = model.predict(features)

    return predictions.tolist()
