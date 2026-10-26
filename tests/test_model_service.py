from pathlib import Path

import pandas as pd
import pytest

from production_app.config import load_settings
from production_app.model_service import load_model, predict


@pytest.fixture
def trained_model():
    """Load the saved model once for each test that needs it."""
    settings = load_settings()
    model, selected_features = load_model(settings.model_path)

    return model, selected_features


def test_model_loads_successfully(trained_model):
    """Verify that the model and selected features load correctly."""
    model, selected_features = trained_model

    assert model is not None
    assert len(selected_features) == 3
    assert model.n_features_in_ == len(selected_features)


def test_model_prediction(trained_model):
    """Verify that the model produces a valid classification."""
    model, selected_features = trained_model

    sample = pd.DataFrame(
        [[0.1471, 25.38, 0.2654]],
        columns=[
            "mean concave points",
            "worst radius",
            "worst concave points",
        ],
    )

    assert list(sample.columns) == selected_features

    prediction = predict(model, sample)

    assert isinstance(prediction, list)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_missing_model_file():
    """Verify that a missing model file raises an error."""
    missing_path = Path("nonexistent_test_model.joblib")

    with pytest.raises(FileNotFoundError):
        load_model(missing_path)
