import logging as log

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

logger = log.getLogger(__name__)


def evaluation_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a fitted model using the test dataset."""
    logger.info("Starting model evaluation")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=["malignant", "benign"],
        output_dict=True,
        zero_division=0,
    )

    results = {
        "accuracy": accuracy,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    logger.info("Model evaluation completed: accuracy=%.4f", accuracy)

    return results
