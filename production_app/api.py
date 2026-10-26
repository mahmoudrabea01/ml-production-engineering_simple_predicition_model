import logging as log

from fastapi import FastAPI, HTTPException

from production_app.config import load_settings
from production_app.logging_config import setup_logging
from production_app.model_service import load_model, predict
from production_app.schemas import PredictionRequest, PredictionResponse

logger = log.getLogger(__name__)

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for predictions using a saved machine learning pipeline.",
    version="1.0.0",
)

settings = load_settings()
setup_logging(settings.log_level)


model = None

try:
    model = load_model(settings.model_path)
except Exception:
    logger.exception("Failed to load the saved model")


@app.get("/health")
def health_check() -> dict:
    """Report whether the API model is available."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is unavailable because the model could not be loaded.",
        )

    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def prediction_endpoint(request: PredictionRequest) -> dict:
    """Predict the class for one validated observation."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is unavailable.",
        )

    try:
        features = request.to_dataframe()
        prediction = predict(model, features)

    except (ValueError, TypeError):
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed.",
        ) from None

    predicted_class = int(prediction[0])

    class_labels = {
        0: "malignant",
        1: "benign",
    }

    if predicted_class not in class_labels:
        logger.error("Unexpected prediction class: %s", predicted_class)
        raise HTTPException(
            status_code=500,
            detail="Prediction returned an unexpected class.",
        )

    return {
        "predicted_class": predicted_class,
        "predicted_label": class_labels[predicted_class],
    }
