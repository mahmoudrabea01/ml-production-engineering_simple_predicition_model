import logging as log

from production_app.config import load_settings
from production_app.data import load_data, split_data
from production_app.evaluation import evaluation_model
from production_app.logging_config import setup_logging
from production_app.model_service import load_model, predict, save_model
from production_app.training import train_model

logger = log.getLogger(__name__)


def main() -> None:
    """Run the complete machine learning workflow."""
    settings = load_settings()
    setup_logging(settings.log_level)

    logger.info("Application started")
    logger.info("Environment: %s", settings.app_env)

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y=y,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    model = train_model(X_train, y_train)

    results = evaluation_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    logger.info("Test accuracy: %.4f", results["accuracy"])
    logger.info("Confusion matrix: %s", results["confusion_matrix"])

    print("Model save path:", settings.model_path.resolve())
    save_model(model, settings.model_path)

    logger.info("Application completed successfully")


def predict_main() -> None:
    """Load the saved model and predict without retraining."""
    settings = load_settings()
    setup_logging(settings.log_level)

    logger.info("Starting CLI prediction")

    # Load the existing trained pipeline.
    model = load_model(settings.model_path)

    # Prepare one sample with the original feature names.
    X, _ = load_data()
    sample = X.iloc[[0]]

    # Generate a prediction using the saved model.
    prediction = predict(model, sample)

    predicted_class = int(prediction[0])
    class_labels = {0: "malignant", 1: "benign"}

    print("Predicted class:", predicted_class)
    print("Predicted label:", class_labels[predicted_class])

    logger.info("CLI prediction completed successfully")


if __name__ == "__main__":
    main()
