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

    loaded_model = load_model(settings.model_path)

    sample = X_test.iloc[[0]]
    predictions = predict(loaded_model, sample)

    logger.info("Sample prediction: %s", predictions[0])
    logger.info("Application completed successfully")


if __name__ == "__main":
    main()
