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

    # Select features and train the final model.
    model, selected_features = train_model(
        X_train,
        y_train,
        random_state=settings.random_state,
    )

    # Keep the same selected feature names and order.
    X_test_selected = X_test.loc[:, selected_features]

    results = evaluation_model(
        model=model,
        X_test=X_test_selected,
        y_test=y_test,
    )

    logger.info("Selected features: %s", selected_features)
    logger.info("Feature count: %d", len(selected_features))
    logger.info("Test accuracy: %.4f", results["accuracy"])
    logger.info("Confusion matrix: %s", results["confusion_matrix"])

    # Save the fitted model and its selected feature names.
    save_model(
        model,
        settings.model_path,
        selected_features=selected_features,
    )

    logger.info("Model saved to: %s", settings.model_path)
    logger.info("Application completed successfully")


def predict_main() -> None:
    """Load the saved model and predict without retraining."""
    settings = load_settings()
    setup_logging(settings.log_level)

    logger.info("Starting CLI prediction")

    # Load the existing trained pipeline.
    model, selected_features = load_model(settings.model_path)

    # Prepare a sample using only the selected features.
    X, _ = load_data()
    sample = X.iloc[[0]].loc[:, selected_features]

    # Generate a prediction using the saved model.
    prediction = predict(model, sample)

    predicted_class = int(prediction[0])
    class_labels = {0: "malignant", 1: "benign"}

    print("Predicted class:", predicted_class)
    print("Predicted label:", class_labels[predicted_class])

    logger.info("CLI prediction completed successfully")


if __name__ == "__main__":
    main()
