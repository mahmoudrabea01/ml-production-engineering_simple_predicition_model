from production_app.config import load_settings
from production_app.data import load_data, split_data
from production_app.model_service import load_model, predict, save_model
from production_app.training import train_model

settings = load_settings()

X, y = load_data()

X_train, X_test, y_train, y_test = split_data(
    X=X,
    y=y,
    test_size=settings.test_size,
    random_state=settings.random_state,
)


# Train the pipeline.
model = train_model(X_train, y_train)

# Save the fitted pipeline.
save_model(model, settings.model_path)

# Load it back from disk.
loaded_model = load_model(settings.model_path)

# Predict using the loaded pipeline.
predictions = predict(loaded_model, X_test.iloc[:5])

print("Saved model:", settings.model_path)
print("Predictions:", predictions)
print("Number of predictions:", len(predictions))
