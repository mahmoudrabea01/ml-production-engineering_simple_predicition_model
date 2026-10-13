from production_app.config import load_settings
from production_app.data import load_data, split_data
from production_app.evaluation import evaluation_model
from production_app.training import train_model

settings = load_settings()

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

print("Accuracy:", round(results["accuracy"], 4))
print("Confusion Matrix:")
print(results["confusion_matrix"])

print("Malignant metrics:")
print(results["classification_report"]["malignant"])

print("Benign metrics:")
print(results["classification_report"]["benign"])
