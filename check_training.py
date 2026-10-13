from production_app.config import load_settings
from production_app.data import load_data, split_data
from production_app.training import train_model

settings = load_settings()


X, y = load_data()

X_train, X_test, y_train, y_test = split_data(
    X=X, y=y, test_size=settings.test_size, random_state=settings.random_state
)

model = train_model(X_train, y_train)

predictions = model.predict(X_test)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Predictions:", predictions[:10])
print("Model type:", type(model).__name__)
