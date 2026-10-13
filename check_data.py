from production_app.config import load_settings
from production_app.data import load_data, split_data

settings = load_settings()

X, y = load_data()

X_train, X_test, y_train, y_test = split_data(
    X=X,
    y=y,
    test_size=settings.test_size,
    random_state=settings.random_state,
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
