print("Script started")

from production_app.config import load_settings

settings = load_settings()

print("Environment:", settings.app_env)
print("Model path:", settings.model_path)
print("Log level:", settings.log_level)
print("Test size:", settings.test_size)
print("Random state:", settings.random_state)
