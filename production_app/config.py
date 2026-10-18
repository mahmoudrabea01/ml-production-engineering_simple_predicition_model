# we centralize them in one configuration module.

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Locate the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load local environment variables.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    app_env: str
    model_path: Path
    log_level: str
    test_size: float
    random_state: int


def load_settings() -> Settings:
    """Load and validate application settings."""

    try:
        test_size = float(os.getenv("TEST_SIZE", "0.2"))
        random_state = int(os.getenv("RANDOM_STATE", "48"))
    except ValueError as exc:
        raise ValueError(
            "TEST_SIZE must be a number and RANDOM_STATE must be an integer."
        ) from exc

    if not 0 < test_size < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1.")

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(f"Invalid LOG_LEVEL: {log_level}")

    model_path = Path(
        os.getenv(
            "MODEL_PATH",
            "model/breast_cancer_pipeline.joblib",
        )
    )

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        model_path=model_path,
        log_level=log_level,
        test_size=test_size,
        random_state=random_state,
    )
