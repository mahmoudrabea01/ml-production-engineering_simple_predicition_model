# Breast Cancer Classification — ML Production Engineering

## Project Overview

This project demonstrates how to organize a machine learning workflow into a reusable Python application.

The project uses the Breast Cancer Wisconsin dataset and a Logistic Regression classifier. It covers data preparation, model training, evaluation, model persistence, and prediction through a command-line application and a Streamlit web interface.

**Disclaimer:** This project is for educational purposes only. It is not a medical diagnostic tool.

## Features

- Load and inspect the Breast Cancer Wisconsin dataset.
- Split the data into training and testing sets.
- Build a preprocessing and classification pipeline using `RobustScaler` and `LogisticRegression`.
- Train and evaluate the model.
- Save and reload the fitted pipeline using `joblib`.
- Generate predictions and class probabilities.
- Configure application settings through environment variables.
- Use logging and exception handling.
- Provide an interactive web interface using Streamlit.
- Apply code-quality tools such as Ruff and Black.

## Project Structure

```text
ml-production-project/
├── notebooks/
│   └── 01_exploration.ipynb
├── production_app/
│   ├── __init__.py
│   ├── config.py
│   ├── logging_config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   ├── model_service.py
│   └── main.py
├── models/
├── streamlit_app.py
├── run.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Technologies

- Python 3.12
- uv
- pandas and NumPy
- scikit-learn
- joblib
- Streamlit
- python-dotenv
- Ruff and Black

## Getting Started

### 1. Prerequisites

Install Python 3.12, Git, and uv.

### 2. Clone the Repository

Replace the repository URL with the URL of your GitHub repository.

```bash
git clone https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
cd <REPO_NAME>
```

### 3. Install Dependencies

```bash
uv sync
```

### 4. Configure Environment Variables

On Windows PowerShell, create a local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

The application settings include:

| Variable | Description | Default |
|---|---|---|
| `APP_ENV` | Application environment | `development` |
| `MODEL_PATH` | Saved model location | `models/breast_cancer_pipeline.joblib` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `TEST_SIZE` | Test-set proportion | `0.2` |
| `RANDOM_STATE` | Random seed | `42` |

### 5. Train and Evaluate the Model

Run the main application:

```bash
uv run python run.py
```

This runs the training and evaluation workflow, saves the fitted pipeline, reloads it, and demonstrates prediction.

The saved model is configured to use:

```text
models/breast_cancer_pipeline.joblib
```

### 6. Launch the Streamlit Interface

After the model has been generated, start the web application:

```bash
uv run streamlit run streamlit_app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

The interface allows users to enter feature values, request a prediction, and view the predicted class probabilities.

The default input values are based on feature medians from the example dataset. They are provided for demonstration and do not represent an actual patient's measurements.

## Model Workflow

1. Load the dataset.
2. Inspect the data and separate features from the target.
3. Split the data into training and testing sets.
4. Build a scikit-learn pipeline containing `RobustScaler` and `LogisticRegression`.
5. Fit the pipeline using the training data.
6. Evaluate predictions on the test set.
7. Save and reload the complete fitted pipeline.
8. Generate predictions through the application and Streamlit interface.

Using a pipeline keeps preprocessing and model prediction together and helps prevent data leakage when preprocessing is fitted only on the training data.

## Code Quality

Check the project with Ruff:

```bash
uv run ruff check .
```

Format Python files with Black:

```bash
uv run black .
```

Check formatting without modifying files:

```bash
uv run black --check .
```

## Notes

- Do not commit your local `.env` file or virtual environment.
- Run the training workflow before launching Streamlit if the saved model does not exist.
- This project is intended to demonstrate machine learning application structure and deployment preparation, not clinical use.
