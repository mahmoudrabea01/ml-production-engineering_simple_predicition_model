
# Breast Cancer Classification — ML Production Engineering

## Project Overview

This project demonstrates how to structure a machine learning workflow into a reusable Python application and expose a trained model through a validated REST API.

The project uses the Breast Cancer Wisconsin dataset and a Logistic Regression classifier. It covers data preparation, preprocessing, model training, evaluation, model persistence, API development, input validation, logging, and an interactive Streamlit frontend.

The fitted scikit-learn pipeline is saved and reused for predictions. The API does not retrain the model when a prediction request is received.

**Disclaimer:** This project is for educational purposes only. It is not a medical diagnostic tool.

## Features

- Load and inspect the Breast Cancer Wisconsin dataset.
- Split data into training and testing sets.
- Build a machine learning pipeline using `RobustScaler` and `LogisticRegression`.
- Train and evaluate the model.
- Save and load the fitted pipeline using `joblib`.
- Generate predictions using the saved model.
- Configure application settings through environment variables.
- Use structured logging and exception handling.
- Expose predictions through a REST API built with FastAPI.
- Validate prediction requests using Pydantic.
- Provide a health endpoint to report model availability.
- Provide an interactive Streamlit frontend that communicates with the API.
- Apply code-quality checks using Ruff, Black, and pre-commit.

## Project Structure

```text
ml-production-project/
├── notebooks/
│   └── 01_exploration.ipynb
├── production_app/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── logging_config.py
│   ├── main.py
│   ├── model_service.py
│   ├── preprocessing.py
│   ├── schemas.py
│   └── training.py
├── model/
│   └── breast_cancer_pipeline.joblib
├── streamlit_app.py
├── run.py
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
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
- FastAPI
- Pydantic
- Uvicorn
- Requests
- Streamlit
- python-dotenv
- Ruff
- Black
- pre-commit

## Getting Started

### 1. Prerequisites

Install Python compatible with the project's Python requirement, Git, and uv.

### 2. Clone the Repository

```bash
git clone https://github.com/mahmoudrabea01/ml-production-engineering_simple_predicition_model.git
cd ml-production-engineering_simple_predicition_model
```

### 3. Install Dependencies

Install the project dependencies using the committed lockfile:

```bash
uv sync --locked
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
| `MODEL_PATH` | Saved model location | `model/breast_cancer_pipeline.joblib` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `TEST_SIZE` | Test-set proportion | `0.2` |
| `RANDOM_STATE` | Random seed | `42` |

Keep `.env` local and do not commit secrets or environment-specific configuration.

### 5. Train and Evaluate the Model

If the saved model artifact does not exist, run the training workflow:

```bash
uv run python run.py
```

This command loads the dataset, prepares the features and target, splits the data, trains and evaluates the pipeline, and saves the fitted model to the configured path.

The default model path is:

```text
model/breast_cancer_pipeline.joblib
```

Training and prediction are separate operations. The API loads the saved model rather than training a new model for each request.

## Running the Application

The application consists of a FastAPI backend and a Streamlit frontend. Run them in two separate terminals from the project root.

### 1. Start the FastAPI Server

In the first terminal, run:

```bash
uv run uvicorn production_app.api:app --reload
```

The API will usually be available at:

http://127.0.0.1:8000

### 2. Check API Health

Open the health endpoint:

http://127.0.0.1:8000/health

When the model is loaded successfully, the endpoint returns a response similar to:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

If the model cannot be loaded, the endpoint returns HTTP `503 Service Unavailable`.

### 3. Explore the API Documentation

Open the interactive Swagger UI:

http://127.0.0.1:8000/docs

Use this page to inspect the available endpoints, submit prediction requests, and review their responses.

### 4. Start the Streamlit Frontend

In a second terminal, run:

```bash
uv run streamlit run streamlit_app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

Enter the feature values and submit the form. Streamlit sends the input data to the FastAPI prediction endpoint and displays the returned class and label.

The frontend requires the API server to be running.

## API Endpoints

### `GET /health`

Reports whether the API can access the loaded prediction model.

- **HTTP 200:** The model is available.
- **HTTP 503:** The model could not be loaded, so prediction service is unavailable.

### `POST /predict`

Accepts one observation containing all 30 features expected by the trained pipeline.

The request body uses JSON, with feature names represented in `snake_case`, such as `mean_radius` and `mean_texture`.

Pydantic validates the input before the prediction function is called. Missing fields, invalid values, and unexpected extra fields are rejected.

A successful response has the following structure:

```json
{
  "predicted_class": 1,
  "predicted_label": "benign"
}
```

The predicted class is `0` for malignant or `1` for benign. These labels describe the model's output for this educational dataset; they are not a medical diagnosis.

### Error Handling

| HTTP status | Meaning |
|---|---|
| `200` | Request processed successfully |
| `422` | Request validation failed |
| `500` | Prediction could not be completed |
| `503` | Prediction model is unavailable |

Error responses provide a client-facing message. Internal model-loading details and diagnostic logs should remain on the server.

## Model Workflow

### Training Workflow

1. Load the dataset.
2. Separate features from the target.
3. Split the data into training and testing sets.
4. Build a scikit-learn pipeline containing `RobustScaler` and `LogisticRegression`.
5. Fit the pipeline on the training data.
6. Evaluate the model on the test set.
7. Save the fitted pipeline using `joblib`.

### Prediction Workflow

1. The API loads the saved pipeline when the application starts.
2. The client submits feature values as a JSON request.
3. Pydantic validates the request.
4. The validated values are converted into a DataFrame with the expected feature names and order.
5. The existing fitted pipeline generates a prediction.
6. The API returns the predicted class and readable label as JSON.

The Streamlit frontend communicates with the API rather than loading the prediction model directly. This separation keeps the prediction logic in the backend and provides a clear interface between the frontend and model service.

Using a fitted pipeline keeps preprocessing and classification together. Fitting preprocessing only on training data also helps prevent data leakage.

## Model Evaluation

One recorded run produced the following test results:

| Metric | Result |
|---|---:|
| Test samples | 114 |
| Accuracy | 0.9737 (97.37%) |

Confusion matrix:

```text
[[42, 0],
 [3, 69]]
```

The matrix follows scikit-learn's convention: rows represent true labels and columns represent predicted labels. The class order is `[0, 1]`.

These results describe one recorded run. They should not be interpreted as evidence of clinical suitability or medical diagnostic performance.

## Code Quality

Check the project with Ruff:

```bash
uv run ruff check .
```

Check formatting without modifying files:

```bash
uv run black --check .
```

Format Python files when necessary:

```bash
uv run black .
```

Run all configured pre-commit hooks against project files:

```bash
uv run pre-commit run --all-files
```

Install the hooks to run them automatically before Git commits:

```bash
uv run pre-commit install
```

## Notes

- Do not commit your local `.env` file or virtual environment.
- Train the model before starting the API if the saved model artifact does not exist.
- The saved model path can be customized using `MODEL_PATH`.
- Start the FastAPI server before using the Streamlit frontend.
- The API expects all 30 features used by the existing fitted pipeline.
- This project demonstrates machine learning application structure and production-oriented practices. It is not intended for clinical use.
