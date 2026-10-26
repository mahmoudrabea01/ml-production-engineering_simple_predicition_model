
# Breast Cancer Classification — ML Production Engineering

## Project Overview

This project demonstrates how to transform an experimental machine learning workflow into a structured, reusable, and containerized Python application.

The project uses the Breast Cancer Wisconsin dataset and Logistic Regression to demonstrate:

- Data preprocessing and feature selection.
- Model training and evaluation.
- Model persistence using Joblib.
- Command-line training and prediction.
- REST API development with FastAPI.
- Input and output validation using Pydantic.
- Interactive frontend development using Streamlit.
- Docker containerization.
- Automated testing using pytest.
- Dependency management using uv.
- Code quality and version control.

The application separates training, model serving, and frontend responsibilities.

**Disclaimer:** This project is for educational purposes only. It is not a medical diagnostic tool.

---

## Features

- Load and inspect the Breast Cancer Wisconsin dataset.
- Split data into training and testing sets.
- Apply Embedded Feature Selection using L1-regularized Logistic Regression.
- Select relevant features without using the test dataset.
- Build a classification pipeline using MinMaxScaler and Logistic Regression.
- Train and evaluate the classification model.
- Save the fitted pipeline and selected feature names using Joblib.
- Generate predictions without retraining the model.
- Provide separate CLI commands for training and prediction.
- Expose prediction services through FastAPI.
- Validate API requests and responses using Pydantic.
- Reject missing, unexpected, and invalid numerical inputs.
- Provide API health monitoring.
- Connect Streamlit to FastAPI through HTTP requests.
- Containerize the FastAPI backend using Docker.
- Run automated model and API tests using pytest.
- Apply Ruff, Black, and pre-commit quality checks.
- Manage Python dependencies using uv and uv.lock.

---

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
│   ├── feature_selection.py
│   ├── logging_config.py
│   ├── main.py
│   ├── model_service.py
│   ├── preprocessing.py
│   ├── schemas.py
│   └── training.py
├── model/
│   └── breast_cancer_pipeline.joblib
├── tests/
│   ├── test_api.py
│   └── test_model_service.py
├── streamlit_app.py
├── run.py
├── Dockerfile
├── .dockerignore
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Technologies

### Machine Learning
- Python 3.12
- scikit-learn
- pandas
- NumPy
- Logistic Regression
- MinMaxScaler
- SelectFromModel
- Joblib

### Backend
- FastAPI
- Pydantic
- Uvicorn

### Frontend
- Streamlit
- Requests

### Production Engineering
- Docker
- uv
- Git
- GitHub
- python-dotenv
- Python logging

### Testing and Code Quality
- pytest
- httpx
- Ruff
- Black
- pre-commit

---

# Feature Selection

## Overview

The original Breast Cancer Wisconsin dataset contains:

| Property | Value |
|---|---|
| Observations | 569 |
| Original features | 30 |
| Target classes | 2 |
| Malignant | 0 |
| Benign | 1 |

Feature selection aims to identify a relevant subset of input features while reducing unnecessary complexity.

Potential benefits include:

- Reducing the number of model inputs.
- Simplifying model deployment.
- Reducing redundant information.
- Improving model interpretability.
- Potentially reducing computational requirements.

Feature selection does not automatically improve predictive accuracy. Its effectiveness must be evaluated experimentally.

## Feature Selection Methods

There are three main categories of feature selection.

### 1. Filter Methods

Filter methods evaluate features using statistical properties, generally without training the final prediction model.

Examples:

- Pearson correlation.
- Chi-square test.
- ANOVA F-test.
- Mutual information.

**Advantages:**
- Computationally inexpensive.
- Fast on high-dimensional datasets.
- Independent of the final estimator.

**Disadvantages:**
- Often evaluate features individually.
- May miss interactions between features.
- Do not directly optimize the final estimator's performance.

### 2. Wrapper Methods

Wrapper methods evaluate different subsets of features by repeatedly fitting a machine learning estimator.

Examples:

- Recursive Feature Elimination (RFE).
- Sequential Forward Selection.
- Sequential Backward Selection.
- RFECV.

**Advantages:**
- Consider the behavior of the selected estimator.
- Can evaluate feature subsets based on predictive performance.
- May produce strong feature subsets.

**Disadvantages:**
- Computationally more expensive.
- Require repeated model fitting.
- Can increase experimentation time.
- Require careful cross-validation to avoid overfitting during selection.

### 3. Embedded Methods

Embedded methods perform feature selection as part of fitting a machine learning estimator.

Examples:

- L1-regularized Logistic Regression.
- Lasso.
- L1-regularized Linear SVM.
- Tree-based feature importance with SelectFromModel.

**Advantages:**
- Feature selection is connected to model training.
- Usually requires fewer model fits than wrapper approaches.
- Can reduce unnecessary features.
- Integrates naturally into machine learning workflows.

**Disadvantages:**
- Depends on the selected estimator.
- May be sensitive to regularization strength.
- Can be unstable when features are highly correlated.
- Does not guarantee better test accuracy.

---

## Selected Method: Embedded Feature Selection

For this project, I chose **Embedded Feature Selection using L1-regularized Logistic Regression and SelectFromModel**.

### Why I Chose Embedded Feature Selection

**1. Dataset size and dimensionality**

The dataset contains only 30 features, rather than thousands of features.

Therefore, the computational advantage of very inexpensive statistical filtering is not essential for this project.

Filter methods remain valid for smaller datasets, but I preferred a model-aware selection approach rather than relying primarily on independent statistical scores.

**2. Computational resources**

Wrapper methods such as RFE and RFECV repeatedly train a model to evaluate feature subsets.

Although the dataset is relatively small, I wanted to avoid unnecessary repeated model fitting and keep the workflow computationally efficient.

Embedded methods offer a practical balance between computational cost and model-aware feature selection.

**3. Model simplicity and implementation**

The classification model is Logistic Regression, which supports regularization.

L1 regularization can shrink some coefficients to zero, making it suitable for selecting relevant features.

The method is straightforward to implement using scikit-learn and integrates naturally into the existing production workflow.

### Final Decision

Embedded Feature Selection was selected because it provides:

- Model-aware feature selection.
- Reasonable computational efficiency.
- Compatibility with Logistic Regression.
- Straightforward implementation.
- A smaller feature set for the API and frontend.
- A practical balance between simplicity and performance.

This decision reflects the engineering requirements of this project rather than a claim that Embedded methods are universally superior.

---

## Feature Selection Implementation

The implementation is located in:

`production_app/feature_selection.py`

The workflow uses:

1. MinMaxScaler for feature scaling.
2. Logistic Regression with L1 regularization.
3. SelectFromModel for selecting features based on nonzero coefficients.

### Selection Configuration

```python
selector = SelectFromModel(
    estimator=LogisticRegression(
        l1_ratio=1.0,
        solver="liblinear",
        C=0.1,
        max_iter=1000,
        random_state=48,
    ),
    threshold=1e-8,
)
```

### How It Works

**Step 1 — Split the dataset**

The dataset is divided into training and testing sets before feature selection.

**Step 2 — Fit the feature-selection scaler**

MinMaxScaler is fitted using training data only.

**Step 3 — Fit L1 Logistic Regression**

The estimator learns coefficients for the original features.

L1 regularization encourages some coefficients to become zero.

**Step 4 — Select features**

SelectFromModel identifies features whose coefficient magnitudes exceed the configured threshold.

**Step 5 — Train the final classifier**

The selected columns are used to fit the final MinMaxScaler and Logistic Regression pipeline.

**Step 6 — Evaluate**

The trained classifier is evaluated using the selected columns from the untouched test dataset.

### Preventing Data Leakage

Feature selection must not use information from the test set.

In this project:

- The dataset is split before feature selection.
- The feature-selection scaler is fitted on training data.
- The feature selector is fitted on training data.
- The final classification pipeline is fitted on selected training features.
- The test dataset is used only for final evaluation.

For future hyperparameter tuning or cross-validation, feature selection should be performed inside each training fold, ideally using a unified scikit-learn pipeline.

---

## Selected Features

The Embedded Feature Selection method selected 3 out of the original 30 features.

| Feature | Description |
|---|---|
| `mean concave points` | Mean number of concave portions of the contour |
| `worst radius` | Largest/worst radius measurement |
| `worst concave points` | Worst concave-point measurement |

### Feature Reduction

- Original features: 30
- Selected features: 3
- Removed features: 27
- Feature count reduction: 90%

The final prediction interface requires only these three features.

The selected feature names and their order are saved together with the trained model.

---

# Model Architecture

## Final Classification Pipeline

The final model uses:

```text
Selected Input Features (3)
          |
          v
     MinMaxScaler
          |
          v
  Logistic Regression
          |
          v
  Predicted Class
```

### Preprocessing

MinMaxScaler transforms features according to the minimum and maximum values learned during fitting.

This helps put features with different numerical scales into comparable ranges.

### Classification

Logistic Regression is used as the final binary classification estimator.

The final classifier is trained using the selected feature subset.

### Model Persistence

The model artifact is stored at:

`model/breast_cancer_pipeline.joblib`

The saved artifact contains:

```python
{
    "model": trained_pipeline,
    "selected_features": selected_features,
}
```

This allows the application to load both the fitted pipeline and the expected feature list.

---

# Model Evaluation

## Previous Baseline

A previous baseline experiment used all 30 features with RobustScaler and Logistic Regression.

| Metric | Result |
|---|---:|
| Features | 30 |
| Test accuracy | 97.37% |

Confusion matrix:

```text
[[42, 0],
 [ 3, 69]]
```

## Final Feature-Selected Model

The updated model uses MinMaxScaler and three selected features.

| Metric | Result |
|---|---:|
| Original features | 30 |
| Selected features | 3 |
| Test samples | 114 |
| Test accuracy | 96.49% |

Confusion matrix:

```text
[[39, 3],
 [ 1, 71]]
```

The class order is:

- 0 — malignant
- 1 — benign

Rows represent actual labels, and columns represent predicted labels.

### Evaluation Interpretation

The feature-selected model achieved 96.49% accuracy while using only three input features.

This demonstrates a substantial reduction in input dimensionality with relatively high classification accuracy.

However, the previous baseline used RobustScaler, while the updated model uses MinMaxScaler.

Therefore, the accuracy difference cannot be attributed solely to feature selection.

A controlled comparison using the same preprocessing, data split, and model settings is required to isolate the effect of feature selection.

The feature-selected model also produced three malignant cases incorrectly classified as benign in the recorded test run. This is an important limitation, especially in a medical context.

These evaluation results demonstrate an educational machine learning workflow, not clinical suitability.

---

# Getting Started

## 1. Prerequisites

Install:

- Python 3.12
- Git
- uv
- Docker Desktop (for containerized execution)

## 2. Clone the Repository

```bash
git clone https://github.com/mahmoudrabea01/ml-production-engineering_simple_predicition_model.git
cd ml-production-engineering_simple_predicition_model
```

## 3. Install Dependencies

Synchronize the Python environment using the committed lockfile:

```bash
uv sync --locked
```

## 4. Configure Environment Variables

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Application configuration:

| Variable | Description | Default |
|---|---|---|
| `APP_ENV` | Application environment | `development` |
| `MODEL_PATH` | Model artifact path | `model/breast_cancer_pipeline.joblib` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `TEST_SIZE` | Test proportion | `0.2` |
| `RANDOM_STATE` | Random seed | `48` |
| `API_URL` | Streamlit prediction endpoint | `http://127.0.0.1:8000/predict` |

Keep `.env` local and do not commit secrets.

---

# Command-Line Workflow

## Train and Evaluate

```bash
uv run python run.py train
```

The training workflow:

1. Loads the dataset.
2. Splits the dataset into training and testing sets.
3. Performs Embedded Feature Selection using training data.
4. Selects the relevant input columns.
5. Builds the final classification pipeline.
6. Fits the pipeline on selected training features.
7. Evaluates predictions on selected test features.
8. Saves the model and selected feature list using Joblib.

Training creates or overwrites the configured model artifact.

## Generate CLI Predictions

```bash
uv run python run.py predict
```

The prediction workflow:

1. Loads the saved artifact.
2. Retrieves the fitted model and selected features.
3. Prepares a sample observation.
4. Selects the required input columns in the correct order.
5. Generates a prediction.
6. Displays the predicted class and label.

CLI prediction does not retrain the model.

---

# Running FastAPI

Start the backend:

```bash
uv run uvicorn production_app.api:app --reload
```

API base URL:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

## GET /health

Checks whether the saved prediction model is available.

Successful response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

Expected status codes:

- HTTP 200 — Model is available.
- HTTP 503 — Model is unavailable.

## POST /predict

Accepts three numerical features.

Example request:

```json
{
  "mean_concave_points": 0.1471,
  "worst_radius": 25.38,
  "worst_concave_points": 0.2654
}
```

### Request Validation

Pydantic validates the request using `PredictionRequest`.

Requirements:

- All three features must be provided.
- `worst_radius` must be greater than zero.
- Concave-point features must be non-negative.
- Missing fields are rejected.
- Unexpected fields are rejected.
- NaN and Infinity are rejected.

### Response Schema

Successful responses follow `PredictionResponse`.

Example structure:

```json
{
  "predicted_class": 1,
  "predicted_label": "benign"
}
```

The actual prediction depends on the submitted values and saved model.

### API Error Handling

| Status | Meaning |
|---|---|
| 200 | Successful prediction |
| 422 | Invalid request |
| 500 | Prediction processing failure |
| 503 | Model unavailable |

FastAPI performs input validation and model inference.

The frontend does not directly execute the machine learning model.

---

# Running Streamlit

Start FastAPI first.

Then open another terminal and run:

```bash
uv run streamlit run streamlit_app.py
```

Streamlit is usually available at:

http://localhost:8501

The interface:

1. Displays the three selected features.
2. Provides editable numerical inputs.
3. Uses dataset medians as example values.
4. Sends an HTTP POST request to FastAPI.
5. Displays the predicted class and label.
6. Handles connection failures, timeouts, and API errors.

### Frontend and Backend Separation

```text
Streamlit Frontend
       |
       | HTTP POST /predict
       v
FastAPI Backend
       |
       | Pydantic Validation
       v
Saved ML Pipeline
       |
       v
Prediction Response
       |
       v
Streamlit Result
```

The Streamlit frontend communicates with FastAPI using Requests.

The saved model is loaded and executed by the backend.

---

# Docker Containerization

The FastAPI backend is packaged into a Docker image.

The image contains:

- Python runtime.
- uv dependency manager.
- Locked runtime dependencies.
- FastAPI application.
- Saved model artifact.
- Selected feature metadata.

Streamlit continues to run locally for this deployment configuration.

## Build the Docker Image

From the project root:

```bash
docker build -t breast-cancer-api:1.1 .
```

## Run the Container

```bash
docker run -d --name breast-cancer-container -p 8000:8000 breast-cancer-api:1.1
```

The `-p 8000:8000` option maps the container's port 8000 to port 8000 on the host.

## Verify the Container

```bash
docker ps
```

View logs:

```bash
docker logs breast-cancer-container
```

Open the API documentation:

http://localhost:8000/docs

Verify that `/health` and `/predict` respond correctly.

## Connect Streamlit to Docker

With the Docker container running:

```bash
uv run streamlit run streamlit_app.py
```

The Streamlit frontend communicates with the Dockerized API through:

`http://127.0.0.1:8000/predict`

## Stop the Container

```bash
docker stop breast-cancer-container
```

## Remove the Container

```bash
docker rm breast-cancer-container
```

Removing a container does not delete the original local trained model or Docker image.

### Docker Architecture

```text
Windows Host
|
|-- Streamlit Frontend
|      |
|      | HTTP Request
|      v
|-- Docker Container
       |
       |-- FastAPI
       |-- Pydantic
       |-- Saved ML Pipeline
       |-- Selected Features
       |
       v
   Prediction Response
```

---

# Automated Testing

Automated tests are implemented using pytest.

Test files:

```text
tests/
├── test_model_service.py
└── test_api.py
```

## Model Service Tests

The model-service tests verify:

- Successful model loading.
- Correct selected feature count.
- Compatibility between selected features and model inputs.
- Valid model predictions.
- Correct exception handling for missing model files.

## FastAPI Tests

The API tests verify:

- Healthy model endpoint returns HTTP 200.
- Valid prediction returns HTTP 200.
- Invalid prediction returns HTTP 422.
- Unavailable model returns HTTP 503.
- Health endpoint returns HTTP 503 when the model is unavailable.

## Safe Failure Simulation

The HTTP 503 tests use pytest's `monkeypatch` fixture.

The model reference is temporarily replaced with `None` to simulate an unavailable model.

This approach does not delete, rename, or modify the real model artifact.

After the test, monkeypatch restores the original reference.

## Run Model Tests

```bash
uv run python -m pytest tests/test_model_service.py -v
```

## Run API Tests

```bash
uv run python -m pytest tests/test_api.py -v
```

## Run All Tests

```bash
uv run python -m pytest tests/ -v
```

### Recorded Test Results

| Test Group | Passed |
|---|---:|
| Model service | 3 |
| FastAPI | 5 |
| Total | 8 |

All eight automated tests passed during local verification.

---

# Code Quality

The project uses Ruff, Black, and pre-commit to maintain consistent code quality.

## Ruff

```bash
uv run ruff check .
```

## Black

Check formatting:

```bash
uv run black --check .
```

Apply formatting:

```bash
uv run black .
```

## Pre-commit

Install hooks:

```bash
uv run pre-commit install
```

Run all configured hooks:

```bash
uv run pre-commit run --all-files
```

---

# Production Engineering Principles

This project demonstrates several production-oriented engineering practices.

## 1. Separation of Responsibilities

Training, prediction, API handling, and frontend presentation are separated into dedicated modules.

## 2. Reproducible Environments

uv and uv.lock manage Python dependencies.

Docker packages the backend and model into a reproducible execution environment.

## 3. Model Persistence

The trained model is saved and loaded instead of retraining for each prediction.

## 4. Feature Consistency

The selected feature list is saved alongside the trained model.

The API and frontend use the same three-feature contract.

## 5. Data Leakage Prevention

Feature selection and preprocessing are fitted using training data only.

## 6. API Validation

Pydantic validates incoming requests and successful responses.

## 7. Error Handling

The application handles invalid inputs, unavailable models, and connection failures.

## 8. Automated Testing

pytest verifies model functionality and API response behavior.

## 9. Version Control

Git and GitHub track source code, dependency definitions, configuration examples, and project documentation.

---

# Final Verification

Before committing or deploying a new version:

### 1. Synchronize Dependencies

```bash
uv sync --locked
```

### 2. Train the Model

```bash
uv run python run.py train
```

### 3. Verify CLI Prediction

```bash
uv run python run.py predict
```

### 4. Run Automated Tests

```bash
uv run python -m pytest tests/ -v
```

### 5. Check Code Quality

```bash
uv run ruff check .
uv run black --check .
uv run pre-commit run --all-files
```

### 6. Build the Docker Image

```bash
docker build -t breast-cancer-api:1.1 .
```

### 7. Run the Dockerized API

```bash
docker run -d --name breast-cancer-container -p 8000:8000 breast-cancer-api:1.1
```

### 8. Verify API Endpoints

Check:

- GET /health — HTTP 200.
- POST /predict with valid inputs — HTTP 200.
- POST /predict with invalid inputs — HTTP 422.
- Automated unavailable-model tests — HTTP 503.

### 9. Verify Streamlit

```bash
uv run streamlit run streamlit_app.py
```

Confirm that Streamlit communicates successfully with the Dockerized FastAPI backend.

---

# Limitations and Future Improvements

Potential improvements include:

- Comparing the 30-feature and 3-feature models using identical preprocessing.
- Evaluating feature selection using cross-validation.
- Tuning the L1 regularization strength.
- Evaluating recall and false-negative rates more carefully.
- Adding probability-based model evaluation.
- Expanding automated tests to cover additional edge cases.
- Adding continuous integration using GitHub Actions.
- Introducing model versioning.
- Improving monitoring and observability.
- Containerizing the frontend if required.
- Evaluating deployment to a cloud platform.

---

# Notes

- This project is intended for educational use.
- It is not suitable for clinical diagnosis.
- Do not commit local secrets or `.env` files.
- Do not commit the local `.venv` directory.
- The model must be trained before prediction if the artifact is unavailable.
- Feature selection must be fitted using training data only.
- The API requires the three selected features in the correct schema.
- The Docker image must include the saved model artifact.
- CLI prediction does not retrain the model.
- The Streamlit frontend requires the FastAPI backend to be running.
- Automated tests should pass before committing changes.

---

## Repository

GitHub:

https://github.com/mahmoudrabea01/ml-production-engineering_simple_predicition_model
