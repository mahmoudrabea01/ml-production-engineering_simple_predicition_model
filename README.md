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
- Save and load the fitted pipeline using `joblib`.
- Generate predictions and class probabilities.
- Configure application settings through environment variables.
- Use logging and exception handling.
- Provide an interactive web interface using Streamlit.
- Apply code-quality checks using Ruff, Black, and pre-commit.

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
| `MODEL_PATH` | Saved model location | `model/breast_cancer_pipeline.joblib` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `TEST_SIZE` | Test-set proportion | `0.2` |
| `RANDOM_STATE` | Random seed | `42` |

### 5. Train and Evaluate the Model

Run the training workflow:

```bash
uv run python run.py
```

This command loads the dataset, splits it into training and testing sets, trains the pipeline, evaluates its performance, and saves the fitted pipeline to the configured model path.

The default model path is:

```text
model/breast_cancer_pipeline.joblib
```

Training is separate from making predictions through the Streamlit interface.

### 6. Launch the Streamlit Interface

After a trained model has been saved, start the web application:

```bash
uv run streamlit run streamlit_app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

The interface loads the previously saved model and uses it to generate predictions and class probabilities. It does not retrain the model when a prediction is requested.

The default input values are based on feature medians from the example dataset. They are provided for demonstration and do not represent an actual patient's measurements.

If the saved model does not exist, run the training workflow first.

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

1. Load the previously saved pipeline.
2. Receive feature values through the Streamlit interface.
3. Generate a prediction and class probabilities using the loaded pipeline.

Using a pipeline keeps preprocessing and classification together and helps ensure consistent transformations during training and prediction. Fitting preprocessing on the training data helps prevent data leakage.

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
- Run the training workflow before launching Streamlit if the saved model does not exist.
- The saved model path can be customized using `MODEL_PATH`.
- This project demonstrates machine learning application structure and production-oriented practices. It is not intended for clinical use.