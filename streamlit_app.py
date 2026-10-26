import os

import requests
import streamlit as st
from sklearn.datasets import load_breast_cancer

# Configurable FastAPI prediction endpoint.
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict",
)

# Must match the saved model and FastAPI schema.
FEATURE_NAMES = [
    "mean concave points",
    "worst radius",
    "worst concave points",
]

st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide",
)

st.title("Breast Cancer Classification")

st.write(
    "A machine learning demonstration using Embedded Feature "
    "Selection, MinMaxScaler, Logistic Regression, and FastAPI."
)

st.warning(
    "Educational demonstration only. "
    "This application is not a medical diagnostic tool."
)


@st.cache_data
def get_default_values() -> dict[str, float]:
    """Get example input values for selected features."""
    dataset = load_breast_cancer(as_frame=True)

    defaults = dataset.data[FEATURE_NAMES].median()

    return defaults.to_dict()


defaults = get_default_values()

st.subheader("Enter Feature Values")

st.caption(
    "This model uses 3 selected features out of the original 30. "
    "The default values are dataset medians for demonstration."
)

with st.form("prediction_form"):
    values = {}

    columns = st.columns(3)

    for index, feature in enumerate(FEATURE_NAMES):
        with columns[index % 3]:
            values[feature] = st.number_input(
                label=feature.title(),
                min_value=0.0,
                value=float(defaults[feature]),
                format="%.4f",
                key=feature,
            )

    submitted = st.form_submit_button(
        "Predict",
        type="primary",
        use_container_width=True,
    )


if submitted:
    # Convert sklearn feature names into Pydantic field names.
    payload = {feature.replace(" ", "_"): value for feature, value in values.items()}

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()

            predicted_class = result["predicted_class"]
            predicted_label = result["predicted_label"]

            st.subheader("Prediction Result")

            st.success(
                f"Predicted class: {predicted_label.title()} " f"({predicted_class})"
            )

            st.caption("This prediction is for educational purposes only.")

        elif response.status_code == 422:
            st.error(
                "The API rejected the submitted features. "
                "Please check the input values."
            )

        elif response.status_code == 503:
            st.error(
                "The prediction service is unavailable. "
                "Please verify that FastAPI loaded the saved model."
            )

        else:
            st.error(
                f"Prediction request failed. " f"HTTP status: {response.status_code}"
            )

    except requests.exceptions.Timeout:
        st.error("The prediction request timed out. " "Please try again.")

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to FastAPI. " "Make sure the API server is running."
        )

    except (requests.exceptions.RequestException, ValueError, KeyError):
        st.error("An unexpected error occurred while processing " "the API response.")
