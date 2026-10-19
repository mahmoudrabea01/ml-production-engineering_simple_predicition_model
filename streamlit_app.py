import requests
import streamlit as st
from sklearn.datasets import load_breast_cancer

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide",
)

st.title("Breast Cancer Classification")
st.write(
    "A machine learning demonstration using "
    "Logistic Regression and a saved preprocessing pipeline."
)

st.warning(
    "Educational demonstration only. "
    "This application is not a medical diagnostic tool."
)


@st.cache_data
def get_dataset():
    """Load feature names and default input values."""
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    defaults = features.median()

    return list(features.columns), defaults


feature_names, defaults = get_dataset()

st.subheader("Enter Feature Values")
st.caption(
    "The default values are feature medians from the example dataset. "
    "Change them to explore different predictions."
)

with st.form("prediction_form"):
    values = {}
    columns = st.columns(3)

    for index, feature in enumerate(feature_names):
        with columns[index % 3]:
            values[feature] = st.number_input(
                label=feature,
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
    # Convert feature names to the field names expected by Pydantic.
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

            st.caption(
                "This is an educational model prediction, " "not a medical diagnosis."
            )

        elif response.status_code == 422:
            st.error(
                "The submitted features were rejected by the API. "
                "Please check the input data."
            )

        elif response.status_code == 503:
            st.error(
                "The prediction service is currently unavailable. "
                "Please ensure the saved model is loaded."
            )

        else:
            st.error(
                f"The prediction request failed "
                f"with status code {response.status_code}."
            )

    except requests.exceptions.Timeout:
        st.error("The API request timed out. " "Please try again.")

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure the FastAPI server is running."
        )

    except (requests.exceptions.RequestException, ValueError, KeyError):
        st.error(
            "An unexpected response was received from the API. " "Please try again."
        )


# import streamlit as st

# st.set_page_config(page_title="Test App")

# st.title("My ML Project")
# st.write("Streamlit is working!")
# st.success("The page is rendering correctly.")
