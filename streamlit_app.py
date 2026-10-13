import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

from production_app.config import load_settings
from production_app.model_service import load_model

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


@st.cache_resource
def get_model(model_path: str):
    """Load and cache the saved model pipeline."""
    return load_model(__import__("pathlib").Path(model_path))


@st.cache_data
def get_dataset():
    """Load feature names and default input values."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    defaults = X.median()
    return list(X.columns), defaults, list(dataset.target_names)


settings = load_settings()
feature_names, defaults, target_names = get_dataset()

try:
    model = get_model(str(settings.model_path.resolve()))
except (OSError, ValueError, EOFError, TypeError) as exc:
    st.error(
        "Could not load the saved model. "
        "Make sure the model file exists and is valid."
    )
    st.exception(exc)
    st.stop()


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
    input_data = pd.DataFrame(
        [values],
        columns=feature_names,
    )

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted class: {target_names[prediction]}")

    probability_columns = st.columns(len(model.classes_))

    for column, class_id, probability in zip(
        probability_columns,
        model.classes_,
        probabilities,
    ):
        with column:
            class_name = target_names[int(class_id)]
            st.metric(
                label=f"{class_name.title()} probability",
                value=f"{probability:.2%}",
            )

    st.caption("Probabilities are model outputs, not medical certainty.")


# import streamlit as st

# st.set_page_config(page_title="Test App")

# st.title("My ML Project")
# st.write("Streamlit is working!")
# st.success("The page is rendering correctly.")
