from typing import Annotated

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# Selected features from the trained model.
# Order must match the saved model artifact.
FEATURE_NAMES = [
    "mean concave points",
    "worst radius",
    "worst concave points",
]

PositiveFeature = Annotated[
    float,
    Field(gt=0, allow_inf_nan=False),
]

NonNegativeFeature = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False),
]


class PredictionRequest(BaseModel):
    """Validate the selected input features."""

    model_config = ConfigDict(extra="forbid")

    mean_concave_points: NonNegativeFeature
    worst_radius: PositiveFeature
    worst_concave_points: NonNegativeFeature

    def to_dataframe(self) -> pd.DataFrame:
        """Convert validated inputs to the model's feature order."""

        features = {
            name.replace("_", " "): value for name, value in self.model_dump().items()
        }

        return pd.DataFrame(
            [[features[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
        )


class PredictionResponse(BaseModel):
    """Define the expected API prediction response."""

    predicted_class: int
    predicted_label: str
