from typing import Annotated

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
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
    """Validate input features for breast cancer prediction."""

    model_config = ConfigDict(extra="forbid")

    mean_radius: PositiveFeature
    mean_texture: PositiveFeature
    mean_perimeter: PositiveFeature
    mean_area: PositiveFeature
    mean_smoothness: PositiveFeature
    mean_compactness: NonNegativeFeature
    mean_concavity: NonNegativeFeature
    mean_concave_points: NonNegativeFeature
    mean_symmetry: PositiveFeature
    mean_fractal_dimension: PositiveFeature

    radius_error: NonNegativeFeature
    texture_error: NonNegativeFeature
    perimeter_error: NonNegativeFeature
    area_error: NonNegativeFeature
    smoothness_error: NonNegativeFeature
    compactness_error: NonNegativeFeature
    concavity_error: NonNegativeFeature
    concave_points_error: NonNegativeFeature
    symmetry_error: NonNegativeFeature
    fractal_dimension_error: NonNegativeFeature

    worst_radius: PositiveFeature
    worst_texture: PositiveFeature
    worst_perimeter: PositiveFeature
    worst_area: PositiveFeature
    worst_smoothness: PositiveFeature
    worst_compactness: NonNegativeFeature
    worst_concavity: NonNegativeFeature
    worst_concave_points: NonNegativeFeature
    worst_symmetry: PositiveFeature
    worst_fractal_dimension: PositiveFeature

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the validated request into model input."""

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
