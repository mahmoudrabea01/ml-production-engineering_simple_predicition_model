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


class PredictionRequest(BaseModel):
    """Validate input features for breast cancer prediction."""

    model_config = ConfigDict(extra="forbid")

    mean_radius: float = Field(gt=0)
    mean_texture: float = Field(gt=0)
    mean_perimeter: float = Field(gt=0)
    mean_area: float = Field(gt=0)
    mean_smoothness: float = Field(gt=0)
    mean_compactness: float = Field(ge=0)
    mean_concavity: float = Field(ge=0)
    mean_concave_points: float = Field(ge=0)
    mean_symmetry: float = Field(gt=0)
    mean_fractal_dimension: float = Field(gt=0)

    radius_error: float = Field(ge=0)
    texture_error: float = Field(ge=0)
    perimeter_error: float = Field(ge=0)
    area_error: float = Field(ge=0)
    smoothness_error: float = Field(ge=0)
    compactness_error: float = Field(ge=0)
    concavity_error: float = Field(ge=0)
    concave_points_error: float = Field(ge=0)
    symmetry_error: float = Field(ge=0)
    fractal_dimension_error: float = Field(ge=0)

    worst_radius: float = Field(gt=0)
    worst_texture: float = Field(gt=0)
    worst_perimeter: float = Field(gt=0)
    worst_area: float = Field(gt=0)
    worst_smoothness: float = Field(gt=0)
    worst_compactness: float = Field(ge=0)
    worst_concavity: float = Field(ge=0)
    worst_concave_points: float = Field(ge=0)
    worst_symmetry: float = Field(gt=0)
    worst_fractal_dimension: float = Field(gt=0)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the validated request into model input."""

        features = {
            name.replace("_", " "): value for name, value in self.model_dump().items()
        }

        return pd.DataFrame(
            [[features[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
        )
