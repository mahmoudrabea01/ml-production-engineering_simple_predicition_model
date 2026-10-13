from sklearn.preprocessing import RobustScaler


def build_preprocessor() -> RobustScaler:
    """Create the feature scaling step."""
    return RobustScaler()
