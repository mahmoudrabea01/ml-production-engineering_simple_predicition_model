from sklearn.preprocessing import MinMaxScaler


def build_preprocessor() -> MinMaxScaler:
    """Create the MinMax feature scaling step."""
    return MinMaxScaler()
