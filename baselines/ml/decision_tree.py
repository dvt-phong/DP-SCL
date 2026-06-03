from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_model(seed=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            class_weight="balanced",
            random_state=seed,
        )),
    ])
