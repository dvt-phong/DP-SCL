from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def build_model(seed=42):
    stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
    kwargs = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "random_state": seed,
    }
    if "estimator" in AdaBoostClassifier().get_params():
        kwargs["estimator"] = stump
    else:
        kwargs["base_estimator"] = stump
    return AdaBoostClassifier(
        **kwargs,
    )
