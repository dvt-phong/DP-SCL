from sklearn.ensemble import GradientBoostingClassifier


def build_model(seed=42):
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=seed,
    )

