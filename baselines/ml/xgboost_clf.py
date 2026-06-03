import numpy as np
from importlib.util import find_spec


class ImbalanceAwareXGBClassifier:
    def __init__(self, seed=42):
        self.seed = seed
        self.model = None

    def fit(self, X, y):
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "xgboost is not installed. Install it with: pip install xgboost"
            ) from exc

        y = np.asarray(y)
        positives = float(np.sum(y == 1))
        negatives = float(np.sum(y == 0))
        scale_pos_weight = negatives / positives if positives > 0 else 1.0

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.seed,
            n_jobs=4,
            tree_method="hist",
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


def build_model(seed=42):
    if find_spec("xgboost") is None:
        raise ModuleNotFoundError(
            "xgboost is not installed. Install it with: pip install xgboost"
        )

    return ImbalanceAwareXGBClassifier(seed)
