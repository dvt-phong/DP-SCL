from .adaboost import build_model as build_adaboost
from .decision_tree import build_model as build_decision_tree
from .gbdt import build_model as build_gbdt
from .knn import build_model as build_knn
from .logistic_regression import build_model as build_logistic_regression
from .random_forest import build_model as build_random_forest
from .svm import build_model as build_svm
from .xgboost_clf import build_model as build_xgboost


ML_BASELINE_REGISTRY = {
    "ml_lr": build_logistic_regression,
    "ml_dt": build_decision_tree,
    "ml_rf": build_random_forest,
    "ml_xgb": build_xgboost,
    "ml_svm": build_svm,
    "ml_gbdt": build_gbdt,
    "ml_ada": build_adaboost,
    "ml_knn": build_knn,
}


DEFAULT_ML_ORDER = (
    "ml_lr",
    "ml_dt",
    "ml_rf",
    "ml_xgb",
    "ml_svm",
    "ml_gbdt",
    "ml_ada",
    "ml_knn",
)


__all__ = ["DEFAULT_ML_ORDER", "ML_BASELINE_REGISTRY"]
