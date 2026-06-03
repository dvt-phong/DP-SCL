from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_model(seed=42):
    base_model = LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000)
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(base_model, cv=3))

