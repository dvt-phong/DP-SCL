from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(seed=42):
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
    )

