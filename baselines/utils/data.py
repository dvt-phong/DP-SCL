import os

import numpy as np


def load_npz_baseline_data(input_dir, ds_config):
    npz_path = os.path.join(input_dir, "datastore", ds_config["npz_filename"])
    data = np.load(npz_path)

    X_train = data["t_data"]
    y_train = data["t_label"].astype(int)
    X_test = data["v_data"]
    y_test = data["v_label"].astype(int)

    if X_train.ndim != 4 or X_test.ndim != 4:
        raise ValueError(
            f"Expected t_data/v_data with 4 dims (N,W,D,F), got "
            f"{X_train.shape} and {X_test.shape}"
        )

    return X_train, y_train, X_test, y_test


def flatten_temporal_data(X_train, X_test):
    return X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)

