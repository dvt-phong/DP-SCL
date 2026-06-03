"""
Baselines package — Phương pháp cơ sở so sánh với DP-SCL trên XuetangX.

Cấu trúc:
    baselines/
    ├── utils/           — Data loading, metrics, result writing
    ├── ml/              — Machine Learning (sklearn / xgboost)
    │   ├── lr.py        — Logistic Regression
    │   ├── svm.py       — SVM / LinearSVC
    │   ├── knn.py       — k-Nearest Neighbors
    │   ├── random_forest.py  — Random Forest
    │   ├── gbdt.py      — Gradient Boosting (GBDT)
    │   ├── xgboost_clf.py   — XGBoost
    │   └── adaboost.py  — AdaBoost
    └── dl/              — Deep Learning (PyTorch) — tích hợp vào train.py
        ├── cnn.py           — Pure CNN (mode: dl_cnn)
        ├── cnn_lstm.py      — CNN + LSTM (mode: dl_cnn_lstm)
        ├── cnn_gru.py       — CNN + GRU (mode: dl_cnn_gru)
        ├── cnn_rnn.py       — CNN + RNN (mode: dl_cnn_rnn)
        └── cnn_lstm_attn.py — CNN + LSTM + Self-Attention (mode: dl_cnn_lstm_at1 / at2)

Cách chạy:
    # ML baselines (tất cả 7 methods):
    python train_ml_baseline.py --dataset xuetangx

    # DL baselines (từng method qua train.py):
    python train.py -mode dl_cnn --dataset xuetangx
    python train.py -mode dl_cnn_lstm --dataset xuetangx
    ...
"""
