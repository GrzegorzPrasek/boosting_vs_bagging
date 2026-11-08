"""Bagging modeli GBDT: Baseline, +Adaptive Sampling, +Ensemble-Aware ES, +oba (Etapy 3, 5, 6)."""


class BaggedGBDT:
    """Bagging B modeli Gradient Boosting.

    Parametry: task, B, M, max_depth, learning_rate,
    adaptive_sampling (mod. 1), alpha, beta,
    ensemble_es (mod. 2), k, p, random_state.

    fit(X_train, y_train, X_val, y_val) -> self
    predict(X), predict_proba(X) (tylko clf)
    Atrybuty po treningu: n_trees_total_, fit_time_
    """
