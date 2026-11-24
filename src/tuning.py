"""Strojenie hiperparametrów (plan, rozdz. 6.3) i pojedyncze zadanie eksperymentu.

Procedura w każdym foldzie zewnętrznym:
  - losujemy N_CONFIGS konfiguracji z przestrzeni danej metody (random search),
  - każdą trenujemy na części treningowej (80%) i oceniamy na walidacji wewnętrznej
    (20%) metryką główną: Macro F1 (clf) / MAE (reg),
  - najlepszy model (już wytrenowany) oceniamy raz na foldzie testowym.

Wspólne losowanie (common random numbers): w danym foldzie wszystkie 4 warianty
BaggedGBDT dostają te same konfiguracje bazowe (B, M, max_depth, learning_rate);
+1 i +1+2 te same (alpha, beta); +2 i +1+2 te same (k, p). Różnice między
wariantami wynikają więc z modyfikacji, a nie z innego wylosowania ustawień.
Każda metoda ma ten sam budżet: N_CONFIGS konfiguracji.
"""

from __future__ import annotations

import itertools
import json
import time
from functools import lru_cache

import numpy as np

from .config import CV_SEEDS, N_FOLDS
from .data import impute_median, load_dataset
from .evaluation import (MAIN_METRIC, ROW_COLUMNS, VARIANT_FLAGS, _fit, _tree_stats,
                         compute_metrics, iter_splits, make_model)

N_CONFIGS = 8

# --------------------------------------------------------------------------- #
# Przestrzenie przeszukiwania
# --------------------------------------------------------------------------- #

BASE_SPACE = {                        # wspólne dla 4 wariantów BaggedGBDT (plan 6.3)
    "B": [5, 10],
    "M": [100, 200],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
}
AS_PAIRS = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (2.0, 0.5), (0.5, 2.0)]  # (alpha, beta)
ES_SPACE = {"k": [10, 20], "p": [2, 3]}

GBDT_SPACE = {"M": [100, 200], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]}
BAGGING_SPACE = {"n_estimators": [100, 200], "max_samples": [0.5, 1.0],
                 "min_samples_leaf": [1, 3]}
RF_SPACE = {"n_estimators": [100, 200], "max_features": ["sqrt", 1.0],
            "min_samples_leaf": [1, 3]}


def _grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*space.values())]


def _sample(space_or_list, n: int, rng: np.random.Generator) -> list:
    """n elementów bez powtórzeń (z powtórzeniami, gdy przestrzeń jest mniejsza)."""
    items = _grid(space_or_list) if isinstance(space_or_list, dict) else list(space_or_list)
    idx = rng.choice(len(items), size=n, replace=len(items) < n)
    return [items[i] for i in idx]


def sample_configs(method: str, seed: int, fold: int, n: int = N_CONFIGS) -> list[dict]:
    """Konfiguracje dla (metoda, seed, fold); deterministyczne, wspólne losowanie."""
    def rng(tag: int):
        return np.random.default_rng([seed, fold, tag])

    if method in VARIANT_FLAGS:
        base = _sample(BASE_SPACE, n, rng(0))
        ab = _sample(AS_PAIRS, n, rng(1))
        kp = _sample(ES_SPACE, n, rng(2))
        flags = VARIANT_FLAGS[method]
        out = []
        for b, (a, be), e in zip(base, ab, kp):
            cfg = dict(b)
            # parametry wyłączonej modyfikacji nie mają wpływu na model
            if method == "Baseline+1(e)":         # ablacja: tylko błąd
                cfg.update(alpha=a, beta=0.0)
            elif method == "Baseline+1(d)":       # ablacja: tylko rozbieżność
                cfg.update(alpha=0.0, beta=be)
            elif flags["adaptive_sampling"]:
                cfg.update(alpha=a, beta=be)
            else:
                cfg.update(alpha=1.0, beta=1.0)
            if flags["ensemble_es"]:
                cfg.update(e)
            else:
                cfg.update(k=20, p=2)
            out.append(cfg)
        return out
    space = {"GBDT": GBDT_SPACE, "Bagging": BAGGING_SPACE, "RF": RF_SPACE}[method]
    return _sample(space, n, rng(10 + ["GBDT", "Bagging", "RF"].index(method)))


# --------------------------------------------------------------------------- #
# Jedno zadanie: (zbiór, metoda, seed, fold)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=4)
def _dataset(name: str):
    X, y, task = load_dataset(name)
    splits = {(s, f): (tr, va, te) for s, f, tr, va, te in iter_splits(X, y, task)}
    return X, y, task, splits


def _score(task: str, y_true, y_pred) -> float:
    """Metryka główna jako wartość do maksymalizacji."""
    m = compute_metrics(task, y_true, y_pred)[MAIN_METRIC[task]]
    return m if task == "clf" else -m


def add_label_noise(task: str, y_tr, y_val, rate: float, seed: int, fold: int):
    """Szum w etykietach części treningowej i walidacyjnej (fold testowy zostaje czysty).

    Każda obserwacja z prawdopodobieństwem `rate` dostaje losową etykietę:
    clf – inną klasę (jednostajnie spośród pozostałych), reg – wartość celu losowo
    wybranej obserwacji z części treningowej (zachowuje rozkład brzegowy y).
    Deterministyczne dla (seed, fold, rate) – wszystkie metody widzą ten sam szum.
    """
    if rate <= 0:
        return y_tr, y_val
    rng = np.random.default_rng([seed, fold, int(round(rate * 1000)), 7])
    y = np.concatenate([y_tr, y_val]).copy()
    hit = np.flatnonzero(rng.random(len(y)) < rate)
    if task == "clf":
        classes = np.unique(y)
        for i in hit:
            others = classes[classes != y[i]]
            y[i] = others[rng.integers(len(others))]
    else:
        y[hit] = rng.choice(y_tr, size=len(hit))
    return y[:len(y_tr)], y[len(y_tr):]


def run_task(dataset: str, method: str, seed: int, fold: int,
             noise: float = 0.0, n_configs: int = N_CONFIGS) -> dict:
    """Strojenie + ocena jednej metody na jednym foldzie. Zwraca wiersz wyników."""
    X, y, task, splits = _dataset(dataset)
    tr, va, te = splits[(seed, fold)]
    X_tr, X_val, X_te = impute_median(X[tr], X[va], X[te])
    y_tr, y_val, y_te = y[tr], y[va], y[te]
    y_tr, y_val = add_label_noise(task, y_tr, y_val, noise, seed, fold)

    best = None
    tune_time = 0.0
    trees_tuning = 0
    for i, cfg in enumerate(sample_configs(method, seed, fold, n_configs)):
        model = make_model(method, task, cfg, seed=seed * 100 + fold)
        t0 = time.perf_counter()
        _fit(model, X_tr, y_tr, X_val, y_val)
        fit_time = time.perf_counter() - t0
        tune_time += fit_time
        trees_tuning += _tree_stats(model)[1]
        val = _score(task, y_val, model.predict(X_val))
        if best is None or val > best[0]:
            best = (val, i, cfg, model, fit_time)

    val, idx, cfg, model, fit_time = best
    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    predict_time = time.perf_counter() - t0
    n_trees, n_trees_trained, n_stages = _tree_stats(model)

    row = {"dataset": dataset, "task": task, "method": method, "seed": seed, "fold": fold,
           "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           **compute_metrics(task, y_te, y_pred),
           "fit_time": fit_time, "predict_time": predict_time,
           "n_trees": n_trees, "n_trees_trained": n_trees_trained,
           "n_stages_mean": n_stages,
           "val_score": val if task == "clf" else -val,
           "tune_time": tune_time, "n_trees_tuning": trees_tuning,
           "best_config": idx, "params": json.dumps(cfg), "noise": noise}
    return row


FULL_COLUMNS = ROW_COLUMNS + ["val_score", "tune_time", "n_trees_tuning",
                              "best_config", "params"]


def all_tasks(datasets, methods, seeds=CV_SEEDS, n_folds: int = N_FOLDS, noises=(0.0,)):
    return [(d, m, s, f, nz) for nz in noises for d in datasets for m in methods
            for s in seeds for f in range(n_folds)]
