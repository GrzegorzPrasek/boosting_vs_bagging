"""Powtarzana walidacja krzyżowa, metryki i zapis wyników.

Procedura (plan, rozdz. 5.2):
  - 2x 5-fold CV (ziarna 42 i 2026); StratifiedKFold dla clf, KFold dla reg,
  - w każdym foldzie zewnętrznym 20% części treningowej -> walidacja wewnętrzna
    (stratyfikowana dla clf); ten sam podział dla wszystkich metod,
  - wszystkie metody trenują na tej samej części treningowej (80%),
    walidację wewnętrzną widzą tylko mechanizmy, które jej potrzebują (ES),
  - fold testowy służy wyłącznie do oceny końcowej.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             root_mean_squared_error)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import CV_SEEDS, INNER_VAL_SIZE, N_FOLDS, REF_PARAMS
from .data import impute_median, load_dataset
from .models import BaggedGBDT

# --------------------------------------------------------------------------- #
# Metody
# --------------------------------------------------------------------------- #

BAGGED_VARIANTS = {
    "Baseline": dict(adaptive_sampling=False, ensemble_es=False),
    "Baseline+1": dict(adaptive_sampling=True, ensemble_es=False),
    "Baseline+2": dict(adaptive_sampling=False, ensemble_es=True),
    "Baseline+1+2": dict(adaptive_sampling=True, ensemble_es=True),
}
REFERENCE_METHODS = ("GBDT", "Bagging", "RF")
ALL_METHODS = REFERENCE_METHODS + tuple(BAGGED_VARIANTS)


def make_model(method: str, task: str, params: dict, seed: int):
    """Tworzy nowy (niewytrenowany) model danej metody."""
    clf = task == "clf"
    if method in BAGGED_VARIANTS:
        keys = ("B", "M", "max_depth", "learning_rate", "alpha", "beta", "k", "p")
        return BaggedGBDT(task=task, random_state=seed, **BAGGED_VARIANTS[method],
                          **{k: params[k] for k in keys})
    if method == "GBDT":
        cls = GradientBoostingClassifier if clf else GradientBoostingRegressor
        return cls(n_estimators=params["M"], max_depth=params["max_depth"],
                   learning_rate=params["learning_rate"], subsample=1.0, random_state=seed)
    if method == "Bagging":
        cls, tree = (BaggingClassifier, DecisionTreeClassifier) if clf \
            else (BaggingRegressor, DecisionTreeRegressor)
        return cls(estimator=tree(), n_estimators=REF_PARAMS["n_estimators"], random_state=seed)
    if method == "RF":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        return cls(n_estimators=REF_PARAMS["n_estimators"], random_state=seed)
    raise ValueError(f"Nieznana metoda: {method}")


def _fit(model, X_tr, y_tr, X_val, y_val):
    if isinstance(model, BaggedGBDT):
        return model.fit(X_tr, y_tr, X_val, y_val)
    return model.fit(X_tr, y_tr)


def _tree_stats(model) -> tuple[int, int, float]:
    """(drzewa w modelu, drzewa zbudowane w treningu, śr. liczba iteracji boostingu na GBDT)."""
    if isinstance(model, BaggedGBDT):
        return model.n_trees_total_, model.n_trees_trained_, float(model.n_stages_.mean())
    if isinstance(model, (GradientBoostingClassifier, GradientBoostingRegressor)):
        n = int(model.estimators_.size)
        return n, n, float(model.n_estimators_)
    n = len(model.estimators_)
    return n, n, np.nan


# --------------------------------------------------------------------------- #
# Metryki
# --------------------------------------------------------------------------- #

def compute_metrics(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if task == "clf":
        return {"f1_macro": f1_score(y_true, y_pred, average="macro"),
                "accuracy": accuracy_score(y_true, y_pred)}
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred)}


MAIN_METRIC = {"clf": "f1_macro", "reg": "mae"}

# Stały układ kolumn CSV (clf i reg w jednym pliku; metryki drugiego typu = NaN)
ROW_COLUMNS = ["dataset", "task", "method", "seed", "fold", "n_train", "n_val", "n_test",
               "f1_macro", "accuracy", "mae", "rmse",
               "fit_time", "predict_time", "n_trees", "n_trees_trained", "n_stages_mean"]


# --------------------------------------------------------------------------- #
# Podziały
# --------------------------------------------------------------------------- #

def iter_splits(X: np.ndarray, y: np.ndarray, task: str,
                seeds=CV_SEEDS, n_folds: int = N_FOLDS):
    """Zwraca (seed, fold, idx_train, idx_val, idx_test) dla każdego foldu zewnętrznego."""
    for seed in seeds:
        cv = (StratifiedKFold if task == "clf" else KFold)(
            n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(cv.split(X, y)):
            tr, val = train_test_split(
                train, test_size=INNER_VAL_SIZE, random_state=seed + fold,
                stratify=y[train] if task == "clf" else None)
            yield seed, fold, tr, val, test


# --------------------------------------------------------------------------- #
# Uruchomienie CV
# --------------------------------------------------------------------------- #

def run_cv(dataset: str, methods=ALL_METHODS, params: dict | None = None,
           out_csv: str | Path | None = None, seeds=CV_SEEDS, n_folds: int = N_FOLDS,
           log: Callable[[str], None] | None = print) -> pd.DataFrame:
    """Uruchamia powtarzaną CV dla jednego zbioru i listy metod.

    Każdy wiersz = (zbiór, metoda, seed, fold). Przy podanym `out_csv` wiersze są
    dopisywane na bieżąco, więc przerwany przebieg nie traci wyników.
    """
    from .config import PILOT_PARAMS
    params = params or PILOT_PARAMS
    X, y, task = load_dataset(dataset)
    rows = []

    for seed, fold, tr, val, te in iter_splits(X, y, task, seeds, n_folds):
        X_tr, X_val, X_te = impute_median(X[tr], X[val], X[te])
        y_tr, y_val, y_te = y[tr], y[val], y[te]

        for method in methods:
            model = make_model(method, task, params, seed=seed * 100 + fold)
            t0 = time.perf_counter()
            _fit(model, X_tr, y_tr, X_val, y_val)
            fit_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            y_pred = model.predict(X_te)
            predict_time = time.perf_counter() - t0
            n_trees, n_trees_trained, n_stages = _tree_stats(model)

            row = {"dataset": dataset, "task": task, "method": method,
                   "seed": seed, "fold": fold,
                   "n_train": len(tr), "n_val": len(val), "n_test": len(te),
                   **compute_metrics(task, y_te, y_pred),
                   "fit_time": fit_time, "predict_time": predict_time,
                   "n_trees": n_trees, "n_trees_trained": n_trees_trained,
                   "n_stages_mean": n_stages}
            rows.append(row)
            if out_csv is not None:
                _append_row(out_csv, row)
            if log:
                m = MAIN_METRIC[task]
                log(f"{dataset:14s} s={seed} f={fold} {method:13s} "
                    f"{m}={row[m]:.4f} t={fit_time:.2f}s trees={n_trees}")

    return pd.DataFrame(rows).reindex(columns=ROW_COLUMNS)


def _append_row(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).reindex(columns=ROW_COLUMNS).to_csv(
        path, mode="a", header=not path.exists(), index=False)


# --------------------------------------------------------------------------- #
# Podsumowania
# --------------------------------------------------------------------------- #

def summarize(df: pd.DataFrame, value: str | None = None) -> pd.DataFrame:
    """Tabela metoda x zbiór z 'średnia ± std' wybranej kolumny.

    Domyślnie metryka główna (F1 macro / MAE); df powinien zawierać jeden typ zadania.
    """
    value = value or MAIN_METRIC[df["task"].iloc[0]]
    g = df.groupby(["dataset", "method"])[value].agg(["mean", "std"])
    cell = g["mean"].map("{:.4f}".format) + " ± " + g["std"].map("{:.4f}".format)
    order = [m for m in ALL_METHODS if m in set(df["method"])]
    return cell.unstack("method")[order]
