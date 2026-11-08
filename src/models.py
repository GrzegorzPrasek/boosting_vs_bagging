"""Bagging modeli GBDT: Baseline, +Adaptive Sampling, +Ensemble-Aware ES, +oba.

Etap 3: Baseline (zwykły bootstrap, stała liczba drzew M).
Etap 5: Adaptive Sampling  -> `_sample_indices`.
Etap 6: Ensemble-Aware ES   -> `_fit_one`.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class BaggedGBDT(BaseEstimator):
    """Bagging B modeli Gradient Boosting.

    Modele trenujemy sekwencyjnie (model b może korzystać z modeli 1..b-1,
    co jest potrzebne obu modyfikacjom). Predykcje są uśredniane:
    w regresji wartości, w klasyfikacji prawdopodobieństwa klas.

    Po `fit` dostępne są:
      estimators_       lista B wytrenowanych modeli GBDT
      inbag_            macierz bool (B, n): czy obserwacja trafiła do próbki modelu b
      classes_          etykiety klas (clf)
      n_stages_         liczba iteracji boostingu każdego modelu
      n_trees_total_    łączna liczba drzew (w klasyfikacji wieloklasowej K drzew na iterację)
      fit_time_         czas treningu w sekundach (z narzutem modyfikacji)
    """

    def __init__(self, task: str = "clf", B: int = 10, M: int = 200,
                 max_depth: int = 3, learning_rate: float = 0.1,
                 adaptive_sampling: bool = False, alpha: float = 1.0, beta: float = 1.0,
                 ensemble_es: bool = False, k: int = 20, p: int = 2,
                 random_state: int | None = None):
        self.task = task
        self.B = B
        self.M = M
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.adaptive_sampling = adaptive_sampling
        self.alpha = alpha
        self.beta = beta
        self.ensemble_es = ensemble_es
        self.k = k
        self.p = p
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        if self.task not in ("clf", "reg"):
            raise ValueError(f"task must be 'clf' or 'reg', got {self.task!r}")
        if self.ensemble_es and (X_val is None or y_val is None):
            raise ValueError("ensemble_es=True wymaga X_val i y_val")

        t0 = time.perf_counter()
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        if self.task == "clf":
            self.classes_ = np.unique(y)

        self.estimators_ = []
        self.inbag_ = np.zeros((self.B, n), dtype=bool)

        for b in range(self.B):
            idx = self._sample_indices(b, X, y, rng)
            self.inbag_[b, np.unique(idx)] = True
            seed = int(rng.integers(2**31 - 1))
            model = self._fit_one(X[idx], y[idx], X_val, y_val, seed)
            self.estimators_.append(model)

        self.n_stages_ = np.array([m.n_estimators_ for m in self.estimators_])
        self.n_trees_total_ = int(sum(m.estimators_.size for m in self.estimators_))
        self.fit_time_ = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task == "clf":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        return self._aggregate(self.estimators_, X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "clf":
            raise AttributeError("predict_proba dostępne tylko dla task='clf'")
        return self._aggregate(self.estimators_, X)

    # ------------------------------------------------------------------ #
    # Elementy wymieniane przez modyfikacje
    # ------------------------------------------------------------------ #

    def _sample_indices(self, b: int, X: np.ndarray, y: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
        """Indeksy próbki dla modelu b. Baseline: bootstrap 100% ze zwracaniem."""
        probs = None  # Etap 5: P(i) z Adaptive Sampling
        return self._draw(len(y), y, probs, rng)

    def _fit_one(self, X: np.ndarray, y: np.ndarray,
                 X_val: np.ndarray | None, y_val: np.ndarray | None,
                 seed: int):
        """Trening jednego GBDT. Baseline: stałe M drzew."""
        model = self._new_gbdt(seed)  # Etap 6: Ensemble-Aware ES
        return model.fit(X, y)

    # ------------------------------------------------------------------ #
    # Pomocnicze
    # ------------------------------------------------------------------ #

    def _new_gbdt(self, seed: int, n_estimators: int | None = None, warm_start: bool = False):
        cls = GradientBoostingClassifier if self.task == "clf" else GradientBoostingRegressor
        return cls(n_estimators=n_estimators or self.M, max_depth=self.max_depth,
                   learning_rate=self.learning_rate, subsample=1.0,
                   warm_start=warm_start, random_state=seed)

    def _draw(self, n: int, y: np.ndarray, probs: np.ndarray | None,
              rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
        """Losuje n indeksów ze zwracaniem. W klasyfikacji próbka musi mieć >= 2 klasy."""
        for _ in range(max_tries):
            idx = rng.choice(n, size=n, replace=True, p=probs)
            if self.task == "reg" or np.unique(y[idx]).size >= 2:
                return idx
        raise RuntimeError("Nie udało się wylosować próbki z co najmniej 2 klasami")

    def _model_output(self, model, X: np.ndarray) -> np.ndarray:
        """Predykcja pojedynczego modelu; w clf proba rozszerzona do wszystkich classes_."""
        if self.task == "reg":
            return model.predict(X)
        proba = np.zeros((X.shape[0], self.classes_.size))
        cols = np.searchsorted(self.classes_, model.classes_)
        proba[:, cols] = model.predict_proba(X)
        return proba

    def _aggregate(self, models, X: np.ndarray) -> np.ndarray:
        return np.mean([self._model_output(m, X) for m in models], axis=0)
