"""Bagging modeli GBDT: Baseline, +Adaptive Sampling, +Ensemble-Aware ES, +oba.

Etap 3: Baseline (zwykły bootstrap, stała liczba drzew M).
Etap 5: Adaptive Sampling  -> `_sample_indices`.
Etap 6: Ensemble-Aware ES   -> `_fit_one`.
"""

from __future__ import annotations

import copy
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
      n_trees_total_    łączna liczba drzew w zespole (w clf wieloklasowej K drzew na iterację)
      n_trees_trained_  łączna liczba zbudowanych drzew (z ES: także te za najlepszym checkpointem)
      sampling_stats_   diagnostyka rozkładu P(i) (Adaptive Sampling)
      es_history_       lista (liczba iteracji, strata walidacyjna zespołu) na model (ES)
      fit_time_         czas treningu w sekundach (z narzutem modyfikacji)
    """

    def __init__(self, task: str = "clf", B: int = 10, M: int = 200,
                 max_depth: int = 3, learning_rate: float = 0.1,
                 adaptive_sampling: bool = False, alpha: float = 1.0, beta: float = 1.0,
                 ensemble_es: bool = False, k: int = 20, p: int = 2,
                 es_metric: str = "auto", es_tol: float = 1e-3, random_state: int | None = None):
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
        self.es_metric = es_metric
        self.es_tol = es_tol
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
        self.sampling_stats_ = []
        self.es_history_ = []
        self._train_outputs = []  # predykcje modeli na X (do OOB w Adaptive Sampling)
        self._val_sum = None      # suma predykcji gotowych modeli na X_val (do ES)
        self._n_trained = 0

        for b in range(self.B):
            idx = self._sample_indices(b, X, y, rng)
            self.inbag_[b, np.unique(idx)] = True
            seed = int(rng.integers(2**31 - 1))
            model = self._fit_one(X[idx], y[idx], X_val, y_val, seed)
            self.estimators_.append(model)
            if self.adaptive_sampling and b < self.B - 1:
                self._train_outputs.append(self._model_output(model, X))
            if self.ensemble_es:
                out = self._model_output(model, X_val)
                self._val_sum = out if self._val_sum is None else self._val_sum + out
        self.n_trees_trained_ = self._n_trained
        del self._train_outputs, self._val_sum, self._n_trained

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
        """Indeksy próbki dla modelu b.

        Baseline (oraz model 1 w Adaptive Sampling): bootstrap 100% ze zwracaniem.
        Adaptive Sampling: losowanie ze zwracaniem z P(i) = w_i / sum(w),
        w_i = 1 + alpha * e_i + beta * d_i (sygnały z predykcji OOB modeli 1..b-1).
        """
        probs = None
        if self.adaptive_sampling and b > 0:
            e, d, has_oob = self._oob_signals(y, b)
            w = 1.0 + self.alpha * e + self.beta * d
            probs = w / w.sum()
            self.sampling_stats_.append(_sampling_stats(b, probs, has_oob))
        return self._draw(len(y), y, probs, rng)

    def _oob_signals(self, y: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Błąd e_i i rozbieżność d_i z predykcji OOB modeli 0..b-1, znormalizowane do [0, 1].

        Dla obserwacji i bierzemy tylko modele, których próbka jej nie zawierała.
          reg: e_i = |y_i - średnia OOB|,      d_i = std predykcji OOB
          clf: e_i = 1 - p_OOB(prawdziwa klasa), d_i = średnia po klasach wariancji proba
        Obserwacje bez żadnego modelu OOB: e_i = d_i = 0 (waga 1).
        d_i wymaga co najmniej 2 modeli OOB, inaczej 0.
        """
        out = np.stack(self._train_outputs[:b])          # (b, n) albo (b, n, K)
        oob = ~self.inbag_[:b]                           # (b, n)
        cnt = oob.sum(axis=0)                            # (n,)
        has_oob = cnt > 0
        mask = oob if out.ndim == 2 else oob[..., None]
        denom = np.maximum(cnt, 1) if out.ndim == 2 else np.maximum(cnt, 1)[:, None]

        mean = (out * mask).sum(axis=0) / denom
        var = np.maximum((out**2 * mask).sum(axis=0) / denom - mean**2, 0.0)

        if self.task == "reg":
            e = np.abs(y - mean)
            d = np.sqrt(var)
        else:
            true_col = np.searchsorted(self.classes_, y)
            e = 1.0 - mean[np.arange(len(y)), true_col]
            d = var.mean(axis=1)

        e = np.where(has_oob, e, 0.0)
        d = np.where(cnt >= 2, d, 0.0)
        return _normalize(e, has_oob), _normalize(d, cnt >= 2), has_oob

    def _fit_one(self, X: np.ndarray, y: np.ndarray,
                 X_val: np.ndarray | None, y_val: np.ndarray | None,
                 seed: int):
        """Trening jednego GBDT.

        Baseline: stałe M drzew.
        Ensemble-Aware ES: co k iteracji checkpoint – strata walidacyjna całego zespołu
        (gotowe modele + bieżący). Poprawa = spadek straty o więcej niż es_tol
        (względnie, np. 1e-3 = 0.1%). Przy poprawie zapamiętujemy kopię modelu; po p
        checkpointach bez poprawy (lub po M iteracjach) zwracamy najlepszą kopię.
        """
        if not self.ensemble_es:
            model = self._new_gbdt(seed).fit(X, y)
            self._n_trained += model.estimators_.size
            return model

        n_done = len(self.estimators_)
        model = self._new_gbdt(seed, n_estimators=self.k, warm_start=True)
        best_loss, best_model, no_improve, history = np.inf, None, 0, []
        while True:
            model.fit(X, y)
            out = self._model_output(model, X_val)
            ens = out if self._val_sum is None else (self._val_sum + out) / (n_done + 1)
            loss = self._es_loss(y_val, ens)
            history.append((model.n_estimators_, loss))
            if loss < best_loss * (1.0 - self.es_tol) or best_model is None:
                best_loss, best_model, no_improve = loss, copy.deepcopy(model), 0
            else:
                no_improve += 1
            if no_improve >= self.p or model.n_estimators_ >= self.M:
                break
            model.n_estimators = min(model.n_estimators_ + self.k, self.M)

        self._n_trained += model.estimators_.size
        self.es_history_.append(history)
        best_model.warm_start = False
        return best_model

    def _es_loss(self, y_val: np.ndarray, ens: np.ndarray) -> float:
        """Strata walidacyjna zespołu (mniejsza = lepsza).

        es_metric="auto": log-loss (clf) / MAE (reg). Log-loss zamiast Macro F1, bo F1
        na małej walidacji jest skokowe i często stoi w miejscu -> zbyt wczesny stop.
        Dostępne też: "f1" (clf), "mae", "mse" (reg).
        """
        metric = self.es_metric
        if metric == "auto":
            metric = "logloss" if self.task == "clf" else "mae"
        if self.task == "clf":
            cols = np.searchsorted(self.classes_, y_val)
            if metric == "logloss":
                p_true = np.clip(ens[np.arange(len(y_val)), cols], 1e-15, 1.0)
                return float(-np.mean(np.log(p_true)))
            if metric == "f1":
                from sklearn.metrics import f1_score
                return -f1_score(y_val, self.classes_[ens.argmax(1)], average="macro")
        else:
            if metric == "mae":
                return float(np.mean(np.abs(y_val - ens)))
            if metric == "mse":
                return float(np.mean((y_val - ens) ** 2))
        raise ValueError(f"Nieobsługiwana es_metric={self.es_metric!r} dla task={self.task!r}")

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


# --------------------------------------------------------------------------- #
# Funkcje pomocnicze Adaptive Sampling
# --------------------------------------------------------------------------- #

NORM_QUANTILE = 0.99


def _normalize(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Skaluje do [0, 1]: dzieli przez kwantyl 99% (wśród `valid`) i obcina do 1.

    Kwantyl zamiast maksimum, żeby pojedyncza obserwacja odstająca (np. bardzo duży
    błąd w regresji) nie spłaszczyła wag wszystkich pozostałych do ~0.
    """
    if not valid.any():
        return np.zeros_like(x)
    scale = np.quantile(x[valid], NORM_QUANTILE)
    if scale <= 0:
        scale = x[valid].max()
    if scale <= 0:
        return np.zeros_like(x)
    return np.clip(x / scale, 0.0, 1.0)


def _sampling_stats(b: int, probs: np.ndarray, has_oob: np.ndarray) -> dict:
    """Diagnostyka rozkładu P(i): czy nie jest zdegenerowany."""
    n = len(probs)
    return {
        "model": b,
        "p_min_x_n": float(probs.min() * n),   # 1.0 = jak w zwykłym bootstrapie
        "p_max_x_n": float(probs.max() * n),
        "max_min_ratio": float(probs.max() / probs.min()),
        "ess_frac": float(1.0 / (probs**2).sum() / n),  # efektywna liczność / n
        "oob_coverage": float(has_oob.mean()),
    }
