"""Wspólna konfiguracja eksperymentów."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Walidacja: 2x powtórzone 5-fold CV
CV_SEEDS = (42, 2026)
N_FOLDS = 5
INNER_VAL_SIZE = 0.2

# Stałe hiperparametry pilotażu (bez strojenia)
PILOT_PARAMS = {
    "B": 10,             # liczba modeli GBDT w baggingu
    "M": 200,            # maks. liczba drzew w jednym GBDT
    "max_depth": 3,
    "learning_rate": 0.1,
    "alpha": 1.0,        # Adaptive Sampling: waga błędu
    "beta": 1.0,         # Adaptive Sampling: waga rozbieżności
    "k": 20,             # Early Stopping: checkpoint co k drzew
    "p": 2,              # Early Stopping: cierpliwość (checkpointy bez poprawy)
}

# Referencje: Bagging drzew decyzyjnych i Random Forest (parametry domyślne sklearn)
REF_PARAMS = {
    "n_estimators": 100,
}

PILOT_DATASETS = {
    "clf": ["wine", "breast_cancer", "banknote"],
    "reg": ["yacht", "auto_mpg", "concrete"],
}
