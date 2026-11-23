"""Ładowanie zbiorów z UCI i preprocessing.

Każdy zbiór jest opisany w DATASETS. Surowe dane (wybrane cechy + cel) są
cache'owane w data/<name>.csv, więc UCI odpytujemy tylko raz.

Preprocessing w load_dataset (bez wycieku informacji z foldów testowych):
  - kategoryczne -> one-hot; brak wartości to osobna kategoria,
  - liczbowe -> zostają z NaN; medianę uzupełnia `impute_median`
    dopasowane na części treningowej foldu,
  - cel klasyfikacji -> etykiety 0..K-1.
"""

from __future__ import annotations

import io
import warnings
import zipfile
from dataclasses import dataclass, field
from typing import Callable
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import DATA_DIR

TARGET = "__target__"


@dataclass(frozen=True)
class DatasetSpec:
    task: str                              # "clf" | "reg"
    uci_id: int
    target: str | None = None              # kolumna celu (None = jedyna kolumna targets)
    drop: tuple[str, ...] = ()             # kolumny do usunięcia (ID, przecieki)
    categorical: tuple[str, ...] = ()      # liczbowe kolumny traktowane jako kategoryczne
    loader: Callable[[], pd.DataFrame] | None = field(default=None, compare=False)


# --------------------------------------------------------------------------- #
# Loadery niestandardowe (zbiory niedostępne w ucimlrepo)
# --------------------------------------------------------------------------- #

def _load_yacht() -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip"
    with urlopen(url, timeout=60) as r, zipfile.ZipFile(io.BytesIO(r.read())) as z:
        raw = z.read("yacht_hydrodynamics.data").decode()
    cols = ["lc_position", "prismatic_coef", "length_displacement",
            "beam_draught", "length_beam", "froude_number", TARGET]
    return pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None, names=cols)


def _load_wine_quality_white() -> pd.DataFrame:
    """ucimlrepo łączy wina czerwone i białe; plan: tylko winequality-white."""
    from ucimlrepo import fetch_ucirepo

    df = fetch_ucirepo(id=186).data.original
    df = df[df["color"] == "white"].drop(columns=["color"])
    return df.rename(columns={"quality": TARGET}).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Rejestr zbiorów (plan, tabele 1 i 2)
# --------------------------------------------------------------------------- #

DATASETS: dict[str, DatasetSpec] = {
    # klasyfikacja
    "wine": DatasetSpec("clf", 109),
    "breast_cancer": DatasetSpec("clf", 17),
    "banknote": DatasetSpec("clf", 267),
    "ionosphere": DatasetSpec("clf", 52),
    "car": DatasetSpec("clf", 19),
    "spambase": DatasetSpec("clf", 94),
    "dry_bean": DatasetSpec("clf", 602),
    "glass": DatasetSpec("clf", 42),
    "german_credit": DatasetSpec("clf", 144),
    "sonar": DatasetSpec("clf", 151),
    "rice": DatasetSpec("clf", 545),
    "mushroom": DatasetSpec("clf", 73),
    # regresja
    "yacht": DatasetSpec("reg", 243, loader=_load_yacht),
    "auto_mpg": DatasetSpec("reg", 9, categorical=("origin",)),
    "concrete": DatasetSpec("reg", 165),
    "abalone": DatasetSpec("reg", 1),
    "energy": DatasetSpec("reg", 242, target="Y1"),                  # Heating Load
    "airfoil": DatasetSpec("reg", 291),
    "ccpp": DatasetSpec("reg", 294),
    "real_estate": DatasetSpec("reg", 477),
    "forest_fires": DatasetSpec("reg", 162),
    "wine_quality_white": DatasetSpec("reg", 186, loader=_load_wine_quality_white),
    "student_por": DatasetSpec("reg", 320, target="G3"),              # G1, G2 są celami, nie cechami
    "computer_hardware": DatasetSpec("reg", 29, target="PRP",
                                     drop=("ERP", "VendorName", "ModelName")),
}


def _fetch_from_uci(spec: DatasetSpec) -> pd.DataFrame:
    """Cechy z ucimlrepo + kolumna TARGET.

    Cel bierzemy z `targets`; gdy zbiór go tam nie ma (Computer Hardware: PRP jest
    wśród cech), wyciągamy go z cech. Pozostałe kolumny celów nie trafiają do cech.
    """
    from ucimlrepo import fetch_ucirepo

    d = fetch_ucirepo(id=spec.uci_id)
    X, y = d.data.features.copy(), d.data.targets
    if spec.target is not None and spec.target in X.columns:
        X[TARGET] = X.pop(spec.target)
        return X
    target = spec.target or y.columns[0]
    X[TARGET] = y[target].values
    return X


def load_raw(name: str) -> pd.DataFrame:
    """Surowa ramka: cechy + kolumna TARGET, po usunięciu kolumn z `drop`."""
    spec = DATASETS[name]
    path = DATA_DIR / f"{name}.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = spec.loader() if spec.loader else _fetch_from_uci(spec)
        df = df.drop(columns=[c for c in spec.drop if c in df.columns])
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Zwraca (X, y, task).

    X: float64, kategoryczne zakodowane one-hot; liczbowe mogą mieć NaN
       (uzupełnić `impute_median` w obrębie foldu).
    y: int 0..K-1 dla "clf", float64 dla "reg".
    """
    spec = DATASETS[name]
    df = load_raw(name)
    df = df.dropna(subset=[TARGET])
    y_raw = df.pop(TARGET)

    cat_cols = [c for c in df.columns
                if c in spec.categorical or not pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols:
        df[cat_cols] = df[cat_cols].astype("string")
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False, dtype=float)
        # brak wartości w kategorycznych -> osobna kategoria (wszystkie dummies = 0)
    X = df.to_numpy(dtype=np.float64)

    if spec.task == "clf":
        _, y = np.unique(y_raw.to_numpy(), return_inverse=True)
        y = y.astype(np.int64)
    else:
        y = y_raw.to_numpy(dtype=np.float64)
    return X, y, spec.task


def impute_median(X_train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Uzupełnia NaN medianą z X_train (kolumny w całości NaN -> 0)."""
    if not (np.isnan(X_train).any() or any(np.isnan(X).any() for X in others)):
        return (X_train, *others)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # kolumny w całości NaN
        med = np.nan_to_num(np.nanmedian(X_train, axis=0), nan=0.0)

    def fill(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        rows, cols = np.where(np.isnan(X))
        X[rows, cols] = med[cols]
        return X

    return (fill(X_train), *(fill(X) for X in others))


def dataset_summary(name: str) -> dict:
    X, y, task = load_dataset(name)
    info = {"name": name, "task": task, "n": X.shape[0], "d": X.shape[1],
            "nan": int(np.isnan(X).sum())}
    if task == "clf":
        info["classes"] = np.bincount(y).tolist()
    else:
        info["y_mean"], info["y_std"] = round(float(y.mean()), 3), round(float(y.std()), 3)
    return info
