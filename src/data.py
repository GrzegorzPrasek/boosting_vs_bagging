"""Ładowanie zbiorów z UCI i preprocessing (Etap 2)."""

import numpy as np


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Zwraca (X, y, task), gdzie task to "clf" albo "reg". X bez NaN, po one-hot."""
    raise NotImplementedError
