# Adaptive Bagged Boosting

Projekt z Ensemble Learning (PW, 2026/2027): bagging modeli Gradient Boosting z dwoma modyfikacjami:

1. **Adaptive Sampling** – próbki dla kolejnych modeli losowane z wagami `w_i = 1 + α·e_i + β·d_i` (błąd i rozbieżność predykcji OOB dotychczasowego zespołu),
2. **Ensemble-Aware Early Stopping** – liczba drzew każdego GBDT dobierana na podstawie jakości całego zespołu na walidacji wewnętrznej.

Plan projektu: `uczenie_zespołowe (1).pdf`. Stan na checkpoint: `docs/checkpoint_2026-11-10.md`.

## Struktura

```
src/
  config.py       ziarna CV, hiperparametry pilotażu, ścieżki
  data.py         loadery zbiorów UCI + preprocessing (cache w data/)
  models.py       BaggedGBDT: Baseline, +1, +2, +1+2
  evaluation.py   2x5-fold CV, metryki, referencje (GBDT, Bagging, RF), zapis CSV
scripts/
  run_pilot.py    pilotaż: 6 zbiorów x 7 metod x 10 przebiegów
tests/            testy poprawności (pytest)
results/          wyniki (pilot.csv, pilot_summary.md, pilot_meta.json)
docs/             materiały na checkpointy
```

## Uruchomienie

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
.venv/Scripts/python.exe -m pytest -q                          # testy
.venv/Scripts/python.exe scripts/run_pilot.py --fresh          # pilotaż od zera (~15 min)
.venv/Scripts/python.exe scripts/run_pilot.py --report         # tylko raport z istniejącego CSV
```

Na Windows przy przekierowaniu wyjścia do pliku ustaw `PYTHONIOENCODING=utf-8`.

## Użycie modelu

```python
from src.data import load_dataset, impute_median
from src.models import BaggedGBDT

X, y, task = load_dataset("wine")
model = BaggedGBDT(task=task, B=10, M=200, adaptive_sampling=True, ensemble_es=True,
                   random_state=0)
model.fit(X_train, y_train, X_val, y_val)   # walidacja wymagana tylko dla ensemble_es
model.predict(X_test)
```

Dodanie zbioru: jedna linia w `DATASETS` w `src/data.py` (ID z UCI, typ zadania, ewentualnie kolumny do usunięcia).
