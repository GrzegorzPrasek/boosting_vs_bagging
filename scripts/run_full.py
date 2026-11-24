"""Pełne eksperymenty: 24 zbiory x 7 metod x 10 foldów, ze strojeniem (random search).

Zadania (zbiór, metoda, seed, fold) liczą się równolegle (joblib, domyślnie 8 procesów,
każdy jednowątkowy). Wyniki dopisywane na bieżąco; przerwany przebieg można wznowić.

    .venv/Scripts/python.exe scripts/run_full.py                 # start / wznowienie
    .venv/Scripts/python.exe scripts/run_full.py --fresh         # od zera
    .venv/Scripts/python.exe scripts/run_full.py --datasets wine yacht --n-configs 2   # test

Wyniki:
    results/full.csv         wiersz na (zbiór, metoda, seed, fold) + wybrane parametry
    results/full_meta.json   ustawienia, wersje bibliotek, sprzęt, czas przebiegu
    results/full_errors.txt  zadania zakończone błędem (jeśli są)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from joblib import Parallel, delayed, parallel_config  # noqa: E402

from src.config import CV_SEEDS, N_FOLDS, RESULTS_DIR  # noqa: E402
from src.data import DATASETS, load_dataset  # noqa: E402
from src.evaluation import ALL_METHODS, VARIANT_FLAGS  # noqa: E402
from src.tuning import (AS_PAIRS, BAGGING_SPACE, BASE_SPACE, ES_SPACE,  # noqa: E402
                        FULL_COLUMNS, GBDT_SPACE, N_CONFIGS, RF_SPACE, all_tasks, run_task)


def _safe_task(*args, **kwargs):
    """Błąd jednego zadania nie przerywa całego przebiegu."""
    try:
        return run_task(*args, **kwargs), None
    except Exception:  # noqa: BLE001
        return None, (args, traceback.format_exc())


def _cost_key(task, sizes):
    """Szacowany koszt zadania – najdłuższe uruchamiamy najpierw (lepsze upakowanie)."""
    ds, method = task[0], task[1]
    n, d, k = sizes[ds]
    base = n * np.log2(max(n, 2)) * d * (k if k > 2 else 1)
    return -base * (10 if method in VARIANT_FLAGS else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--methods", nargs="*", default=list(ALL_METHODS))
    ap.add_argument("--n-configs", type=int, default=N_CONFIGS)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", default=str(RESULTS_DIR / "full.csv"))
    ap.add_argument("--noise", nargs="*", type=float, default=[0.0],
                    help="poziomy szumu w etykietach (train+val), np. 0.1 0.2")
    ap.add_argument("--exclude", nargs="*", default=[], help="zbiory do pominięcia")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.fresh and out.exists():
        out.unlink()

    datasets = [d for d in args.datasets if d not in args.exclude]
    tasks = all_tasks(datasets, args.methods, noises=args.noise)
    columns = FULL_COLUMNS + ["noise"]
    if out.exists():
        done = pd.read_csv(out)
        columns = list(done.columns)                  # zgodność z istniejącym plikiem
        nz = done["noise"] if "noise" in done else pd.Series(0.0, index=done.index)
        done_set = set(zip(done.dataset, done.method, done.seed, done.fold, nz.astype(float)))
        tasks = [t for t in tasks if t not in done_set]

    sizes = {}
    for ds in {t[0] for t in tasks}:
        X, y, task = load_dataset(ds)
        sizes[ds] = (X.shape[0], X.shape[1], len(np.unique(y)) if task == "clf" else 1)
    tasks.sort(key=lambda t: _cost_key(t, sizes))

    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "n_configs": args.n_configs, "n_jobs": args.n_jobs, "cv_seeds": CV_SEEDS,
        "n_folds": N_FOLDS, "datasets": datasets, "methods": args.methods,
        "noise": args.noise,
        "spaces": {"bagged_base": BASE_SPACE, "as_pairs": AS_PAIRS, "es": ES_SPACE,
                   "GBDT": GBDT_SPACE, "Bagging": BAGGING_SPACE, "RF": RF_SPACE},
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
    }
    print(f"Zadań do policzenia: {len(tasks)} (n_jobs={args.n_jobs}, "
          f"konfiguracji={args.n_configs})", flush=True)

    t_start = time.time()
    n_ok = n_err = 0
    err_path = out.with_name(out.stem + "_errors.txt")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
            delayed(_safe_task)(*t, n_configs=args.n_configs) for t in tasks)
        for row, err in results:
            if err is not None:
                n_err += 1
                with open(err_path, "a", encoding="utf-8") as f:
                    f.write(f"{err[0]}\n{err[1]}\n")
                print(f"[BŁĄD] {err[0]}", flush=True)
                continue
            n_ok += 1
            pd.DataFrame([row]).reindex(columns=columns).to_csv(
                out, mode="a", header=not out.exists(), index=False)
            el = time.time() - t_start
            eta = el / (n_ok + n_err) * (len(tasks) - n_ok - n_err)
            print(f"[{n_ok + n_err}/{len(tasks)}] {row['dataset']:18s} {row['method']:13s} "
                  f"s={row['seed']} f={row['fold']} nz={row['noise']} tune={row['tune_time']:.0f}s "
                  f"| {el / 60:.0f} min, ETA ~{eta / 60:.0f} min", flush=True)

    meta.update(finished=datetime.now().isoformat(timespec="seconds"),
                wall_time_min=round((time.time() - t_start) / 60, 1),
                tasks_ok=n_ok, tasks_failed=n_err)
    out.with_name(out.stem + f"_meta_{datetime.now():%H%M%S}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Koniec: {n_ok} OK, {n_err} błędów, {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
