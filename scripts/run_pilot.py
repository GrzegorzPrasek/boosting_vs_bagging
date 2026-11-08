"""Pilotaż: 6 zbiorów x 7 metod x 10 foldów (2x 5-fold CV), stałe hiperparametry.

Uruchomienie:
    .venv/Scripts/python.exe scripts/run_pilot.py            # wznawia, jeśli CSV istnieje
    .venv/Scripts/python.exe scripts/run_pilot.py --fresh    # od zera
    .venv/Scripts/python.exe scripts/run_pilot.py --report   # tylko tabele z istniejącego CSV

Wyniki:
    results/pilot.csv         wiersz na (zbiór, metoda, seed, fold)
    results/pilot_summary.md  tabele jakości, kosztu, rankingi, szacunek czasu
    results/pilot_meta.json   parametry, wersje bibliotek, sprzęt
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from src.config import (CV_SEEDS, N_FOLDS, PILOT_DATASETS, PILOT_PARAMS,  # noqa: E402
                        REF_PARAMS, RESULTS_DIR)
from src.data import load_dataset  # noqa: E402
from src.evaluation import (ALL_METHODS, BAGGED_VARIANTS, MAIN_METRIC,  # noqa: E402
                            run_cv, summarize)

CSV = RESULTS_DIR / "pilot.csv"
SUMMARY = RESULTS_DIR / "pilot_summary.md"
META = RESULTS_DIR / "pilot_meta.json"
N_RUNS = len(CV_SEEDS) * N_FOLDS

# --------------------------------------------------------------------------- #
# Pozostałe 18 zbiorów z planu: (n, przybliżona liczba cech po one-hot, K klas)
# Liczba cech jest szacunkowa (dokładna wyjdzie po napisaniu loaderów).
# --------------------------------------------------------------------------- #
OTHER_DATASETS = {
    "clf": {
        "ionosphere": (351, 34, 2), "car": (1728, 21, 4), "spambase": (4601, 57, 2),
        "dry_bean": (13611, 16, 7), "glass": (214, 9, 6), "german_credit": (1000, 61, 2),
        "sonar": (208, 60, 2), "rice": (3810, 7, 2), "mushroom": (8124, 117, 2),
    },
    "reg": {
        "abalone": (4177, 10, 1), "energy": (768, 8, 1), "airfoil": (1503, 5, 1),
        "ccpp": (9568, 4, 1), "real_estate": (414, 6, 1), "forest_fires": (517, 29, 1),
        "wine_quality_white": (4898, 11, 1), "student_por": (649, 56, 1),
        "computer_hardware": (209, 6, 1),
    },
}
TUNING_CONFIGS = 8  # random search na zbiór x fold (plan, rozdz. 6.3)


# --------------------------------------------------------------------------- #
# Uruchomienie
# --------------------------------------------------------------------------- #

def run(fresh: bool) -> None:
    if fresh and CSV.exists():
        CSV.unlink()
    done = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=["dataset", "method"])
    counts = done.groupby(["dataset", "method"]).size()

    for task, names in PILOT_DATASETS.items():
        for name in names:
            todo = [m for m in ALL_METHODS if counts.get((name, m), 0) < N_RUNS]
            if not todo:
                print(f"[skip] {name}: komplet")
                continue
            # niekompletne pary (przerwany przebieg) liczymy od nowa
            if CSV.exists():
                df = pd.read_csv(CSV)
                df = df[~((df.dataset == name) & df.method.isin(todo))]
                df.to_csv(CSV, index=False)
            print(f"[run] {name}: {', '.join(todo)}", flush=True)
            run_cv(name, methods=todo, params=PILOT_PARAMS, out_csv=CSV,
                   log=lambda s: print(s, flush=True))


# --------------------------------------------------------------------------- #
# Raport
# --------------------------------------------------------------------------- #

def avg_ranks(df: pd.DataFrame, task: str) -> pd.Series:
    metric = MAIN_METRIC[task]
    means = df.groupby(["dataset", "method"])[metric].mean().unstack("method")
    ranks = means.rank(axis=1, ascending=(task == "reg"))
    order = [m for m in ALL_METHODS if m in ranks.columns]
    return ranks.mean()[order]


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    order = [m for m in ALL_METHODS if m in set(df.method)]
    g = df.groupby(["dataset", "method"]).agg(
        trees=("n_trees", "mean"), trained=("n_trees_trained", "mean"),
        stages=("n_stages_mean", "mean"), fit=("fit_time", "mean"),
        pred=("predict_time", "mean"))
    out = pd.DataFrame({
        "drzewa (zbud.)": g.trained.round(0).astype(int).astype(str),
        "fit [s]": g.fit.map("{:.2f}".format),
    })
    return out.unstack("method").swaplevel(axis=1)[order]


def es_savings(df: pd.DataFrame) -> pd.DataFrame:
    """Oszczędność zbudowanych drzew i czasu wariantów z ES względem odpowiednika bez ES."""
    rows = []
    pairs = [("Baseline+2", "Baseline"), ("Baseline+1+2", "Baseline+1")]
    g = df.groupby(["dataset", "method"])[["n_trees_trained", "fit_time"]].mean()
    for ds in df.dataset.unique():
        for es, ref in pairs:
            if (ds, es) in g.index and (ds, ref) in g.index:
                a, b = g.loc[(ds, es)], g.loc[(ds, ref)]
                rows.append({"dataset": ds, "porównanie": f"{es} vs {ref}",
                             "drzewa": f"{a.n_trees_trained / b.n_trees_trained - 1:+.0%}",
                             "czas": f"{a.fit_time / b.fit_time - 1:+.0%}"})
    return pd.DataFrame(rows)


def time_estimate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Szacunek czasu pełnych eksperymentów (24 zbiory, strojenie).

    Model kosztu: fit_time ~ c_metoda * n_train * log2(n_train) * d * K,
    gdzie K = liczba klas dla clf wieloklasowej (sklearn buduje K drzew na iterację),
    1 dla binarnej i regresji. c_metoda = mediana z pilotażu.
    Strojenie: TUNING_CONFIGS konfiguracji na fold; średnia konfiguracja z siatki
    (B in {5,10}, M in {100,200}) kosztuje ok. 0.75 * 0.75 = 0.56 konfiguracji pilotażowej.
    """
    pilot_shape = {}
    for task, names in PILOT_DATASETS.items():
        for n in names:
            X, y, _ = load_dataset(n)
            k = len(np.unique(y)) if task == "clf" else 1
            pilot_shape[n] = (X.shape[0], X.shape[1], k)

    def complexity(n, d, k):
        n_tr = n * 0.8 * 0.8
        return n_tr * np.log2(n_tr) * d * (k if k > 2 else 1)

    g = df.groupby(["dataset", "method"]).fit_time.mean().reset_index()
    g["c"] = g.apply(lambda r: r.fit_time / complexity(*pilot_shape[r.dataset]), axis=1)
    c = g.groupby("method").c.median()

    all_ds = {**{n: s for n, s in pilot_shape.items()},
              **OTHER_DATASETS["clf"], **OTHER_DATASETS["reg"]}
    tuning_factor = 1 + TUNING_CONFIGS * 0.56
    rows = []
    for ds, shape in all_ds.items():
        per_fit = {m: c[m] * complexity(*shape) for m in c.index}
        rows.append({"dataset": ds, "n": shape[0], "d≈": shape[1], "K": shape[2],
                     "1 przebieg CV, wszystkie metody [min]": sum(per_fit.values()) * N_RUNS / 60})
    est = pd.DataFrame(rows).sort_values("1 przebieg CV, wszystkie metody [min]", ascending=False)
    total_no_tuning = est["1 przebieg CV, wszystkie metody [min]"].sum() / 60
    info = {
        "bez strojenia [h, 1 rdzeń]": total_no_tuning,
        "ze strojeniem [h, 1 rdzeń]": total_no_tuning * tuning_factor,
        "ze strojeniem [h, 8 rdzeni]": total_no_tuning * tuning_factor / 8,
    }
    return est, info


def report() -> None:
    df = pd.read_csv(CSV)
    lines = [f"# Pilotaż – wyniki\n",
             f"Wygenerowano: {datetime.now():%Y-%m-%d %H:%M}. "
             f"2×5-fold CV (ziarna {CV_SEEDS}), stałe hiperparametry: `{PILOT_PARAMS}`. "
             f"Referencje Bagging/RF: `{REF_PARAMS}`.\n"]

    for task, title, arrow in (("clf", "Klasyfikacja – Macro F1", "↑"),
                               ("reg", "Regresja – MAE", "↓")):
        d = df[df.task == task]
        if d.empty:
            continue
        lines += [f"## {title} ({arrow}), średnia ± std z 10 przebiegów\n",
                  summarize(d).to_markdown(), "",
                  "Średnia pozycja w rankingu (1 = najlepsza):\n",
                  avg_ranks(d, task).round(2).to_frame("ranga").T.to_markdown(), ""]

    lines += ["## Koszt: zbudowane drzewa i czas treningu (średnia na fold)\n",
              cost_table(df).to_markdown(), "",
              "## Oszczędność Ensemble-Aware ES\n",
              es_savings(df).to_markdown(index=False), ""]

    est, info = time_estimate(df)
    lines += ["## Szacunek czasu pełnych eksperymentów\n",
              "Model: czas ~ c_metoda · n·log n · d · K, c z pilotażu; d dla nowych zbiorów "
              "przybliżone. Wynik orientacyjny.\n",
              est.round(1).to_markdown(index=False), "",
              *[f"- **{k}**: {v:.1f}" for k, v in info.items()], ""]

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"Zapisano {SUMMARY}")


def write_meta() -> None:
    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "params": PILOT_PARAMS, "ref_params": REF_PARAMS,
        "cv_seeds": CV_SEEDS, "n_folds": N_FOLDS,
        "datasets": PILOT_DATASETS,
        "methods": list(ALL_METHODS), "bagged_variants": BAGGED_VARIANTS,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="usuń istniejący CSV i licz od zera")
    ap.add_argument("--report", action="store_true", help="tylko wygeneruj raport z CSV")
    args = ap.parse_args()
    if not args.report:
        write_meta()
        run(args.fresh)
    report()
