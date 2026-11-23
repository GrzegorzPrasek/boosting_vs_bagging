"""Analiza wyników (plan, rozdz. 5.4): tabele, rangi, testy Friedmana i Wilcoxona, koszt.

    .venv/Scripts/python.exe scripts/analyze.py                         # results/full.csv
    .venv/Scripts/python.exe scripts/analyze.py --csv results/pilot.csv --out results/_test

Wyniki (w katalogu --out, domyślnie results/):
    summary.md           wszystkie tabele i testy
    per_dataset_*.csv    średnia i std metryk dla (zbiór, metoda)

Zasady:
  - klasyfikacja i regresja analizowane osobno,
  - jednostką testów statystycznych jest zbiór danych (średnia z 10 przebiegów),
  - w regresji do testów i porównań przekrojowych używamy MAE znormalizowanego:
    MAE / std(y w części treningowej foldu), bo surowe MAE ma różne skale,
  - Friedman (alpha = 0,05) dla wszystkich 7 metod; potem Wilcoxon dla par z planu,
    z korektą Holma na wielokrotne porównania (podajemy też surowe p).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.evaluation import ALL_METHODS, iter_splits  # noqa: E402

ALPHA = 0.05
PAIRS = [  # (A, B): plan 5.4
    ("GBDT", "Baseline"),
    ("Baseline", "Baseline+1"),
    ("Baseline", "Baseline+2"),
    ("Baseline", "Baseline+1+2"),
    ("Baseline+1", "Baseline+1+2"),
    ("Baseline+2", "Baseline+1+2"),
]
TASK_INFO = {
    "clf": {"metric": "f1_macro", "label": "Macro F1", "higher": True, "aux": "accuracy"},
    "reg": {"metric": "nmae", "label": "MAE / std(y)", "higher": False, "aux": "rmse"},
}


# --------------------------------------------------------------------------- #
# Przygotowanie danych
# --------------------------------------------------------------------------- #

def add_normalized_mae(df: pd.DataFrame) -> pd.DataFrame:
    """nMAE = MAE / std(y_train) dla każdego (zbiór, seed, fold)."""
    df = df.copy()
    df["nmae"] = np.nan
    for ds in df.loc[df.task == "reg", "dataset"].unique():
        X, y, task = load_dataset(ds)
        sd = {(s, f): y[tr].std() for s, f, tr, _, _ in iter_splits(X, y, task)}
        m = df.dataset == ds
        df.loc[m, "nmae"] = df.loc[m, "mae"] / [sd[(s, f)] for s, f in
                                               zip(df.loc[m, "seed"], df.loc[m, "fold"])]
    return df


def methods_in(df):
    return [m for m in ALL_METHODS if m in set(df.method)]


def means(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Macierz zbiór x metoda: średnia z 10 przebiegów."""
    return df.groupby(["dataset", "method"])[col].mean().unstack("method")[methods_in(df)]


def ranks(m: pd.DataFrame, higher: bool) -> pd.DataFrame:
    return m.rank(axis=1, ascending=not higher)


# --------------------------------------------------------------------------- #
# Testy
# --------------------------------------------------------------------------- #

def friedman(m: pd.DataFrame) -> tuple[float, float]:
    stat, p = stats.friedmanchisquare(*[m[c].values for c in m.columns])
    return float(stat), float(p)


def holm(pvals: list[float]) -> list[float]:
    """Skorygowane p-wartości metodą Holma–Bonferroniego."""
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def wilcoxon_pairs(m: pd.DataFrame, higher: bool) -> pd.DataFrame:
    rows = []
    for a, b in PAIRS:
        if a not in m or b not in m:
            continue
        diff = m[b] - m[a]
        better = diff > 0 if higher else diff < 0
        nz = diff[diff != 0]
        p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 2 else np.nan
        rows.append({"porównanie": f"{b} vs {a}",
                     "B lepsze / gorsze / remis": f"{int(better.sum())} / "
                     f"{int(((diff != 0) & ~better).sum())} / {int((diff == 0).sum())}",
                     "mediana różnicy (B − A)": float(diff.median()),
                     "p": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        valid = out["p"].notna()
        out.loc[valid, "p (Holm)"] = holm(out.loc[valid, "p"].tolist())
        out["istotne (Holm, 0.05)"] = out["p (Holm)"] < ALPHA
    return out


# --------------------------------------------------------------------------- #
# Tabele
# --------------------------------------------------------------------------- #

def mean_std_table(df: pd.DataFrame, col: str, fmt: str) -> pd.DataFrame:
    g = df.groupby(["dataset", "method"])[col].agg(["mean", "std"])
    cell = g["mean"].map(fmt.format) + " ± " + g["std"].map(fmt.format)
    return cell.unstack("method")[methods_in(df)]


def bold_best(table: pd.DataFrame, m: pd.DataFrame, higher: bool) -> pd.DataFrame:
    t = table.copy()
    best = m.idxmax(axis=1) if higher else m.idxmin(axis=1)
    for ds, meth in best.items():
        t.loc[ds, meth] = f"**{t.loc[ds, meth]}**"
    return t


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = {"n_trees": "drzewa w modelu", "n_trees_trained": "drzewa zbudowane",
            "fit_time": "trening [s]", "predict_time": "predykcja [s]"}
    if "tune_time" in df:
        cols["tune_time"] = "strojenie [s]"
    g = df.groupby("method")[list(cols)].mean().rename(columns=cols).loc[methods_in(df)]
    # względem Baseline, liczone na zbiorach i uśredniane (każdy zbiór waży tyle samo)
    if "Baseline" in set(df.method):
        per_ds = df.groupby(["dataset", "method"])[["n_trees_trained", "fit_time"]].mean()
        base = per_ds.xs("Baseline", level="method")
        rel = per_ds.div(base, level="dataset").groupby("method").mean()
        g["drzewa zbud. / Baseline"] = rel["n_trees_trained"].reindex(g.index)
        g["trening / Baseline"] = rel["fit_time"].reindex(g.index)
    return g


def es_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Ile iteracji boostingu zostawia ES (średnio na model GBDT), względem limitu M."""
    d = df[df.method.isin(["Baseline+2", "Baseline+1+2"])].copy()
    if d.empty or "params" not in d:
        return pd.DataFrame()
    d["M"] = d["params"].map(lambda s: json.loads(s)["M"])
    d["frac"] = d["n_stages_mean"] / d["M"]
    return d.groupby(["task", "method"]).agg(
        **{"śr. iteracji / M": ("frac", "mean"),
           "folds z ES < M (%)": ("frac", lambda x: 100 * (x < 0.999).mean())}).round(3)


def chosen_params(df: pd.DataFrame) -> pd.DataFrame:
    """Jak często wybierano poszczególne wartości hiperparametrów (wariant +1+2 i GBDT)."""
    if "params" not in df:
        return pd.DataFrame()
    rows = []
    for meth in ("GBDT", "Baseline", "Baseline+1+2"):
        p = pd.DataFrame([json.loads(s) for s in df.loc[df.method == meth, "params"]])
        for col in p.columns:
            if meth != "Baseline+1+2" and col in ("alpha", "beta", "k", "p"):
                continue
            vc = p[col].astype(str).value_counts(normalize=True)
            rows.append({"metoda": meth, "parametr": col,
                         "rozkład wyborów": ", ".join(f"{k}: {v:.0%}" for k, v in vc.items())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Raport
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(RESULTS_DIR / "full.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = add_normalized_mae(pd.read_csv(args.csv))
    n_runs = df.groupby(["dataset", "method"]).size()
    L = [f"# Wyniki eksperymentów\n",
         f"Źródło: `{args.csv}` – {len(df)} przebiegów, {df.dataset.nunique()} zbiorów, "
         f"{df.method.nunique()} metod, {int(n_runs.min())}–{int(n_runs.max())} przebiegów "
         f"na (zbiór, metoda).\n"]

    for task in ("clf", "reg"):
        d = df[df.task == task]
        if d.empty:
            continue
        info = TASK_INFO[task]
        m = means(d, info["metric"])
        r = ranks(m, info["higher"])
        title = "Klasyfikacja" if task == "clf" else "Regresja"
        arrow = "↑" if info["higher"] else "↓"

        main_col = "f1_macro" if task == "clf" else "mae"
        raw = means(d, main_col)
        L += [f"## {title}\n",
              f"### {('Macro F1' if task == 'clf' else 'MAE')} ({arrow}), średnia ± std "
              f"z 10 przebiegów (pogrubione: najlepsza metoda)\n",
              bold_best(mean_std_table(d, main_col, "{:.4f}"), raw, info["higher"]).to_markdown(),
              ""]
        aux = info["aux"]
        L += [f"### {aux} (metryka pomocnicza)\n",
              mean_std_table(d, aux, "{:.4f}").to_markdown(), ""]
        if task == "reg":
            L += ["### MAE / std(y) – błąd znormalizowany (do porównań między zbiorami)\n",
                  m.round(4).to_markdown(),
                  "",
                  "Średnia po zbiorach: " + ", ".join(f"{c}: {v:.4f}" for c, v in m.mean().items()),
                  ""]

        stat, p = friedman(m)
        L += ["### Średnie rangi (1 = najlepsza)\n",
              r.mean().round(2).to_frame("średnia ranga").T.to_markdown(), "",
              f"**Test Friedmana** ({len(m)} zbiorów, {m.shape[1]} metod): "
              f"χ² = {stat:.2f}, p = {p:.4g} → "
              + ("różnice istotne (p < 0,05)." if p < ALPHA else "brak istotnych różnic (p ≥ 0,05)."),
              ""]
        w = wilcoxon_pairs(m, info["higher"])
        L += [f"### Testy Wilcoxona dla par z planu (na {info['label']})\n",
              w.to_markdown(index=False, floatfmt=".4g"), "",
              "Uwaga: plan przewiduje testy parami po wykryciu różnic testem Friedmana; "
              "podajemy je zawsze, interpretując zgodnie z wynikiem Friedmana.\n"]

        L += ["### Koszt obliczeniowy (średnio na fold)\n",
              cost_table(d).round(3).to_markdown(), ""]

        L += [f"### Szczegóły wyników ({title.lower()})\n"]
        cols = [c for c in ("f1_macro", "accuracy", "mae", "rmse", "nmae", "fit_time",
                            "tune_time", "n_trees_trained") if c in d and d[c].notna().any()]
        tbl = d.groupby(["dataset", "method"])[cols].agg(["mean", "std"])
        tbl.round(5).to_csv(out / f"per_dataset_{task}.csv")
        L += [f"Pełna tabela: `per_dataset_{task}.csv`\n"]

    es = es_stats(df)
    if not es.empty:
        L += ["## Ensemble-Aware Early Stopping – ile iteracji zostaje\n",
              es.to_markdown(), ""]
    cp = chosen_params(df)
    if not cp.empty:
        L += ["## Wybierane hiperparametry (strojenie)\n", cp.to_markdown(index=False), ""]

    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Zapisano {out / 'summary.md'}")


if __name__ == "__main__":
    main()
