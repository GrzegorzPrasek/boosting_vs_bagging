"""Analiza eksperymentów dodatkowych: szum w etykietach i ablacja Adaptive Sampling.

    .venv/Scripts/python.exe scripts/analyze_extra.py

Dane: results/extra.csv (szum 10% i 20%, warianty ablacji) + results/full.csv (szum 0%,
te same zbiory – bez Dry Bean, który pominęliśmy w eksperymentach dodatkowych).
Wyniki: results/extra_summary.md, results/figures/fig_noise.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from analyze import TASK_INFO, add_normalized_mae, friedman, holm  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402

MAIN = ["GBDT", "Bagging", "RF", "Baseline", "Baseline+1", "Baseline+2", "Baseline+1+2"]
ABL = ["Baseline+1(e)", "Baseline+1(d)"]
MAIN_PAIRS = [("Baseline", "Baseline+1"), ("Baseline", "Baseline+2"),
              ("Baseline", "Baseline+1+2"), ("Baseline+1", "Baseline+1+2"),
              ("Baseline+2", "Baseline+1+2"), ("GBDT", "Baseline+1"),
              ("GBDT", "Baseline+1+2"), ("RF", "Baseline+1+2")]
ABL_PAIRS = [("Baseline", "Baseline+1(e)"), ("Baseline", "Baseline+1(d)"),
             ("Baseline+1(e)", "Baseline+1"), ("Baseline+1(d)", "Baseline+1"),
             ("Baseline+1(e)", "Baseline+1(d)")]
NOISE_LABEL = {0.0: "0%", 0.1: "10%", 0.2: "20%"}


def load() -> pd.DataFrame:
    full = pd.read_csv(RESULTS_DIR / "full.csv")
    extra = pd.read_csv(RESULTS_DIR / "extra.csv")
    full = full[full.dataset.isin(extra.dataset.unique())].assign(noise=0.0)
    df = pd.concat([full, extra], ignore_index=True)
    return add_normalized_mae(df)


def matrix(df, task, noise, methods):
    d = df[(df.task == task) & np.isclose(df.noise, noise) & df.method.isin(methods)]
    m = d.groupby(["dataset", "method"])[TASK_INFO[task]["metric"]].mean().unstack("method")
    return m[[x for x in methods if x in m]].dropna()


def pairs_table(m, higher, pairs):
    rows = []
    for a, b in pairs:
        if a not in m or b not in m:
            continue
        diff = m[b] - m[a]
        better = diff > 0 if higher else diff < 0
        nz = diff[diff != 0]
        p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 2 else np.nan
        rel = (diff / m[a]).mean() * 100 * (1 if higher else -1)
        rows.append({"porównanie": f"{a} → {b}",
                     "lepsze / gorsze": f"{int(better.sum())} / {int(((diff != 0) & ~better).sum())}",
                     "śr. zmiana [%]": round(rel, 2), "p": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p (Holm)"] = holm(out["p"].tolist())
        out["p"] = out["p"].round(4)
        out["p (Holm)"] = out["p (Holm)"].round(4)
    return out


def rel_to_baseline(df, task, methods):
    """Średnia (po zbiorach) względna różnica metryki vs Baseline [%], + = lepiej."""
    higher = TASK_INFO[task]["higher"]
    out = {}
    for nz in sorted(df.noise.unique()):
        m = matrix(df, task, nz, ["Baseline"] + methods)
        out[NOISE_LABEL.get(nz, nz)] = {x: ((m[x] - m["Baseline"]) / m["Baseline"]).mean() * 100
                                        * (1 if higher else -1) for x in methods if x in m}
    return pd.DataFrame(out).round(2)


def degradation(df, task):
    """Względna zmiana metryki przy 20% szumu względem 0% (− = pogorszenie)."""
    higher = TASK_INFO[task]["higher"]
    m0, m2 = matrix(df, task, 0.0, MAIN + ABL), matrix(df, task, 0.2, MAIN + ABL)
    cols = [c for c in m0 if c in m2]
    r = ((m2[cols] - m0[cols]) / m0[cols]).mean() * 100 * (1 if higher else -1)
    return r.round(2)


def es_trees(df):
    g = df[df.method.isin(["Baseline", "Baseline+2", "Baseline+1+2"])]
    g = g.groupby(["task", "noise", "dataset", "method"]).n_trees_trained.mean().unstack("method")
    out = {}
    for (task, nz), d in g.groupby(level=[0, 1]):
        out[(task, NOISE_LABEL.get(nz, nz))] = {
            m: round(float(np.exp(np.log(d[m] / d["Baseline"]).mean())), 3)
            for m in ("Baseline+2", "Baseline+1+2")}
    return pd.DataFrame(out).T


def fig_noise(df, out):
    series = [("Baseline+1", "#2a78d6", "-"), ("Baseline+1(d)", "#7a4fd0", "--"),
              ("Baseline+2", "#eb6834", "-"), ("Baseline+1+2", "#1baf7a", "-"),
              ("GBDT", "#eda100", "-"), ("RF", "#e87ba4", "-")]
    tasks = [t for t in ("clf", "reg") if (df.task == t).any()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.7 * len(tasks), 3.4))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, tasks):
        r = rel_to_baseline(df, t, [s for s, _, _ in series])
        x = np.arange(r.shape[1])
        for name, col, ls in series:
            if name in r.index:
                ax.plot(x, r.loc[name].values, color=col, linewidth=2, marker="o", linestyle=ls,
                        markersize=5, markeredgecolor="white", label=name)
        ax.axhline(0, color="#52514e", linewidth=0.8, linestyle="--")
        ax.set_xticks(x, [f"szum {c}" for c in r.columns])
        if t == tasks[0]:
            ax.set_ylabel("różnica vs Baseline [%]  (+ = lepiej)")
        n = df[df.task == t].dataset.nunique()
        ax.set_title(f"{'Klasyfikacja' if t == 'clf' else 'Regresja'} ({n} zbiorów, "
                     f"{TASK_INFO[t]['label']})", fontsize=9, loc="left")
        ax.grid(color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, lab = axes[0].get_legend_handles_labels()
    lab = [x.replace("Baseline+1(d)", "Baseline+1(d) – tylko rozbieżność") for x in lab]
    fig.legend(h, lab, frameon=False, loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main():
    df = load()
    md = ["# Eksperymenty dodatkowe: szum w etykietach i ablacja Adaptive Sampling", "",
          f"Zbiory: {df.dataset.nunique()} (bez Dry Bean). Szum 0% z `full.csv`, "
          f"10% i 20% z `extra.csv`. Przebiegów: {len(df)}.", ""]
    for task in ("clf", "reg"):
        info = TASK_INFO[task]
        md += [f"## {'Klasyfikacja' if task == 'clf' else 'Regresja'} ({info['label']})", ""]
        rank_rows = {}
        for nz in sorted(df.noise.unique()):
            m = matrix(df, task, nz, MAIN)
            r = m.rank(axis=1, ascending=not info["higher"]).mean()
            stat, p = friedman(m)
            rank_rows[f"szum {NOISE_LABEL[nz]} (Friedman p={p:.3f})"] = r.round(2)
        md += ["### Średnie rangi 7 metod", "", pd.DataFrame(rank_rows).T.to_markdown(), ""]
        md += ["### Względna różnica vs Baseline [%] (+ = lepiej), średnia po zbiorach", "",
               rel_to_baseline(df, task, MAIN[:3] + MAIN[4:] + ABL).to_markdown(), ""]
        for nz in sorted(df.noise.unique()):
            m = matrix(df, task, nz, MAIN + ABL)
            md += [f"### Testy Wilcoxona – szum {NOISE_LABEL[nz]}", "",
                   pairs_table(m, info["higher"], MAIN_PAIRS).to_markdown(index=False), "",
                   "Ablacja (osobna rodzina testów, Holm):", "",
                   pairs_table(m, info["higher"], ABL_PAIRS).to_markdown(index=False), ""]
        md += ["### Odporność: zmiana metryki przy 20% szumu względem 0% [%] (− = pogorszenie)", "",
               degradation(df, task).to_frame("zmiana [%]").to_markdown(), ""]
    md += ["## Early Stopping przy szumie: zbudowane drzewa / Baseline (średnia geom.)", "",
           es_trees(df).to_markdown(), ""]
    (RESULTS_DIR / "extra_summary.md").write_text("\n".join(md), encoding="utf-8")
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    fig_noise(df, RESULTS_DIR / "figures" / "fig_noise.png")
    print("Zapisano results/extra_summary.md i results/figures/fig_noise.png")


if __name__ == "__main__":
    main()
