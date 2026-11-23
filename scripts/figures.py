"""Wykresy do raportu (PNG, druk / PDF).

    .venv/Scripts/python.exe scripts/figures.py                          # results/full.csv
    .venv/Scripts/python.exe scripts/figures.py --csv results/pilot.csv --out results/_fig

Rysunki:
    fig_ranks.png     średnie rangi metod (klasyfikacja | regresja)
    fig_tradeoff.png  jakość (średnia ranga) vs czas treningu względem Baseline
    fig_es.png        Early Stopping: odsetek zbudowanych drzew względem Baseline, per zbiór
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze import TASK_INFO, add_normalized_mae, means, methods_in, ranks  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.evaluation import BAGGED_VARIANTS  # noqa: E402

# Paleta referencyjna (skill dataviz, light): sloty 1–2 zweryfikowane walidatorem
OURS = "#2a78d6"      # warianty BaggedGBDT
REF = "#eb6834"       # metody referencyjne
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"
TITLES = {"clf": "Klasyfikacja ({n} zbiorów, Macro F1)",
          "reg": "Regresja ({n} zbiorów, MAE / std(y))"}


def _title(ax, d, t):
    ax.set_title(TITLES[t].format(n=d.dataset.nunique()), fontsize=9, color=INK, loc="left")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 220,
})


def _color(m):
    return OURS if m in BAGGED_VARIANTS else REF


def _legend(fig):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=OURS, label="bagging GBDT (nasze warianty)"),
                        Patch(color=REF, label="metody referencyjne")],
               frameon=False, loc="upper center", ncol=2, fontsize=8)


def fig_ranks(df, out):
    tasks = [t for t in ("clf", "reg") if (df.task == t).any()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.0), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, tasks):
        d = df[df.task == t]
        info = TASK_INFO[t]
        r = ranks(means(d, info["metric"]), info["higher"]).mean().sort_values()
        y = np.arange(len(r))[::-1]
        ax.barh(y, r.values, height=0.6, color=[_color(m) for m in r.index],
                edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, r.values):
            ax.text(v + 0.08, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, r.index)
        ax.set_xlim(0, len(r) + 0.6)
        ax.set_xlabel("średnia ranga (1 = najlepsza)")
        _title(ax, d, t)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    _legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "fig_ranks.png")
    plt.close(fig)


def fig_tradeoff(df, out):
    tasks = [t for t in ("clf", "reg") if (df.task == t).any()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.1))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, tasks):
        d = df[df.task == t]
        info = TASK_INFO[t]
        r = ranks(means(d, info["metric"]), info["higher"]).mean()
        ft = d.groupby(["dataset", "method"]).fit_time.mean().unstack("method")
        rel = (ft.div(ft["Baseline"], axis=0)).apply(np.log).mean().apply(np.exp)  # średnia geom.
        placed = []
        for m in sorted(methods_in(d), key=lambda m: (r[m], rel[m])):
            ax.scatter(rel[m], r[m], s=42, color=_color(m), edgecolor=SURFACE,
                       linewidth=1.5, zorder=3)
            # etykiety punktów leżących blisko siebie przesuwamy w dół, by się nie nakładały
            close = sum(abs(np.log(rel[m] / x)) < 0.25 and abs(r[m] - y) < 0.12
                        for x, y in placed)
            placed.append((rel[m], r[m]))
            ax.annotate(m, (rel[m], r[m]), xytext=(5, 3 - 10 * close),
                        textcoords="offset points", fontsize=7.5, color=INK)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        lo = min(rel[m] for m in methods_in(d))
        ticks = [v for v in (0.1, 0.2, 0.5, 1.0) if v >= lo * 0.8]
        ax.set_xticks(ticks, [f"{v:g}×" for v in ticks])
        ax.set_xlim(lo * 0.8, 1.6)
        ax.set_xlabel("czas treningu / Baseline (skala log)")
        ax.set_ylabel("średnia ranga (wyżej = lepiej)")
        ax.invert_yaxis()
        _title(ax, d, t)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    _legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out / "fig_tradeoff.png")
    plt.close(fig)


def fig_es(df, out):
    """Zbudowane drzewa: Baseline+2 / Baseline, per zbiór (1.0 = brak oszczędności)."""
    g = df.groupby(["dataset", "method"]).n_trees_trained.mean().unstack("method")
    if "Baseline+2" not in g:
        return
    ratio = (g["Baseline+2"] / g["Baseline"]).sort_values()
    task = df.groupby("dataset").task.first()
    fig, ax = plt.subplots(figsize=(6.4, 0.22 * len(ratio) + 0.9))
    y = np.arange(len(ratio))[::-1]
    ax.barh(y, ratio.values, height=0.62, color=OURS, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, ratio.values):
        ax.text(1.04, yi, f"{v:.0%}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [f"{ds} ({'clf' if task[ds] == 'clf' else 'reg'})" for ds in ratio.index],
                  fontsize=7.5)
    ax.axvline(1.0, color=INK2, linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("zbudowane drzewa: Baseline+2 jako % Baseline (100% = brak oszczędności)")
    ax.set_title("Ensemble-Aware Early Stopping – oszczędność drzew per zbiór",
                 fontsize=9, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "fig_es.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(RESULTS_DIR / "full.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = add_normalized_mae(pd.read_csv(args.csv))
    fig_ranks(df, out)
    fig_tradeoff(df, out)
    fig_es(df, out)
    print(f"Zapisano rysunki w {out}")


if __name__ == "__main__":
    main()
