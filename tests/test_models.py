"""Testy poprawności: uruchomienie `.venv/Scripts/python.exe -m pytest -q`."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from src.data import impute_median
from src.evaluation import ROW_COLUMNS, iter_splits, run_cv
from src.models import BaggedGBDT, _truncate_gbdt


@pytest.fixture(scope="module")
def clf3():
    X, y = make_classification(n_samples=240, n_features=8, n_informative=5,
                               n_classes=3, random_state=0)
    return X[:160], y[:160], X[160:200], y[160:200], X[200:], y[200:]


@pytest.fixture(scope="module")
def reg():
    X, y = make_regression(n_samples=240, n_features=6, noise=10.0, random_state=0)
    return X[:160], y[:160], X[160:200], y[160:200], X[200:], y[200:]


# --------------------------------------------------------------------------- #
# Early Stopping: przycinanie modelu
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cls,data", [(GradientBoostingClassifier, "clf3"),
                                      (GradientBoostingRegressor, "reg")])
def test_truncate_equals_model_trained_on_n_stages(cls, data, request):
    X, y, _, _, X_te, _ = request.getfixturevalue(data)
    grown = cls(n_estimators=20, max_depth=2, random_state=7, warm_start=True).fit(X, y)
    for n in (40, 60):
        grown.n_estimators = n
        grown.fit(X, y)
    truncated = _truncate_gbdt(grown, 40)
    fresh = cls(n_estimators=40, max_depth=2, random_state=7).fit(X, y)
    np.testing.assert_allclose(truncated.predict(X_te), fresh.predict(X_te))
    if cls is GradientBoostingClassifier:
        np.testing.assert_allclose(truncated.predict_proba(X_te), fresh.predict_proba(X_te))
    assert truncated.n_estimators_ == 40


def test_es_selects_best_checkpoint_and_respects_M(clf3):
    X, y, X_val, y_val, _, _ = clf3
    m = BaggedGBDT(task="clf", B=4, M=100, k=10, p=2, ensemble_es=True,
                   random_state=0).fit(X, y, X_val, y_val)
    assert (m.n_stages_ <= 100).all()
    assert m.n_trees_trained_ >= m.n_trees_total_
    for hist, n in zip(m.es_history_, m.n_stages_):
        stages = [s for s, _ in hist]
        assert n in stages
        # po wybranym checkpoincie było co najwyżej p checkpointów (albo koniec na M)
        assert len(stages) - stages.index(n) - 1 <= 2 or stages[-1] == 100


# --------------------------------------------------------------------------- #
# Baseline i Adaptive Sampling
# --------------------------------------------------------------------------- #

def test_reproducible_and_proba_valid(clf3):
    X, y, _, _, X_te, _ = clf3
    a = BaggedGBDT(task="clf", B=3, M=30, random_state=1).fit(X, y).predict_proba(X_te)
    b = BaggedGBDT(task="clf", B=3, M=30, random_state=1).fit(X, y).predict_proba(X_te)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (len(X_te), 3)
    np.testing.assert_allclose(a.sum(axis=1), 1.0)


def test_adaptive_sampling_first_model_is_plain_bootstrap(reg):
    X, y, *_ = reg
    base = BaggedGBDT(task="reg", B=3, M=20, random_state=3).fit(X, y)
    ada = BaggedGBDT(task="reg", B=3, M=20, adaptive_sampling=True, random_state=3).fit(X, y)
    np.testing.assert_array_equal(base.inbag_[0], ada.inbag_[0])
    assert not np.array_equal(base.inbag_[1:], ada.inbag_[1:])


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (2.0, 0.5)])
def test_adaptive_weights_bounded(clf3, alpha, beta):
    X, y, *_ = clf3
    m = BaggedGBDT(task="clf", B=5, M=20, adaptive_sampling=True, alpha=alpha, beta=beta,
                   random_state=0).fit(X, y)
    for st in m.sampling_stats_:
        # w_i w [1, 1 + alpha + beta]  =>  max/min <= 1 + alpha + beta
        assert 1.0 <= st["max_min_ratio"] <= 1.0 + alpha + beta + 1e-9
        assert 0.0 < st["ess_frac"] <= 1.0


# --------------------------------------------------------------------------- #
# Dane i ewaluacja
# --------------------------------------------------------------------------- #

def test_impute_median_uses_train_only():
    X_tr = np.array([[1.0], [3.0], [np.nan]])
    X_te = np.array([[np.nan], [100.0]])
    a, b = impute_median(X_tr, X_te)
    assert a[2, 0] == 2.0 and b[0, 0] == 2.0


@pytest.mark.parametrize("task", ["clf", "reg"])
def test_splits_disjoint_and_cover(task):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = rng.integers(0, 2, 100) if task == "clf" else rng.normal(size=100)
    splits = list(iter_splits(X, y, task))
    assert len(splits) == 10
    for _, _, tr, val, te in splits:
        assert not (set(tr) & set(val) or set(tr) & set(te) or set(val) & set(te))
        assert len(tr) + len(val) + len(te) == 100
    for seed in (42, 2026):
        te_all = np.concatenate([te for s, _, _, _, te in splits if s == seed])
        assert sorted(te_all) == list(range(100))


def test_csv_columns_fixed_for_both_tasks(tmp_path):
    out = tmp_path / "r.csv"
    params = dict(B=2, M=10, max_depth=2, learning_rate=0.1, alpha=1, beta=1, k=5, p=1)
    run_cv("wine", methods=("GBDT", "Baseline+1+2"), params=params, out_csv=out,
           seeds=(42,), n_folds=2, log=None)
    run_cv("yacht", methods=("GBDT", "Baseline+1+2"), params=params, out_csv=out,
           seeds=(42,), n_folds=2, log=None)
    df = pd.read_csv(out)
    assert list(df.columns) == ROW_COLUMNS
    assert df.loc[df.task == "clf", "f1_macro"].notna().all()
    assert df.loc[df.task == "clf", "mae"].isna().all()
    assert df.loc[df.task == "reg", "mae"].notna().all()
