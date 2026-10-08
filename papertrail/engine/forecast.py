"""Demand forecasting: category seasonality x per-SKU smoothed level.

15,000 SKUs with sparse, lumpy weekly sales is a bad fit for one model per SKU.
Seasonality is a category-level phenomenon (back-to-school lifts every pen at
once), so we estimate it where the signal is strong — category totals — and
let each SKU carry only its own level, fit by exponential smoothing on the
de-seasonalised series. Cheap, explainable, and it backtests well.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ALPHA = 0.25          # level smoothing for SKUs selling >= 3 units/week
ALPHA_SLOW = 0.07     # longer memory for slow movers
SMOOTH = 3            # centred moving-average window for the seasonal index


def weekly_matrix(lines: pd.DataFrame, orders: pd.DataFrame, skus: pd.DataFrame,
                  start: pd.Timestamp, n_weeks: int, value: str = "qty") -> np.ndarray:
    """SKU x week matrix of units (value='qty') or pick lines (value='lines')."""
    df = lines.merge(orders[["order_id", "created_at"]], on="order_id")
    wk = ((df.created_at - start).dt.days // 7).to_numpy()
    keep = (wk >= 0) & (wk < n_weeks)
    sku_pos = pd.Index(skus.sku).get_indexer(df.sku)
    vals = df.qty.to_numpy() if value == "qty" else np.ones(len(df))
    mat = np.zeros((len(skus), n_weeks))
    np.add.at(mat, (sku_pos[keep], wk[keep]), vals[keep])
    return mat


def seasonal_index(mat: np.ndarray, cat_codes: np.ndarray, n_cats: int, iso_weeks: np.ndarray,
                   years: list[slice]) -> np.ndarray:
    """Category x ISO-week (1..53) multiplicative index, averaged over the given years.
    Each year is normalised by its own mean so growth doesn't leak into seasonality."""
    idx = np.zeros((n_cats, 54))
    cnt = np.zeros(54)
    for ys in years:
        cat_tot = np.zeros((n_cats, mat[:, ys].shape[1]))
        np.add.at(cat_tot, cat_codes, mat[:, ys])
        cat_tot = cat_tot / np.maximum(cat_tot.mean(axis=1, keepdims=True), 1e-9)
        for j, w in enumerate(iso_weeks[ys]):
            idx[:, w] += cat_tot[:, j]
        np.add.at(cnt, iso_weeks[ys], 1)
    idx[:, 1:] /= np.maximum(cnt[1:], 1)
    idx[:, 53] = np.where(cnt[53] == 0, idx[:, 52], idx[:, 53])
    # circular smoothing over weeks 1..52
    core = idx[:, 1:53]
    pad = np.concatenate([core[:, -1:], core, core[:, :1]], axis=1)
    k = np.ones(SMOOTH) / SMOOTH
    sm = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, pad)
    idx[:, 1:53] = sm / sm.mean(axis=1, keepdims=True)
    idx[:, 53] = idx[:, 52]
    return np.clip(idx, 0.15, None)


@dataclass
class Fit:
    level: np.ndarray          # de-seasonalised weekly level per SKU (end of history)
    resid_sd: np.ndarray       # robust sigma of one-step-ahead errors (units/week), recent 26 weeks
    fitted: np.ndarray         # one-step-ahead fitted values, SKU x week


def fit_level(mat: np.ndarray, season: np.ndarray) -> Fit:
    """Vectorised simple exponential smoothing on y / season."""
    n, T = mat.shape
    des = mat / season
    # slow, lumpy SKUs need a longer memory or a single quiet month zeroes them out
    alpha = np.where(mat.mean(axis=1) >= 3, ALPHA, ALPHA_SLOW)
    level = des[:, :8].mean(axis=1)
    fitted = np.zeros_like(mat)
    for t in range(T):
        fitted[:, t] = level * season[:, t]
        level = alpha * des[:, t] + (1 - alpha) * level
    resid = mat[:, -26:] - fitted[:, -26:]
    # sigma ~= 1.25 x mean absolute error (Silver, Pyke & Peterson). Robust to the
    # occasional wholesale case order that would otherwise blow up safety stock.
    return Fit(level=level, resid_sd=1.25 * np.abs(resid).mean(axis=1), fitted=fitted)


def forecast(level: np.ndarray, season_future: np.ndarray) -> np.ndarray:
    """level (n,) x season_future (n, H) -> (n, H) units/week."""
    return level[:, None] * season_future


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(actual - pred).sum() / max(actual.sum(), 1e-9))


def backtest(mat: np.ndarray, cat_codes: np.ndarray, n_cats: int, iso_weeks: np.ndarray,
             abc: np.ndarray, holdout: int = 8) -> pd.DataFrame:
    """Train on weeks [0, T-holdout), forecast the last `holdout` weeks from one origin.
    Seasonality comes from year one only, so nothing from the test window leaks in."""
    T = mat.shape[1]
    train = mat[:, : T - holdout]
    test = mat[:, T - holdout:]
    idx = seasonal_index(train, cat_codes, n_cats, iso_weeks, [slice(0, 52)])
    s_train = idx[cat_codes][:, iso_weeks[: T - holdout]]
    s_test = idx[cat_codes][:, iso_weeks[T - holdout:]]
    fit = fit_level(train, s_train)
    model = forecast(fit.level, s_test)
    snaive = mat[:, T - holdout - 52: T - 52] * (train[:, -52:].sum() / max(train[:, -104:-52].sum(), 1))
    ma8 = np.repeat(train[:, -8:].mean(axis=1, keepdims=True), holdout, axis=1)

    rows = []
    for label, mask in [("All SKUs", np.ones(len(abc), bool)), ("A items", abc == "A"),
                        ("B items", abc == "B"), ("C items", abc == "C")]:
        rows.append({
            "segment": label, "skus": int(mask.sum()),
            "model_wape": wape(test[mask], model[mask]),
            "seasonal_naive_wape": wape(test[mask], snaive[mask]),
            "moving_avg_8wk_wape": wape(test[mask], ma8[mask]),
        })
    # the number planners actually care about: weekly category totals
    cat_test = np.zeros((n_cats, holdout)); np.add.at(cat_test, cat_codes, test)
    cat_model = np.zeros((n_cats, holdout)); np.add.at(cat_model, cat_codes, model)
    cat_ma = np.zeros((n_cats, holdout)); np.add.at(cat_ma, cat_codes, ma8)
    cat_sn = np.zeros((n_cats, holdout)); np.add.at(cat_sn, cat_codes, snaive)
    rows.append({"segment": "Category x week totals", "skus": int(len(abc)),
                 "model_wape": wape(cat_test, cat_model), "seasonal_naive_wape": wape(cat_test, cat_sn),
                 "moving_avg_8wk_wape": wape(cat_test, cat_ma)})
    return pd.DataFrame(rows)
