"""Reorder points, safety stock, stock status and suggested purchase quantities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from papertrail import config as C

REVIEW_PERIOD_DAYS = 7          # buyers place POs weekly
OVERSTOCK_DAYS = 120
STATUS_ORDER = ["Stockout", "Reorder now", "Watch", "Healthy", "Overstock", "Slow mover"]


def abc_classes(lines_per_week: np.ndarray) -> np.ndarray:
    """A = SKUs making up the first 80% of pick lines, B the next 15%, C the rest."""
    order = np.argsort(-lines_per_week)
    cum = np.cumsum(lines_per_week[order]) / max(lines_per_week.sum(), 1e-9)
    cls = np.empty(len(lines_per_week), dtype=object)
    cls[order] = np.where(cum <= 0.80, "A", np.where(cum <= 0.95, "B", "C"))
    return cls


def policy(skus: pd.DataFrame, daily_fcst: np.ndarray, sd_week: np.ndarray, abc: np.ndarray,
           on_order: pd.Series, next_eta: pd.Series, now: pd.Timestamp) -> pd.DataFrame:
    df = skus[["sku", "name", "category", "product_type", "variant", "color", "zone", "location",
               "supplier", "lead_time_days",
               "case_pack", "unit_cost", "unit_price", "on_hand", "allocated"]].copy()
    lt = df.lead_time_days.to_numpy()
    z = np.vectorize(C.SERVICE_Z.get)(abc)
    sd_day = sd_week / np.sqrt(7)

    df["abc"] = abc
    df["daily_demand"] = np.round(daily_fcst, 2)
    df["safety_stock"] = np.ceil(z * sd_day * np.sqrt(lt)).astype(int)
    df["reorder_point"] = np.ceil(daily_fcst * lt + df.safety_stock).astype(int)
    df["available"] = (df.on_hand - df.allocated).clip(lower=0)
    df["on_order"] = df.sku.map(on_order).fillna(0).astype(int)
    df["next_po_eta"] = df.sku.map(next_eta)
    df["position"] = df.available + df.on_order

    order_up_to = daily_fcst * (lt + REVIEW_PERIOD_DAYS) + df.safety_stock
    raw = np.maximum(order_up_to - df.position, 0)
    df["suggested_qty"] = (np.ceil(raw / df.case_pack) * df.case_pack).astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        cover = np.where(daily_fcst > 0.01, df.available / daily_fcst, np.inf)
    df["days_of_cover"] = np.round(cover, 1)
    stockout_day = np.where(np.isfinite(cover), np.floor(cover), np.nan)
    df["projected_stockout"] = [
        (now.normalize() + pd.Timedelta(days=int(d))).date().isoformat() if np.isfinite(d) and d < 120 else None
        for d in stockout_day
    ]

    has_demand = daily_fcst > 0.02
    status = np.select(
        [
            ~has_demand,
            df.available <= 0,
            df.position <= df.reorder_point,
            df.position <= df.reorder_point + daily_fcst * REVIEW_PERIOD_DAYS,
            (cover > np.maximum(OVERSTOCK_DAYS, lt + 90)) & (df.on_hand >= 2 * df.case_pack),
        ],
        ["Slow mover", "Stockout", "Reorder now", "Watch", "Overstock"],
        "Healthy",
    )
    df["status"] = status
    df["suggested_po_value"] = np.round(df.suggested_qty * df.unit_cost, 2)
    # revenue at risk: days uncovered before the next PO (or a new PO) can land
    eta_days = (pd.to_datetime(df.next_po_eta) - now.normalize()).dt.days
    gap = np.where(df.on_order > 0, eta_days.fillna(pd.Series(lt, index=df.index)), lt) - np.where(np.isfinite(cover), cover, 1e9)
    df["revenue_at_risk"] = np.round(np.clip(gap, 0, None) * daily_fcst * df.unit_price, 2)
    df["overstock_value"] = np.where(
        df.status == "Overstock",
        np.round((df.on_hand - daily_fcst * (OVERSTOCK_DAYS / 2)).clip(lower=0) * df.unit_cost, 2), 0.0)
    return df
