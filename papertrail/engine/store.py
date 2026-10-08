"""One object that loads the raw data, runs every engine step once, and serves results.

Both the agent tools and the API read from a single Store, so the numbers a
picker sees on the pick list, the numbers the dashboard shows and the numbers
the agent quotes are always the same numbers.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

from papertrail import config as C
from papertrail.engine import forecast as F
from papertrail.engine import inventory as INV
from papertrail.engine import orders as ORD
from papertrail.engine import picking as PK
from papertrail.engine import slotting as SL

FUTURE_WEEKS = 8


class Store:
    def __init__(self, raw_dir=C.RAW_DIR, now=C.NOW):
        self.now = pd.Timestamp(now)
        self.skus = pd.read_parquet(raw_dir / "skus.parquet")
        self.locations = pd.read_parquet(raw_dir / "locations.parquet")
        self.orders = pd.read_parquet(raw_dir / "orders.parquet")
        self.lines = pd.read_parquet(raw_dir / "order_lines.parquet")
        self.pos = pd.read_parquet(raw_dir / "purchase_orders.parquet")
        self.pack_x = float((self.locations.x.min() + self.locations.x.max()) / 2)
        self.history_end = pd.Timestamp(C.HISTORY_END)
        self.start = self.history_end - pd.Timedelta(days=7 * C.HISTORY_WEEKS - 1)
        self.cats = sorted(self.skus.category.unique())
        self.cat_codes = pd.Categorical(self.skus.category, categories=self.cats).codes
        mondays = pd.date_range(self.start, periods=C.HISTORY_WEEKS, freq="7D")
        self.week_starts = mondays
        self.iso = np.array([min(d.isocalendar().week, 53) for d in mondays])
        fut = pd.date_range(self.history_end + pd.Timedelta(days=1), periods=FUTURE_WEEKS, freq="7D")
        self.future_week_starts = fut
        self.iso_future = np.array([min(d.isocalendar().week, 53) for d in fut])

    # ---------------------------------------------------------------- history
    @cached_property
    def units(self) -> np.ndarray:
        return F.weekly_matrix(self.lines, self.orders, self.skus, self.start, C.HISTORY_WEEKS, "qty")

    @cached_property
    def pick_lines(self) -> np.ndarray:
        return F.weekly_matrix(self.lines, self.orders, self.skus, self.start, C.HISTORY_WEEKS, "lines")

    @cached_property
    def abc(self) -> np.ndarray:
        return INV.abc_classes(self.pick_lines[:, -26:].mean(axis=1))

    # ---------------------------------------------------------------- forecasts
    @cached_property
    def season(self) -> np.ndarray:
        return F.seasonal_index(self.units, self.cat_codes, len(self.cats), self.iso,
                                [slice(0, 52), slice(52, 104)])

    @cached_property
    def season_lines(self) -> np.ndarray:
        return F.seasonal_index(self.pick_lines, self.cat_codes, len(self.cats), self.iso,
                                [slice(0, 52), slice(52, 104)])

    @cached_property
    def unit_fit(self) -> F.Fit:
        return F.fit_level(self.units, self.season[self.cat_codes][:, self.iso])

    @cached_property
    def units_fcst(self) -> np.ndarray:
        return F.forecast(self.unit_fit.level, self.season[self.cat_codes][:, self.iso_future])

    @cached_property
    def lines_fcst(self) -> np.ndarray:
        fit = F.fit_level(self.pick_lines, self.season_lines[self.cat_codes][:, self.iso])
        return F.forecast(fit.level, self.season_lines[self.cat_codes][:, self.iso_future])

    @cached_property
    def backtest(self) -> pd.DataFrame:
        return F.backtest(self.units, self.cat_codes, len(self.cats), self.iso, self.abc)

    @cached_property
    def dow_share(self) -> np.ndarray:
        h = self.orders[(self.orders.created_at >= self.history_end - pd.Timedelta(weeks=8))
                        & (self.orders.created_at < self.history_end + pd.Timedelta(days=1))]
        s = h.created_at.dt.dayofweek.value_counts().sort_index().to_numpy().astype(float)
        return s / s.sum()

    # ---------------------------------------------------------------- inventory
    @cached_property
    def policy(self) -> pd.DataFrame:
        on_order = self.pos.groupby("sku").qty.sum()
        next_eta = self.pos.groupby("sku").eta.min().dt.date.astype(str)
        daily = self.units_fcst[:, :4].mean(axis=1) / 7
        pol = INV.policy(self.skus, daily, self.unit_fit.resid_sd, self.abc, on_order, next_eta, self.now)
        pol["fcst_4wk_units"] = np.round(self.units_fcst[:, :4].sum(axis=1)).astype(int)
        pol["sold_4wk_units"] = self.units[:, -4:].sum(axis=1).astype(int)
        pol["sold_52wk_units"] = self.units[:, -52:].sum(axis=1).astype(int)
        pol["picks_per_week"] = np.round(self.lines_fcst[:, :6].mean(axis=1), 2)
        return pol

    # ---------------------------------------------------------------- orders
    @cached_property
    def exceptions(self) -> pd.DataFrame:
        return ORD.exceptions(self.orders, self.lines, self.policy, self.pos, self.now, self.history_end)

    @cached_property
    def board(self) -> dict:
        return ORD.status_board(self.orders, self.history_end, self.now)

    @cached_property
    def pickers(self) -> pd.DataFrame:
        return ORD.picker_stats(self.orders, self.lines, self.now)

    def open_orders(self, statuses=("New",)) -> pd.DataFrame:
        live = ORD.live_orders(self.orders, self.history_end)
        return live[live.status.isin(statuses)]

    # ---------------------------------------------------------------- picking
    @cached_property
    def wave_plan(self) -> dict:
        return PK.build_waves(self.open_orders(("New",)), self.lines, self.skus, self.locations,
                              self.pack_x, self.now)

    def plan_waves(self, target_minutes: float = 30, max_orders: int = 50,
                   include_released: bool = False) -> dict:
        st = ("New", "Released") if include_released else ("New",)
        return PK.build_waves(self.open_orders(st), self.lines, self.skus, self.locations,
                              self.pack_x, self.now, target_minutes=target_minutes,
                              max_orders=max_orders)

    @cached_property
    def labor_plan(self) -> list[dict]:
        """Next 5 business days: expected orders, lines, picker-hours and headcount."""
        plan_m_per_line = self.wave_plan["summary"].get("distance_km", 0) * 1000 / max(self.wave_plan["summary"].get("lines", 1), 1)
        single_m_per_line = self.wave_plan["summary"].get("single_order_distance_km", 0) * 1000 / max(self.wave_plan["summary"].get("lines", 1), 1)
        lpo = len(self.lines) / len(self.orders)
        upl = self.lines.qty.mean()
        week_lines = self.lines_fcst.sum(axis=0)          # per future week
        out, d = [], self.now.normalize() + pd.Timedelta(days=1)
        while len(out) < 5:
            if d.dayofweek < 5:
                wk = min((d - self.future_week_starts[0]).days // 7, FUTURE_WEEKS - 1)
                # weekend orders ship Monday, so Monday carries Sat+Sun volume
                share = self.dow_share[d.dayofweek] + (self.dow_share[5] + self.dow_share[6] if d.dayofweek == 0 else 0)
                lines = week_lines[wk] * share
                def hours(m_per_line, sort_s):
                    return (lines * (C.SECONDS_PER_LINE + (upl - 1) * C.SECONDS_PER_EXTRA_UNIT + sort_s)
                            + lines * m_per_line / C.WALK_SPEED_MPS) / 3600
                hb = hours(plan_m_per_line, C.SECONDS_SORT_PER_LINE)
                hs = hours(single_m_per_line, 0)
                out.append({"date": d.date().isoformat(), "weekday": d.strftime("%A"),
                            "orders": int(round(lines / lpo)), "lines": int(round(lines)),
                            "picker_hours_batched": round(hb, 1), "picker_hours_single_order": round(hs, 1),
                            "pickers_needed": int(np.ceil(hb / 7.0)),
                            "pickers_needed_single_order": int(np.ceil(hs / 7.0))})
            d += pd.Timedelta(days=1)
        return out

    # ---------------------------------------------------------------- slotting
    @cached_property
    def velocity(self) -> pd.Series:
        return pd.Series(self.lines_fcst[:, :6].mean(axis=1), index=self.skus.sku)

    @cached_property
    def batch_walk_factor(self) -> float:
        """Metres walked per line with batch picking / with single-order picking.
        Slot-move savings are computed on single trips, so we scale them by this."""
        s = self.wave_plan["summary"]
        return s["distance_km"] / max(s["single_order_distance_km"], 1e-9) if s else 1.0

    @cached_property
    def slotting(self) -> pd.DataFrame:
        plan = SL.swap_plan(self.skus, self.locations, self.velocity)
        plan["hours_saved_per_week_batched"] = (plan.hours_saved_per_week * self.batch_walk_factor).round(3)
        return plan

    @cached_property
    def zone_heat(self) -> pd.DataFrame:
        return SL.zone_heat(self.skus, self.locations, self.velocity)

    # ---------------------------------------------------------------- briefing
    def briefing(self) -> dict:
        pol, ex, sl = self.policy, self.exceptions, self.slotting
        stat = pol.status.value_counts().to_dict()
        reorder = pol[pol.status.isin(["Stockout", "Reorder now"]) & (pol.suggested_qty > 0)]
        top20 = sl.head(20)
        coach = self.pickers[self.pickers.flag.str.contains("coaching")]
        actions = []
        for kind in ["Late", "At risk", "Short", "Stuck", "On hold"]:
            n = int((ex.type == kind).sum()) if not ex.empty else 0
            if n:
                actions.append({"area": "Orders", "text": f"{n} {kind.lower()} order exception{'s' if n != 1 else ''}",
                                "detail": f"First: {ex[ex.type == kind].iloc[0].order_id} — {ex[ex.type == kind].iloc[0].action}"})
        if len(reorder):
            actions.append({"area": "Inventory", "text": f"Approve replenishment for {len(reorder):,} SKUs across "
                                                         f"{reorder.supplier.nunique()} suppliers (${reorder.suggested_po_value.sum():,.0f})",
                            "detail": f"{stat.get('Stockout', 0)} SKUs are out of stock; "
                                      f"${pol.revenue_at_risk.sum():,.0f} revenue at risk before replenishment lands."})
        wp = self.wave_plan["summary"]
        if wp:
            actions.append({"area": "Picking", "text": f"Release {wp['waves']} waves for {wp['orders']} new orders",
                            "detail": f"Batch + zone picking walks {wp['walking_saved_pct']}% less than single-order picking."})
        actions.append({"area": "Slotting", "text": f"Do the top 20 slot swaps — pickers visit these {top20.picks_per_week.sum():,.0f}×/week",
                        "detail": f"First move: {top20.iloc[0].sku} {top20.iloc[0].from_location} → {top20.iloc[0].to_location}."})
        if len(coach):
            actions.append({"area": "People", "text": f"Check in with {', '.join(coach.picker.head(3))}",
                            "detail": "; ".join(f"{r.picker}: {r.flag.lower()}" for r in coach.head(3).itertuples())})
        return {
            "as_of": self.now.isoformat(), "board": self.board,
            "inventory": {k: int(stat.get(k, 0)) for k in INV.STATUS_ORDER},
            "reorder_value": round(float(reorder.suggested_po_value.sum()), 2),
            "revenue_at_risk": round(float(pol.revenue_at_risk.sum()), 2),
            "overstock_value": round(float(pol.overstock_value.sum()), 2),
            "exceptions": ex.type.value_counts().to_dict() if not ex.empty else {},
            "waves": wp, "tomorrow": self.labor_plan[0], "actions": actions,
        }
