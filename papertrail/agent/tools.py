"""The agent's tools: small, JSON-returning functions over the Store.

The language model never touches raw tables. It calls these, gets compact
answers with the numbers already computed, and explains them. That keeps
answers grounded (every figure traces to engine output) and keeps token use low.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from papertrail.engine.store import Store


def _clean(obj: Any) -> Any:
    """Make pandas/numpy values JSON-safe."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else round(f, 2)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    return obj


def _records(df: pd.DataFrame, cols: list[str], limit: int) -> list[dict]:
    return _clean(df[cols].head(limit).to_dict("records"))


class Tools:
    def __init__(self, store: Store):
        self.s = store

    # ------------------------------------------------------------ overview
    def daily_briefing(self) -> dict:
        """Today's operations summary: order board, exceptions, inventory health, waves, tomorrow's labor."""
        return _clean(self.s.briefing())

    # ------------------------------------------------------------ orders
    def lookup_order(self, order_id: str) -> dict:
        """Full status of one order: lines, allocation, picker, wave, any exception."""
        oid = order_id.strip().upper()
        if not oid.startswith("SO-"):
            oid = "SO-" + re.sub(r"\D", "", oid)
        o = self.s.orders[self.s.orders.order_id == oid]
        if o.empty:
            return {"error": f"No order {oid}."}
        o = o.iloc[0]
        ln = self.s.lines[self.s.lines.order_id == oid].merge(
            self.s.skus[["sku", "name", "location"]], on="sku")
        ex = self.s.exceptions
        ex = ex[ex.order_id == oid] if not ex.empty else ex
        return _clean({
            "order_id": oid, "status": o.status, "channel": o.channel, "ship_method": o.ship_method,
            "ship_state": o.ship_state, "created_at": o.created_at, "ship_by": o.ship_by.date().isoformat(),
            "shipped_at": None if pd.isna(o.shipped_at) else o.shipped_at, "picker": o.picker or None,
            "wave_id": o.wave_id or None, "hold_reason": o.hold_reason or None,
            "lines": ln[["line_no", "sku", "name", "qty", "qty_allocated", "location"]].to_dict("records"),
            "exceptions": [] if ex.empty else ex[["type", "detail", "action"]].to_dict("records"),
        })

    def list_exceptions(self, type: str | None = None, limit: int = 15) -> dict:
        """Open order exceptions (Late, At risk, Short, Stuck, On hold) with recommended actions, most urgent first."""
        ex = self.s.exceptions
        if ex.empty:
            return {"total": 0, "items": []}
        if type:
            ex = ex[ex.type.str.lower() == type.lower()]
        return {"total": int(len(ex)), "by_type": ex.type.value_counts().to_dict(),
                "items": _records(ex, ["order_id", "type", "status", "ship_method", "ship_by", "detail", "action"], limit)}

    def order_board(self) -> dict:
        """Counts of today's and open orders by pipeline status."""
        return _clean(self.s.board)

    # ------------------------------------------------------------ inventory
    def sku_info(self, query: str, limit: int = 5) -> dict:
        """Look up SKUs by id (PT-12345) or by words in the name; returns stock, forecast, reorder point and slot."""
        p = self.s.policy
        q = query.strip()
        if re.fullmatch(r"(?i)pt-?\d{5}", q):
            hit = p[p.sku == q.upper().replace("PT", "PT-").replace("--", "-")]
        else:
            words = [w for w in re.findall(r"[a-z0-9#.]+", q.lower()) if len(w) > 1]
            mask = np.ones(len(p), bool)
            low = p.name.str.lower()
            for w in words:
                mask &= low.str.contains(re.escape(w)).to_numpy()
            hit = p[mask].sort_values("picks_per_week", ascending=False)
        if hit.empty:
            return {"error": f"No SKU matches '{query}'."}
        cols = ["sku", "name", "category", "abc", "status", "location", "on_hand", "allocated", "available",
                "on_order", "next_po_eta", "daily_demand", "fcst_4wk_units", "sold_4wk_units", "reorder_point",
                "safety_stock", "suggested_qty", "days_of_cover", "projected_stockout", "supplier",
                "lead_time_days", "unit_price"]
        return {"matches": int(len(hit)), "items": _records(hit, cols, limit)}

    def reorder_recommendations(self, category: str | None = None, supplier: str | None = None,
                                limit: int = 20) -> dict:
        """SKUs that need a purchase order now (stockouts first, then by revenue at risk), with suggested quantities rounded to case packs."""
        p = self.s.policy
        r = p[p.status.isin(["Stockout", "Reorder now"]) & (p.suggested_qty > 0)]
        if category:
            r = r[r.category.str.lower().str.contains(category.lower())]
        if supplier:
            r = r[r.supplier.str.lower().str.contains(supplier.lower())]
        r = r.assign(_s=(r.status == "Stockout").astype(int)).sort_values(["_s", "revenue_at_risk"], ascending=False)
        by_sup = r.groupby("supplier").agg(skus=("sku", "size"), units=("suggested_qty", "sum"),
                                           value=("suggested_po_value", "sum")).round(2).reset_index()
        return {
            "skus": int(len(r)), "total_value": round(float(r.suggested_po_value.sum()), 2),
            "by_supplier": _clean(by_sup.to_dict("records")),
            "items": _records(r, ["sku", "name", "status", "abc", "available", "on_order", "reorder_point",
                                  "suggested_qty", "suggested_po_value", "supplier", "lead_time_days",
                                  "revenue_at_risk", "projected_stockout"], limit),
        }

    def stock_health(self, status: str | None = None, category: str | None = None, limit: int = 15) -> dict:
        """Inventory health counts by status (Stockout, Reorder now, Watch, Healthy, Overstock, Slow mover), optionally listing SKUs in one status."""
        p = self.s.policy
        if category:
            p = p[p.category.str.lower().str.contains(category.lower())]
        out = {"counts": p.status.value_counts().to_dict(),
               "overstock_value": round(float(p.overstock_value.sum()), 2),
               "revenue_at_risk": round(float(p.revenue_at_risk.sum()), 2)}
        if status:
            sel = p[p.status.str.lower() == status.lower()]
            key = "overstock_value" if status.lower() == "overstock" else "revenue_at_risk"
            sel = sel.sort_values([key, "picks_per_week"], ascending=False)
            out["items"] = _records(sel, ["sku", "name", "category", "abc", "on_hand", "available", "days_of_cover",
                                          "daily_demand", "overstock_value", "revenue_at_risk", "location"], limit)
            out["by_category"] = sel.category.value_counts().head(8).to_dict()
        return _clean(out)

    def forecast_demand(self, query: str, weeks: int = 8) -> dict:
        """Weekly unit forecast for a SKU (PT-12345) or a whole category name, next 1-8 weeks, with last year's same weeks for context."""
        s = self.s
        weeks = int(max(1, min(weeks, 8)))
        labels = [d.date().isoformat() for d in s.future_week_starts[:weeks]]
        ly = slice(s.units.shape[1] - 52, s.units.shape[1] - 52 + weeks)
        cat = [c for c in s.cats if query.lower() in c.lower()]
        if re.fullmatch(r"(?i)pt-?\d{5}", query.strip()):
            sku_id = "PT-" + re.sub(r"\D", "", query)
            i = s.skus.index[s.skus.sku == sku_id]
            if len(i) == 0:
                return {"error": f"No SKU {query}."}
            i = i[0]
            return _clean({"sku": s.skus.sku[i], "name": s.skus.name[i], "week_starting": labels,
                           "forecast_units": s.units_fcst[i, :weeks].round(1).tolist(),
                           "same_weeks_last_year": s.units[i, ly].tolist(),
                           "last_8_weeks_actual": s.units[i, -8:].tolist()})
        if cat:
            m = s.skus.category.to_numpy() == cat[0]
            return _clean({"category": cat[0], "week_starting": labels,
                           "forecast_units": s.units_fcst[m, :weeks].sum(0).round().tolist(),
                           "same_weeks_last_year": s.units[m][:, ly].sum(0).tolist(),
                           "last_8_weeks_actual": s.units[m][:, -8:].sum(0).tolist(),
                           "backtest_note": "Category-week forecasts backtested at "
                                            f"{s.backtest.iloc[-1].model_wape:.1%} WAPE over the last 8 weeks."})
        return {"error": f"'{query}' is not a SKU id or a category. Categories: {', '.join(s.cats)}."}

    # ------------------------------------------------------------ picking
    def plan_waves(self, max_orders_per_wave: int = 50, include_released: bool = False) -> dict:
        """Build pick waves for unreleased orders: priority order, zone pick lists in walking order, distance saved vs single-order picking."""
        plan = self.s.plan_waves(max_orders=int(max_orders_per_wave), include_released=include_released)
        slim = [{k: w[k] for k in ("wave_id", "orders", "lines", "units", "due_today", "overnight", "expedited",
                                   "distance_m", "single_order_distance_m", "est_minutes")}
                | {"zones": [f"{p['zone']}: {len(p['stops'])} stops, {p['est_minutes']} min" for p in w["pick_lists"]]}
                for w in plan["waves"]]
        return _clean({"summary": plan["summary"], "waves": slim})

    def pick_list(self, wave_id: str, zone: str) -> dict:
        """Stop-by-stop pick list for one zone of one wave (from the current wave plan)."""
        for w in self.s.wave_plan["waves"]:
            if w["wave_id"].lower() == wave_id.lower():
                for p in w["pick_lists"]:
                    if p["zone"].lower() == zone.strip().lower()[:1]:
                        return _clean(p)
                return {"error": f"Wave {wave_id} has no picks in zone {zone}."}
        return {"error": f"No wave {wave_id}. Current waves: {[w['wave_id'] for w in self.s.wave_plan['waves']]}"}

    def labor_plan(self) -> dict:
        """Next 5 business days: expected orders/lines and pickers needed with batch vs single-order picking."""
        return {"days": _clean(self.s.labor_plan)}

    def picker_performance(self) -> dict:
        """Last 4 weeks per picker: lines/hour, mispick rate, and coaching flags."""
        return {"pickers": _records(self.s.pickers, ["picker", "orders", "lines", "lines_per_hour",
                                                     "mispick_rate_pct", "flag"], 40)}

    # ------------------------------------------------------------ slotting
    def slotting_moves(self, zone: str | None = None, limit: int = 15) -> dict:
        """Ranked slot swaps that move fast movers closer to pack-out, with weekly walking saved."""
        sl = self.s.slotting
        if zone:
            sl = sl[sl.zone == zone.strip().upper()[:1]]
        heat = self.s.zone_heat
        return {
            "total_moves_available": int(len(sl)),
            "zone_health": _clean(heat.to_dict("records")),
            "note": "Savings are single-trip equivalents; hours_saved_per_week_batched applies today's batch-picking ratio.",
            "moves": _records(sl, ["rank", "zone", "sku", "name", "picks_per_week", "from_location", "to_location",
                                   "swap_sku", "swap_picks_per_week", "metres_saved_per_week",
                                   "hours_saved_per_week_batched"], limit),
        }


def tool_specs(tools: Tools) -> list[dict]:
    """Claude tool definitions generated from the methods above."""
    spec = {
        "daily_briefing": {},
        "lookup_order": {"order_id": ("string", "Order id like SO-1546575", True)},
        "list_exceptions": {"type": ("string", "Late | At risk | Short | Stuck | On hold", False),
                            "limit": ("integer", "Max rows (default 15)", False)},
        "order_board": {},
        "sku_info": {"query": ("string", "SKU id (PT-12345) or words from the product name", True),
                     "limit": ("integer", "Max matches (default 5)", False)},
        "reorder_recommendations": {"category": ("string", "Filter by category", False),
                                    "supplier": ("string", "Filter by supplier", False),
                                    "limit": ("integer", "Max rows (default 20)", False)},
        "stock_health": {"status": ("string", "Stockout | Reorder now | Watch | Healthy | Overstock | Slow mover", False),
                         "category": ("string", "Filter by category", False),
                         "limit": ("integer", "Max rows (default 15)", False)},
        "forecast_demand": {"query": ("string", "SKU id or category name", True),
                            "weeks": ("integer", "1-8 (default 8)", False)},
        "plan_waves": {"max_orders_per_wave": ("integer", "Default 50", False),
                       "include_released": ("boolean", "Re-plan released-but-unpicked orders too", False)},
        "pick_list": {"wave_id": ("string", "e.g. W-1007-N01", True), "zone": ("string", "A-F", True)},
        "labor_plan": {},
        "picker_performance": {},
        "slotting_moves": {"zone": ("string", "A-F", False), "limit": ("integer", "Max rows (default 15)", False)},
    }
    out = []
    for name, params in spec.items():
        fn: Callable = getattr(tools, name)
        out.append({
            "name": name,
            "description": (fn.__doc__ or "").strip(),
            "input_schema": {
                "type": "object",
                "properties": {k: {"type": t, "description": d} for k, (t, d, _) in params.items()},
                "required": [k for k, (_, _, req) in params.items() if req],
            },
        })
    return out


def call(tools: Tools, name: str, args: dict) -> str:
    try:
        return json.dumps(getattr(tools, name)(**(args or {})), default=str)
    except Exception as e:  # tools must never crash the agent loop
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
