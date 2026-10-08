"""Run the engine once and write compact JSON for the web interface.

    python -m papertrail.export          -> ui/data/{meta,skus,sales,orders}.json

The UI is static: it reads these files, so it can be hosted anywhere (GitHub
Pages, a Claude artifact) and still show the full picture. The FastAPI server
serves the same files and adds the Python agent on /api/agent.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from papertrail import config as C
from papertrail.engine import inventory as INV
from papertrail.engine.store import Store


def _j(x):
    if isinstance(x, dict):
        return {str(k): _j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if x is None or x is pd.NaT:
        return None
    return x


def _dump(obj, name):
    C.UI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = C.UI_DATA_DIR / name
    path.write_text(json.dumps(_j(obj), separators=(",", ":")))
    return path.stat().st_size


SKELETON_HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                 '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                 '<style>:root{padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}'
                 'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')


def wrap_page(content: str) -> str:
    """ui/index.html is written as page content (the form Claude artifacts publish).
    Browsers need a doctype and head around it everywhere else."""
    return SKELETON_HEAD + content + "</body></html>"


def write_site(out_dir) -> None:
    """A static copy for GitHub Pages: docs/index.html + docs/data/*.json"""
    import shutil
    from pathlib import Path
    out = Path(out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(wrap_page((C.ROOT / "ui" / "index.html").read_text()))
    for f in C.UI_DATA_DIR.glob("*.json"):
        shutil.copy(f, out / "data" / f.name)


def build(store: Store | None = None) -> dict:
    s = store or Store()
    pol = s.policy
    cats = s.cats
    sups = sorted(pol.supplier.unique())
    statuses = INV.STATUS_ORDER

    # ---- SKUs, columnar --------------------------------------------------
    cover = pol.days_of_cover.replace([np.inf, -np.inf], np.nan)
    skus = {
        "cats": cats, "suppliers": sups, "statuses": statuses,
        "sku": pol.sku.tolist(), "name": pol.name.tolist(),
        "cat": [cats.index(c) for c in pol.category], "abc": "".join(pol.abc),
        "st": [statuses.index(x) for x in pol.status], "loc": pol.location.tolist(),
        "oh": pol.on_hand.tolist(), "av": pol.available.astype(int).tolist(), "oo": pol.on_order.tolist(),
        "eta": [x if isinstance(x, str) else None for x in pol.next_po_eta],
        "dd": pol.daily_demand.round(2).tolist(), "rop": pol.reorder_point.tolist(),
        "ss": pol.safety_stock.tolist(), "sq": pol.suggested_qty.tolist(),
        "cov": [None if pd.isna(v) else round(float(v), 1) for v in cover],
        "so": pol.projected_stockout.tolist(),
        "sup": [sups.index(x) for x in pol.supplier], "lt": pol.lead_time_days.tolist(),
        "cp": pol.case_pack.tolist(), "price": pol.unit_price.tolist(), "cost": pol.unit_cost.tolist(),
        "ppw": pol.picks_per_week.tolist(), "f4": pol.fcst_4wk_units.tolist(), "s4": pol.sold_4wk_units.tolist(),
        "s52": pol.sold_52wk_units.tolist(), "rar": pol.revenue_at_risk.round(0).tolist(),
        "ovv": pol.overstock_value.round(0).tolist(),
    }

    # ---- weekly sales (last 52) + forecast (8), as compact strings ----------
    hist = s.units[:, -52:].astype(int)
    fc = np.round(s.units_fcst, 1)
    sales = {
        "weeks": [d.date().isoformat() for d in s.week_starts[-52:]],
        "future": [d.date().isoformat() for d in s.future_week_starts],
        "h": [",".join(map(str, r)) for r in hist],
        "f": [",".join(f"{v:g}" for v in r) for r in fc],
    }

    # ---- live week orders (lookup in the browser) -------------------------
    live = s.orders[s.orders.created_at >= s.history_end + pd.Timedelta(days=1)]
    ll = s.lines[s.lines.order_id.isin(live.order_id)]
    lines_by = {oid: g[["sku", "qty", "qty_allocated"]].values.tolist() for oid, g in ll.groupby("order_id")}
    orders = [{
        "id": o.order_id, "st": o.status, "ch": o.channel, "sm": o.ship_method, "state": o.ship_state,
        "c": o.created_at.isoformat(timespec="minutes"), "by": o.ship_by.date().isoformat(),
        "sh": None if pd.isna(o.shipped_at) else o.shipped_at.isoformat(timespec="minutes"),
        "since": None if pd.isna(o.status_since) else o.status_since.isoformat(timespec="minutes"),
        "pk": o.picker or None, "w": o.wave_id or None, "hold": o.hold_reason or None,
        "l": lines_by.get(o.order_id, []),
    } for o in live.itertuples()]

    # ---- category series for the forecast chart --------------------------
    cat_series = []
    for i, c in enumerate(cats):
        m = s.cat_codes == i
        cat_series.append({"category": c, "zone": C.CATEGORIES[c][0],
                           "actual": s.units[m][:, -52:].sum(0).astype(int).tolist(),
                           "last_year": s.units[m][:, -104:-52].sum(0).astype(int).tolist(),
                           "forecast": s.units_fcst[m].sum(0).round().tolist()})
    daily = (s.orders[s.orders.created_at >= s.now.normalize() - pd.Timedelta(weeks=13)]
             .set_index("created_at").resample("D").size())

    wp = s.wave_plan
    ex = s.exceptions
    meta = {
        "now": s.now.isoformat(), "company": "PaperTrail Stationery Co.",
        "cutoff_hour": C.CARRIER_CUTOFF_HOUR,
        "zones": C.ZONE_NAMES, "pack_x": s.pack_x,
        "briefing": s.briefing(),
        "exceptions": ex.to_dict("records") if not ex.empty else [],
        "waves": wp["waves"], "wave_summary": wp["summary"],
        "labor": s.labor_plan,
        "pickers": s.pickers.to_dict("records"),
        "slotting": s.slotting.head(300).to_dict("records"),
        "slotting_total_moves": int(len(s.slotting)),
        "batch_walk_factor": s.batch_walk_factor,
        "zone_heat": s.zone_heat.to_dict("records"),
        "backtest": s.backtest.to_dict("records"),
        "categories": cat_series, "weeks": sales["weeks"], "future_weeks": sales["future"],
        "daily_orders": {"dates": [d.date().isoformat() for d in daily.index], "counts": daily.tolist()},
        "locations_by_zone": (s.locations.groupby("zone")
                              .agg(aisles=("aisle", "max"), x0=("x", "min"), x1=("x", "max")).reset_index()
                              .to_dict("records")),
        "dims": {"bays": C.BAYS_PER_SIDE, "levels": C.LEVELS, "aisle_m": C.AISLE_SPACING_M, "bay_m": C.BAY_DEPTH_M},
        "history": {"weeks": C.HISTORY_WEEKS, "orders": int(len(s.orders)), "lines": int(len(s.lines)),
                    "skus": int(len(s.skus)), "slots": int(len(s.locations))},
    }
    sizes = {n: _dump(o, n) for n, o in [("meta.json", meta), ("skus.json", skus),
                                          ("sales.json", sales), ("orders.json", {"orders": orders})]}
    return sizes


if __name__ == "__main__":
    import sys
    for name, size in build().items():
        print(f"{name:12s} {size / 1e6:6.2f} MB")
    if "--site" in sys.argv:
        write_site(C.ROOT / "docs")
        print("static site written to docs/ (point GitHub Pages at it)")
