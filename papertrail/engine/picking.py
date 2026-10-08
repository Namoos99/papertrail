"""Wave planning, batch pick lists, and S-shape routing.

Today's pickers walk one order at a time. The plan here:
  1. Prioritise open orders by ship-by date, ship method, then age.
  2. Cut them into waves of at most 50 orders / ~30 minutes per zone picker.
  3. Inside a wave, merge identical SKUs (pick once, sort at pack-out) and split
     the work by zone so each picker stays in one part of the building.
  4. Sequence every zone pick list in S-shape (serpentine) order.
Distances use the same geometry as the warehouse model, so "before vs after"
is an apples-to-apples estimate rather than a marketing number.
"""
from __future__ import annotations

import pandas as pd

from papertrail import config as C

AISLE_DEPTH_M = 2.0 + C.BAYS_PER_SIDE * C.BAY_DEPTH_M + 1.0
METHOD_RANK = {"Overnight": 0, "Expedited": 1, "Ground": 2}


def s_shape(stops: pd.DataFrame, pack_x: float) -> tuple[pd.DataFrame, float]:
    """Order stops serpentine-style and return (ordered stops, metres walked).
    stops needs columns x, y, aisle_key."""
    if stops.empty:
        return stops, 0.0
    aisles = sorted(stops.aisle_key.unique(), key=lambda a: stops.loc[stops.aisle_key == a, "x"].iloc[0])
    ordered, vertical = [], 0.0
    n = len(aisles)
    for i, a in enumerate(aisles):
        grp = stops[stops.aisle_key == a]
        last_odd = (i == n - 1) and (n % 2 == 1)
        if last_odd:
            ordered.append(grp.sort_values("y"))
            vertical += 2 * grp.y.max()
        else:
            ordered.append(grp.sort_values("y", ascending=(i % 2 == 0)))
            vertical += AISLE_DEPTH_M
    xs = stops.x
    horizontal = abs(pack_x - xs.min()) + (xs.max() - xs.min()) + abs(xs.max() - pack_x)
    return pd.concat(ordered), float(horizontal + vertical)


def single_order_distance(order_stops: pd.DataFrame, pack_x: float) -> float:
    return s_shape(order_stops, pack_x)[1]


def pick_minutes(metres: float, lines: int, units: int, sort_lines: int = 0) -> float:
    """sort_lines: order lines that must be sorted into totes at pack-out (batch picking)."""
    return (metres / C.WALK_SPEED_MPS + lines * C.SECONDS_PER_LINE + sort_lines * C.SECONDS_SORT_PER_LINE
            + max(units - lines, 0) * C.SECONDS_PER_EXTRA_UNIT) / 60


def prioritise(open_orders: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    o = open_orders.copy()
    o["method_rank"] = o.ship_method.map(METHOD_RANK)
    o["due_today"] = o.ship_by <= now.normalize()
    return o.sort_values(["ship_by", "method_rank", "created_at"])


def build_waves(open_orders: pd.DataFrame, lines: pd.DataFrame, skus: pd.DataFrame,
                locations: pd.DataFrame, pack_x: float, now: pd.Timestamp,
                target_minutes: float = 30, max_orders: int = 50,
                pickers_per_wave: int | None = None) -> dict:
    """open_orders: orders with status New (and released-but-unpicked if you want to re-plan)."""
    if open_orders.empty:
        return {"waves": [], "summary": {}}
    pri = prioritise(open_orders, now)
    loc = locations.set_index("location")
    sk = skus.set_index("sku")
    ol = lines[lines.order_id.isin(pri.order_id) & (lines.qty_allocated > 0)].copy()
    ol["location"] = ol.sku.map(sk.location)
    ol = ol.join(loc[["zone", "aisle", "x", "y", "level"]], on="location")
    ol["aisle_key"] = ol.zone + "-" + ol.aisle.astype(str)

    # rough per-order effort to size waves: ~2 min walking amortised + line time
    per_order = ol.groupby("order_id").agg(lines=("sku", "size"), units=("qty_allocated", "sum"))
    per_order["est_min"] = 0.6 + per_order.lines * C.SECONDS_PER_LINE / 60
    pri = pri[pri.order_id.isin(per_order.index)]

    waves, cur, cur_min = [], [], 0.0
    zones_in_play = max(1, ol.zone.nunique())
    budget = target_minutes * (pickers_per_wave or zones_in_play)
    for oid in pri.order_id:
        m = per_order.at[oid, "est_min"]
        if cur and (cur_min + m > budget or len(cur) >= max_orders):
            waves.append(cur); cur, cur_min = [], 0.0
        cur.append(oid); cur_min += m
    if cur:
        waves.append(cur)

    stamp = now.strftime("%m%d")
    out, tot_batch, tot_single = [], 0.0, 0.0
    for wi, oids in enumerate(waves, start=1):
        wl = ol[ol.order_id.isin(oids)]
        single = sum(single_order_distance(g, pack_x) for _, g in wl.groupby("order_id"))
        pick_lists, wave_m = [], 0.0
        for zone, zl in wl.groupby("zone"):
            merged = (zl.groupby(["sku", "location", "aisle_key", "x", "y", "level"], as_index=False)
                        .agg(qty=("qty_allocated", "sum"), orders=("order_id", lambda s: sorted(set(s)))))
            route, metres = s_shape(merged, pack_x)
            wave_m += metres
            stops = [{
                "seq": i + 1, "location": r.location, "sku": r.sku, "name": sk.at[r.sku, "name"],
                "qty": int(r.qty), "orders": r.orders[:6], "n_orders": len(r.orders),
            } for i, r in enumerate(route.itertuples())]
            pick_lists.append({
                "zone": zone, "zone_name": C.ZONE_NAMES[zone], "stops": stops,
                "lines": int(len(zl)), "units": int(zl.qty_allocated.sum()),
                "distance_m": round(metres), "est_minutes": round(pick_minutes(metres, len(merged), int(zl.qty_allocated.sum()), len(zl)), 1),
            })
        meta = pri.set_index("order_id").loc[oids]
        out.append({
            "wave_id": f"W-{stamp}-N{wi:02d}",
            "orders": len(oids), "order_ids": oids,
            "lines": int(len(wl)), "units": int(wl.qty_allocated.sum()),
            "due_today": int(meta.due_today.sum()),
            "overnight": int((meta.ship_method == "Overnight").sum()),
            "expedited": int((meta.ship_method == "Expedited").sum()),
            "distance_m": round(wave_m), "single_order_distance_m": round(single),
            "pick_lists": sorted(pick_lists, key=lambda p: p["zone"]),
            "est_minutes": round(max(p["est_minutes"] for p in pick_lists), 1),
        })
        tot_batch += wave_m; tot_single += single

    return {
        "waves": out,
        "summary": {
            "waves": len(out), "orders": int(sum(w["orders"] for w in out)),
            "lines": int(sum(w["lines"] for w in out)),
            "distance_km": round(tot_batch / 1000, 2),
            "single_order_distance_km": round(tot_single / 1000, 2),
            "walking_saved_pct": round(100 * (1 - tot_batch / max(tot_single, 1)), 1),
            "walking_hours_saved": round((tot_single - tot_batch) / C.WALK_SPEED_MPS / 3600, 1),
        },
    }
