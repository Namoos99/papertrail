"""Live order pipeline: status board, exceptions with recommended actions, picker stats."""
from __future__ import annotations

import numpy as np
import pandas as pd

from papertrail import config as C

PIPELINE = ["New", "Released", "Picking", "Packed", "Shipped", "On hold"]
STUCK_HOURS = {"Picking": 4.0, "Packed": 3.0, "Released": 6.0}
SEVERITY = {"Late": 1, "Short": 2, "At risk": 2, "Stuck": 3, "On hold": 3}

HOLD_ACTION = {
    "Address failed validation": "Call/email the customer to confirm the address, then release.",
    "Payment under review": "Check the payment processor; release or cancel within 24h.",
    "Customer requested change": "Apply the change in the order, re-allocate, and release.",
    "PO number missing (wholesale)": "Ask the account rep for the PO number; release once received.",
}


def live_orders(orders: pd.DataFrame, history_end) -> pd.DataFrame:
    return orders[orders.created_at >= pd.Timestamp(history_end) + pd.Timedelta(days=1)]


def substitutes(pol: pd.DataFrame) -> dict:
    """Same product type and variant in a different colour, in stock -> a sensible sub.
    Returns {(product_type, variant): [(sku, color, available), ...]}."""
    df = pol[pol.available > 0].sort_values("available", ascending=False)
    return {k: list(zip(g.sku, g.color, g.available))[:4]
            for k, g in df.groupby(["product_type", "variant"])}


def exceptions(orders: pd.DataFrame, lines: pd.DataFrame, pol: pd.DataFrame,
               pos: pd.DataFrame, now: pd.Timestamp, history_end) -> pd.DataFrame:
    live = live_orders(orders, history_end)
    open_ = live[live.status != "Shipped"].copy()
    today = now.normalize()
    hours_to_cutoff = C.CARRIER_CUTOFF_HOUR - (now.hour + now.minute / 60)
    pol_i = pol.set_index("sku")
    subs = substitutes(pol)
    po_next = pos.sort_values("eta").groupby("sku").first()
    rows = []

    def add(o, kind, detail, action, sku=None):
        rows.append({
            "order_id": o.order_id, "type": kind, "severity": SEVERITY[kind], "status": o.status,
            "ship_method": o.ship_method, "ship_by": o.ship_by.date().isoformat(),
            "channel": o.channel, "detail": detail, "action": action, "sku": sku,
            "age_hours": round((now - o.created_at).total_seconds() / 3600, 1),
        })

    short = lines[lines.order_id.isin(open_.order_id) & (lines.qty_allocated < lines.qty)]
    short_by_order = short.groupby("order_id")
    for o in open_.itertuples():
        if o.status == "On hold":
            add(o, "On hold", o.hold_reason, HOLD_ACTION.get(o.hold_reason, "Review and release."))
            continue
        if o.ship_by < today:
            upgrade = {"Ground": "Upgrade to Expedited at our cost",
                       "Expedited": "Upgrade to Overnight at our cost"}.get(o.ship_method, "Ship on the next truck")
            add(o, "Late", f"Ship-by was {o.ship_by:%a %b %d}; still {o.status}.",
                f"{upgrade}, pull into the next wave, and email the customer a new delivery date.")
        elif o.ship_by == today and o.status in ("New", "Released"):
            add(o, "At risk", f"Must ship today; {hours_to_cutoff:.1f}h to the 5:30pm carrier cutoff.",
                "Pull into the next wave now (priority lane).")
        lim = STUCK_HOURS.get(o.status)
        in_state = (now - o.status_since).total_seconds() / 3600
        if lim and in_state > lim:
            who = f" with {o.picker}" if o.picker else ""
            add(o, "Stuck", f"{o.status} for {in_state:.0f}h{who} (normal < {lim:.0f}h).",
                f"Check with {o.picker or 'the wave lead'}; likely a missing tote or an unscanned pack-out.")
        if o.order_id in short_by_order.groups:
            for ln in short_by_order.get_group(o.order_id).itertuples():
                sk = pol_i.loc[ln.sku]
                miss = int(ln.qty - ln.qty_allocated)
                if ln.sku in po_next.index:
                    p = po_next.loc[ln.sku]
                    eta = pd.Timestamp(p.eta)
                    if eta <= o.ship_by + pd.Timedelta(days=2):
                        act = f"Hold line for {p.po_id} (ETA {eta:%a %b %d}), ship complete."
                    else:
                        act = f"Ship available now, backorder {miss} — {p.po_id} lands {eta:%b %d}."
                else:
                    alt = [s for s in subs.get((sk.product_type, sk.variant), []) if s[0] != ln.sku]
                    act = (f"Offer substitute {alt[0][0]} ({alt[0][1]}, {alt[0][2]} on hand)."
                           if alt else "Ship available now, backorder the rest, raise a PO today.")
                add(o, "Short", f"{ln.sku} short {miss} of {ln.qty} ({sk['name']}).", act, ln.sku)
    ex = pd.DataFrame(rows)
    if ex.empty:
        return ex
    return ex.sort_values(["severity", "ship_by", "age_hours"], ascending=[True, True, False]).reset_index(drop=True)


def status_board(orders: pd.DataFrame, history_end, now: pd.Timestamp) -> dict:
    live = live_orders(orders, history_end)
    today = live[live.created_at >= now.normalize()]
    shipped_today = live[(live.status == "Shipped") & (live.shipped_at >= now.normalize())]
    open_ = live[live.status != "Shipped"]
    return {
        "orders_today": int(len(today)),
        "shipped_today": int(len(shipped_today)),
        "open": int(len(open_)),
        "by_status": {s: int((open_.status == s).sum()) for s in PIPELINE if s != "Shipped"},
        "due_today_open": int((open_.ship_by <= now.normalize()).sum()),
        "week_to_date_orders": int(len(live)),
    }


def picker_stats(orders: pd.DataFrame, lines: pd.DataFrame, now: pd.Timestamp, weeks: int = 4) -> pd.DataFrame:
    since = now.normalize() - pd.Timedelta(weeks=weeks)
    o = orders[(orders.status == "Shipped") & (orders.created_at >= since) & (orders.picker != "")]
    nl = lines[lines.order_id.isin(o.order_id)].groupby("order_id").size()
    o = o.assign(lines=o.order_id.map(nl))
    g = o.groupby("picker").agg(orders=("order_id", "size"), lines=("lines", "sum"),
                                minutes=("pick_minutes", "sum"), mispicks=("mispick", "sum"))
    g["lines_per_hour"] = (g.lines / (g.minutes / 60)).round(1)
    g["mispick_rate_pct"] = (100 * g.mispicks / g.orders).round(2)
    med_rate, med_lph = g.mispick_rate_pct.median(), g.lines_per_hour.median()
    g["flag"] = np.select(
        [g.mispick_rate_pct > 1.6 * med_rate, g.lines_per_hour < 0.88 * med_lph,
         (g.mispick_rate_pct < 0.7 * med_rate) & (g.lines_per_hour > med_lph)],
        ["Accuracy coaching", "Pace coaching", "Top performer — pair as mentor"], "")
    return g.reset_index().sort_values("lines_per_hour", ascending=False)
