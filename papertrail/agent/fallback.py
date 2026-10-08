"""Local answers when Claude isn't available.

A keyword router maps the question to one tool and formats the result as short
markdown. It can't hold a conversation, but the floor never goes dark because
an API call failed — pick lists, exceptions and reorder lists keep working.
"""
from __future__ import annotations

import re

from papertrail.agent.tools import Tools


def _money(x) -> str:
    return f"${x:,.0f}"


def _rows(items: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)
    body = "\n".join("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |" for r in items)
    return f"{head}\n{body}"


def answer(tools: Tools, q: str):
    from papertrail.agent.agent import Reply   # local import avoids a cycle
    ql = q.lower()

    m = re.search(r"so-?\s?(\d{7})", ql)
    if m:
        r = tools.lookup_order(m.group(1))
        if "error" in r:
            return Reply(r["error"], ["lookup_order"], "local")
        txt = (f"**{r['order_id']}** — {r['status']} · {r['ship_method']} · ship by {r['ship_by']}"
               + (f" · picker {r['picker']}" if r["picker"] else "") + "\n\n"
               + _rows(r["lines"], ["sku", "name", "qty", "qty_allocated", "location"]))
        for e in r["exceptions"]:
            txt += f"\n\n⚠️ **{e['type']}:** {e['detail']} → {e['action']}"
        return Reply(txt, ["lookup_order"], "local")

    m = re.search(r"\bpt-?\s?(\d{5})\b", ql)
    if m and ("forecast" in ql or "demand" in ql):
        r = tools.forecast_demand(f"PT-{m.group(1)}")
        if "error" in r:
            return Reply(r["error"], ["forecast_demand"], "local")
        rows = [{"week": w, "forecast": f, "last year": ly}
                for w, f, ly in zip(r["week_starting"], r["forecast_units"], r["same_weeks_last_year"])]
        return Reply(f"**{r['sku']}** {r['name']}\n\n" + _rows(rows, ["week", "forecast", "last year"]),
                     ["forecast_demand"], "local")
    if m:
        r = tools.sku_info(f"PT-{m.group(1)}")
        if "error" in r:
            return Reply(r["error"], ["sku_info"], "local")
        s = r["items"][0]
        return Reply(
            f"**{s['sku']}** {s['name']}\n\n"
            f"- **{s['status']}** · class {s['abc']} · slot `{s['location']}`\n"
            f"- On hand {s['on_hand']} (available {s['available']}), on order {s['on_order']}"
            + (f" arriving {s['next_po_eta']}" if s["next_po_eta"] else "") + "\n"
            f"- Demand ≈ {s['daily_demand']}/day · reorder point {s['reorder_point']} · cover {s['days_of_cover']} days\n"
            + (f"- **Order {s['suggested_qty']}** from {s['supplier']} ({s['lead_time_days']}-day lead time)"
               if s["suggested_qty"] else "- No purchase needed this week"),
            ["sku_info"], "local")

    if any(w in ql for w in ("wave", "pick list", "picklist", "release")):
        r = tools.plan_waves()
        s = r["summary"]
        txt = (f"**{s['waves']} waves** for {s['orders']} orders ({s['lines']} lines). "
               f"Batch + zone picking walks **{s['walking_saved_pct']}% less** "
               f"({s['distance_km']} km vs {s['single_order_distance_km']} km one-order-at-a-time).\n\n")
        txt += _rows(r["waves"], ["wave_id", "orders", "lines", "due_today", "overnight", "est_minutes"])
        return Reply(txt, ["plan_waves"], "local")

    if any(w in ql for w in ("reorder", "purchase", " po", "buy", "replenish")):
        r = tools.reorder_recommendations(limit=10)
        txt = f"**{r['skus']:,} SKUs need a PO** — {_money(r['total_value'])} total.\n\n"
        txt += _rows(r["by_supplier"], ["supplier", "skus", "units", "value"]) + "\n\n**Most urgent:**\n\n"
        txt += _rows(r["items"], ["sku", "status", "available", "suggested_qty", "supplier", "projected_stockout"])
        return Reply(txt, ["reorder_recommendations"], "local")

    if any(w in ql for w in ("stockout", "out of stock", "overstock", "low stock", "inventory", "stock")):
        status = "Overstock" if "overstock" in ql else "Stockout" if ("out of stock" in ql or "stockout" in ql) else None
        r = tools.stock_health(status=status, limit=10)
        txt = "**Inventory health:** " + " · ".join(f"{k} {v:,}" for k, v in r["counts"].items())
        txt += f"\n\nRevenue at risk {_money(r['revenue_at_risk'])} · tied up in overstock {_money(r['overstock_value'])}"
        if status:
            txt += "\n\n" + _rows(r["items"], ["sku", "name", "on_hand", "days_of_cover", "location"])
        return Reply(txt, ["stock_health"], "local")

    if any(w in ql for w in ("slot", "move", "closer", "layout", "fast mover")):
        r = tools.slotting_moves(limit=10)
        txt = "**Zone slotting today** (share of picks in the best 20% of slots vs ideal):\n\n"
        txt += _rows(r["zone_health"], ["zone", "zone_name", "volume_in_prime_pct", "ideal_volume_in_prime_pct"])
        txt += "\n\n**Top swaps:**\n\n" + _rows(r["moves"], ["rank", "sku", "picks_per_week", "from_location", "to_location", "swap_sku"])
        return Reply(txt, ["slotting_moves"], "local")

    if any(w in ql for w in ("tomorrow", "staff", "headcount", "labor", "labour", "how many")):
        r = tools.labor_plan()
        return Reply("**Labor plan:**\n\n" + _rows(r["days"], ["weekday", "orders", "lines", "pickers_needed", "pickers_needed_single_order"]),
                     ["labor_plan"], "local")

    if any(w in ql for w in ("picker", "coach", "mispick", "accuracy", "productivity")):
        r = tools.picker_performance()
        return Reply("**Pickers, last 4 weeks:**\n\n" + _rows(r["pickers"], ["picker", "lines_per_hour", "mispick_rate_pct", "flag"]),
                     ["picker_performance"], "local")

    if any(w in ql for w in ("exception", "late", "stuck", "hold", "short", "problem", "issue", "at risk")):
        kind = next((k for k in ("Late", "At risk", "Short", "Stuck", "On hold") if k.lower().split()[-1] in ql), None)
        r = tools.list_exceptions(type=kind, limit=12)
        txt = f"**{r['total']} open exceptions** " + " · ".join(f"{k} {v}" for k, v in r.get("by_type", {}).items()) + "\n\n"
        txt += _rows(r["items"], ["order_id", "type", "detail", "action"])
        return Reply(txt, ["list_exceptions"], "local")

    m = re.search(r"forecast|demand", ql)
    if m:
        for cat in tools.s.cats:
            if any(w in ql for w in cat.lower().replace("&", " ").split() if len(w) > 3):
                r = tools.forecast_demand(cat)
                rows = [{"week": w, "forecast": f, "last year": ly}
                        for w, f, ly in zip(r["week_starting"], r["forecast_units"], r["same_weeks_last_year"])]
                return Reply(f"**{cat}** — weekly units\n\n" + _rows(rows, ["week", "forecast", "last year"]),
                             ["forecast_demand"], "local")

    # default: the briefing
    b = tools.daily_briefing()
    bd = b["board"]
    txt = (f"**Today so far:** {bd['orders_today']} orders in, {bd['shipped_today']} shipped, {bd['open']} open "
           f"({bd['due_today_open']} due today).\n\n**Do next:**\n")
    txt += "\n".join(f"{i}. **{a['text']}** — {a['detail']}" for i, a in enumerate(b["actions"], 1))
    t = b["tomorrow"]
    txt += f"\n\n**{t['weekday']}:** ~{t['orders']} orders → {t['pickers_needed']} pickers with batch picking."
    return Reply(txt, ["daily_briefing"], "local")
