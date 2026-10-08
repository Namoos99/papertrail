"""Velocity-based slotting: put the SKUs pickers visit most where the walk is shortest.

Each slot has a cost in metre-equivalents (round trip to the pack station plus a
penalty for floor-level and top-shelf picks). Within each zone, the ideal is to
rank SKUs by forecast pick lines/week and assign them to slots ranked by cost.
Re-slotting a whole zone at once isn't realistic, so we turn the gap into
*swaps*: a hot SKU sitting in a bad slot trades places with a slow SKU sitting
in a prime slot. Each swap's benefit = (v_hot - v_slow) x (cost_bad - cost_good),
in metres per week. The list is ranked so the team can do the top 20 on a
Saturday and capture most of the value.

Caveat (also in DECISIONS.md): savings are measured on single-command trips.
With batch picking the real saving is smaller, because one walk serves many
lines. Treat the number as a ranking signal, not a promise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from papertrail import config as C


def swap_plan(skus: pd.DataFrame, locations: pd.DataFrame, velocity: pd.Series,
              prime_share: float = 0.12) -> pd.DataFrame:
    """velocity: forecast pick lines/week indexed by sku."""
    loc = locations.set_index("location")
    df = skus[["sku", "name", "zone", "location", "category"]].copy()
    df["velocity"] = df.sku.map(velocity).fillna(0).to_numpy()
    df["slot_cost"] = df.location.map(loc.slot_cost_m)

    moves = []
    for zone, z in df.groupby("zone"):
        k = max(1, int(len(z) * prime_share))
        prime_slots = loc[loc.zone == zone].sort_values("slot_cost_m").index[:k]
        hot = z.nlargest(k, "velocity")
        move_in = hot[~hot.location.isin(prime_slots)].sort_values("velocity", ascending=False)
        occupants = z[z.location.isin(prime_slots) & ~z.sku.isin(hot.sku)]
        # free prime slots (not occupied by any SKU) can be used without a swap
        empty_prime = [s for s in prime_slots if s not in set(z.location)]
        targets = (
            [(loc.at[s, "slot_cost_m"], s, None) for s in empty_prime]
            + [(r.slot_cost, r.location, r) for r in occupants.itertuples()]
        )
        targets.sort(key=lambda t: t[0])
        for hot_row, (t_cost, t_loc, occ) in zip(move_in.itertuples(), targets):
            v_out = occ.velocity if occ is not None else 0.0
            gain = (hot_row.velocity - v_out) * (hot_row.slot_cost - t_cost)
            if gain <= 0:
                continue
            moves.append({
                "zone": zone,
                "sku": hot_row.sku, "name": hot_row.name, "picks_per_week": round(hot_row.velocity, 1),
                "from_location": hot_row.location, "to_location": t_loc,
                "swap_sku": None if occ is None else occ.sku,
                "swap_name": None if occ is None else occ.name,
                "swap_picks_per_week": None if occ is None else round(v_out, 1),
                "metres_saved_per_week": round(gain, 1),
            })
    plan = pd.DataFrame(moves).sort_values("metres_saved_per_week", ascending=False).reset_index(drop=True)
    plan["hours_saved_per_week"] = (plan.metres_saved_per_week / C.WALK_SPEED_MPS / 3600).round(2)
    plan.insert(0, "rank", np.arange(1, len(plan) + 1))
    return plan


def zone_heat(skus: pd.DataFrame, locations: pd.DataFrame, velocity: pd.Series) -> pd.DataFrame:
    """How well each zone is slotted today: share of pick volume sitting in the
    cheapest 20% of slots (ideal = most of it)."""
    loc = locations.set_index("location")
    df = skus[["sku", "zone", "location"]].copy()
    df["velocity"] = df.sku.map(velocity).fillna(0).to_numpy()
    df["slot_cost"] = df.location.map(loc.slot_cost_m)
    rows = []
    for zone, z in df.groupby("zone"):
        cut = loc[loc.zone == zone].slot_cost_m.quantile(0.2)
        in_prime = z.loc[z.slot_cost <= cut, "velocity"].sum() / max(z.velocity.sum(), 1e-9)
        ideal = z.velocity.sort_values(ascending=False).iloc[: int((loc.zone == zone).sum() * 0.2)].sum() / max(z.velocity.sum(), 1e-9)
        rows.append({"zone": zone, "zone_name": C.ZONE_NAMES[zone], "skus": len(z),
                     "picks_per_week": round(z.velocity.sum()),
                     "volume_in_prime_pct": round(100 * in_prime, 1),
                     "ideal_volume_in_prime_pct": round(100 * min(ideal, 1), 1)})
    return pd.DataFrame(rows)
