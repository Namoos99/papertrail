"""Synthetic history for a 15,000-SKU stationery distributor.

Produces, deterministically from one seed:
  data/raw/skus.parquet          catalog + warehouse slot + supplier terms + on-hand
  data/raw/locations.parquet     every slot in the building with coordinates
  data/raw/orders.parquet        104 weeks of shipped orders + this week's live pipeline
  data/raw/order_lines.parquet   order lines (sku, qty, qty_allocated)
  data/raw/purchase_orders.parquet  open inbound POs

The demand model is built so the agent has something real to learn:
  * heavy-tailed velocity (a few thousand SKUs drive most picks)
  * category seasonality (back-to-school, holiday, planner season, tax season)
  * day-of-week pattern, mild growth trend, and occasional promo spikes
  * slots assigned the way a growing warehouse ends up: mostly at random
    inside each zone, so slotting has genuine wins to find

Run:  python -m papertrail.data.generate
"""
from __future__ import annotations

import itertools
import math
from datetime import timedelta

import numpy as np
import pandas as pd

from papertrail import config as C

# ---------------------------------------------------------------------------
# Catalog vocabulary
# ---------------------------------------------------------------------------
BRANDS = [
    "Inkwell", "Northpoint", "Lumen", "Paperbark", "Quill & Co", "Brightline",
    "Atlas Office", "Cedar & Sage", "Studio Mono", "Kestrel", "Fieldnote",
    "Maple Row", "Copperleaf", "Ridgeway",
]

PRODUCT_TYPES = {
    "Pens": ["Gel Pen", "Ballpoint Pen", "Rollerball Pen", "Fineliner", "Fountain Pen", "Brush Pen", "Retractable Gel Pen"],
    "Pencils": ["#2 Pencil", "Mechanical Pencil", "Colored Pencil Set", "Graphite Pencil Set", "Pencil Lead Refill", "Carpenter Pencil"],
    "Markers & Highlighters": ["Highlighter", "Permanent Marker", "Dry Erase Marker", "Washable Marker Set", "Paint Marker", "Chisel Highlighter"],
    "Notebooks & Journals": ["Spiral Notebook", "Composition Book", "Dot Grid Journal", "Hardcover Journal", "Subject Notebook", "Pocket Notebook"],
    "Paper & Pads": ["Copy Paper Ream", "Legal Pad", "Sticky Notes", "Graph Pad", "Cardstock Pack", "Index Cards"],
    "Envelopes & Mailers": ["#10 Envelope Box", "Padded Mailer", "Kraft Envelope", "Poly Mailer", "Catalog Envelope", "Greeting Envelope"],
    "Binders & Folders": ["3-Ring Binder", "Pocket Folder", "Expanding File", "Presentation Binder", "Sheet Protectors", "Hanging Folder"],
    "Filing & Storage": ["File Box", "Storage Bin", "Magazine File", "Document Case", "Archive Box", "Drawer Organizer"],
    "Labels & Tape": ["Shipping Labels", "Address Labels", "Packing Tape", "Washi Tape", "Label Maker Tape", "Masking Tape"],
    "Staplers & Fasteners": ["Stapler", "Staples Box", "Binder Clips", "Paper Clips", "Push Pins", "Hole Punch"],
    "Desk Accessories": ["Desk Organizer", "Pen Cup", "Letter Tray", "Desk Pad", "Monitor Stand", "Bookends"],
    "Planners & Calendars": ["Weekly Planner", "Daily Planner", "Wall Calendar", "Desk Calendar", "Academic Planner", "Undated Planner"],
    "Art & Craft Supplies": ["Watercolor Set", "Sketchbook", "Acrylic Paint Set", "Craft Paper Roll", "Brush Set", "Glitter Glue"],
    "Greeting Cards & Gift Wrap": ["Boxed Holiday Cards", "Birthday Card", "Gift Wrap Roll", "Gift Tags", "Thank You Cards", "Tissue Paper"],
}

COLORS = ["Black", "Blue", "Red", "Green", "Purple", "Pink", "Teal", "Orange", "Gray", "Navy",
          "Assorted", "Pastel", "Neon", "Kraft", "White", "Sage", "Mustard", "Coral"]

# Second variant dimension and pack sizes that make sense for each category
_TIPS = ["Extra Fine", "Fine", "Medium", "Bold", "0.38mm", "0.5mm", "0.7mm", "1.0mm"]
_PAPER = ["Mini", "Pocket", "A5", "A4", "Letter", "Legal", "Large", "Jumbo"]
_MATERIAL = ["Bamboo", "Metal Mesh", "Acrylic", "Recycled", "Walnut", "Matte", "Linen", "Clear"]
_ENVELOPE = ["#6", "#9", "#10", "6x9", "9x12", "10x13", "A2", "A7"]
_CRAFT = ["Student", "Studio", "Artist Grade", "Travel", "Classroom", "Mini", "Large", "Jumbo"]
_TAPE = ["1/2 in", "3/4 in", "1 in", "1.88 in", "2 in", "3 in", "Narrow", "Wide"]
VARIANTS = {
    "Pens": (_TIPS, [1, 2, 3, 4, 6, 8, 10, 12, 24, 36]),
    "Pencils": (["HB", "2B", "4B", "H", "0.5mm", "0.7mm", "0.9mm", "Soft"], [1, 2, 4, 6, 10, 12, 24, 48, 72]),
    "Markers & Highlighters": (["Fine", "Chisel", "Bullet", "Brush", "Jumbo", "Ultra Fine", "Twin Tip", "Medium"], [1, 2, 4, 5, 6, 8, 10, 12, 24]),
    "Notebooks & Journals": (_PAPER, [1, 2, 3, 4, 5, 6, 12]),
    "Paper & Pads": (_PAPER, [1, 2, 3, 4, 5, 6, 10, 12, 24]),
    "Envelopes & Mailers": (_ENVELOPE, [10, 25, 50, 100, 250, 500]),
    "Binders & Folders": (["0.5 in", "1 in", "1.5 in", "2 in", "3 in", "Letter", "Legal", "A4"], [1, 2, 4, 6, 10, 12, 25, 50]),
    "Filing & Storage": (_MATERIAL, [1, 2, 3, 4, 6]),
    "Labels & Tape": (_TAPE, [1, 2, 3, 4, 6, 10, 12, 24, 36]),
    "Staplers & Fasteners": (["Mini", "Standard", "Heavy Duty", "Small", "Medium", "Large", "Jumbo", "Assorted"], [1, 2, 6, 12, 50, 100, 1000, 5000]),
    "Desk Accessories": (_MATERIAL, [1, 2, 3, 4]),
    "Planners & Calendars": (["2026", "2027", "2026-27 Academic", "Undated", "Pocket", "A5", "Letter", "Large"], [1, 2, 3]),
    "Art & Craft Supplies": (_CRAFT, [1, 2, 3, 6, 12, 24, 36]),
    "Greeting Cards & Gift Wrap": (["Floral", "Classic", "Modern", "Kids", "Foil", "Botanical", "Minimal", "Vintage"], [1, 4, 6, 8, 10, 12, 20, 24, 50]),
}

SUPPLIERS = {
    "Domestic (Ohio)": (5, 9),
    "Domestic (Texas)": (6, 10),
    "Mexico": (12, 18),
    "Vietnam": (35, 50),
    "China": (40, 60),
    "Germany": (25, 35),
    "Japan": (30, 45),
}

US_STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ",
             "MA", "TN", "IN", "MO", "MD", "WI", "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK",
             "CT", "UT", "IA", "NV", "AR", "KS", "NM", "NE", "ID", "WV", "HI", "NH", "ME", "MT",
             "RI", "DE", "SD", "ND", "AK", "VT", "WY"]


# ---------------------------------------------------------------------------
# Seasonality profiles, indexed by ISO week (1..53)
# ---------------------------------------------------------------------------
def _bump(weeks: np.ndarray, centre: float, width: float, height: float) -> np.ndarray:
    # circular distance on a 52-week year so Dec/Jan bumps wrap correctly
    d = np.minimum(np.abs(weeks - centre), 52 - np.abs(weeks - centre))
    return height * np.exp(-0.5 * (d / width) ** 2)


def seasonality_profiles() -> dict[str, np.ndarray]:
    w = np.arange(1, 54, dtype=float)
    base = 1.0 - _bump(w, 52, 0.8, 0.35)                      # quiet Christmas week
    profiles = {
        "base": base,
        "school": base + _bump(w, 32, 3.0, 1.9) + _bump(w, 2, 1.5, 0.3),   # back-to-school, Jan restart
        "holiday": base + _bump(w, 47.5, 2.6, 1.7),
        "holiday_light": base + _bump(w, 47.5, 3.0, 0.6),
        "planner": base + _bump(w, 51, 3.5, 1.4) + _bump(w, 32, 2.0, 0.7),
        "tax": base + _bump(w, 9, 5.0, 0.5),
    }
    return {k: v / v.mean() for k, v in profiles.items()}


# ---------------------------------------------------------------------------
# Warehouse
# ---------------------------------------------------------------------------
def build_locations(zone_counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    x_cursor = 0.0
    per_aisle = C.BAYS_PER_SIDE * 2 * C.LEVELS
    for zone in sorted(zone_counts):
        n_slots = math.ceil(zone_counts[zone] * C.SLOT_HEADROOM)
        n_aisles = math.ceil(n_slots / per_aisle)
        for a in range(1, n_aisles + 1):
            ax = x_cursor + (a - 1) * C.AISLE_SPACING_M
            for side in ("L", "R"):
                for bay in range(1, C.BAYS_PER_SIDE + 1):
                    for level in range(1, C.LEVELS + 1):
                        rows.append((f"{zone}-{a:02d}-{side}{bay:02d}-{level}", zone, a, side, bay,
                                     level, ax, 2.0 + bay * C.BAY_DEPTH_M))
        x_cursor += n_aisles * C.AISLE_SPACING_M + C.ZONE_GAP_M
    loc = pd.DataFrame(rows, columns=["location", "zone", "aisle", "side", "bay", "level", "x", "y"])
    pack_x = (loc.x.min() + loc.x.max()) / 2
    loc["dist_to_pack_m"] = (loc.x - pack_x).abs() + loc.y
    loc["slot_cost_m"] = 2 * loc.dist_to_pack_m + loc.level.map(C.LEVEL_PENALTY_M)
    loc.attrs["pack_x"] = pack_x
    return loc


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
def build_catalog(rng: np.random.Generator) -> pd.DataFrame:
    cats = list(C.CATEGORIES)
    shares = np.array([C.CATEGORIES[c][1] for c in cats])
    counts = np.floor(shares / shares.sum() * C.N_SKUS).astype(int)
    counts[np.argmax(counts)] += C.N_SKUS - counts.sum()

    records = []
    for cat, n in zip(cats, counts):
        zone, _, (pmin, pmax), profile, case_pack = C.CATEGORIES[cat]
        variants, packs = VARIANTS[cat]
        combos = list(itertools.product(BRANDS, PRODUCT_TYPES[cat], COLORS, variants, packs))
        pick = rng.choice(len(combos), size=n, replace=False)
        for i in pick:
            brand, ptype, color, size, pack = combos[i]
            records.append((cat, zone, profile, brand, ptype, color, size, pack, pmin, pmax, case_pack))
    df = pd.DataFrame(records, columns=["category", "zone", "season_profile", "brand", "product_type",
                                        "color", "variant", "pack", "pmin", "pmax", "case_pack"])
    df = df.sample(frac=1, random_state=C.SEED).reset_index(drop=True)
    df.insert(0, "sku", [f"PT-{i:05d}" for i in range(10001, 10001 + len(df))])
    df["name"] = (df.brand + " " + df.product_type + ", " + df.variant + ", " + df.color
                  + np.where(df.pack > 1, ", " + df.pack.astype(str) + "-pack", ""))

    # price: log-uniform inside the category range, nudged up for bigger packs
    u = rng.random(len(df))
    price = np.exp(np.log(df.pmin) + u * (np.log(df.pmax) - np.log(df.pmin))) * (1 + np.log1p(df.pack) * 0.08)
    df["unit_price"] = np.round(price, 2) - 0.01
    df["unit_cost"] = np.round(df.unit_price * rng.uniform(0.38, 0.58, len(df)), 2)

    sup_p = np.array([0.22, 0.18, 0.12, 0.16, 0.22, 0.05, 0.05])
    df["supplier"] = rng.choice(list(SUPPLIERS), size=len(df), p=sup_p)
    lo = df.supplier.map(lambda s: SUPPLIERS[s][0])
    hi = df.supplier.map(lambda s: SUPPLIERS[s][1])
    df["lead_time_days"] = (lo + rng.random(len(df)) * (hi - lo)).round().astype(int)

    # Base weekly demand in *lines* — lognormal gives the long tail real catalogs have
    df["base_lines_wk"] = rng.lognormal(mean=-0.95, sigma=1.35, size=len(df))
    # ~3% of SKUs are new launches: no history in year one
    df["launch_week"] = np.where(rng.random(len(df)) < 0.03, rng.integers(40, 95, len(df)), 0)
    # ~2% are being phased out: demand fades through year two
    df["phase_out"] = rng.random(len(df)) < 0.02
    return df.drop(columns=["pmin", "pmax"])


def assign_slots(skus: pd.DataFrame, loc: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Random within zone, on purpose: this is how slotting looks after years of growth."""
    out = pd.Series(index=skus.index, dtype=object)
    for zone, idx in skus.groupby("zone").groups.items():
        slots = loc.loc[loc.zone == zone, "location"].to_numpy()
        chosen = rng.choice(slots, size=len(idx), replace=False)
        out.loc[idx] = chosen
    return out


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------
DOW_FACTOR = np.array([1.22, 1.14, 1.08, 1.03, 0.95, 0.78, 0.80])   # Mon..Sun
DOW_FACTOR = DOW_FACTOR / DOW_FACTOR.mean()
HOUR_WEIGHTS = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 4, 3, 2], dtype=float)
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()


def ship_by(created: pd.Series, method: pd.Series) -> pd.Series:
    """Ship-by date: Overnight same business day (if before 1pm), Expedited next business day,
    Ground within two business days."""
    d = created.dt.normalize()
    add = np.select([method == "Overnight", method == "Expedited"], [0, 1], 2)
    add = np.where((method == "Overnight") & (created.dt.hour >= 13), 1, add)
    out = pd.to_datetime([np.busday_offset(np.datetime64(x.date()), int(a), roll="forward")
                          for x, a in zip(d, add)])
    return pd.Series(out, index=created.index)


def generate_orders(skus: pd.DataFrame, rng: np.random.Generator):
    profiles = seasonality_profiles()
    start = C.HISTORY_END - timedelta(days=7 * C.HISTORY_WEEKS - 1)
    days = pd.date_range(start, C.NOW.replace(hour=0, minute=0), freq="D")

    prof_idx = skus.season_profile.to_numpy()
    base = skus.base_lines_wk.to_numpy()
    launch = skus.launch_week.to_numpy()
    phase = skus.phase_out.to_numpy()
    case = skus.case_pack.to_numpy()
    n_sku = len(skus)

    # promo spikes: ~400 SKUs get a 2-week promo at a random point
    promo_sku = rng.choice(n_sku, 400, replace=False)
    promo_start = rng.integers(0, C.HISTORY_WEEKS - 2, 400)

    mean_lines_per_order = 2.9
    order_rows, line_rows = [], []
    order_seq = 1_000_000
    for d in days:
        week_i = (d - start).days // 7
        woy = min(d.isocalendar().week, 53)
        season = np.empty(n_sku)
        for p, arr in profiles.items():
            season[prof_idx == p] = arr[woy - 1]
        trend = 1 + 0.07 * week_i / 52
        w = base * season * trend
        w = np.where(week_i < launch, 0, w)
        w = np.where(phase, w * np.clip(1 - max(0, week_i - 52) / 40, 0.05, 1), w)
        on_promo = promo_sku[(week_i >= promo_start) & (week_i < promo_start + 2)]
        w[on_promo] *= 3.5

        exp_lines = w.sum() / 7 * DOW_FACTOR[d.dayofweek]
        n_orders = rng.poisson(exp_lines / mean_lines_per_order)
        if d.date() == C.NOW.date():              # today: only orders up to "now"
            frac = HOUR_WEIGHTS[: C.NOW.hour].sum() + HOUR_WEIGHTS[C.NOW.hour] * C.NOW.minute / 60
            n_orders = int(n_orders * frac)
        n_lines = 1 + rng.poisson(mean_lines_per_order - 1, n_orders)
        total = int(n_lines.sum())
        sku_idx = rng.choice(n_sku, size=total, p=w / w.sum())
        order_of_line = np.repeat(np.arange(n_orders), n_lines)

        hours = rng.choice(24, size=n_orders, p=HOUR_WEIGHTS)
        mins = rng.integers(0, 60, n_orders)
        created = d + pd.to_timedelta(hours, "h") + pd.to_timedelta(mins, "m")
        channel = rng.choice(["Web store", "Marketplace", "Wholesale"], n_orders, p=[0.55, 0.38, 0.07])
        method = rng.choice(["Ground", "Expedited", "Overnight"], n_orders, p=[0.80, 0.15, 0.05])
        ids = np.arange(order_seq, order_seq + n_orders)
        order_seq += n_orders
        order_rows.append(pd.DataFrame({
            "order_id": [f"SO-{i}" for i in ids], "created_at": created, "channel": channel,
            "ship_method": method, "ship_state": rng.choice(US_STATES, n_orders),
        }))
        is_wholesale = channel[order_of_line] == "Wholesale"
        qty = np.where(is_wholesale, case[sku_idx] * rng.integers(1, 4, total), 1 + rng.geometric(0.62, total) - 1)
        line_rows.append(pd.DataFrame({"order_id": [f"SO-{i}" for i in ids[order_of_line]],
                                       "sku_idx": sku_idx, "qty": qty}))

    orders = pd.concat(order_rows, ignore_index=True)
    lines = pd.concat(line_rows, ignore_index=True)
    lines = lines.groupby(["order_id", "sku_idx"], as_index=False).qty.sum()   # merge dup skus
    lines["sku"] = skus.sku.to_numpy()[lines.sku_idx]
    lines = lines.drop(columns="sku_idx")
    lines["line_no"] = lines.groupby("order_id").cumcount() + 1
    orders["ship_by"] = ship_by(orders.created_at, orders.ship_method)
    return orders, lines


def simulate_fulfilment(orders: pd.DataFrame, lines: pd.DataFrame, skus: pd.DataFrame,
                        rng: np.random.Generator):
    """History: everything shipped (with a realistic late rate and picker attribution).
    Live week: statuses reflect where each order sits at C.NOW, including exceptions."""
    n = len(orders)
    now = pd.Timestamp(C.NOW)
    hist = orders.created_at < pd.Timestamp(C.HISTORY_END + timedelta(days=1))

    # pickers have different speed and accuracy — the agent should notice
    p_speed = dict(zip(C.PICKERS, rng.normal(1.0, 0.12, len(C.PICKERS)).clip(0.75, 1.3)))
    p_err = dict(zip(C.PICKERS, rng.gamma(2.0, 0.0022, len(C.PICKERS)).clip(0.0008, 0.012)))
    orders["picker"] = rng.choice(C.PICKERS, n)
    n_lines = lines.groupby("order_id").size().reindex(orders.order_id).to_numpy()
    speed = orders.picker.map(p_speed).to_numpy()
    err = orders.picker.map(p_err).to_numpy()
    orders["mispick"] = rng.random(n) < 1 - (1 - err) ** n_lines   # cleared below for unpicked orders
    # single-order picking: a walk out and back per order plus travel between lines
    orders["pick_minutes"] = np.round((60 + n_lines * (C.SECONDS_PER_LINE + 75)) / 60 / speed
                                      * rng.lognormal(0, 0.18, n), 1)

    # history: shipped, ~3.5% after ship-by (more in peak weeks)
    lag_h = rng.gamma(2.2, 7.0, n)
    peak = orders.created_at.dt.isocalendar().week.isin(range(29, 37)).to_numpy()
    lag_h = np.where(peak, lag_h * 1.35, lag_h)
    shipped_at = orders.created_at + pd.to_timedelta(lag_h, "h")
    shipped_at = shipped_at.where(shipped_at.dt.dayofweek < 5,
                                  shipped_at + pd.to_timedelta(7 - shipped_at.dt.dayofweek, "D"))
    orders["shipped_at"] = shipped_at.where(hist)
    orders["status"] = np.where(hist, "Shipped", "")
    orders["exception"] = ""
    orders["hold_reason"] = ""
    orders["status_since"] = orders.shipped_at
    orders["wave_id"] = ""

    # ---- live pipeline -----------------------------------------------------
    live = ~hist
    age_h = ((now - orders.created_at).dt.total_seconds() / 3600).to_numpy()
    r = rng.random(n)
    status = np.full(n, "", dtype=object)
    status[live & (age_h > 20)] = "Shipped"
    slipped = live & (age_h > 20) & (r < 0.008)           # a few older orders fell through the cracks
    status[slipped] = rng.choice(["Released", "Picking"], slipped.sum())
    m = live & (age_h <= 20) & (age_h > 7)
    status[m] = np.where(r[m] < 0.55, "Shipped", np.where(r[m] < 0.8, "Packed", "Picking"))
    m = live & (age_h <= 7) & (age_h > 3)
    status[m] = np.where(r[m] < 0.2, "Packed", np.where(r[m] < 0.65, "Picking", "Released"))
    m = live & (age_h <= 3)
    status[m] = np.where(r[m] < 0.15, "Released", "New")
    orders.loc[live, "status"] = status[live]
    orders.loc[live & (orders.status == "Shipped"), "shipped_at"] = \
        orders.created_at + pd.to_timedelta(rng.uniform(3, 18, n), "h")
    orders.loc[live & (orders.status == "Shipped") & (orders.shipped_at > now), "shipped_at"] = now - pd.Timedelta("30min")
    orders.loc[live & (orders.status != "Shipped"), "picker"] = np.where(
        orders.loc[live & (orders.status != "Shipped"), "status"].isin(["Picking", "Packed"]),
        orders.loc[live & (orders.status != "Shipped"), "picker"], "")

    orders.loc[live & ~orders.status.isin(["Shipped", "Packed"]), "mispick"] = False

    # status_since: when it entered its current state
    since_h = np.clip(np.minimum(age_h * rng.uniform(0.1, 0.6, n), rng.uniform(0.2, 2.2, n)), 0.05, None)
    live_open = live & (orders.status != "Shipped")
    orders.loc[live_open, "status_since"] = now - pd.to_timedelta(since_h, "h")[live_open.to_numpy()]
    orders.loc[live & (orders.status == "Shipped"), "status_since"] = orders.shipped_at

    # holds: address verification / payment review / customer request
    hold = live_open & (rng.random(n) < 0.012)
    orders.loc[hold, "status"] = "On hold"
    orders.loc[hold, "hold_reason"] = rng.choice(
        ["Address failed validation", "Payment under review", "Customer requested change",
         "PO number missing (wholesale)"], hold.sum())

    # stuck: picking/packed far longer than it should be
    stuck = live_open & orders.status.isin(["Picking", "Packed"]) & (rng.random(n) < 0.03)
    orders.loc[stuck, "status_since"] = now - pd.to_timedelta(rng.uniform(5, 26, stuck.sum()), "h")

    # waves for released / picking orders
    rel = live_open & orders.status.isin(["Released", "Picking", "Packed"])
    orders.loc[rel, "wave_id"] = ("W-" + orders.loc[rel, "created_at"].dt.strftime("%m%d")
                                  + "-" + (orders.loc[rel, "created_at"].dt.hour // 2).astype(str).str.zfill(2))
    return orders


def build_inventory(skus: pd.DataFrame, lines: pd.DataFrame, orders: pd.DataFrame,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Current on-hand. Drawn as 'days of cover' against recent demand so the agent
    sees the full spread: stockouts, below reorder point, healthy and overstock."""
    recent = lines.merge(orders[["order_id", "created_at"]], on="order_id")
    cutoff = pd.Timestamp(C.HISTORY_END) - pd.Timedelta(days=181)   # buyers plan off ~6 months
    recent = recent[(recent.created_at > cutoff) & (recent.created_at <= pd.Timestamp(C.HISTORY_END + timedelta(days=1)))]
    daily = recent.groupby("sku").qty.sum().reindex(skus.sku).fillna(0).to_numpy() / 182

    lt = skus.lead_time_days.to_numpy()
    u = rng.random(len(skus))
    # 3.5% out, ~6% inside lead time (need a PO now), ~83% healthy, ~8% overstocked
    cover = np.select(
        [u < 0.035, u < 0.095, u < 0.92],
        [np.zeros(len(skus)), rng.uniform(0.2, 1.0, len(skus)) * lt,
         lt * 1.3 + rng.uniform(15, 55, len(skus))],
        rng.uniform(130, 320, len(skus)),
    )
    on_hand = np.round(daily * cover + np.where(u < 0.035, 0, rng.integers(0, 4, len(skus))))
    # slow movers with no recent sales still have a few units on the shelf
    on_hand = np.where((daily == 0) & (u >= 0.035), rng.integers(2, 30, len(skus)), on_hand)
    return pd.Series(on_hand.astype(int), index=skus.index)


def build_purchase_orders(skus: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Open inbound POs on ~35% of SKUs (more likely when stock is thin)."""
    p = np.where(skus.on_hand == 0, 0.55, 0.32)
    has = rng.random(len(skus)) < p
    s = skus[has]
    days_out = rng.integers(2, s.lead_time_days.clip(lower=3).to_numpy() + 1)
    eta = pd.Timestamp(C.NOW.date()) + pd.to_timedelta(days_out, "D")
    qty = (np.ceil(s.base_lines_wk * 2 * rng.uniform(3, 8, len(s)) / s.case_pack) * s.case_pack).astype(int)
    qty = np.maximum(qty, s.case_pack)
    return pd.DataFrame({
        "po_id": [f"PO-{60000 + i}" for i in range(len(s))],
        "sku": s.sku.to_numpy(), "supplier": s.supplier.to_numpy(),
        "qty": qty.to_numpy(), "eta": eta,
    })


def allocate_live(orders: pd.DataFrame, lines: pd.DataFrame, skus: pd.DataFrame) -> pd.DataFrame:
    """FIFO allocate open orders against on-hand; anything short becomes a SHORT exception."""
    open_ids = orders.loc[~orders.status.isin(["Shipped", ""]), ["order_id", "created_at"]]
    lines["qty_allocated"] = lines.qty
    ol = lines.merge(open_ids, on="order_id").sort_values("created_at")
    avail = skus.set_index("sku").on_hand.to_dict()
    alloc = []
    for sku, q in zip(ol.sku.to_numpy(), ol.qty.to_numpy()):
        a = min(q, avail[sku])
        avail[sku] -= a
        alloc.append(a)
    ol["qty_allocated"] = alloc
    lines = lines.merge(ol[["order_id", "sku", "qty_allocated"]], on=["order_id", "sku"], how="left",
                        suffixes=("_x", ""))
    lines["qty_allocated"] = lines.qty_allocated.fillna(lines.qty_allocated_x).astype(int)
    lines = lines.drop(columns="qty_allocated_x")
    skus["allocated"] = skus.on_hand - skus.sku.map(avail)
    return lines


def main() -> None:
    rng = np.random.default_rng(C.SEED)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("· catalog")
    skus = build_catalog(rng)
    print("· warehouse")
    loc = build_locations(skus.zone.value_counts().to_dict())
    skus["location"] = assign_slots(skus, loc, rng)

    print("· orders (this takes ~a minute)")
    orders, lines = generate_orders(skus, rng)
    orders = simulate_fulfilment(orders, lines, skus, rng)
    skus["on_hand"] = build_inventory(skus, lines, orders, rng)
    pos = build_purchase_orders(skus, rng)
    lines = allocate_live(orders, lines, skus)

    skus = skus.drop(columns=["base_lines_wk", "launch_week", "phase_out"])   # the agent must learn these
    skus.to_parquet(C.RAW_DIR / "skus.parquet", index=False)
    loc.to_parquet(C.RAW_DIR / "locations.parquet", index=False)
    orders.to_parquet(C.RAW_DIR / "orders.parquet", index=False)
    lines.to_parquet(C.RAW_DIR / "order_lines.parquet", index=False)
    pos.to_parquet(C.RAW_DIR / "purchase_orders.parquet", index=False)

    hist = orders[orders.status == "Shipped"]
    weeks = (hist.created_at.max() - hist.created_at.min()).days / 7
    print(f"SKUs {len(skus):,} | slots {len(loc):,} | orders {len(orders):,} "
          f"(~{len(hist) / weeks:,.0f}/wk) | lines {len(lines):,} | open POs {len(pos):,}")
    print(orders[orders.status != "Shipped"].status.value_counts().to_string())


if __name__ == "__main__":
    main()
