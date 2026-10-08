"""Invariants the floor depends on. Run: pytest -q  (generates data first if missing)."""
import pandas as pd

from papertrail import config as C
from papertrail.engine import picking as PK


def test_history_shape(store):
    assert len(store.skus) == C.N_SKUS
    assert store.skus.sku.is_unique and store.skus.location.is_unique
    hist = store.orders[store.orders.created_at <= pd.Timestamp(C.HISTORY_END) + pd.Timedelta(days=1)]
    per_week = len(hist) / C.HISTORY_WEEKS
    assert 4_500 < per_week < 6_000, per_week


def test_reorder_policy(store):
    p = store.policy
    assert (p.reorder_point >= p.safety_stock).all()
    assert (p.suggested_qty % p.case_pack == 0).all()
    assert (p.loc[p.status == "Stockout", "available"] == 0).all()
    reorder = p[p.status == "Reorder now"]
    assert (reorder.position <= reorder.reorder_point).all()


def test_forecast_beats_baselines_where_it_matters(store):
    bt = store.backtest.set_index("segment")
    cat = bt.loc["Category x week totals"]
    assert cat.model_wape < cat.seasonal_naive_wape and cat.model_wape < cat.moving_avg_8wk_wape
    assert cat.model_wape < 0.15
    assert bt.loc["A items"].model_wape < bt.loc["A items"].moving_avg_8wk_wape


def test_waves_cover_new_orders_and_walk_less(store):
    plan = store.wave_plan
    new = store.open_orders(("New",))
    waved = {o for w in plan["waves"] for o in w["order_ids"]}
    assert waved <= set(new.order_id)
    assert all(w["orders"] <= 50 for w in plan["waves"])
    assert plan["summary"]["distance_km"] < plan["summary"]["single_order_distance_km"]
    # every allocated line appears in exactly one pick list stop
    stops_units = sum(s["qty"] for w in plan["waves"] for p in w["pick_lists"] for s in p["stops"])
    lines = store.lines[store.lines.order_id.isin(waved)]
    assert stops_units == lines.qty_allocated.sum()


def test_s_shape_visits_every_stop():
    stops = pd.DataFrame({"x": [0, 0, 3, 9, 9], "y": [5, 20, 10, 3, 30],
                          "aisle_key": ["A-1", "A-1", "A-2", "A-4", "A-4"]})
    route, metres = PK.s_shape(stops, pack_x=4.5)
    assert len(route) == 5 and metres > 0
    # first aisle goes up the aisle, second comes back down
    assert list(route.y[:2]) == [5, 20]


def test_slotting_swaps_are_beneficial_and_disjoint(store):
    sl = store.slotting
    assert (sl.metres_saved_per_week > 0).all()
    assert sl.to_location.is_unique and sl.sku.is_unique
    assert (sl.zone == sl.from_location.str[0]).all() and (sl.zone == sl.to_location.str[0]).all()


def test_exceptions_have_actions(store):
    ex = store.exceptions
    assert not ex.empty
    assert ex.action.str.len().gt(10).all()
    assert set(ex.type) <= {"Late", "At risk", "Short", "Stuck", "On hold"}
