"""The agent's tools and its offline fallback, plus the API surface."""
import json

import pytest
from fastapi.testclient import TestClient

from papertrail.agent.agent import Agent
from papertrail.agent.tools import call, tool_specs


@pytest.fixture(scope="module")
def agent(store):
    return Agent(store=store, api_key="")      # force local mode


def test_tool_specs_match_methods(agent):
    specs = tool_specs(agent.tools)
    assert len(specs) >= 12
    for sp in specs:
        assert callable(getattr(agent.tools, sp["name"]))
        assert sp["description"] and sp["input_schema"]["type"] == "object"


def test_tools_return_json(agent):
    oid = agent.store.exceptions.iloc[0].order_id
    for name, args in [("daily_briefing", {}), ("lookup_order", {"order_id": oid}),
                       ("list_exceptions", {"type": "Short"}), ("sku_info", {"query": "gel pen"}),
                       ("reorder_recommendations", {"supplier": "china", "limit": 5}),
                       ("forecast_demand", {"query": "Planners"}), ("plan_waves", {}),
                       ("labor_plan", {}), ("picker_performance", {}), ("slotting_moves", {"zone": "A"})]:
        out = json.loads(call(agent.tools, name, args))
        assert "error" not in out, (name, out)


def test_tool_errors_are_returned_not_raised(agent):
    out = json.loads(call(agent.tools, "lookup_order", {"order_id": "SO-0000000"}))
    assert "error" in out


@pytest.mark.parametrize("q,tool", [
    ("what should I do first?", "daily_briefing"),
    ("plan the next waves", "plan_waves"),
    ("which SKUs need a PO", "reorder_recommendations"),
    ("how many pickers tomorrow", "labor_plan"),
    ("who needs coaching", "picker_performance"),
    ("any stuck orders?", "list_exceptions"),
])
def test_fallback_routes(agent, q, tool):
    r = agent.ask(q)
    assert r.mode == "local" and r.tools_used == [tool] and r.text


def test_api():
    from papertrail.api.server import app
    c = TestClient(app)
    assert c.get("/api/health").json()["ok"]
    r = c.post("/api/agent", json={"question": "plan the next waves"})
    assert r.status_code == 200 and "waves" in r.json()["text"]
    assert c.get("/").text.startswith("<!doctype html>")
    assert c.get("/data/meta.json").status_code == 200


def test_claude_tool_loop_with_fake_client(store):
    """Drives Agent._ask_claude with a stand-in client: one tool call, then an answer."""
    from types import SimpleNamespace as NS

    class FakeMessages:
        def __init__(self):
            self.calls = 0
            self.seen_tool_result = None

        def create(self, **kw):
            self.calls += 1
            if self.calls == 1:
                assert any(t["name"] == "list_exceptions" for t in kw["tools"])
                return NS(stop_reason="tool_use", content=[
                    NS(type="text", text="Checking."),
                    NS(type="tool_use", id="t1", name="list_exceptions", input={"type": "Late"})])
            self.seen_tool_result = kw["messages"][-1]["content"][0]
            return NS(stop_reason="end_turn", content=[NS(type="text", text="Two late orders; expedite them.")])

    a = Agent(store=store, api_key="")
    fake = FakeMessages()
    a.client = NS(messages=fake)
    r = a.ask("anything late?")
    assert r.mode == "claude" and r.tools_used == ["list_exceptions"]
    assert "late" in r.text.lower()
    payload = json.loads(fake.seen_tool_result["content"])
    assert payload["total"] == int((store.exceptions.type == "Late").sum())
