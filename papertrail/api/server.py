"""FastAPI server: the dashboard, its data, and the Python agent.

    uvicorn papertrail.api.server:app --reload
    open http://localhost:8000

With ANTHROPIC_API_KEY set, the chat panel talks to Claude through
papertrail.agent; without it, the same endpoint answers locally.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from papertrail import config as C
from papertrail.agent.agent import Agent
from papertrail.agent.tools import Tools
from papertrail.export import wrap_page

app = FastAPI(title="PaperTrail", version="1.0")


@lru_cache(maxsize=1)
def agent() -> Agent:
    return Agent()


def tools() -> Tools:
    return agent().tools


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)


@app.get("/api/health")
def health():
    return {"ok": True, "llm": agent().client is not None, "as_of": C.NOW.isoformat()}


@app.post("/api/agent")
def ask(body: Ask):
    hist = [h for h in body.history if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str)][-8:]
    if hist and hist[0]["role"] == "assistant":
        hist = hist[1:]
    r = agent().ask(body.question, hist)
    return {"text": r.text, "tools_used": r.tools_used, "mode": r.mode}


@app.get("/api/briefing")
def briefing():
    return tools().daily_briefing()


@app.get("/api/orders/{order_id}")
def order(order_id: str):
    r = tools().lookup_order(order_id)
    if "error" in r:
        raise HTTPException(404, r["error"])
    return r


@app.get("/api/exceptions")
def exceptions(type: str | None = None, limit: int = 50):
    return tools().list_exceptions(type=type, limit=limit)


@app.get("/api/skus")
def skus(q: str, limit: int = 10):
    return tools().sku_info(q, limit=limit)


@app.get("/api/reorder")
def reorder(category: str | None = None, supplier: str | None = None, limit: int = 50):
    return tools().reorder_recommendations(category=category, supplier=supplier, limit=limit)


@app.get("/api/waves")
def waves(max_orders_per_wave: int = 50, include_released: bool = False):
    return tools().plan_waves(max_orders_per_wave=max_orders_per_wave, include_released=include_released)


@app.get("/api/waves/{wave_id}/{zone}")
def pick_list(wave_id: str, zone: str):
    r = tools().pick_list(wave_id, zone)
    if "error" in r:
        raise HTTPException(404, r["error"])
    return r


@app.get("/api/slotting")
def slotting(zone: str | None = None, limit: int = 25):
    return tools().slotting_moves(zone=zone, limit=limit)


@app.get("/api/labor")
def labor():
    return tools().labor_plan()


@app.get("/api/pickers")
def pickers():
    return tools().picker_performance()


@app.get("/api/forecast/{query}")
def forecast(query: str, weeks: int = 8):
    return tools().forecast_demand(query, weeks=weeks)


UI = C.ROOT / "ui"
if (UI / "data").exists():
    app.mount("/data", StaticFiles(directory=UI / "data"), name="data")


@app.get("/")
def index():
    return HTMLResponse(wrap_page((UI / "index.html").read_text()), headers={"Cache-Control": "no-cache"})
