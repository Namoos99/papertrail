"""PaperTrail agent: Claude reasons over the warehouse through tools; a local
rule-based router answers when no API key is set or the API is unreachable.

Usage:
    from papertrail.agent.agent import Agent
    agent = Agent()                       # uses ANTHROPIC_API_KEY if present
    agent.ask("What should I deal with first today?")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from papertrail.agent import fallback
from papertrail.agent.tools import Tools, call, tool_specs
from papertrail.engine.store import Store

MODEL = os.getenv("PAPERTRAIL_MODEL", "claude-haiku-5-5")
MAX_TOOL_ROUNDS = 6

SYSTEM = """You are PaperTrail, the operations agent for a stationery distributor that ships about
5,000 orders a week from a 15,000-SKU warehouse (zones A-F, pack station at the front).
You help the operations lead run the floor hands-off and help pickers do better work.

How to work:
- Always get numbers from tools. Never invent order ids, SKUs, quantities or locations.
- Lead with the decision or action, then the reason in one line. Keep answers short and scannable.
- When something needs a human (approving POs, releasing a hold, coaching a picker), say who and what.
- Pick lists are for people on the floor: give location, SKU, quantity, in walking order.
- Be candid about uncertainty: SKU-level weekly forecasts are noisy for slow movers; category totals are reliable.
- Current time: {now}. Carrier pickup is 5:30 pm."""


@dataclass
class Reply:
    text: str
    tools_used: list[str] = field(default_factory=list)
    mode: str = "claude"          # "claude" or "local"


class Agent:
    def __init__(self, store: Store | None = None, api_key: str | None = None):
        self.store = store or Store()
        self.tools = Tools(self.store)
        self.specs = tool_specs(self.tools)
        key = os.getenv("ANTHROPIC_API_KEY") if api_key is None else api_key   # "" forces local mode
        self.client = None
        if key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=key)
            except ImportError:
                self.client = None

    def ask(self, question: str, history: list[dict] | None = None) -> Reply:
        if self.client is None:
            return fallback.answer(self.tools, question)
        try:
            return self._ask_claude(question, history or [])
        except Exception as e:      # network down, rate limit, bad key: degrade, don't die
            r = fallback.answer(self.tools, question)
            r.text = f"_(Claude unavailable: {type(e).__name__}; answered locally.)_\n\n" + r.text
            return r

    def _ask_claude(self, question: str, history: list[dict]) -> Reply:
        messages = [*history, {"role": "user", "content": question}]
        used: list[str] = []
        system = SYSTEM.format(now=self.store.now.strftime("%A %b %d, %I:%M %p"))
        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.client.messages.create(model=MODEL, max_tokens=1200, system=system,
                                               tools=self.specs, messages=messages)
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                return Reply(text=text.strip(), tools_used=used)
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    used.append(b.name)
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": call(self.tools, b.name, b.input)})
            messages.append({"role": "user", "content": results})
        return Reply(text="I hit my tool-call limit on that one — try a narrower question.", tools_used=used)


if __name__ == "__main__":
    import sys
    a = Agent()
    q = " ".join(sys.argv[1:]) or "Give me today's briefing."
    r = a.ask(q)
    print(f"[{r.mode}] tools: {', '.join(r.tools_used) or '-'}\n")
    print(r.text)
