# PaperTrail

An operations agent for a high-volume stationery warehouse: 15,000 SKUs, about 5,000 orders a week, six pick zones, sixteen pickers. It watches the live order pipeline, tells the operations lead what to deal with first, builds pick waves with walking-order pick lists, keeps reorder points honest, and finds the slot moves that cut the most walking. A Claude-powered assistant answers questions by calling the same tools the dashboard uses, and a local rule-based fallback keeps the floor running when the API isn't reachable.

The business is simulated. Two years of history (546,700 orders, 1.58 million lines) are generated from one seed with the patterns a real stationery distributor lives with: a long-tailed catalog, back-to-school and holiday peaks, planner season, wholesale case orders, promos, launches and phase-outs.

![Today view](docs/screenshot.png)

## What it does

| Job | What the agent produces |
|---|---|
| **Order status and exceptions** | Every open order classified as Late, At risk (must ship before the 5:30 pm truck), Short, Stuck, or On hold, each with a specific next action: upgrade shipping, hold for an inbound PO, offer an in-stock substitute in another colour, or call the picker who has it. |
| **Pick waves** | Unreleased orders prioritised by ship-by date and service level, cut into waves of 50, merged by SKU and split by zone, then sequenced in serpentine order. Today's five waves walk 16 km instead of 72 km picked one order at a time (78% less). |
| **Reorder and stockout alerts** | Per-SKU safety stock and reorder points from forecast demand, lead time and an ABC service level; purchase quantities rounded to case packs and grouped by supplier; revenue at risk until stock lands. |
| **Slotting** | A ranked swap list that moves next-six-weeks fast movers into the cheapest slots in their zone. Today only 17–24% of pick visits land in the closest fifth of slots; the ideal is 72–76%. |
| **Forecast** | Category seasonality times a per-SKU smoothed level, backtested on a held-out 8 weeks. |
| **Staffing and people** | Pickers needed for the next five shipping days, and a scorecard that flags accuracy or pace coaching against the team median. |

### Forecast accuracy (8-week holdout, WAPE, lower is better)

| Level | This model | Same week last year | 8-week average |
|---|---|---|---|
| Category × week totals | **6.4%** | 23.5% | 23.1% |
| A items (SKU × week) | **78.3%** | 111.3% | 84.7% |
| All SKUs (SKU × week) | **91.6%** | 129.2% | 98.4% |

Category totals are where buying and staffing decisions are made, and the model is reliable there. SKU-week numbers for slow movers are mostly noise; the reorder logic absorbs that with safety stock rather than pretending a single week is predictable. See `DECISIONS.md`.

## Using the app (plain-English guide)

The app has an in-product guide: press **? Guide** at the top right (or the `?` key). It has the daily routine, a picker's quick guide, example questions for the assistant, and a glossary. Every page also has a "How to use this page" box. The short version:

1. **Today.** Read "Do next" from the top and tick items off.
2. **Orders.** Fix the problem orders, most urgent first. Each row says what to do.
3. **Pick waves.** Open a wave, choose a zone, and hand the list to a picker. Pickers switch on **Floor view** and work top to bottom.
4. **Inventory.** Keep "Reorder now" selected, copy the purchase plan as CSV, and send it to buying.
5. **Slotting and Forecast.** Weekly: move the top few items closer to the pack station, and check the forecast before busy seasons.

The assistant (right side, or the **Ask PaperTrail** tab on a phone) takes plain questions such as "Which orders will miss the truck?" or an order or item number.

PaperTrail only gives advice. It never changes anything in your real systems, and your ticks are saved in the browser only.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m papertrail.data.generate     # ~10 s: catalog, slots, 104 weeks of orders, live pipeline
python -m papertrail.export            # runs the engine, writes ui/data/*.json
uvicorn papertrail.api.server:app --reload
# open http://localhost:8000
```

Set `ANTHROPIC_API_KEY` (see `.env.example`) and the chat panel answers through Claude with tool use. Without it, the same panel answers locally from the engine output.

Ask from the terminal:

```bash
python -m papertrail.agent.agent "Which orders will miss the truck?"
```

Run the checks: `pytest -q`

Static copy for GitHub Pages: `python -m papertrail.export --site` writes `docs/`. Hosted that way the chat uses local answers, since a static page has nowhere to keep an API key.

## Put it on GitHub

You need a free GitHub account and Git installed (check with `git --version`). These steps go once, in order.

1. **Unzip** `papertrail.zip` and open a terminal in the new `papertrail` folder.
2. **Create an empty repository on GitHub.** Go to <https://github.com/new>, name it `papertrail`, leave "Add a README", ".gitignore" and "license" all **unticked**, and press **Create repository**. Keep that page open.
3. **Turn the folder into a repo and save the first commit:**
   ```bash
   git init
   git add .
   git commit -m "PaperTrail: warehouse operations agent"
   ```
   If Git says it doesn't know who you are, run `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`, then repeat the commit.
4. **Connect it to GitHub and upload** (replace `YOUR-USERNAME`):
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/papertrail.git
   git push -u origin main
   ```
   When asked to sign in, use your GitHub username and a **personal access token** as the password (GitHub > Settings > Developer settings > Personal access tokens, with the `repo` permission). Or run `gh auth login` first if you use the GitHub CLI.
5. **Refresh the GitHub page.** Your code is there, and the **Actions** tab shows the tests running automatically.

**Later changes:** `git add .`, `git commit -m "what changed"`, `git push`.

**Optional, a live web page.** Run `python -m papertrail.export --site`, commit the new `docs/` folder and push. On GitHub open **Settings > Pages**, set Source to "Deploy from a branch", choose `main` and `/docs`, and save. Your app appears at `https://YOUR-USERNAME.github.io/papertrail/` after a minute. (The page works without a server, so it uses the built-in local assistant. To use Claude, run the server version above.)

**Keep secrets out.** Your `ANTHROPIC_API_KEY` lives in a `.env` file, which is already ignored by Git. Never paste a key into a file you commit.

## How it fits together

```
data/generate.py ──► data/raw/*.parquet ──► engine/store.py ─┬─► agent/tools.py ──► agent/agent.py (Claude tool loop)
                                                            │                     └► agent/fallback.py (offline)
                                                            ├─► api/server.py (FastAPI: /api/*, /api/agent)
                                                            └─► export.py ──► ui/data/*.json ──► ui/index.html
```

* `engine/forecast.py` seasonal index per category, exponential smoothing per SKU, robust σ, backtest
* `engine/inventory.py` ABC classes, safety stock, reorder point, order-up-to quantity, stock status
* `engine/orders.py` exception rules and recommended actions, picker scorecard
* `engine/picking.py` wave building, SKU merge, zone split, S-shape routing, distance model
* `engine/slotting.py` slot cost model and the swap list
* `agent/tools.py` thirteen small tools that return computed JSON; the model never sees raw tables

The model reads numbers and explains them. Every figure it quotes comes from a tool call, so the dashboard, the pick list and the assistant always agree.

## Using real data

The generator writes five tables; replace them with exports from your WMS or ERP and everything downstream runs unchanged:

| File | Columns the engine uses |
|---|---|
| `skus.parquet` | sku, name, category, product_type, variant, color, zone, location, supplier, lead_time_days, case_pack, unit_price, unit_cost, on_hand, allocated |
| `locations.parquet` | location, zone, aisle, side, bay, level, x, y, dist_to_pack_m, slot_cost_m |
| `orders.parquet` | order_id, created_at, channel, ship_method, ship_state, ship_by, status, status_since, shipped_at, picker, wave_id, hold_reason, pick_minutes, mispick |
| `order_lines.parquet` | order_id, line_no, sku, qty, qty_allocated |
| `purchase_orders.parquet` | po_id, sku, supplier, qty, eta |

## Project layout

```
papertrail/
  config.py           business constants: zones, geometry, service levels, the simulation clock
  data/generate.py    synthetic history
  engine/             forecasting, inventory policy, orders, picking, slotting, store
  agent/              tools, Claude loop, offline fallback
  api/server.py       FastAPI app
  export.py           engine → JSON for the UI
ui/index.html         single-file dashboard (no build step)
tests/                engine invariants, tool contracts, fallback routing, API, Claude loop with a fake client
DECISIONS.md          the design choices, written as interview answers
```
