# Decisions

Each entry is a choice I made, written the way I'd answer if asked about it in an interview.

---

### Why generate data instead of using a public dataset?

Public retail datasets give you sales, but not the warehouse: no slots, no pickers, no ship-by dates, no purchase orders in flight. An agent that is supposed to run a floor needs all of those to agree with each other. So I wrote a generator that builds the whole business from one seed: a 15,000-SKU catalog with category-appropriate variants, 17,280 slots with real coordinates, 104 weeks of orders, the live state of the current week, and open POs.

I made it realistic where it matters to the decisions. Demand per SKU is lognormal, so about 4,400 SKUs carry 80% of pick lines. Each category has its own season: back-to-school lifts writing instruments and notebooks almost threefold in August, holiday lifts cards and art supplies in November, planners peak in December and January. Wholesale orders arrive in case packs, which makes demand lumpy. Slots were assigned at random inside each zone, the way a growing warehouse actually ends up, so the slotting engine has genuine wins to find.

The generator deliberately drops its own ground truth (base demand, launch week, phase-out flag) before writing, so the engine has to learn those from history like it would with a real WMS export.

### Why two years of history when one would show seasonality?

Honesty in the backtest. With one year, the seasonal index for the test weeks is estimated from the test weeks themselves. With two, I fit seasonality on year one only, hold out the last eight weeks, and score the forecast on data it has never seen. That's the number in the README.

### Why category seasonality times a SKU level, instead of a model per SKU?

Most SKUs sell a handful of units a week. Fitting a seasonal model to each one fits noise. Seasonality is a category phenomenon (back-to-school lifts every pen at once), so I estimate it where the signal is strong, on category totals, and let each SKU carry only its own level, fit by exponential smoothing on the de-seasonalised series. Slow movers get a longer memory (α = 0.07 vs 0.25) so one quiet month doesn't zero them out.

Result on the holdout: 6.4% WAPE on category-week totals, against 23–24% for both baselines. At the SKU-week level it's 92%, which sounds bad until you compare it with the baselines (98% and 129%) and remember what a SKU-week is for a C item: zero, one or two units. I put both numbers on the Forecast page on purpose. The planning decisions (buying, staffing) sit at the level where the model is reliable; the SKU-level decisions (reorder points) are protected by safety stock.

### Why 1.25 × MAE for sigma instead of the standard deviation?

The first version used the standard deviation of forecast errors and flagged 22% of the catalog as "reorder now". The cause was wholesale case orders: one 36-unit order in a SKU that normally sells two a week inflates σ enough to double its safety stock. 1.25 × mean absolute error is the textbook robust estimate (Silver, Pyke & Peterson) and brought the reorder list down to a size a buyer can act on, without hiding real stockouts.

### How are reorder points set?

Safety stock = z × σ_daily × √lead time, with z from a service level by ABC class: 95% for A items, 90% for B, 85% for C. Reorder point = forecast daily demand × lead time + safety stock. Buyers review weekly, so the suggested quantity orders up to lead time + 7 days of demand plus safety stock, rounded up to the case pack. Inventory position counts open POs, so a SKU that's out of stock but has a PO landing inside its lead time isn't double-ordered.

The SKU drawer in the UI prints this arithmetic for every item. If a buyer disagrees with a number, they can see exactly which input to argue with.

### Why batch-and-zone picking, and how do you know it saves 78%?

Today's pickers take one order at a time, and a typical order touches three SKUs in different parts of a 246 m wide building. Batching 50 orders, merging identical SKUs (pick once, sort into totes at pack-out) and giving each zone its own picker means one walk serves many lines.

The 78% comes from the same distance model in both cases: S-shape routing over the real slot coordinates, once per order versus once per zone per wave. I add 8 seconds per line for sorting at pack-out so batching isn't credited with free work. Serpentine routing is not optimal, but it is what pickers can follow without a device telling them every turn, and it's the standard comparison in the warehousing literature.

### Why does slotting show swaps instead of a full re-slot?

Nobody moves 15,000 SKUs. A swap is two people, two slots, ten minutes. I rank swaps by (picks/week of hot item − picks/week of cold item) × (slot cost difference), so the top 20 can be done on a Saturday and capture a real share of the benefit.

Two calls worth defending. Velocity uses the next six weeks of forecast pick lines, not last month, so holiday items move forward before the rush instead of after it. And the hours saved are scaled down by the batch-picking walk ratio, because slot savings measured on single trips overstate what you get once a single walk serves many lines. I'd rather show 3.5 hours a week for the top 20 swaps and be believed than show 15 and be wrong.

### Why give the language model tools instead of the data?

Two reasons: grounding and cost. The model never sees raw tables. It calls thirteen tools that return small, already-computed JSON (an order's lines with slots, the reorder list grouped by supplier, a pick list in walking order) and its job is to choose the right tool and explain the result. Every number it quotes traces back to engine output, so the assistant, the dashboard and the printed pick list can't disagree. And a typical answer costs a few thousand tokens instead of a context window full of CSV, which is why the default model is Haiku.

### What happens when Claude isn't available?

The floor can't stop because an API call failed. `agent/fallback.py` routes the question by keywords to one tool and formats the answer as markdown. It can't hold a conversation, but "plan the next waves", "which POs today" and "status of SO-1546575" all work offline. The web UI has the same fallback in JavaScript, so the hosted dashboard degrades the same way when a viewer hasn't allowed Claude.

### What would I do next with real data?

1. Swap the generator for WMS/ERP exports (the README lists the columns) and recalibrate the time-per-line constants against real pick logs.
2. Close the loop on waves: mark waves released and picked in the WMS so the agent re-plans from actual floor state.
3. Measure forecast value, not just accuracy: simulate the reorder policy over the holdout and report stockout days and inventory dollars against the current buying rule.
4. Add markdown recommendations for the back-to-school overhang the agent already flags as overstock.
