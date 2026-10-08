"""Shared constants for PaperTrail.

Everything that defines the simulated business lives here so the generator,
the engine and the agent agree on one picture of the warehouse.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
UI_DATA_DIR = ROOT / "ui" / "data"

SEED = 42
N_SKUS = 15_000
HISTORY_WEEKS = 104                     # two years, so forecasts can be backtested honestly
TARGET_ORDERS_PER_WEEK = 5_000

# "Now" in the simulation. History runs up to the Sunday before; the current
# week (Mon -> now) is the live order pipeline the agent manages.
NOW = datetime(2026, 10, 7, 16, 57)
HISTORY_END = datetime(2026, 10, 4)     # last day of history (Sunday)
CARRIER_CUTOFF_HOUR = 17.5              # 5:30 pm carrier pickup

# Category -> (zone, share of catalog, price range, seasonality profile, case pack)
CATEGORIES = {
    "Pens":                    ("A", 0.12, (1.5, 28),  "school",   12),
    "Pencils":                 ("A", 0.06, (0.8, 18),  "school",   24),
    "Markers & Highlighters":  ("A", 0.09, (1.5, 35),  "school",   12),
    "Notebooks & Journals":    ("B", 0.11, (2.5, 40),  "school",   10),
    "Paper & Pads":            ("B", 0.08, (3.0, 55),  "base",     10),
    "Envelopes & Mailers":     ("B", 0.05, (4.0, 45),  "holiday_light", 20),
    "Binders & Folders":       ("C", 0.08, (1.5, 30),  "school",   12),
    "Filing & Storage":        ("C", 0.06, (6.0, 80),  "tax",      6),
    "Labels & Tape":           ("C", 0.06, (2.0, 40),  "base",     12),
    "Staplers & Fasteners":    ("C", 0.04, (1.5, 45),  "base",     12),
    "Desk Accessories":        ("D", 0.06, (5.0, 90),  "holiday_light", 6),
    "Planners & Calendars":    ("D", 0.06, (8.0, 45),  "planner",  8),
    "Art & Craft Supplies":    ("E", 0.08, (2.0, 70),  "holiday",  6),
    "Greeting Cards & Gift Wrap": ("F", 0.05, (2.5, 25), "holiday", 24),
}

ZONE_NAMES = {
    "A": "Writing instruments",
    "B": "Paper goods",
    "C": "Office & filing",
    "D": "Desk & planners",
    "E": "Art & craft",
    "F": "Cards & gift wrap",
}

# Warehouse geometry (metres). Zones sit side by side along x; the pack
# station is at the front, centred.
BAYS_PER_SIDE = 30
LEVELS = 4
AISLE_SPACING_M = 3.0
BAY_DEPTH_M = 1.2
ZONE_GAP_M = 6.0
SLOT_HEADROOM = 1.12                    # spare slots per zone

# Labour model used to turn metres into hours
WALK_SPEED_MPS = 1.2
SECONDS_PER_LINE = 14
SECONDS_PER_EXTRA_UNIT = 2
SECONDS_SORT_PER_LINE = 8              # pick-and-sort: putting batch-picked items into order totes
LEVEL_PENALTY_M = {1: 2.0, 2: 0.0, 3: 4.0, 4: 10.0}   # bending / ladder cost in metre-equivalents

# Service levels by ABC class -> z-score
SERVICE_Z = {"A": 1.65, "B": 1.28, "C": 1.04}

PICKERS = [
    "Maya", "Diego", "Priya", "Tom", "Aisha", "Ben", "Lena", "Omar",
    "Grace", "Kenji", "Rosa", "Sam", "Ivy", "Marcus", "Noor", "Eli",
]
