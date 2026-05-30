"""
game_context.py
================
Pulls game-level factors that aren't player stats but materially affect
HR and K projections.

History note: this module used to expose umpire k_factor, catcher framing,
team defense OAA, Vegas totals, pitcher workload, times-through-order
multiplier, and a separate park-handedness factor table. Most of those were
fetched but never actually plugged into any calculation in app.py — they
appeared in the UI but didn't influence numbers.

Current scope (May 2026): only Vegas totals are kept here, and even those
are currently disabled by default in app.py (use_vegas = False) because
park × weather + opp team K%/HR% + pitcher quality already capture what
Vegas totals represent for HR/K props.

The remaining functions are kept so the import block in app.py doesn't
break, but most are no-ops that return empty/neutral data.
"""

from __future__ import annotations

from datetime import datetime
import pandas as pd
import requests
import streamlit as st


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/csv, text/html, */*",
}

CURRENT_SEASON = datetime.now().year


# ---------------------------------------------------------------------------
# Stub / compatibility — these used to do real work, now no-ops
# ---------------------------------------------------------------------------

def get_umpire_for_game(game_pk: int) -> dict:
    """Disabled — no UmpScorecards lookup table available. Returns neutral."""
    return {"name": None, "k_factor": 1.0, "bb_factor": 1.0}


def get_catcher_framing(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Disabled — framing factor was fetched but never wired into k_total_projection."""
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Vegas implied totals — pulled from ESPN. Currently disabled in app.py
# (use_vegas = False) but kept here in case we want to re-enable.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def get_vegas_totals(game_date: str) -> pd.DataFrame:
    """
    Pull MLB game totals (O/U) and moneylines from ESPN's public API.

    Returns df with: home_team, away_team, total, away_ml, home_ml,
    home_implied, away_implied.

    home_implied = total/2 - spread/2 (negative spread = home favored)
    away_implied = total/2 + spread/2

    Verified math: total=9, spread=-1.5 (home favored) →
      home_implied = 4.5 - (-0.75) = 5.25 ✓
      away_implied = 4.5 + (-0.75) = 3.75 ✓
    """
    try:
        date_str = game_date.replace("-", "")
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
            f"?dates={date_str}"
        )
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        rows = []
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            comps = comp.get("competitors", [])
            home = next((c for c in comps if c.get("homeAway") == "home"), {})
            away = next((c for c in comps if c.get("homeAway") == "away"), {})
            row = {
                "gamePk": event.get("id"),
                "home_team": home.get("team", {}).get("abbreviation"),
                "away_team": away.get("team", {}).get("abbreviation"),
                "total": None,
                "home_implied": None,
                "away_implied": None,
            }
            odds_list = comp.get("odds", [])
            if odds_list:
                o = odds_list[0]
                total = o.get("overUnder")
                if total:
                    row["total"] = float(total)
                spread = o.get("spread")
                if spread is not None and total:
                    spread = float(spread)
                    row["home_implied"] = round((total / 2) - (spread / 2), 2)
                    row["away_implied"] = round((total / 2) + (spread / 2), 2)
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Times-Through-Order penalty — exposed for completeness, not currently called
# ---------------------------------------------------------------------------

def ttop_multiplier(lineup_pos: int, expected_ip: float = 5.5) -> float:
    """
    Hitter HR multiplier based on times-through-order.
    First time through: ~1.00. Second: ~1.05. Third: ~1.12.
    Currently not called from the main pipeline — kept for future use.
    """
    if lineup_pos is None or lineup_pos < 1 or lineup_pos > 9:
        return 1.0
    # Each lineup spot sees the pitcher at slightly different time-through-order
    # First 9 batters = 1st TTOP, batters 10-18 = 2nd, 19+ = 3rd
    # Approximate via expected_ip * batters-per-inning
    avg_batters_per_inning = 4.2
    total_batters_pitcher = expected_ip * avg_batters_per_inning
    # Which time-through does this lineup spot see?
    # Spot 1 hits at PA 1, 10, 19 etc; spot 9 at 9, 18, 27
    times_seen = int(total_batters_pitcher / 9)
    if times_seen <= 1:
        return 1.00
    if times_seen == 2:
        return 1.05
    return 1.12
