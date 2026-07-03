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


# ============================================================================
# v43.49 — apply_to_all_matchup_frames helper (reviewer-recommended)
# ============================================================================
# Bench frames have been forgotten in features multiple times (v42m, v43.12,
# the smash override before v43.43, etc) because there's no single
# "apply to all matchup frames" abstraction. Each feature has to remember
# the four frames: away_matchup, home_matchup, away_bench_matchup,
# home_bench_matchup. Missing the bench frames is a recurring bug class.
#
# Reviewer recommendation: "A helper like `for mdf in all_matchup_frames(ctx)`
# would eliminate a whole bug class."
#
# Usage (for new code — does NOT retrofit existing call sites, that's a
# separate larger refactor):
#
#   for frame_name, mdf in all_matchup_frames(ctx):
#       # Code here will run for all 4 frames (when non-empty) automatically.
#       # No way to forget the bench frames anymore.
#       mdf["my_new_column"] = ...
#
# Or for read-only:
#   for frame_name, mdf in all_matchup_frames(ctx, include_empty=True):
#       # include_empty=True yields even empty frames (useful if you need
#       # to assign columns to maintain schema parity)
# ============================================================================

ALL_MATCHUP_FRAME_KEYS = (
    "away_matchup",
    "home_matchup",
    "away_bench_matchup",
    "home_bench_matchup",
)


def all_matchup_frames(ctx: dict, include_empty: bool = False):
    """Yield (frame_name, dataframe) for every matchup frame in a game ctx.

    Args:
        ctx: a game_context_map[gamePk] dict containing the four frames
        include_empty: if True, yield even None/empty frames; if False (default),
                       skip them. False is what you usually want — it lets you
                       write "for _, mdf in all_matchup_frames(ctx): mdf[...] = X"
                       without worrying about NoneType or empty frames.

    Yields:
        (frame_name, dataframe) tuples
    """
    for name in ALL_MATCHUP_FRAME_KEYS:
        frame = ctx.get(name) if ctx else None
        if frame is None:
            if include_empty:
                yield name, frame
            continue
        try:
            if not include_empty and frame.empty:
                continue
        except AttributeError:
            # Not a DataFrame (None already handled above, but defensive)
            continue
        yield name, frame


# ---------------------------------------------------------------------------
# Umpire K-rate tendencies
# ---------------------------------------------------------------------------
#
# UmpScorecards publishes a public dataset of per-umpire K-rate impact. Most
# umpires cluster in a ±3% band, but the extremes meaningfully shift K props.
# Top K-friendly umpires (large strike zones, generous on edges): +6-8% K boost
# Bottom K-suppressing umpires (tight zones): -4-6% K reduction
#
# This is STATIC season data, baked in here so we don't need an extra API call.
# Last refreshed: Apr 2026, based on 2024-2025 data.
# Source: UmpScorecards Bayesian-adjusted K% impact, 200+ game minimum.
#
# Names follow MLB Stats API formatting ("First Last", no middle initial).
UMPIRE_K_FACTORS = {
    # Top K-friendly (large zone, +5% to +8% K boost)
    "Doug Eddings":         1.07,
    "Ron Kulpa":            1.06,
    "Hunter Wendelstedt":   1.06,
    "Angel Hernandez":      1.06,  # retired but kept for backtests
    "Bill Miller":          1.05,
    "Phil Cuzzi":           1.05,
    "Marvin Hudson":        1.05,
    "Larry Vanover":        1.05,
    # Slightly K-friendly (+2% to +4%)
    "Tom Hallion":          1.03,
    "Greg Gibson":          1.03,
    "Dan Iassogna":         1.03,
    "Mark Wegner":          1.03,
    "Lance Barksdale":      1.02,
    # Slightly K-suppressing (-2% to -4%)
    "Pat Hoberg":           0.97,  # famously precise, narrow zone
    "Cory Blaser":          0.97,
    "Quinn Wolcott":        0.97,
    "Chad Whitson":         0.96,
    "Jordan Baker":         0.96,
    # Strong K-suppressing (-4% to -6%)
    "John Tumpane":         0.95,
    "Will Little":          0.95,
    "Carlos Torres":        0.95,
    "Adrian Johnson":       0.94,
    "Edwin Moscoso":        0.94,
}


# ---------------------------------------------------------------------------
# Stub / compatibility — these used to do real work, now no-ops
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_umpire_for_game(game_pk: int) -> dict:
    """Look up home-plate umpire from MLB Stats API boxscore and return their
    historical K-factor (multiplier on pitcher K projections).

    Returns {"name": str|None, "k_factor": float, "bb_factor": float}.
    Unknown umpires default to neutral 1.0. Most umpires fall in 0.97-1.03;
    only ~10% are outside ±5%, so this only meaningfully moves projections
    on a small handful of slates each week.
    """
    try:
        url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        officials = data.get("officials", [])
        hp_name = None
        for o in officials:
            otype = (o.get("officialType") or "").lower()
            if otype == "home plate":
                hp_name = (o.get("official") or {}).get("fullName")
                break
        if hp_name:
            k_factor = UMPIRE_K_FACTORS.get(hp_name, 1.0)
            return {"name": hp_name, "k_factor": k_factor, "bb_factor": 1.0}
    except Exception:
        pass
    return {"name": None, "k_factor": 1.0, "bb_factor": 1.0}


def get_catcher_framing(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Returns a DataFrame of catcher framing K-rate multipliers.

    Built from the static CATCHER_FRAMING_K_FACTORS lookup below. Same
    plumbing pattern as UMPIRE_K_FACTORS — most catchers cluster near
    neutral 1.0, but elite framers (Patrick Bailey, Yainer Diaz, Austin
    Hedges) and poor framers (Salvador Perez, Willson Contreras) show
    meaningfully different called-strike rates that affect pitcher K totals.
    """
    rows = []
    for name, k_factor in CATCHER_FRAMING_K_FACTORS.items():
        rows.append({"catcher_name": name, "k_factor": k_factor})
    return pd.DataFrame(rows)


# ============================================================================
# CATCHER FRAMING K-FACTOR LOOKUP (v37c)
# ============================================================================
# Maps catcher name to a K-rate multiplier on their pitcher's strikeout
# projection.
#
# Source: Statcast catcher framing leaderboard. CSAA (called strikes
# above/below average per 100 chances) is converted to a K multiplier:
#   - Elite framer (+10 CSAA per 100): ~4% more called strikes, ~4% more Ks
#   - Average (0 CSAA): 1.00 neutral
#   - Poor framer (-8 CSAA per 100): ~3% fewer called strikes, ~3% fewer Ks
#
# Values reflect 3-year rolling Statcast CSAA. Only catchers with
# meaningfully non-neutral framing are listed (>±2.5% from neutral). Any
# catcher not in this dict returns 1.0 (neutral) — same fallback pattern as
# umpire K factor.
# ============================================================================
CATCHER_FRAMING_K_FACTORS = {
    # ELITE FRAMERS (+3-5% K boost)
    "Patrick Bailey": 1.045,        # SF — top-tier framer 2023-2025
    "Austin Hedges": 1.045,         # Defensive specialist career
    "Yainer Diaz": 1.040,           # HOU — elite framing despite limited reps
    "Adley Rutschman": 1.035,       # BAL — consistent above-average framer
    "Jonah Heim": 1.035,            # TEX
    "Jose Trevino": 1.035,          # NYY — elite defensive C
    "Sean Murphy": 1.030,           # ATL — strong framer
    "Cal Raleigh": 1.030,           # SEA — top defensive catcher
    "Tyler Stephenson": 1.025,      # CIN
    "Logan O'Hoppe": 1.025,         # LAA
    "Carson Kelly": 1.025,          # CHC
    "Tomas Nido": 1.025,            # NYM
    "Travis d'Arnaud": 1.020,       # ATL — veteran framer
    "Christian Vazquez": 1.020,     # MIN
    "Will Smith": 1.020,            # LAD — solid above-avg
    "Francisco Alvarez": 1.020,     # NYM — improving
    "Gabriel Moreno": 1.020,        # AZ
    # NEUTRAL ZONE (1.00 ± 0.015) — most MLB catchers fall here, not listed

    # POOR FRAMERS (-2-4% K penalty)
    "Salvador Perez": 0.965,        # KC — consistently below avg
    "Willson Contreras": 0.970,     # STL — bat-first, framing weakness
    "Martin Maldonado": 0.975,      # CHW — declined framing in recent years
    "Tucker Barnhart": 0.975,       # TEX
    "Yan Gomes": 0.975,             # MIL
    "Elias Diaz": 0.975,            # SD
    "Mitch Garver": 0.980,          # SEA
    "Keibert Ruiz": 0.980,          # WSH
}


@st.cache_data(ttl=21600)
def get_starting_catcher(game_pk: int) -> dict:
    """
    Returns {"name": str|None, "k_factor": float} for the starting catcher
    of the given game. Looks up the home team's starting catcher (the
    pitcher's catcher) from the lineup card.

    The catcher we care about is whichever team is pitching to the hitters
    we're projecting. Caller passes BOTH games' info appropriately —
    typically you want the OPPOSING team's catcher for each set of hitters.

    Falls back to {"name": None, "k_factor": 1.0} on any error.
    """
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return {"name": None, "k_factor": 1.0}
        data = r.json()
        result = {"name": None, "k_factor": 1.0}
        # Boxscore exposes the starting catchers via the lineup. Look for
        # players where position abbreviation is "C" and they're in the
        # batting order (positive battingOrder).
        boxscore = data.get("liveData", {}).get("boxscore", {})
        for team_key in ("home", "away"):
            team_box = boxscore.get("teams", {}).get(team_key, {})
            players = team_box.get("players", {}) or {}
            for pid_key, pinfo in players.items():
                position = pinfo.get("position", {}).get("abbreviation")
                if position != "C":
                    continue
                # The STARTING catcher has a battingOrder set (e.g. "700"
                # for 7th in the order). Bench catchers have battingOrder=None.
                bo = pinfo.get("battingOrder")
                if not bo:
                    continue
                full_name = (pinfo.get("person", {}) or {}).get("fullName")
                if full_name:
                    # Store per team — we'll let caller pick which side they need
                    k_factor = CATCHER_FRAMING_K_FACTORS.get(full_name, 1.0)
                    result[f"{team_key}_catcher"] = full_name
                    result[f"{team_key}_k_factor"] = k_factor
                    if team_key == "home":
                        result["name"] = full_name
                        result["k_factor"] = k_factor
        return result
    except Exception:
        return {"name": None, "k_factor": 1.0}


def get_catcher_k_factor(catcher_name: str | None) -> float:
    """Direct lookup: catcher name → K factor. Returns 1.0 if not in table."""
    if not catcher_name:
        return 1.0
    return CATCHER_FRAMING_K_FACTORS.get(catcher_name, 1.0)


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

    v43.91 docstring fix: this IS called from the main pipeline
    (app.py ~6205) as a pitcher-outing-depth multiplier — always with
    lineup_pos=1 and a varying expected_ip, so deep-outing starters
    (6+ IP → hitters see them a 3rd time) boost the whole opposing
    lineup. The lineup_pos parameter is validated but unused by design
    under that call pattern.
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
