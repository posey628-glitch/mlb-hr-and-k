"""
game_context.py
================
Pulls game-level factors that aren't player stats but materially affect
HR and K projections:

  - Umpire strike-zone tendencies (K-friendly vs hitter-friendly)
  - Starting catcher framing (CSAA)
  - Team defensive OAA (affects BABIP / HR-distance outs)
  - Vegas implied totals (market consensus on run environment)
  - Pitcher rest days + recent pitch counts (workload)
  - Times Through Order Penalty multiplier

Most of these load fast (single API call). Vegas + umpire involve some
HTML parsing that can break if upstream changes — wrapped in try/except
so a failure doesn't break the rest of the app.
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timedelta

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
# Umpire — from MLB Stats API game preview + reference tables
# ---------------------------------------------------------------------------

# Umpire historical K% impact - rough averages for known umps.
# Real data would come from UmpScorecards but that requires scraping.
# This is a fallback table; gets enriched if we can scrape live data.
UMP_DEFAULT = {"k_factor": 1.00, "bb_factor": 1.00}


@st.cache_data(ttl=3600)
def get_umpire_for_game(game_pk: int) -> dict:
    """
    Pull HP umpire name from MLB Stats API boxscore.
    Returns {name, k_factor, bb_factor} where factors are multipliers on
    league-average rates (1.0 = neutral).
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        officials = r.json().get("liveData", {}).get("boxscore", {}).get("officials", [])
        hp_ump = next(
            (o for o in officials if o.get("officialType") == "Home Plate"),
            None,
        )
        if hp_ump:
            name = hp_ump.get("official", {}).get("fullName", "TBD")
            # In production, look up name in scraped UmpScorecards data
            factors = UMP_DEFAULT.copy()
            return {"name": name, **factors}
    except Exception:
        pass
    return {"name": "TBD", **UMP_DEFAULT}


# ---------------------------------------------------------------------------
# Catcher framing - season CSAA leaderboard from Savant
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_catcher_framing(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Catcher framing runs / CSAA per catcher this season.
    Higher = steals more strikes = boosts K rate for whoever's pitching.
    Returns: player_id, player_name, runs_extra_strikes, called_strike_rate, etc.
    """
    url = (
        "https://baseballsavant.mlb.com/leaderboard/catcher-framing"
        f"?year={season}&team=&min=q&sort=4,1&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "last_name, first_name" in df.columns:
            df["player_name"] = df["last_name, first_name"].apply(
                lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
                if isinstance(s, str) and "," in s else s
            )
        # Compute K-rate multiplier - good framer = ~+3% K rate
        if "runs_extra_strikes" in df.columns:
            df["framing_k_factor"] = (1 + df["runs_extra_strikes"] / 25).clip(0.92, 1.08)
        return df
    except Exception as e:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Team OAA defense (affects BABIP and HR-distance fly balls)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_team_defense(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Team-level OAA. Bad outfield = more 'HR-distance outs' become hits/HRs."""
    url = (
        "https://baseballsavant.mlb.com/leaderboard/outs_above_average"
        f"?year={season}&type=Fielder&startDate=&endDate=&split=no&team=yes"
        "&range=year&min=q&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Vegas implied totals - best single environmental predictor
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800)
def get_vegas_totals(game_date: str) -> pd.DataFrame:
    """
    Pull MLB game totals (O/U) and moneylines from ESPN's public API,
    then compute implied team runs.

    Returns df with: home_team, away_team, total, away_ml, home_ml,
    away_implied, home_implied.
    """
    # ESPN scoreboard - free, no key, returns odds when available
    yyyymmdd = game_date.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={yyyymmdd}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        rows = []
        for event in data.get("events", []):
            competitions = event.get("competitions", [])
            if not competitions:
                continue
            comp = competitions[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})
            odds = comp.get("odds", [])
            row = {
                "home_team": home.get("team", {}).get("displayName"),
                "home_abbr": home.get("team", {}).get("abbreviation"),
                "away_team": away.get("team", {}).get("displayName"),
                "away_abbr": away.get("team", {}).get("abbreviation"),
                "total": None,
                "away_ml": None, "home_ml": None,
                "away_implied": None, "home_implied": None,
            }
            if odds:
                o = odds[0]
                total = o.get("overUnder")
                if total:
                    row["total"] = float(total)
                # Spread tells us implied team totals
                spread = o.get("spread")
                if spread is not None and total:
                    # spread is from home perspective (negative = home favored)
                    row["home_implied"] = round((total / 2) - (spread / 2), 2)
                    row["away_implied"] = round((total / 2) + (spread / 2), 2)
                row["away_ml"] = (o.get("awayTeamOdds") or {}).get("moneyLine")
                row["home_ml"] = (o.get("homeTeamOdds") or {}).get("moneyLine")
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Pitcher rest / workload
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_pitcher_workload(pitcher_id: int, season: int = CURRENT_SEASON) -> dict:
    """
    Returns days rest since last start + recent pitch counts.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
        f"?stats=gameLog&group=pitching&season={season}&sportId=1"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        starts = [
            s for s in splits
            if int(s.get("stat", {}).get("gamesStarted", 0)) > 0
        ]
        if not starts:
            return {}
        last_start = starts[-1]
        last_date = last_start.get("date")
        days_rest = None
        if last_date:
            d = datetime.strptime(last_date, "%Y-%m-%d").date()
            days_rest = (datetime.now().date() - d).days
        # Pitch counts last 3 starts
        recent_starts = starts[-3:]
        pitch_counts = []
        for s in recent_starts:
            pc = s.get("stat", {}).get("numberOfPitches", 0)
            if pc:
                pitch_counts.append(int(pc))
        avg_pitches = sum(pitch_counts) / len(pitch_counts) if pitch_counts else None
        return {
            "days_rest": days_rest,
            "last_start_date": last_date,
            "recent_pitch_counts": pitch_counts,
            "avg_recent_pitches": round(avg_pitches, 1) if avg_pitches else None,
            "workload_flag": (
                "🚩 High" if avg_pitches and avg_pitches > 100
                else "Normal" if avg_pitches else "—"
            ),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Times Through Order Penalty (TTOP)
# ---------------------------------------------------------------------------

def ttop_multiplier(lineup_pos: int, expected_ip: float = 5.5) -> float:
    """
    Multiplier on HR probability based on times the pitcher will see this
    spot in the order.

    Top of order sees pitcher 3+ times; bottom sees 2 times.
    Penalty: 1.0 (1st TTO), 1.05 (2nd), 1.15 (3rd+)
    """
    if expected_ip < 3:
        return 1.0  # Bullpen game
    # Top 4 of order get 3 looks if pitcher goes 6 IP
    if expected_ip >= 5.5 and lineup_pos <= 4:
        return 1.10
    if expected_ip >= 5 and lineup_pos <= 6:
        return 1.06
    return 1.0


# ---------------------------------------------------------------------------
# Pull-side HR factor: hitter's pull-air% × park's pull-side dimensions
# ---------------------------------------------------------------------------

# Approximate park HR factors by hitter handedness
# Source: Statcast park factors by handedness, 3-yr rolling
PARK_HAND_FACTORS = {
    "Yankee Stadium":           {"L": 1.18, "R": 1.05},  # short porch LF/RF
    "Fenway Park":              {"L": 0.92, "R": 1.15},  # Green Monster
    "Camden Yards":             {"L": 1.08, "R": 0.95},
    "Oriole Park at Camden Yards": {"L": 1.08, "R": 0.95},
    "Citizens Bank Park":       {"L": 1.12, "R": 1.10},
    "Coors Field":              {"L": 1.22, "R": 1.20},
    "Great American Ball Park": {"L": 1.18, "R": 1.12},
    "Wrigley Field":            {"L": 1.05, "R": 1.05},  # depends on wind
    "Oracle Park":              {"L": 0.72, "R": 0.92},  # death to lefty pulls
    "Petco Park":               {"L": 0.92, "R": 0.94},
    "T-Mobile Park":            {"L": 0.90, "R": 0.92},
    "Dodger Stadium":           {"L": 0.95, "R": 0.98},
    "Globe Life Field":         {"L": 1.05, "R": 1.10},
    "Truist Park":              {"L": 1.00, "R": 1.05},
    "Chase Field":              {"L": 1.02, "R": 1.04},
    "Rogers Centre":            {"L": 1.05, "R": 1.05},
    "loanDepot park":           {"L": 0.92, "R": 0.95},
    "Comerica Park":            {"L": 0.95, "R": 0.95},
    "Kauffman Stadium":         {"L": 0.95, "R": 0.95},
    "Tropicana Field":          {"L": 0.95, "R": 0.95},
    "Progressive Field":        {"L": 0.98, "R": 0.98},
    "American Family Field":    {"L": 0.98, "R": 1.00},
    "Target Field":             {"L": 1.00, "R": 1.00},
    "Citi Field":               {"L": 0.95, "R": 0.98},
    "Nationals Park":           {"L": 1.00, "R": 1.00},
    "PNC Park":                 {"L": 0.95, "R": 0.95},
    "Busch Stadium":            {"L": 0.95, "R": 0.95},
    "Angel Stadium":            {"L": 0.95, "R": 0.95},
}


def park_hand_factor(venue: str, bats: str) -> float:
    """HR factor adjusted for hitter handedness in this park."""
    if not venue or not bats:
        return 1.0
    park = PARK_HAND_FACTORS.get(venue, {})
    return park.get(bats, 1.0)
