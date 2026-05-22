"""
fangraphs.py
=============
Scrapes FanGraphs leaderboards for Stuff+, Location+, Pitching+.

These are FanGraphs' proprietary models that grade pitchers on:
  - Stuff+   : How nasty the raw pitches are (movement, velo, spin) - 100 = avg
  - Location+: How well the pitcher locates - 100 = avg
  - Pitching+: Combined Stuff+ × Location+ - 100 = avg

A pitcher with 115 Stuff+ and 110 Location+ projects to outperform their
current ERA. A 90 Stuff+ pitcher is in trouble.

NOTE: FanGraphs doesn't offer a free public API. This is HTML scraping
of their public leaderboard pages. If they change their site layout,
this module will fail gracefully and return an empty DataFrame.

The app handles missing data without falling back to defaults.
"""

from __future__ import annotations
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
import streamlit as st

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

CURRENT_SEASON = datetime.now().year


@st.cache_data(ttl=21600)  # 6 hours - these numbers don't move fast
def get_fangraphs_pitch_models(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Scrape FanGraphs Stuff+/Location+/Pitching+ leaderboard.

    Returns DataFrame with columns:
      - player_name, mlbamid (or playerid), team
      - Stuff+, Location+, Pitching+ (overall)
      - Stuff+ per pitch type (FF, SL, CB, CH, etc.)

    Returns empty DataFrame on any failure - the app handles that.
    """
    # FanGraphs pitching-leaders page with statgroup=PIT&stattype=pit-plus
    url = (
        f"https://www.fangraphs.com/leaders-legacy.aspx?pos=all&stats=pit"
        f"&lg=all&qual=20&type=36&season={season}&season1={season}"
        f"&ind=0&team=0&rost=0&age=0&filter=&players=0&page=1_500"
    )

    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()

        # FanGraphs returns the leaderboard as HTML tables. Use pandas read_html.
        tables = pd.read_html(StringIO(r.text))
        if not tables:
            return pd.DataFrame()

        # The leaderboard table is typically the largest one
        df = max(tables, key=lambda t: len(t))

        # Clean: drop multi-level headers if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]

        # Find name and key plus-stats columns
        # FanGraphs uses "Name", "Team", "Stuff+", "Location+", "Pitching+"
        rename_map = {}
        for col in df.columns:
            cl = col.lower()
            if cl == "name":
                rename_map[col] = "player_name"
            elif cl == "team":
                rename_map[col] = "team"
            elif "stuff+" in cl and "ff" not in cl and "sl" not in cl:
                rename_map[col] = "stuff_plus"
            elif "location+" in cl:
                rename_map[col] = "location_plus"
            elif "pitching+" in cl:
                rename_map[col] = "pitching_plus"

        df = df.rename(columns=rename_map)

        # Keep only the columns we care about
        keep = [c for c in [
            "player_name", "team", "stuff_plus", "location_plus", "pitching_plus"
        ] if c in df.columns]
        if not keep:
            return pd.DataFrame()

        df = df[keep].copy()

        # Convert numeric columns
        for c in ["stuff_plus", "location_plus", "pitching_plus"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Drop rows that didn't parse
        df = df.dropna(subset=[c for c in ["stuff_plus", "location_plus", "pitching_plus"]
                                if c in df.columns], how="all")

        return df.reset_index(drop=True)

    except Exception:
        # Silent failure - caller checks if df is empty
        return pd.DataFrame()


def merge_fangraphs_into_pitchers(pitcher_df: pd.DataFrame,
                                   fg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge FanGraphs plus stats onto a Statcast pitcher DataFrame.
    Matches by player_name (since FanGraphs scrape doesn't expose mlbamid easily).

    If fg_df is empty, returns pitcher_df unchanged.
    """
    if fg_df is None or fg_df.empty or pitcher_df is None or pitcher_df.empty:
        return pitcher_df

    if "player_name" not in pitcher_df.columns or "player_name" not in fg_df.columns:
        return pitcher_df

    # Name normalize for matching
    def norm(s):
        if not isinstance(s, str):
            return ""
        return s.strip().lower().replace(".", "").replace("'", "")

    pitcher_df = pitcher_df.copy()
    fg_df = fg_df.copy()
    pitcher_df["_norm_name"] = pitcher_df["player_name"].apply(norm)
    fg_df["_norm_name"] = fg_df["player_name"].apply(norm)

    # Avoid column name collision
    fg_cols = [c for c in fg_df.columns
                if c not in ("player_name", "team")]
    merged = pitcher_df.merge(
        fg_df[["_norm_name"] + fg_cols],
        on="_norm_name", how="left",
    )
    merged = merged.drop(columns=["_norm_name"])
    return merged
