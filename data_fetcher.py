"""
data_fetcher.py
================
Pulls today's MLB slate and all underlying data from free public sources:
  - MLB Stats API (statsapi.mlb.com) - slate, probable pitchers, lineups
  - Baseball Savant (baseballsavant.mlb.com) - Statcast stats, arsenals

No API keys required. All endpoints are publicly documented.
Results are cached via Streamlit's @st.cache_data with TTLs tuned per source:
  - Slate (probable pitchers): 5 min
  - Lineups: 3 min
  - Active rosters: 15 min
  - Season Statcast stats: 1 hour AND rolls over daily at 5 AM ET
"""

from __future__ import annotations

import io
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
import requests
import streamlit as st


def _stats_day_key() -> str:
    """Return a cache-buster string that changes once per day at 5 AM ET.

    Statcast updates with the previous day's batted-ball data around 2-4 AM ET.
    By rolling our cache key over at 5 AM ET, we guarantee that morning users
    always get fresh stats from the new day's update — without waiting for the
    1-hour TTL to expire mid-morning.

    Returns a string like "2026-05-28" (the "stats day"). For times before 5 AM,
    we return yesterday's date so we don't bust the cache prematurely while
    Statcast itself is still updating.
    """
    try:
        try:
            from zoneinfo import ZoneInfo
            now_et = datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            # Fallback: assume UTC-4 (ET in summer); good enough for cache
            now_et = datetime.utcnow() - timedelta(hours=4)
        # If it's before 5 AM ET, Statcast may still be updating — use yesterday
        if now_et.hour < 5:
            return (now_et - timedelta(days=1)).strftime("%Y-%m-%d")
        return now_et.strftime("%Y-%m-%d")
    except Exception:
        return date.today().isoformat()


# Pretend to be a real browser - Savant sometimes 403s on bare requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/csv, */*",
}

CURRENT_SEASON = datetime.now().year


# ----------------------------------------------------------------------------
# Defensive helpers (v42s — reviewer-validated reliability/safety wins)
# ----------------------------------------------------------------------------

def enforce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Force a list of columns to numeric dtype, coercing invalid values to NaN.

    Catches the silent corruption case where Savant/MLB Stats API occasionally
    returns string values like "-.--" or "" in numeric fields. Without this,
    downstream math operations silently fail or produce nonsensical results.
    Modifies df in place (and also returns it for chaining).
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clip_outliers(df: pd.DataFrame, caps: dict | None = None) -> pd.DataFrame:
    """Clamp Statcast/MLB stats to physically-plausible ranges.

    Savant occasionally returns extreme outlier values (e.g., a 90 mph average
    exit velo for a pitcher with 1 batted ball event). These pollute downstream
    ranking and modeling. Caps are calibrated to the realistic envelope of MLB
    performance — values outside these ranges are not skill, they're data noise.
    Modifies df in place (and also returns it for chaining).
    """
    if caps is None:
        caps = {
            "barrel_pct": (0, 30),         # league max ~25%, allow buffer
            "hard_hit": (0, 75),           # league max ~65%
            "iso": (0, 0.500),             # league max ~.400
            "la": (-10, 50),               # avg launch angle realistic range
            "avg_ev": (70, 120),           # exit velocity realistic range
            "hr9": (0, 6),                 # already capped in models, belt-and-suspenders
            "barrel_batted_rate": (0, 30),
            "hard_hit_percent": (0, 75),
            "avg_best_speed": (70, 120),
            "launch_speed": (70, 120),
            "launch_angle": (-10, 50),
        }
    for col, (low, high) in caps.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(low, high)
    return df


def safe_request(url: str, timeout: int = 20, max_retries: int = 3,
                  **kwargs) -> requests.Response | None:
    """HTTP GET with exponential backoff retry. Returns None on final failure.

    Replaces patterns like `requests.get(url, headers=HEADERS, timeout=20)` that
    fail hard on the first transient error. Backoff schedule: 1.5s, 3s, 6s.
    Returns None instead of raising so callers can decide whether to fall back
    to a cached value, skip the source, or surface an error. The returned
    Response (if any) is already .raise_for_status()-cleared.

    Existing fetch functions in this file have their own try/except blocks and
    don't need to be retrofitted unless they're observed to fail transiently.
    Use this helper for new fetches or specifically-fragile endpoints.
    """
    import time
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1.5 * (2 ** attempt))
    return None


# ----------------------------------------------------------------------------
# Zone tier fetchers (v43 — experimental, defensive)
# ----------------------------------------------------------------------------
# Savant's swing/take leaderboard categorizes every pitch into one of four
# zone tiers: Heart, Shadow, Chase, Waste. These fetchers pull each pitcher's
# % of pitches by tier and each hitter's run value by tier.
#
# IMPORTANT: I could not validate the exact endpoint URL/schema from my dev
# environment (baseballsavant.mlb.com not in allowlist). These fetchers are
# best-effort based on documented Savant patterns. They return EMPTY DataFrame
# on any failure so the app falls back to the plate_discipline_flag signal
# without crashing.
#
# If the fetch returns data: zone_fit_flag will populate in matchup tables.
# If the fetch fails: the field will be empty, plate_discipline_flag still works,
# and the user can flag the issue for proper investigation.

@st.cache_data(ttl=21600)  # 6hr — zone data updates daily
def get_pitcher_zone_tiers(season: int = CURRENT_SEASON, _stats_day: str = "") -> pd.DataFrame:
    """Pitcher % of pitches by zone tier (heart, shadow, chase, waste).

    Returns DataFrame with columns:
      player_id, pitcher_heart_pct, pitcher_shadow_pct,
      pitcher_chase_pct, pitcher_waste_pct
    Empty on failure — caller treats empty as "no zone data available."
    """
    # Try a few candidate URL patterns. The first that returns parseable CSV wins.
    candidate_urls = [
        # Custom leaderboard with zone selections
        f"https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={season}&type=pitcher&filter=&min=50"
        f"&selections=player_id,player_name,heart_zone_percent,shadow_zone_percent,"
        f"chase_zone_percent,waste_zone_percent"
        f"&chart=false&x=heart_zone_percent&y=shadow_zone_percent&r=no&csv=true",
        # Swing/take leaderboard
        f"https://baseballsavant.mlb.com/leaderboard/swing-take"
        f"?year={season}&type=pitcher&min=100&csv=true",
    ]
    for url in candidate_urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            if not r.text or len(r.text) < 100:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty:
                continue
            # Try to identify columns flexibly — Savant uses various names
            col_map = {}
            for c in df.columns:
                cl = c.lower()
                if "player_id" in cl or cl == "id":
                    col_map["player_id"] = c
                elif ("heart" in cl and "pct" in cl) or ("heart" in cl and "percent" in cl):
                    col_map["heart"] = c
                elif "shadow" in cl and ("pct" in cl or "percent" in cl):
                    col_map["shadow"] = c
                elif "chase" in cl and ("pct" in cl or "percent" in cl):
                    col_map["chase"] = c
                elif "waste" in cl and ("pct" in cl or "percent" in cl):
                    col_map["waste"] = c
            # Need at least player_id + heart to be useful
            if "player_id" not in col_map or "heart" not in col_map:
                continue
            out = pd.DataFrame()
            out["player_id"] = pd.to_numeric(df[col_map["player_id"]], errors="coerce").astype("Int64")
            out["pitcher_heart_pct"] = pd.to_numeric(df[col_map["heart"]], errors="coerce")
            for tier in ("shadow", "chase", "waste"):
                if tier in col_map:
                    out[f"pitcher_{tier}_pct"] = pd.to_numeric(df[col_map[tier]], errors="coerce")
                else:
                    out[f"pitcher_{tier}_pct"] = pd.NA
            return out.dropna(subset=["player_id"])
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=21600)
def get_hitter_zone_tiers(season: int = CURRENT_SEASON, _stats_day: str = "") -> pd.DataFrame:
    """Hitter wOBA or run value by zone tier.

    Returns DataFrame with columns:
      player_id, hitter_heart_woba, hitter_shadow_woba,
      hitter_chase_woba, hitter_waste_woba (or barrel% equivalents)
    Empty on failure.
    """
    candidate_urls = [
        f"https://baseballsavant.mlb.com/leaderboard/swing-take"
        f"?year={season}&type=batter&min=100&csv=true",
        f"https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={season}&type=batter&filter=&min=50"
        f"&selections=player_id,player_name,heart_woba,shadow_woba,chase_woba,waste_woba"
        f"&chart=false&x=heart_woba&y=shadow_woba&r=no&csv=true",
    ]
    for url in candidate_urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            if not r.text or len(r.text) < 100:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty:
                continue
            col_map = {}
            for c in df.columns:
                cl = c.lower()
                if "player_id" in cl or cl == "id":
                    col_map["player_id"] = c
                elif "heart" in cl and ("woba" in cl or "wOBA" in c or "run_value" in cl):
                    col_map["heart"] = c
                elif "shadow" in cl and ("woba" in cl or "run_value" in cl):
                    col_map["shadow"] = c
                elif "chase" in cl and ("woba" in cl or "run_value" in cl):
                    col_map["chase"] = c
                elif "waste" in cl and ("woba" in cl or "run_value" in cl):
                    col_map["waste"] = c
            if "player_id" not in col_map or "heart" not in col_map:
                continue
            out = pd.DataFrame()
            out["player_id"] = pd.to_numeric(df[col_map["player_id"]], errors="coerce").astype("Int64")
            out["hitter_heart_woba"] = pd.to_numeric(df[col_map["heart"]], errors="coerce")
            for tier in ("shadow", "chase", "waste"):
                if tier in col_map:
                    out[f"hitter_{tier}_woba"] = pd.to_numeric(df[col_map[tier]], errors="coerce")
                else:
                    out[f"hitter_{tier}_woba"] = pd.NA
            return out.dropna(subset=["player_id"])
        except Exception:
            continue
    return pd.DataFrame()


# ----------------------------------------------------------------------------
# Per-player zone fetchers (v43+ — wrap bulk leaderboard fetchers)
# ----------------------------------------------------------------------------
# These return a single player's zone data as a dict. They wrap the bulk
# leaderboard fetchers above which are cached, so per-player calls cost
# nothing after the first slate load. Used by the Pitcher Lookup utility
# so a user can scout any pitcher's zone tendencies.
#
# Return {} if no data found for the requested player (bulk fetch failed,
# player not in leaderboard, etc). Callers should treat empty dict as
# "no zone data available, fall back to other signals."

def get_pitcher_zone_distribution(pitcher_id: int,
                                    season: int = CURRENT_SEASON) -> dict:
    """Return one pitcher's heart/shadow/chase/waste % distribution.

    Wraps get_pitcher_zone_tiers (cached 6hr). Returns dict like:
      {
        "player_id": 696285,
        "pitcher_heart_pct": 26.2,
        "pitcher_shadow_pct": 41.5,
        "pitcher_chase_pct": 24.1,
        "pitcher_waste_pct": 8.2,
      }
    Or {} if not found.

    Args:
      pitcher_id: MLB player ID
      season: which season's data (defaults to current)
    """
    try:
        df = get_pitcher_zone_tiers(season=season, _stats_day=_stats_day_key())
        if df is None or df.empty:
            return {}
        rows = df[df["player_id"] == int(pitcher_id)]
        if rows.empty:
            return {}
        return rows.iloc[0].to_dict()
    except Exception:
        return {}


def get_hitter_zone_performance(hitter_id: int,
                                  season: int = CURRENT_SEASON) -> dict:
    """Return one hitter's wOBA by zone tier.

    Wraps get_hitter_zone_tiers (cached 6hr). Returns dict like:
      {
        "player_id": 592450,  # Judge
        "hitter_heart_woba": 0.488,
        "hitter_shadow_woba": 0.310,
        "hitter_chase_woba": 0.140,
        "hitter_waste_woba": 0.060,
      }
    Or {} if not found.

    Args:
      hitter_id: MLB player ID
      season: which season's data (defaults to current)
    """
    try:
        df = get_hitter_zone_tiers(season=season, _stats_day=_stats_day_key())
        if df is None or df.empty:
            return {}
        rows = df[df["player_id"] == int(hitter_id)]
        if rows.empty:
            return {}
        return rows.iloc[0].to_dict()
    except Exception:
        return {}


def zone_fit_score(pitcher_id: int, hitter_id: int,
                     season: int = CURRENT_SEASON) -> dict:
    """Compute zone fit composite for one (pitcher, hitter) matchup.

    Returns dict with:
      - composite_woba: weighted hitter wOBA across the pitcher's tier distribution
      - heart_share: pitcher's heart-of-plate %
      - hitter_heart_woba: hitter's wOBA on heart pitches
      - data_coverage: how much of the pitcher's distribution we have data for
      - assessment: human-readable label
    Or {} if data unavailable for either player.

    This is the heart of zone fit: a pitcher who lives in the heart of the
    plate (high heart_share) facing a hitter who crushes heart pitches
    (high hitter_heart_woba) = HR risk. The composite_woba accounts for the
    full tier distribution, not just the heart.
    """
    p_dist = get_pitcher_zone_distribution(pitcher_id, season)
    h_perf = get_hitter_zone_performance(hitter_id, season)
    if not p_dist or not h_perf:
        return {}
    tiers = ("heart", "shadow", "chase", "waste")
    total_pct = 0.0
    weighted = 0.0
    for t in tiers:
        p_pct = p_dist.get(f"pitcher_{t}_pct")
        h_woba = h_perf.get(f"hitter_{t}_woba")
        if (p_pct is None or pd.isna(p_pct)
            or h_woba is None or pd.isna(h_woba)):
            continue
        try:
            p_pct = float(p_pct); h_woba = float(h_woba)
        except (TypeError, ValueError):
            continue
        weighted += p_pct * h_woba
        total_pct += p_pct
    if total_pct < 50.0:
        return {}  # not enough data coverage to be meaningful
    composite_woba = weighted / total_pct
    # v43.1 calibration: tier-weighted composite is naturally diluted by
    # chase/waste pitches (where every hitter performs poorly). Elite
    # matchups produce composites ~0.350-0.370, not 0.420. League-average
    # matchup composite is ~0.310. Recalibrated thresholds:
    #   >= 0.370: ELITE (top ~5% of matchups — heart-pounder vs heart-crusher)
    #   >= 0.340: STRONG (above neutral, favorable for hitter)
    #   <= 0.260: POOR (below neutral, pitcher dominates the zones hitter struggles in)
    if composite_woba >= 0.370:
        assessment = "💣 elite zone fit"
    elif composite_woba >= 0.340:
        assessment = "🎯 strong zone fit"
    elif composite_woba <= 0.260:
        assessment = "🛡️ poor zone fit"
    else:
        assessment = "neutral"
    return {
        "composite_woba": round(composite_woba, 3),
        "heart_share": round(float(p_dist.get("pitcher_heart_pct", 0)), 1),
        "hitter_heart_woba": round(float(h_perf.get("hitter_heart_woba", 0)), 3),
        "data_coverage_pct": round(total_pct, 1),
        "assessment": assessment,
    }


# ----------------------------------------------------------------------------
# Savant handedness-split Statcast (v43.5 — experimental, URL unverified)
# ----------------------------------------------------------------------------
# MLB Stats API splits give us basic handedness data (AVG, OPS, ISO, HR/PA,
# K%, BB%). But contact-quality metrics by handedness (barrel%, hard_hit%,
# xwOBA, avg_ev) are Statcast-only — they live in Savant.
#
# This is the gap that lets a hitter get an inflated grade against the side
# they actually struggle against. Example: Jo Adell overall barrel% is ~8%,
# but vs-LHP barrel% can be 30%+. Without these splits, power_score scores
# him as a weak HR threat vs lefties when he's actually elite.
#
# DEFENSIVE PATTERN: tries several candidate URLs, returns empty DataFrame on
# any failure. Caller treats empty as "no handedness Statcast data; fall
# back to season-overall." Same pattern as zone tier fetchers.
#
# IMPORTANT: Savant URLs are NOT in the dev environment allowlist, so this
# is NOT testable here. URL patterns must be verified in production. The
# caption status indicator ("Savant handedness: ✅ working / ❌ unavailable")
# tells you immediately if the fetcher connected.

@st.cache_data(ttl=21600)  # 6hr — handedness splits stable day-over-day
def get_hitter_handedness_statcast(season: int = CURRENT_SEASON,
                                     _stats_day: str = "") -> pd.DataFrame:
    """Pull hitter Statcast contact-quality metrics split by pitcher handedness.

    Returns DataFrame with columns:
      player_id,
      vs_lhp_barrel_pct, vs_lhp_hard_hit, vs_lhp_xwoba, vs_lhp_avg_ev,
      vs_rhp_barrel_pct, vs_rhp_hard_hit, vs_rhp_xwoba, vs_rhp_avg_ev
    Empty on failure — caller falls back to season-overall stats.

    Implementation note: Savant exposes handedness splits via the custom
    leaderboard with a `split` parameter (or `pitch_hand` filter). We pull
    vs-LHP and vs-RHP separately, then merge on player_id. Each side may
    succeed or fail independently — partial data is better than none.
    """
    def _fetch_one_side(pitch_hand: str) -> pd.DataFrame:
        """Fetch Statcast contact quality for hitters vs one pitcher hand.

        pitch_hand: 'L' or 'R'
        """
        candidate_urls = [
            # Custom leaderboard with pitch_hand filter and Statcast selections.
            # Savant's custom-leaderboard endpoint accepts ?pitch_hand=L|R and
            # returns a CSV when csv=true is appended.
            f"https://baseballsavant.mlb.com/leaderboard/custom"
            f"?year={season}&type=batter&filter=&min=30"
            f"&pitch_hand={pitch_hand}"
            f"&selections=player_id,player_name,pa,barrel_batted_rate,"
            f"hard_hit_percent,xwoba,launch_speed"
            f"&chart=false&x=barrel_batted_rate&y=xwoba&r=no&csv=true",
            # Statcast leaderboard with split parameter (alternate Savant URL)
            f"https://baseballsavant.mlb.com/leaderboard/statcast"
            f"?type=batter&year={season}&player_type=batter"
            f"&min_results=30&split=vs_{pitch_hand}HP&csv=true",
            # Savant search CSV with pitcher_throws filter
            f"https://baseballsavant.mlb.com/statcast_search/csv"
            f"?all=true&year={season}&pitcher_throws={pitch_hand}"
            f"&player_type=batter&min_pa=30",
        ]
        for url in candidate_urls:
            try:
                r = requests.get(url, headers=HEADERS, timeout=25)
                r.raise_for_status()
                if not r.text or len(r.text) < 100:
                    continue
                df = pd.read_csv(io.StringIO(r.text))
                if df.empty:
                    continue
                # Flexible column matching — Savant column names vary
                col_map = {}
                for c in df.columns:
                    cl = c.lower().replace(" ", "_")
                    if cl in ("player_id", "playerid", "id", "mlb_id"):
                        col_map["player_id"] = c
                    elif "barrel" in cl and ("rate" in cl or "pct" in cl or "percent" in cl):
                        col_map["barrel"] = c
                    elif ("hard_hit" in cl or "hardhit" in cl) and (
                        "rate" in cl or "pct" in cl or "percent" in cl
                    ):
                        col_map["hard_hit"] = c
                    elif cl in ("xwoba", "x_woba", "estimated_woba"):
                        col_map["xwoba"] = c
                    elif "launch_speed" in cl or "avg_ev" in cl or "exit_velocity" in cl:
                        col_map["avg_ev"] = c
                # Need at least player_id + barrel to be useful
                if "player_id" not in col_map or "barrel" not in col_map:
                    continue
                out = pd.DataFrame()
                out["player_id"] = pd.to_numeric(
                    df[col_map["player_id"]], errors="coerce"
                ).astype("Int64")
                out["barrel_pct"] = pd.to_numeric(df[col_map["barrel"]], errors="coerce")
                for stat in ("hard_hit", "xwoba", "avg_ev"):
                    if stat in col_map:
                        out[stat] = pd.to_numeric(df[col_map[stat]], errors="coerce")
                    else:
                        out[stat] = pd.NA
                return out.dropna(subset=["player_id"])
            except Exception:
                continue
        return pd.DataFrame()

    # Fetch both sides; either may fail independently
    df_l = _fetch_one_side("L")
    df_r = _fetch_one_side("R")

    if df_l.empty and df_r.empty:
        return pd.DataFrame()

    # Rename columns with vs_lhp_ / vs_rhp_ prefix
    if not df_l.empty:
        df_l = df_l.rename(columns={
            "barrel_pct": "vs_lhp_barrel_pct",
            "hard_hit": "vs_lhp_hard_hit",
            "xwoba": "vs_lhp_xwoba",
            "avg_ev": "vs_lhp_avg_ev",
        })
    if not df_r.empty:
        df_r = df_r.rename(columns={
            "barrel_pct": "vs_rhp_barrel_pct",
            "hard_hit": "vs_rhp_hard_hit",
            "xwoba": "vs_rhp_xwoba",
            "avg_ev": "vs_rhp_avg_ev",
        })

    # Outer merge on player_id — keep all hitters that appeared on either side
    if df_l.empty:
        return df_r
    if df_r.empty:
        return df_l
    return df_l.merge(df_r, on="player_id", how="outer")


def apply_handedness_overrides(matchup_df: pd.DataFrame,
                                 p_throws: str | None,
                                 min_split_pa: int = 30,
                                 shrinkage_pa: int = 100) -> pd.DataFrame:
    """Per-game: blend season-overall barrel/hard_hit/xwoba/avg_ev/iso
    with handedness-specific values, weighted by sample size.

    v43.64 (reviewer doc fix #10) — IMPORTANT coverage caveat:
    This function CAN blend barrel/hard_hit/xwoba/avg_ev/iso BUT only when
    the corresponding `vs_lhp_*` / `vs_rhp_*` Statcast columns are actually
    present. Those columns only exist if `get_hitter_handedness_statcast`
    succeeded — and that fetcher's Savant URLs are unverified and often
    return empty. In the common case the Savant handedness fetch comes
    back empty, this function only blends `iso` (from the MLB Stats API
    split path, which IS reliable). So "handedness blend" is effectively
    "ISO-only blend" most of the time. The `_hand_statcast_status` caption
    in app.py surfaces whether Statcast actually returned. Tune
    expectations accordingly — Pipeline Health's "handedness coverage"
    line shows what's actually populated.

    v43.43 (reviewer-validated): the original v43.18 implementation did a
    HARD REPLACE of season values with handedness values when split_pa >=
    min_split_pa=30. That let a hitter with 35 PA vs LHP and a flukey ISO
    fully drive their grade. Reviewer caught this: "tiny-sample splits
    driving grades — split_confidence warns but doesn't stop the override
    from affecting the score."

    The fix is a Bayesian shrinkage blend:
      blended = (hand_pa × hand_value + shrinkage_pa × season_value)
                / (hand_pa + shrinkage_pa)

    With shrinkage_pa=100 (default):
      30 PA hand sample  → 23% weight to handedness, 77% to season-overall
      60 PA              → 38% / 62%
      100 PA             → 50% / 50%
      200 PA             → 67% / 33%
      300+ PA            → 75%+ / 25%-

    This means flukey small samples can shift the grade but can't dominate
    it. Established split hitters (200+ PA) still get most of the override
    benefit. No hard cutoff — the blend gracefully scales.

    The min_split_pa=30 floor still applies: below 30 PA, no blend at all
    (just season-overall).

    Returns modified matchup_df. Original column values are not preserved
    (they remain in vs_lhp_* / vs_rhp_* for reference). Switch hitters: the
    vs_lhp/vs_rhp splits naturally reflect which side they batted (splits
    accumulate based on pitcher faced), so the blend works the same way.
    """
    if matchup_df is None or matchup_df.empty:
        return matchup_df
    if not p_throws or p_throws not in ("L", "R"):
        return matchup_df

    side = "lhp" if p_throws == "L" else "rhp"
    pa_col = f"vs_{side}_pa"
    if pa_col not in matchup_df.columns:
        return matchup_df

    df = matchup_df.copy()
    pa_vals = pd.to_numeric(df[pa_col], errors="coerce")
    # Hard floor: below min_split_pa, ignore handedness entirely
    sufficient_mask = pa_vals >= min_split_pa

    # Bayesian blend each stat with shrinkage toward season-overall
    overrides = [
        ("barrel_pct",  f"vs_{side}_barrel_pct"),
        ("hard_hit",    f"vs_{side}_hard_hit"),
        ("xwoba",       f"vs_{side}_xwoba"),
        ("avg_ev",      f"vs_{side}_avg_ev"),
        ("iso",         f"vs_{side}_iso"),
    ]
    for season_col, hand_col in overrides:
        if season_col in df.columns and hand_col in df.columns:
            season_vals = pd.to_numeric(df[season_col], errors="coerce")
            hand_vals = pd.to_numeric(df[hand_col], errors="coerce")
            # Apply blend only where: sufficient PA AND both values valid
            blend_mask = sufficient_mask & hand_vals.notna() & season_vals.notna()
            if blend_mask.any():
                # weight_hand = hand_pa / (hand_pa + shrinkage_pa)
                hand_pa = pa_vals[blend_mask]
                w_hand = hand_pa / (hand_pa + shrinkage_pa)
                w_season = 1 - w_hand
                blended = (
                    hand_vals[blend_mask] * w_hand
                    + season_vals[blend_mask] * w_season
                )
                df.loc[blend_mask, season_col] = blended
    return df


# ----------------------------------------------------------------------------
# Batter vs Pitcher (BvP) history — v43.14
# ----------------------------------------------------------------------------
# IMPORTANT — honest analytics context: published research (FanGraphs,
# Baseball Prospectus, etc.) consistently finds that career BvP has very
# weak predictive power once you control for handedness, pitcher skill,
# and park. The sample sizes are tiny (typical max ~30-50 PA across
# multiple years, often <15 PA) and the noise dominates any signal.
#
# That said, BvP IS contextually interesting and worth surfacing:
#   - Tiebreaker between similar grades
#   - Display value (most user-facing tools show it)
#   - Strong tails (4-for-8 with 3 HR) DO have weak predictive lift
#
# Our approach:
#   1. Fetch via MLB Stats API's vsPlayer split endpoint
#   2. Display raw counts (PA, HR, AB, hits) — let the user judge
#   3. Light-weight scoring incorporation:
#      - Only fires at PA >= 20 (below = treat as no information)
#      - Hard-capped at ±5 bonus points to pick_score
#      - Never overrides the model's underlying signal
#
# Cache duration: 12 hours. BvP changes only when these two specific players
# face off, which happens at most once per series.

@st.cache_data(ttl=43200)  # 12hr
def get_bvp_for_matchup(batter_id: int, pitcher_id: int) -> dict:
    """Fetch career batter-vs-pitcher stats from MLB Stats API.

    Returns dict with:
      pa, ab, hr, h, bb, k, avg, slg, ops, hr_per_pa
    or {} on any failure / no history.

    Endpoint: /api/v1/people/{batter_id}/stats?stats=vsPlayer&opposingPlayerId={pitcher_id}&group=hitting
    """
    if (batter_id is None or pitcher_id is None
            or pd.isna(batter_id) or pd.isna(pitcher_id)):
        return {}
    try:
        bid = int(batter_id)
        pid = int(pitcher_id)
    except (TypeError, ValueError):
        return {}
    if bid <= 0 or pid <= 0:
        return {}

    url = (
        f"https://statsapi.mlb.com/api/v1/people/{bid}/stats"
        f"?stats=vsPlayer&opposingPlayerId={pid}&group=hitting"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return {}
        data = r.json()
        # Aggregate across all season splits returned (career = sum)
        agg_pa = 0; agg_ab = 0; agg_hr = 0; agg_h = 0
        agg_bb = 0; agg_k = 0; agg_2b = 0; agg_3b = 0
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                stat = split.get("stat", {})

                def _to_int(v):
                    if v is None: return 0
                    try: return int(v)
                    except (TypeError, ValueError):
                        try: return int(float(v))
                        except (TypeError, ValueError): return 0

                agg_pa += _to_int(stat.get("plateAppearances") or stat.get("pa"))
                agg_ab += _to_int(stat.get("atBats") or stat.get("ab"))
                agg_hr += _to_int(stat.get("homeRuns") or stat.get("hr"))
                agg_h  += _to_int(stat.get("hits") or stat.get("h"))
                agg_bb += _to_int(stat.get("baseOnBalls") or stat.get("bb"))
                agg_k  += _to_int(stat.get("strikeOuts") or stat.get("so"))
                agg_2b += _to_int(stat.get("doubles") or stat.get("2b"))
                agg_3b += _to_int(stat.get("triples") or stat.get("3b"))

        if agg_pa == 0 and agg_ab == 0:
            return {}

        denom_pa = agg_pa if agg_pa > 0 else agg_ab
        denom_ab = agg_ab if agg_ab > 0 else agg_pa

        # Compute derived stats
        # v43.18 (reviewer-validated defensive): if MLB Stats API ever
        # returns inconsistent data (agg_h < 2b+3b+hr), singles goes negative
        # and SLG becomes nonsense. Clamp at zero.
        singles = max(0, agg_h - agg_2b - agg_3b - agg_hr)
        total_bases = singles + 2 * agg_2b + 3 * agg_3b + 4 * agg_hr
        out = {
            "bvp_pa": agg_pa or agg_ab,
            "bvp_ab": agg_ab,
            "bvp_hr": agg_hr,
            "bvp_h": agg_h,
            "bvp_bb": agg_bb,
            "bvp_k": agg_k,
            "bvp_avg": round(agg_h / denom_ab, 3) if denom_ab > 0 else None,
            "bvp_slg": round(total_bases / denom_ab, 3) if denom_ab > 0 else None,
            "bvp_hr_per_pa": round(agg_hr / denom_pa * 100, 2) if denom_pa > 0 else None,
        }
        return out
    except Exception:
        return {}


def bvp_score_adjustment(bvp_dict: dict, min_pa: int = 20) -> tuple[float, str]:
    """Convert BvP history into a small pick_score adjustment.

    Returns (adjustment_points, descriptive_flag).

    Adjustment is capped at ±5 points (intentionally small — BvP has weak
    predictive power per the research, so we treat it as a tiebreaker, not
    a driver). Returns (0, "") if sample is too small (<min_pa) or no data.

    Flag examples:
      "🎯 BvP +5 (8/15, 3 HR)"  — strong positive history
      "⚠️ BvP -3 (1/18, 0 HR)"  — weak history
      ""                         — insufficient sample or no data
    """
    if not bvp_dict:
        return (0.0, "")
    pa = bvp_dict.get("bvp_pa") or 0
    if pa < min_pa:
        return (0.0, "")

    hr = bvp_dict.get("bvp_hr") or 0
    hr_rate = bvp_dict.get("bvp_hr_per_pa")
    h = bvp_dict.get("bvp_h") or 0
    ab = bvp_dict.get("bvp_ab") or 0
    slg = bvp_dict.get("bvp_slg")
    avg = bvp_dict.get("bvp_avg")

    # Score based on HR rate vs league average (~3% per PA)
    # AND on overall production (SLG/AVG)
    points = 0.0
    if hr_rate is not None:
        # +1 pt per 1% above league avg, capped at +5
        # -1 pt per 1.5% below league avg, capped at -3
        delta = float(hr_rate) - 3.0
        if delta > 0:
            points = min(5.0, delta * 1.0)
        else:
            points = max(-3.0, delta * 0.67)

    # Build descriptive flag
    if hr_rate is None:
        return (0.0, "")
    if points >= 3.0:
        emoji = "💣"
    elif points >= 1.5:
        emoji = "🎯"
    elif points <= -2.0:
        emoji = "⚠️"
    elif points <= -1.0:
        emoji = "📉"
    else:
        emoji = "📊"
    sign = "+" if points >= 0 else ""
    flag = f"{emoji} BvP {sign}{points:.1f} ({h}/{ab}, {hr} HR in {pa} PA)"
    return (round(points, 1), flag)
# ----------------------------------------------------------------------------
# v43.50 (reviewer-validated removal): Batter-park history section removed.
# The v43.26 feature used stats=byVenue which MLB Stats API rejects with
# HTTP 400 ('Invalid Request with value: byVenue'). v43.42 retry was equally
# broken. Feature contributed ~0.04% to HR Score even when working — not
# worth the complexity of switching to game-log-based replacement. If we
# ever want this back, the right approach is per-batter game log filtered
# by venue.id, with diagnostics from day one.
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# Bat tracking / Blast % fetcher — v43.17 (OPT-IN)
# ----------------------------------------------------------------------------
# Statcast added bat tracking in mid-2024. The metric the user requested
# ("Blast %") is a Statcast-defined event: a swing that is both
# (a) squared up (contact quality) AND
# (b) fast (bat speed ≥75 mph).
# It captures roughly the same thing as barrel% but on the SWING level
# rather than the BBE level, so it covers all swings not just contact.
# This is real and predictive — June 2025 Statcast research showed
# blast rate has higher year-over-year stability than barrel rate.
#
# Implementation notes (per user's "opt-in" requirement):
#   - Defaults to DISABLED. User toggles in sidebar.
#   - Caches 24hr (bat tracking changes slowly week-to-week).
#   - Field names tried as a fallback chain (Savant has changed them
#     between 2024 beta and 2025 production). Status reported in caption.
#   - On any failure / empty result, the fetcher returns empty DataFrame
#     and the status caption shows the failure. No silent zero-data.
#   - If enabled and successful, app.py blends `blast_pct` into hr_form.
#     If enabled but fetcher returned empty, hr_form falls back to v43.16
#     weights (no blast component). No silent degradation.

# v43.42 diagnostic: when get_bat_tracking can't find an
# ideal_attack_angle column under any candidate name, it stores Savant's
# ACTUAL response columns here so we can verify the field name. The app
# surfaces this in an expander when IAA is missing for everyone.
_LAST_BAT_TRACKING_COLS: list = []

# v43.69 (production-debug): track what happened in the avg_dist / barrel_count
# fallback fetch. Each entry: {"url": str, "status": int|str, "n_rows": int,
# "cols": [...truncated], "dist_col": str|None, "brl_col": str|None,
# "id_col": str|None, "dist_merged": int, "brl_merged": int}.
# Surfaced via last_distance_diag() so app.py can show in Pipeline Health.
_LAST_DISTANCE_DIAG: list = []

def last_distance_diag() -> list:
    """Returns the most recent avg_dist/barrel_count fallback fetch attempts.

    Each entry is a dict describing one URL attempt. Empty list means the
    fallback didn't run (because data was already present from primary fetch).
    """
    try:
        from_state = st.session_state.get("_last_distance_diag")
        if from_state:
            return list(from_state)
    except Exception:
        pass
    return list(_LAST_DISTANCE_DIAG)

def last_bat_tracking_columns() -> list:
    """Returns the most recent Savant bat-tracking response's column names.

    v43.62 (reviewer-validated fix): prefers st.session_state over the
    module global. The global was set inside `get_bat_tracking` which is
    `@st.cache_data` — on cache hit the function body doesn't run, so the
    global stayed stale. App.py now stashes the returned 3rd tuple element
    into session_state on EVERY call (cache hit or miss), so this reader
    always sees the most recent fetch.

    Empty when IAA was found, fetcher hasn't run, or session_state unavailable.
    """
    try:
        # Prefer session_state (set by app.py from get_bat_tracking's
        # 3rd tuple element — survives the @st.cache_data wrapper)
        cols = st.session_state.get("_last_bat_tracking_cols")
        if cols:
            return list(cols)
    except Exception:
        pass
    # Fall back to module global (legacy path / non-Streamlit callers)
    return list(_LAST_BAT_TRACKING_COLS)

@st.cache_data(ttl=86400)  # 24hr — bat tracking is a slow-moving stat
def get_bat_tracking(season: int | None = None) -> tuple[pd.DataFrame, str, list]:
    """Fetch Statcast bat tracking leaderboard.

    Returns (df, status_message, savant_columns):
      - df: the bat tracking dataframe (empty on failure)
      - status_message: describes what happened (used by app caption)
      - savant_columns: list of column names Savant actually returned
                        (used by the IAA diagnostic to surface field-name drift)

    Possible status messages:
      - "✅ N hitters" — success, df has rows
      - "❌ no rows returned" — endpoint responded but df was empty
      - "❌ no blast column" — endpoint returned data but expected field
                                names weren't there (Savant schema drift)
      - "❌ fetch error: ..." — HTTP / parse error

    v43.62 (reviewer-validated):
      • Fix #1.2: returns savant_columns as part of the tuple. Previously
        the function wrote to a module-level `_LAST_BAT_TRACKING_COLS`
        global, but `@st.cache_data` skips the function body on cache hit
        so the global stays stale. The IAA diagnostic in app.py read that
        global and showed empty/old data after the first slate of the day.
        Returning the column list as part of the cached tuple means it
        survives caching — same data the diagnostic needs, but reliable.
      • Fix #1.6: `season: int = 2025` default removed. Now defaults to
        current year via None sentinel — wouldn't go stale next year.
    """
    # v43.62: resolve default season at call time (was a stale hardcoded 2025)
    if season is None:
        from datetime import date as _date
        season = _date.today().year

    # v43.59: declare global at function entry so all assignment sites
    # downstream (both the v43.57 dropna-failure path and the original
    # v43.42 iaa-missing path) can write without re-declaring.
    # v43.62: still maintained for backward compat with last_bat_tracking_columns(),
    # but the canonical source of truth is now the 3rd tuple element.
    global _LAST_BAT_TRACKING_COLS

    # Field name variants — try each, take whichever returns data
    # v43.58 (production-validated): updated based on actual Savant column
    # list observed June 2026: id, name, swings_competitive,
    # percent_swings_competitive, contact, avg_bat_speed, hard_swing_rate,
    # squared_up_per_bat_contact, squared_up_per_swing, blast_per_bat_contact,
    # blast_per_swing, swing_length, swords, batter_run_value, whiffs,
    # whiff_per_swing, batted_ball_events, batted_ball_event_per_swing.
    # Notably: Savant uses "hard_swing_rate" (not fast_swing*),
    # "blast_per_swing" / "blast_per_bat_contact" (not blast_rate*), and
    # has NO ideal-attack-angle column at all in this endpoint.
    #
    # v43.64 (reviewer fix #9 — calibration pin): per-swing and per-contact
    # have DIFFERENT denominators. Same hitter looks ~20-30% better in one
    # metric than the other. HR Criteria #4's `blast_pct ≥ 5` threshold was
    # calibrated against per-SWING. If we silently switch to per-contact
    # when per-swing is empty, criterion #4 flips meaning. Pin to per-swing
    # candidates only; if Savant drops per-swing, we'll see it in the
    # bat-tracking status and can update deliberately, not silently.
    # Removed from candidates: "blast_per_bat_contact", "blasts_per_bbe".
    blast_candidates = ["blast_rate", "blasts_per_swing", "blast_percent",
                          "blast_per_swing"]
    bat_speed_candidates = ["avg_bat_speed", "bat_speed_avg", "swing_speed",
                              "bat_speed"]
    fast_swing_candidates = ["fast_swing_rate", "fast_swing_percent",
                              "hard_swing_rate"]
    # Same per-swing vs per-contact concern applies; pin to per-swing.
    squared_up_candidates = ["squared_up_rate", "squared_up_per_swing",
                               "squared_up_percent"]
    # v43.36 (user-requested): Ideal Attack Angle — Statcast bat-tracking
    # metric: % of competitive swings where attack angle is 5-20°. This is
    # the "HR launch zone" swing path. League avg ~50-55%, elite ~65%+.
    # v43.42: expanded candidate list — none of the original 6 names
    # matched what Savant returns. Adding common Statcast bat-tracking
    # field naming patterns. If STILL none match, log what columns Savant
    # actually returned so we can fix it on next ship.
    ideal_aa_candidates = [
        # Original candidates (v43.36)
        "attack_angle_in_range_pct", "ideal_attack_angle_pct",
        "attack_angle_ideal", "ideal_swing_pct", "ideal_attack_rate",
        "attack_angle_in_zone_pct",
        # v43.42 additions — common Statcast bat-tracking patterns
        "attack_angle_in_ideal_range_pct", "attack_angle_ideal_pct",
        "ideal_attack_angle_rate", "competitive_swings_in_attack_range_pct",
        "attack_angle_ideal_range", "attack_angle_5_20_pct",
        "swings_in_attack_range_rate", "ideal_attack_swing_rate",
    ]

    # Build a selections string that includes ALL candidates — Savant
    # should return whichever ones it has, ignore the rest.
    all_fields = (["player_id", "player_name"]
                  + blast_candidates + bat_speed_candidates
                  + fast_swing_candidates + squared_up_candidates
                  + ideal_aa_candidates)
    selections = ",".join(all_fields)

    candidate_urls = [
        # Primary: custom leaderboard with bat-tracking selections.
        # Same endpoint that already works for hard_hit/barrel_pct etc.
        f"https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={season}&type=batter&filter=&min=q"
        f"&selections={selections}"
        f"&chart=false&x=player_name&y=player_name&r=no&csv=true",
        # Fallback: dedicated bat-tracking leaderboard URL
        f"https://baseballsavant.mlb.com/leaderboard/bat-tracking"
        f"?attackZone=&batSide=&season={season}&team=&min=q"
        f"&sortColumn={blast_candidates[0]}&sortDirection=desc&csv=true",
    ]

    for url in candidate_urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            if r.status_code != 200:
                continue
            if not r.text or len(r.text) < 100:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty:
                continue

            # v43.59 (production-validated): the previous "first candidate
            # that's in df.columns" pattern failed because Savant returns
            # EVERY column we request in selections=..., regardless of
            # whether the column actually has data populated. In June 2026:
            # `blast_rate` was returned but all 157 values were NaN; the
            # actually-populated column was a different blast variant.
            # Switch to "first candidate that exists AND has non-NaN data".
            def _pick_populated_col(candidates, df):
                """Return the first candidate that has actual data.
                Falls back to first existing column even if all-NaN (still
                better than None for downstream diagnostics).
                """
                for c in candidates:
                    if c in df.columns:
                        try:
                            if df[c].notna().any():
                                return c
                        except Exception:
                            continue
                # Second pass: exists but all NaN — return for diagnostic visibility
                for c in candidates:
                    if c in df.columns:
                        return c
                return None

            # Find which blast field came back AND has data
            blast_col = _pick_populated_col(blast_candidates, df)
            if blast_col is None:
                # Fuzzy match: any column with "blast" in name
                _fuzzy = [c for c in df.columns if "blast" in c.lower()]
                blast_col = _pick_populated_col(_fuzzy, df)
            if blast_col is None:
                continue  # no blast column at all, try next URL
            # If we got the column but it's all-NaN, treat as missing data
            # and try the NEXT URL (in case another URL returns populated data).
            if not df[blast_col].notna().any():
                # blast column exists but empty — this URL is a dud
                _LAST_BAT_TRACKING_COLS = list(df.columns)
                continue

            # Find supporting columns (same data-density logic)
            bs_col = _pick_populated_col(bat_speed_candidates, df)
            if bs_col is None:
                _fuzzy = [c for c in df.columns
                          if "bat_speed" in c.lower() or "swing_speed" in c.lower()]
                bs_col = _pick_populated_col(_fuzzy, df)
            fs_col = _pick_populated_col(fast_swing_candidates, df)
            if fs_col is None:
                _fuzzy = [c for c in df.columns
                          if "fast_swing" in c.lower() or "hard_swing" in c.lower()]
                fs_col = _pick_populated_col(_fuzzy, df)
            sq_col = _pick_populated_col(squared_up_candidates, df)
            if sq_col is None:
                _fuzzy = [c for c in df.columns if "squared_up" in c.lower()]
                sq_col = _pick_populated_col(_fuzzy, df)
            # v43.36: ideal_attack_angle — try exact candidates, then fall
            # back to fuzzy match on "attack" + "angle" in column name.
            iaa_col = _pick_populated_col(ideal_aa_candidates, df)
            if iaa_col is None:
                _fuzzy = [c for c in df.columns
                          if ("attack" in c.lower() and "angle" in c.lower())
                          or "ideal_attack" in c.lower()]
                iaa_col = _pick_populated_col(_fuzzy, df)

            # v43.42: when IAA isn't found, log what Savant ACTUALLY returned
            # to a module-level diagnostic so we can verify the field name
            # offline. Without this we keep guessing field names blindly.
            # v43.59: `global _LAST_BAT_TRACKING_COLS` moved to function entry.
            if not iaa_col:
                _LAST_BAT_TRACKING_COLS = list(df.columns)

            # v43.57: hardier player_id detection — Savant may use a
            # different column name (mlbam_id, id, player, etc) depending on
            # endpoint. Previously we read df.get("player_id") which
            # returned None silently when the field had any other name,
            # making every output row's player_id NaN → dropna killed
            # everything → "no rows after filtering" status.
            pid_candidates = ["player_id", "mlbam_id", "id", "playerid",
                              "MLBAMID", "key_mlbam", "Player ID"]
            pid_col = next((c for c in pid_candidates if c in df.columns),
                           # Fuzzy match: any column with "id" or "mlbam" in lower
                           next((c for c in df.columns
                                 if "mlbam" in c.lower()
                                 or (c.lower().endswith("_id") and "team" not in c.lower())
                                 or c.lower() == "id"), None))

            # Build output frame with normalized column names
            out = pd.DataFrame()
            if pid_col:
                out["player_id"] = pd.to_numeric(df[pid_col], errors="coerce")
            else:
                out["player_id"] = pd.Series(dtype="float64")  # all NaN
            # v43.58 (critical fix): the previous `df.get("player_name") or
            # df.get("name") or df.get("Name")` chain raised ValueError
            # ("truth value of Series is ambiguous") when player_name was
            # missing but name was present. The `or` operator evaluates
            # bool() on each Series, which pandas refuses for len > 1.
            # ValueError got caught by outer except → continue → next URL
            # → same bug → "all endpoints failed". Use explicit None checks.
            for _name_col in ("player_name", "name", "Name"):
                if _name_col in df.columns:
                    out["player_name"] = df[_name_col]
                    break
            else:
                out["player_name"] = pd.Series(dtype="object")
            # v43.65 (reviewer-validated CRITICAL fix #9): Savant returns
            # `blast_per_swing` as a 0-1 DECIMAL (e.g. 0.05 means "5% of
            # swings are blasts"). The rest of the codebase uses 0-100
            # PERCENTAGE convention (barrel_pct, hard_hit, pull_pct all 0-100).
            # The mismatch made HR Criteria #4's threshold (5.0) structurally
            # unreachable — production export showed `hr_crit_iaa` = 0 for
            # all 532 hitters, max criteria_met = 3/4 ever. Multiply by 100
            # at the fetcher to restore convention consistency. App.py's
            # median imputation block is scale-agnostic (works on either
            # scale); only the threshold check in add_hr_criteria was
            # scale-sensitive, and that now sees 0-100 values matching the
            # documented ≥5 threshold.
            _raw_blast = pd.to_numeric(df[blast_col], errors="coerce")
            # Defensive: if Savant ever switches to percent (e.g. 5.0 not
            # 0.05), don't double-multiply. Sniff scale by median —
            # decimal-scale blasts have median ~0.03-0.06; percent-scale
            # would have median ~3-6. Threshold at 1.0 cleanly separates.
            try:
                _med = float(_raw_blast.dropna().median())
                if pd.notna(_med) and _med < 1.0:
                    out["blast_pct"] = (_raw_blast * 100).round(2)
                else:
                    out["blast_pct"] = _raw_blast.round(2)
            except Exception:
                # If sniff fails, default to multiply-by-100 (the empirically
                # observed Savant behavior as of June 2026)
                out["blast_pct"] = (_raw_blast * 100).round(2)
            if bs_col:
                out["bat_speed"] = pd.to_numeric(df[bs_col], errors="coerce")
            # v43.65: same scale normalization as blast_pct (Savant returns
            # 0-1 decimal for all per-swing/per-contact rates; codebase
            # convention is 0-100).
            def _to_percent_scale(col_name):
                """Sniff scale and multiply by 100 if it's a 0-1 decimal."""
                series = pd.to_numeric(df[col_name], errors="coerce")
                try:
                    _med = float(series.dropna().median())
                    if pd.notna(_med) and _med < 1.0:
                        return (series * 100).round(2)
                except Exception:
                    return (series * 100).round(2)
                return series.round(2)

            if fs_col:
                out["fast_swing_pct"] = _to_percent_scale(fs_col)
            if sq_col:
                out["squared_up_pct"] = _to_percent_scale(sq_col)
            if iaa_col:
                out["ideal_attack_angle_pct"] = _to_percent_scale(iaa_col)

            # v43.57: capture pre-dropna shape so a "no rows after filtering"
            # status tells us WHY. Without this, we knew the filter killed
            # everything but not what the input looked like.
            _pre_drop_rows = len(out)
            _pre_drop_pid_valid = int(out["player_id"].notna().sum()) if not out.empty else 0
            _pre_drop_blast_valid = int(out["blast_pct"].notna().sum()) if not out.empty else 0

            # Filter to rows with non-null blast_pct AND valid player_id
            out = out.dropna(subset=["blast_pct", "player_id"])
            out["player_id"] = out["player_id"].astype("Int64")

            if out.empty:
                # v43.57: log column-level failure detail so the app can
                # surface it via last_bat_tracking_columns(). The `global`
                # statement is already at the top of this function.
                _LAST_BAT_TRACKING_COLS = list(df.columns)
                _pid_matched = pid_col if pid_col else "(none — no MLBAM-like column found)"
                _reason = (
                    f"pre-dropna: {_pre_drop_rows} rows, "
                    f"player_id valid: {_pre_drop_pid_valid}, "
                    f"blast_pct valid: {_pre_drop_blast_valid}. "
                    f"Matched player_id col: '{_pid_matched}'. "
                    f"Matched blast col: '{blast_col}'."
                )
                return pd.DataFrame(), f"❌ no rows after filtering — {_reason}", list(df.columns)
            return out, f"✅ {len(out)} hitters", list(df.columns)
        except Exception as e:
            # Try next URL
            continue

    return pd.DataFrame(), "❌ all endpoints failed (Savant schema drift?)", []


# ----------------------------------------------------------------------------
# Safe parsers - handle MLB Stats API "-.--" and other junk values
# ----------------------------------------------------------------------------

def _safe_float(val):
    """Convert API value to float; return None for '-.--', '', or invalid."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            f = float(val)
            return f if not (f != f) else None  # NaN check
        except (TypeError, ValueError):
            return None
    s = str(val).strip()
    if s in ("", "-.--", "--", "-", ".---", "null", "None"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    """Convert API value to int; return None for '-.--', '', or invalid."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = str(val).strip()
    if s in ("", "-.--", "--", "-", "null", "None"):
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------------
# Shared normalizers - keep player_id types consistent across all sources
# ----------------------------------------------------------------------------

def _normalize_player_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure player_id is a consistent integer type so merges work.
    Also rename common column aliases that Savant uses inconsistently.
    """
    if df is None or df.empty:
        return df
    if "player_id" not in df.columns:
        for cand in ["mlb_id", "playerid", "MLBAMID", "mlbam_id", "player_mlb_id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "player_id"})
                break
    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")

    # Savant sometimes returns these under alternate names - normalize so
    # downstream code finds them under one canonical name.
    rename_map = {}
    for alias, canonical in [
        ("isolated_power", "iso"),
        ("la", "launch_angle"),
        ("ev", "launch_speed"),
        ("hardhit_percent", "hard_hit_percent"),
        ("barrel_percent", "barrel_batted_rate"),
        ("xba_diff", "xba_minus_ba_diff"),
    ]:
        if alias in df.columns and canonical not in df.columns:
            rename_map[alias] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _derive_hitter_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute commonly-missing hitter columns from real underlying stats.
    These are math identities (ISO = SLG - AVG), not fake defaults.
    """
    if df is None or df.empty:
        return df

    # ISO from SLG - AVG if missing or all-null
    needs_iso = "iso" not in df.columns or df["iso"].isna().all()
    if needs_iso:
        slg_col = next((c for c in ["slg", "slugging_percentage"] if c in df.columns), None)
        avg_col = next((c for c in ["batting_avg", "avg", "ba"] if c in df.columns), None)
        if slg_col and avg_col:
            df["iso"] = (df[slg_col] - df[avg_col]).round(3)

    return df


# ----------------------------------------------------------------------------
# Slate / probable pitchers (MLB Stats API)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=300)  # 5min — was 30min; probable pitchers change throughout the day
def get_slate(game_date: Optional[str] = None) -> pd.DataFrame:
    """
    Return one row per game for the given date (YYYY-MM-DD, default today).
    Columns: gamePk, gameTime, away_team, home_team, away_team_id, home_team_id,
             away_pitcher, home_pitcher, away_pitcher_id, home_pitcher_id, venue
    """
    if game_date is None:
        game_date = date.today().isoformat()

    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={game_date}"
        "&hydrate=probablePitcher,linescore,team"
    )
    # Retry with exponential backoff. MLB Stats API occasionally times out
    # under load — three quick retries instead of one long timeout reduces
    # the chance of a hard app failure.
    data = None
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            data = r.json()
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt < 2:
                import time as _time
                _time.sleep(0.5 * (2 ** attempt))  # 0.5s, then 1s
            continue
        except Exception as e:
            last_err = e
            break
    if data is None:
        # Raise so Streamlit DOESN'T cache an empty result. The app.py
        # caller catches this and shows a friendly error.
        raise ConnectionError(
            f"Unable to fetch MLB schedule for {game_date} after 3 attempts. "
            f"Last error: {type(last_err).__name__}"
        )

    rows = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away = g["teams"]["away"]
            home = g["teams"]["home"]
            rows.append({
                "gamePk": g["gamePk"],
                "gameTime": g.get("gameDate"),
                "status": g.get("status", {}).get("detailedState"),
                "venue": g.get("venue", {}).get("name"),
                "venue_id": g.get("venue", {}).get("id"),  # v43.26 — for park history
                "away_team": away["team"]["name"],
                "away_team_abbr": away["team"].get("abbreviation",
                                                   away["team"]["name"][:3].upper()),
                "away_team_id": away["team"]["id"],
                "home_team": home["team"]["name"],
                "home_team_abbr": home["team"].get("abbreviation",
                                                   home["team"]["name"][:3].upper()),
                "home_team_id": home["team"]["id"],
                "away_pitcher": (away.get("probablePitcher") or {}).get("fullName", "TBD"),
                "away_pitcher_id": (away.get("probablePitcher") or {}).get("id"),
                "home_pitcher": (home.get("probablePitcher") or {}).get("fullName", "TBD"),
                "home_pitcher_id": (home.get("probablePitcher") or {}).get("id"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["gameTime"] = pd.to_datetime(df["gameTime"], errors="coerce")
    return df


@st.cache_data(ttl=180)  # 3min — lineups are posted/changed throughout afternoon
def get_lineup(game_pk: int, side: str = "home") -> list[dict]:
    """
    Return the projected/actual lineup for a side ("home" or "away").
    Falls back to recent starters if lineup not yet posted.
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        boxscore = r.json().get("liveData", {}).get("boxscore", {})
        team = boxscore.get("teams", {}).get(side, {})
        batting_order = team.get("battingOrder", [])
        players = team.get("players", {})
        out = []
        for pid in batting_order:
            p = players.get(f"ID{pid}", {})
            person = p.get("person", {})
            out.append({
                "id": person.get("id"),
                "name": person.get("fullName"),
                "position": p.get("position", {}).get("abbreviation"),
                "bats": p.get("batSide", {}).get("code"),
            })
        return out
    except Exception:
        return []


@st.cache_data(ttl=900)  # 15min — catches mid-day call-ups, IL moves, demotions
def get_team_roster(team_id: int) -> list[dict]:
    """Active roster + 40-man roster for a team — used as lineup fallback.

    v38h fix: previously only pulled `/roster/active` (28-man), which silently
    excluded freshly-called-up players (Pinango, McAdoo, Crooks, fresh
    September promos) and players DFA'd or transferred mid-season but still
    appearing in recent lineups. Now pulls 40-man roster AND active roster,
    merges them with active flagged first.

    The MLB Stats API has multiple roster types:
      active        — 28-man currently active
      40Man         — full 40-man (includes AAA-stashed players)
      fullSeason    — all players who appeared this year (huge, noisy)

    We use active + 40Man combined. Players on 40-man but not active are
    typically AAA stashes that can be recalled any day — exactly the gap
    that causes Pinango/McAdoo-tier omissions.

    Hydrates with person.bats so bench players have batting handedness.
    """
    seen_ids = set()
    out = []

    def _fetch(roster_type):
        url = (
            f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster/{roster_type}"
            "?hydrate=person"
        )
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json().get("roster", [])
        except Exception:
            return []

    # Pull active first (these are the most reliable), then 40-man (catches AAA stashes)
    for roster_type in ("active", "40Man"):
        roster_data = _fetch(roster_type)
        for p in roster_data:
            pos = p.get("position", {}).get("abbreviation", "")
            if pos in ("P", "SP", "RP"):
                continue
            person = p.get("person", {}) or {}
            pid = person.get("id") or p.get("person", {}).get("id")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            bats = ((person.get("batSide") or {}).get("code")
                    or (p.get("batSide") or {}).get("code"))
            out.append({
                "id": pid,
                "name": person.get("fullName") or p["person"]["fullName"],
                "position": pos,
                "bats": bats,
                "roster_type": roster_type,  # diagnostic — "active" or "40Man"
            })
    return out


@st.cache_data(ttl=900)
def get_team_pitchers(team_id: int) -> list[dict]:
    """v42v: Active + 40-man pitcher roster for a team.

    Used by the Pitcher Lookup utility so users can scout any pitcher,
    not just today's probable starter. Mirrors get_team_roster's logic but
    INCLUDES pitchers (which get_team_roster excludes for the hitter
    fallback flow). Returns dicts with id, name, position (P/SP/RP),
    throws (R/L/?), and roster_type ("active" or "40Man").
    """
    seen_ids = set()
    out = []

    def _fetch(roster_type):
        url = (
            f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster/{roster_type}"
            "?hydrate=person"
        )
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json().get("roster", [])
        except Exception:
            return []

    for roster_type in ("active", "40Man"):
        roster_data = _fetch(roster_type)
        for p in roster_data:
            pos = p.get("position", {}).get("abbreviation", "")
            if pos not in ("P", "SP", "RP"):
                continue
            person = p.get("person", {}) or {}
            pid = person.get("id") or p.get("person", {}).get("id")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            throws = ((person.get("pitchHand") or {}).get("code")
                      or (p.get("pitchHand") or {}).get("code") or "?")
            out.append({
                "id": pid,
                "name": person.get("fullName") or p["person"]["fullName"],
                "position": pos,
                "throws": throws,
                "roster_type": roster_type,
            })
    return out


@st.cache_data(ttl=900)  # 15min refresh — catches IL moves within 15 min
def get_active_roster_ids(team_id: int) -> set:
    """
    Return set of player_ids currently on the 26-man ACTIVE roster.

    Used for hitter IL detection (v39g). The active roster updates within
    minutes of any IL move — much faster than the MLB transactions API
    which can lag 12-24 hours.

    A hitter showing in our slate but NOT in get_active_roster_ids means
    they're either on the IL, optioned, or DFA'd. All three states mean
    they shouldn't play tonight regardless of why our pipeline pulled them.

    Returns empty set on any failure (downstream treats empty as "skip
    this filter" rather than penalizing everyone).
    """
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster/active"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return {
            p["person"]["id"] for p in r.json().get("roster", [])
            if p.get("person", {}).get("id")
        }
    except Exception:
        return set()


@st.cache_data(ttl=3600)
def get_all_team_rosters(slate: pd.DataFrame) -> dict:
    """
    Returns {team_id: [hitter dicts]} for every team on today's slate.
    Used so users can scan any hitter, not just confirmed lineups.
    """
    rosters = {}
    team_ids = set()
    for _, g in slate.iterrows():
        team_ids.add(int(g["away_team_id"]))
        team_ids.add(int(g["home_team_id"]))
    for tid in team_ids:
        rosters[tid] = get_team_roster(tid)
    return rosters


@st.cache_data(ttl=3600)  # 1hr — schedule/lineups don't change for past games
def get_team_recent_starts(team_id: int, lookback_days: int = 14,
                            _cache_version: str = "v1") -> dict:
    """
    v42n: Returns {player_id: starts_count} for a team's recent games.

    Counts how many times each player appeared in the starting lineup over
    the past `lookback_days` days. Used by _fill_to_nine to detect real
    regular starters whose season PA is artificially low due to recent IL
    time (CJ Abrams, Willy Adames archetype) — they'd otherwise be
    demoted to the bench by the naive PA-sort.

    Implementation: single MLB Stats API call per team with
    `?hydrate=lineups` to get every game's lineup in one shot. Cached 1hr
    so repeat slates are free.

    Returns {} on any failure — caller treats empty dict as "no recent
    data, fall back to PA sort."

    Args:
      team_id: MLB team ID
      lookback_days: how far back to count (default 14)
      _cache_version: bump to force cache invalidation
    """
    from datetime import date as _date, timedelta as _td
    today = _date.today()
    start = today - _td(days=lookback_days)
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&teamId={team_id}"
        f"&startDate={start.isoformat()}&endDate={today.isoformat()}"
        f"&hydrate=lineups"
    )
    starts = {}
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            status = (game.get("status", {}) or {}).get("abstractGameState", "")
            # Only count completed games — in-progress / scheduled games
            # haven't generated meaningful starting-lineup data yet
            if status not in ("Final", "Live", "In Progress"):
                continue

            # Determine which side this team was on
            away_id = (game.get("teams", {}).get("away", {})
                            .get("team", {}).get("id"))
            home_id = (game.get("teams", {}).get("home", {})
                            .get("team", {}).get("id"))
            if away_id == team_id:
                side_key = "awayPlayers"
            elif home_id == team_id:
                side_key = "homePlayers"
            else:
                continue

            lineups = game.get("lineups", {}) or {}
            players = lineups.get(side_key, []) or []
            # Lineup hydration returns position players in batting order.
            # Each entry is a person dict — count this as a "start" for them.
            for p in players:
                pid = (p or {}).get("id")
                if pid is None:
                    continue
                starts[pid] = starts.get(pid, 0) + 1
    return starts


# ----------------------------------------------------------------------------
# Hitter season stats (Baseball Savant custom leaderboard)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_hitter_stats(season: int = CURRENT_SEASON, _stats_day: str = "") -> pd.DataFrame:
    """
    Pull season-level Statcast hitter stats with ALL columns needed for the
    dashboard: standard rates, expected stats, batted-ball quality, pulled
    contact, fly ball %, launch angle, swing/miss, etc.

    _stats_day is a cache-buster set by the caller (defaults to today's ET stats
    date via _stats_day_key()). Including it in the signature makes Streamlit
    treat a new day as a cache miss — so morning users get fresh stats without
    waiting for the 1-hour TTL.
    """
    # v43.66 (researcher framework): added avg_hit_distance (for Must-Have
    # threshold ≥315 ft / Nuclear ≥330 ft) and barrels (raw barrel COUNT,
    # used to derive a near_hr_est = max(0, barrels - home_run) for the
    # Nuclear "Near HR ≥3" threshold). Both come from the same Savant
    # custom leaderboard with no extra HTTP call.
    selections = (
        "pa,abs,hits,player_age,k_percent,bb_percent,woba,xwoba,xiso,xba,xslg,xobp,"
        "iso,babip,slg,obp,batting_avg,on_base_plus_slg,home_run,"
        "barrel_batted_rate,barrels,solidcontact_percent,flareburner_percent,"
        "poorlyunder_percent,poorlytopped_percent,poorlyweak_percent,"
        "hard_hit_percent,avg_best_speed,avg_hit_angle,launch_speed,launch_angle,"
        "avg_hit_distance,"
        "whiff_percent,swing_percent,sweet_spot_percent,xwobacon,wobacon,"
        "groundballs_percent,flyballs_percent,linedrives_percent,popups_percent,"
        "pull_percent,straightaway_percent,opposite_percent,"
        "pull_air_percent,straightaway_air_percent,opposite_air_percent,"
        "z_swing_percent,z_swing_miss_percent,oz_swing_percent,oz_swing_miss_percent,"
        "f_strike_percent,zone_percent"
    )
    url = (
        "https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={season}&type=batter&filter=&min=1&selections={selections}"
        "&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&csv=true"
    )
    # Savant occasionally times out under load. Retry with backoff.
    df = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < 2:
                import time as _time
                _time.sleep(1.0 * (2 ** attempt))
            continue
        except Exception:
            break
    if df is None or df.empty:
        # Raise so Streamlit doesn't cache empty. Caller in app.py shows warning.
        raise ConnectionError("Baseball Savant unreachable for hitter stats after 3 attempts")
    if "last_name, first_name" in df.columns:
        df["player_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
            if isinstance(s, str) and "," in s else s
        )
    if "player_id" not in df.columns:
        for cand in ["mlb_id", "playerid", "MLBAMID"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "player_id"})
                break
    # Compute PulledBrl% as approximation. Savant may return pull data under
    # several names depending on endpoint version.
    pull_col = None
    for cand in ["pull_air_percent", "pull_percent", "pull_pct", "pulled_percent"]:
        if cand in df.columns and df[cand].notna().any():
            pull_col = cand
            break
    if pull_col and "barrel_batted_rate" in df.columns:
        df["pulled_brl_pct"] = (
            pd.to_numeric(df[pull_col], errors="coerce") *
            pd.to_numeric(df["barrel_batted_rate"], errors="coerce") / 100
        ).round(2)

    # CRITICAL: Always derive ISO at source if missing/null.
    # Savant's leaderboard sometimes returns 'iso' as an empty column even
    # when requested. SLG-AVG is the math definition of ISO.
    if "iso" not in df.columns or df["iso"].isna().all():
        for slg_col in ["slg", "xslg", "slugging_pct"]:
            if slg_col not in df.columns:
                continue
            for avg_col in ["batting_avg", "avg", "ba", "xba"]:
                if avg_col not in df.columns:
                    continue
                df["iso"] = (df[slg_col] - df[avg_col]).round(3)
                break
            if "iso" in df.columns and df["iso"].notna().any():
                break

    # Clamp ISO at zero. ISO = SLG - AVG, which is mathematically ≥ 0 because
    # every hit contributes to SLG with weight ≥ AVG-weight (a single counts
    # 1 in both, a double counts 2 in SLG vs 1 in AVG, etc). Negative values
    # only appear when Savant returns xISO for tiny samples (expected stats
    # can invert at low PA). Affects ~9 fringe hitters with PA < 80 — they're
    # already excluded from picks via the PA gate, but the displayed value
    # looks like a data error. Clamping fixes the display without altering
    # downstream calculations (negatives never reach the projections).
    if "iso" in df.columns:
        df["iso"] = pd.to_numeric(df["iso"], errors="coerce").clip(lower=0)

    # LA must come from a real Statcast source. NEVER estimated/derived.
    # Try every known column name where Savant might return real LA data.
    la_candidates = [
        "launch_angle", "launch_angle_avg", "avg_hit_angle",
        "la", "la_avg", "attack_angle", "swing_path_tilt",
        "average_la", "avg_la",
    ]
    populated_la = None
    for cand in la_candidates:
        if cand not in df.columns:
            continue
        coerced = pd.to_numeric(df[cand], errors="coerce")
        if coerced.notna().any():
            df[cand] = coerced
            populated_la = cand
            break

    if populated_la and populated_la != "launch_angle":
        df["launch_angle"] = df[populated_la]
    elif populated_la is None and "launch_angle" not in df.columns:
        df["launch_angle"] = pd.NA
    # Real LA only - no FB%/GB% estimation

    # If LA still missing, try a sequence of Savant endpoints known to return it.
    # Each is independent and uses a different URL pattern.
    # v43.68 (production-validated): this same exit-velocity endpoint also
    # returns avg_hit_distance and raw `barrels` count — fields that the
    # custom leaderboard at the top of this function does NOT expose. So we
    # extend this loop to ALSO populate avg_dist and barrel_count when present,
    # even if launch_angle is already populated. Researcher's Nuclear filter
    # needs these for ~3 of its 14 criteria; without them Nuclear grades go dark.
    # Pre-check: are we missing any of {LA, avg_dist, barrel_count}? If so,
    # hit the endpoint.
    needs_la = (
        "launch_angle" in df.columns and df["launch_angle"].isna().all()
    )
    needs_dist = (
        "avg_dist" not in df.columns
        or df.get("avg_dist", pd.Series(dtype=float)).isna().all()
    )
    needs_barrels = (
        "barrel_count" not in df.columns
        or df.get("barrel_count", pd.Series(dtype=float)).isna().all()
    )
    if needs_la or needs_dist or needs_barrels:
        # v43.69 (production-debug): per-URL diagnostic so we can see what's
        # happening when the merge produces all-NaN.
        global _LAST_DISTANCE_DIAG
        _LAST_DISTANCE_DIAG = []
        la_urls = [
            # v43.69: try the canonical Statcast leaderboard URL FIRST with
            # an empty min so we don't accidentally filter out qualified
            # hitters. The previous URL had `min=1` which Savant interprets
            # variably depending on the leaderboard type.
            f"https://baseballsavant.mlb.com/leaderboard/statcast"
            f"?type=batter&year={season}&team=&min=&csv=true",
            # Original exit-velocity + barrels endpoint
            f"https://baseballsavant.mlb.com/leaderboard/statcast"
            f"?type=batter&year={season}&position=&team=&min=1&csv=true",
            # Custom leaderboard with explicit launch_angle selection
            f"https://baseballsavant.mlb.com/leaderboard/custom"
            f"?year={season}&type=batter&filter=&min=1"
            f"&selections=pa,launch_angle,launch_speed,avg_hit_angle"
            f"&chart=false&x=pa&y=pa&r=no&csv=true",
            # Legacy exit-velocity leaderboard
            f"https://baseballsavant.mlb.com/leaderboard/exit-velocity"
            f"?type=batter&year={season}&min=1&csv=true",
        ]
        # Cast df["player_id"] to a stable type for dict lookups. Previous
        # bug: mixing Int64 (from ev_df) and float (from df) made map() miss
        # every row even when the data was present.
        try:
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
        except Exception:
            pass

        for url in la_urls:
            diag_entry = {
                "url": url[:100] + "..." if len(url) > 100 else url,
                "status": None,
                "n_rows": 0,
                "cols_sample": [],
                "id_col": None,
                "la_col": None,
                "dist_col": None,
                "brl_col": None,
                "dist_merged": 0,
                "brl_merged": 0,
                "la_merged": 0,
                "error": None,
            }
            try:
                rr = requests.get(url, headers=HEADERS, timeout=20)
                diag_entry["status"] = rr.status_code
                rr.raise_for_status()
                ev_df = pd.read_csv(io.StringIO(rr.text))
                diag_entry["n_rows"] = len(ev_df)
                diag_entry["cols_sample"] = list(ev_df.columns)[:25]
                if ev_df.empty:
                    _LAST_DISTANCE_DIAG.append(diag_entry)
                    continue
                # Find ID column once for all merges below
                id_col = None
                for cand in ["player_id", "mlb_id", "MLBAMID", "playerid", "id"]:
                    if cand in ev_df.columns:
                        id_col = cand
                        break
                diag_entry["id_col"] = id_col
                if not id_col:
                    _LAST_DISTANCE_DIAG.append(diag_entry)
                    continue
                ev_df[id_col] = pd.to_numeric(
                    ev_df[id_col], errors="coerce"
                ).astype("Int64")

                # v43.72 (production-debug): switched from dict.map() to
                # pd.merge() because Int64 + dict lookup was silently failing
                # for the slate's actual lineup player_ids. The .map worked
                # for 254 df rows but the 254 were NOT the lineup hitters —
                # a pandas Int64 hash mismatch issue. pd.merge handles this
                # correctly because it uses pandas' join semantics with proper
                # dtype harmonization.

                # Build a small subset to merge
                merge_cols = [id_col]
                merge_renames = {id_col: "player_id"}

                # ----- LA (existing logic, unchanged at top) -----
                if needs_la:
                    la_col = None
                    for cand in ["launch_angle", "avg_hit_angle", "angle",
                                  "avg_launch_angle", "avg_la", "la"]:
                        if cand in ev_df.columns:
                            coerced = pd.to_numeric(ev_df[cand], errors="coerce")
                            if coerced.notna().any():
                                ev_df[cand] = coerced
                                la_col = cand
                                break
                    diag_entry["la_col"] = la_col
                    if la_col:
                        merge_cols.append(la_col)
                        merge_renames[la_col] = "_la_from_ev"

                # ----- v43.68: Avg Distance -----
                if needs_dist:
                    dist_col = None
                    for cand in ["avg_hit_distance", "avg_distance",
                                  "distance", "hit_distance", "avg_dist"]:
                        if cand in ev_df.columns:
                            coerced = pd.to_numeric(ev_df[cand], errors="coerce")
                            if coerced.notna().any():
                                ev_df[cand] = coerced
                                dist_col = cand
                                break
                    diag_entry["dist_col"] = dist_col
                    if dist_col:
                        merge_cols.append(dist_col)
                        merge_renames[dist_col] = "_dist_from_ev"

                # ----- v43.68: Barrels (raw count) -----
                if needs_barrels:
                    brl_col = None
                    for cand in ["barrels", "barrel", "n_barrels",
                                  "barrels_total", "barrel_count"]:
                        if cand in ev_df.columns:
                            coerced = pd.to_numeric(ev_df[cand], errors="coerce")
                            if coerced.notna().any():
                                ev_df[cand] = coerced
                                brl_col = cand
                                break
                    diag_entry["brl_col"] = brl_col
                    if brl_col:
                        merge_cols.append(brl_col)
                        merge_renames[brl_col] = "_brl_from_ev"

                # If we picked up at least one mappable column, do a proper merge
                if len(merge_cols) > 1:
                    ev_subset = ev_df[merge_cols].copy().rename(columns=merge_renames)
                    # Drop any rows with NA player_id (they can't merge)
                    ev_subset = ev_subset.dropna(subset=["player_id"])
                    # Deduplicate on player_id (keep last) to handle Savant duplicates
                    ev_subset = ev_subset.drop_duplicates(subset=["player_id"], keep="last")
                    # Merge — pandas handles Int64 alignment correctly
                    df = df.merge(ev_subset, on="player_id", how="left")

                    # Now copy the merged temp columns into the canonical names
                    if needs_la and "_la_from_ev" in df.columns:
                        # Only overwrite NaN cells in launch_angle; preserve any
                        # values already present from primary fetch
                        if "launch_angle" in df.columns:
                            df["launch_angle"] = df["launch_angle"].fillna(df["_la_from_ev"])
                        else:
                            df["launch_angle"] = df["_la_from_ev"]
                        df = df.drop(columns=["_la_from_ev"])
                        diag_entry["la_merged"] = int(df["launch_angle"].notna().sum())
                        if df["launch_angle"].notna().any():
                            needs_la = False

                    if needs_dist and "_dist_from_ev" in df.columns:
                        df["avg_dist"] = df["_dist_from_ev"].round(0)
                        df = df.drop(columns=["_dist_from_ev"])
                        diag_entry["dist_merged"] = int(df["avg_dist"].notna().sum())
                        if df["avg_dist"].notna().any():
                            needs_dist = False

                    if needs_barrels and "_brl_from_ev" in df.columns:
                        df["barrel_count"] = df["_brl_from_ev"].round(0)
                        df = df.drop(columns=["_brl_from_ev"])
                        diag_entry["brl_merged"] = int(df["barrel_count"].notna().sum())
                        if df["barrel_count"].notna().any():
                            needs_barrels = False

                _LAST_DISTANCE_DIAG.append(diag_entry)
                # If everything's populated, stop trying URLs
                if not (needs_la or needs_dist or needs_barrels):
                    break
            except Exception as _url_err:
                diag_entry["error"] = f"{type(_url_err).__name__}: {str(_url_err)[:120]}"
                _LAST_DISTANCE_DIAG.append(diag_entry)
                continue

        # v43.69: also stash to session_state so the diag survives the
        # @st.cache_data wrapper on next call (cache hit doesn't run this code).
        try:
            st.session_state["_last_distance_diag"] = list(_LAST_DISTANCE_DIAG)
        except Exception:
            pass

    # Normalize player_id type for merges + derive missing columns
    df = _normalize_player_df(df)
    df = _derive_hitter_missing(df)

    # Clean up float precision artifacts at source so downstream code doesn't
    # need to repeat rounding. Different metrics get different precision.
    # NOTE: previously this dict had TWO `1:` keys, which silently overwrote
    # the first — meaning all percentage columns (k_percent, barrel_batted_rate,
    # hard_hit_percent, etc.) weren't being rounded at all. Now merged.
    round_specs = {
        # Ratios stored as 0-1 → 3 decimals
        3: ["iso", "xwoba", "xwobacon", "xslg", "xobp", "xba", "slg", "obp",
            "ops", "batting_avg", "babip", "avg", "woba"],
        # Percentages (0-100) and velocities/angles → 1 decimal
        1: ["k_percent", "bb_percent", "whiff_percent", "swing_percent",
            "barrel_batted_rate", "hard_hit_percent", "sweet_spot_percent",
            "flyballs_percent", "groundballs_percent", "linedrives_percent",
            "popups_percent", "pull_percent", "pull_air_percent",
            "straightaway_percent", "opposite_percent", "z_swing_percent",
            "oz_swing_percent", "f_strike_percent", "zone_percent",
            "csw_percent",
            "launch_speed", "launch_angle", "avg_best_speed", "avg_hit_angle"],
    }
    for digits, cols in round_specs.items():
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(digits)

    # v42s: defensive numeric coercion + outlier clipping
    # Catches Savant edge cases where columns occasionally contain string
    # junk values or extreme outliers from tiny-sample players.
    df = enforce_numeric(df, [
        "barrel_batted_rate", "hard_hit_percent", "iso", "xwoba", "xwobacon",
        "launch_angle", "launch_speed", "pull_percent",
        "flyballs_percent", "groundballs_percent", "linedrives_percent",
        "k_percent", "bb_percent", "avg_best_speed",
    ])
    df = clip_outliers(df)
    return df


# ----------------------------------------------------------------------------
# Pitcher season stats
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_pitcher_stats(season: int = CURRENT_SEASON, _stats_day: str = "") -> pd.DataFrame:
    """Season-level Statcast pitcher stats - expanded.

    Now requests stats that let us derive ERA-equivalents independently of
    MLB Stats API: xera, slg (SLG against), obp (OBP against), hits, walks,
    earned_runs, home_run, etc. This way if MLB Stats API is unreachable
    (which happens on Streamlit Cloud), we still have everything we need.
    """
    selections = (
        # Core Statcast
        "pa,k_percent,bb_percent,woba,xwoba,xiso,xba,xslg,xobp,xera,"
        "barrel_batted_rate,hard_hit_percent,avg_best_speed,avg_hit_angle,"
        "whiff_percent,swing_percent,sweet_spot_percent,xwobacon,iso,babip,"
        "launch_speed,launch_angle,p_total_pitches,p_total_swinging_strike,"
        "csw_percent,zone_percent,in_zone_swing_miss_percent,"
        "f_strike_percent,oz_swing_percent,z_swing_percent,"
        "groundballs_percent,flyballs_percent,linedrives_percent,popups_percent,"
        "pull_percent,straightaway_percent,opposite_percent,"
        # Real outcome stats (ERA-equivalent derivation)
        "home_run,slg,obp,batting_avg,hits,walks_drawn,earned_runs,"
        "ab,strikeout,plate_appearances"
    )
    url = (
        "https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={season}&type=pitcher&filter=&min=1&selections={selections}"
        "&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&csv=true"
    )
    # Retry with backoff for transient Savant timeouts
    df = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < 2:
                import time as _time
                _time.sleep(1.0 * (2 ** attempt))
            continue
        except Exception:
            break
    if df is None or df.empty:
        raise ConnectionError("Baseball Savant unreachable for pitcher stats after 3 attempts")
    if "last_name, first_name" in df.columns:
        df["player_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
            if isinstance(s, str) and "," in s else s
        )
    df = _normalize_player_df(df)

    # ------------------------------------------------------------------
    # SAVANT-DERIVED ERA/WHIP equivalents - work even when MLB Stats API dies
    # ------------------------------------------------------------------
    # Statcast 'pa' is plate appearances faced. ~4.3 PA per IP league average.
    if "pa" in df.columns:
        df["ip_savant"] = (df["pa"] / 4.3).round(1)
        # If MLB Stats API gave us real IP later, that will overwrite this.
        # For now, use ip_savant as the IP value.
        df["ip"] = df["ip_savant"]
    # K/9 from K% × 4.3 PA per IP
    if "k_percent" in df.columns:
        df["k9_savant"] = (df["k_percent"] * 4.3 * 9 / 100).round(2)
        df["k9"] = df["k9_savant"]
    # BB/9 from BB% × 4.3 PA per IP
    if "bb_percent" in df.columns:
        df["bb9_savant"] = (df["bb_percent"] * 4.3 * 9 / 100).round(2)
        df["bb9"] = df["bb9_savant"]
    # WHIP from (H + BB) / IP - if Savant gave us hits and walks
    # Cap at 5.0 to prevent absurd values from tiny IP samples (Savant IP is
    # estimated from PA which can be off by ~30% in small samples).
    if "hits" in df.columns and "walks_drawn" in df.columns and "ip_savant" in df.columns:
        whip_raw = (df["hits"] + df["walks_drawn"]) / df["ip_savant"].replace(0, pd.NA)
        df["whip_savant"] = whip_raw.clip(upper=5.0).round(2)
        df["whip"] = df["whip_savant"]
    # ERA from earned_runs × 9 / IP
    # Cap at 12.0: Savant IP is estimated from PA (pa / 4.3) which gets
    # inflated for tiny samples. Jordan Wicks case (May 2026): 24 PA gave
    # ip_savant=5.58 but real IP was 4.1, making ERA derive to 16.62 vs
    # real ~12.4. The cap prevents the display from showing nonsense values.
    # Real ERA gets prioritized in the coalesce step (app.py) when available.
    if "earned_runs" in df.columns and "ip_savant" in df.columns:
        era_raw = df["earned_runs"] * 9 / df["ip_savant"].replace(0, pd.NA)
        df["era_savant"] = era_raw.clip(upper=12.0).round(2)
        df["era"] = df["era_savant"]
    # HR/9 from home_run × 9 / IP — cap at 6.0 for the same reason
    if "home_run" in df.columns and "ip_savant" in df.columns:
        hr9_raw = df["home_run"] * 9 / df["ip_savant"].replace(0, pd.NA)
        df["hr9_savant"] = hr9_raw.clip(upper=6.0).round(2)
        df["hr9"] = df["hr9_savant"]

    # v42s: defensive numeric coercion + outlier clipping (same pattern as hitter side)
    df = enforce_numeric(df, [
        "era", "whip", "hr9", "k9", "bb9",
        "barrel_batted_rate", "hard_hit_percent",
        "launch_angle", "launch_speed", "avg_best_speed", "xera",
        "k_percent", "bb_percent",
    ])
    df = clip_outliers(df)
    return df


# ----------------------------------------------------------------------------
# Pitcher splits vs LHB / vs RHB - critical for handedness matchup analysis
# ----------------------------------------------------------------------------

def _fetch_pitcher_splits_single(pitcher_id: int, season: int) -> dict:
    """
    Fetch a single pitcher's vs-LHB and vs-RHB splits from MLB Stats API.
    Returns dict with keys like vs_lhb_pa, vs_lhb_hr_per_pa, vs_lhb_avg, etc.

    MLB Stats API field names vary; try several variations and derive
    HR% / K% from atBats if plateAppearances isn't returned.
    """
    out = {}
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
        f"?stats=statSplits&sitCodes=vl,vr&group=pitching&season={season}"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return out
        data = r.json()
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                code = split.get("split", {}).get("code", "")
                if code == "vl":
                    label = "lhb"
                elif code == "vr":
                    label = "rhb"
                else:
                    continue
                stat = split.get("stat", {})

                # Try multiple field names (MLB API has variations)
                pa = (stat.get("plateAppearances") or stat.get("battersFaced")
                      or stat.get("totalBattersFaced") or stat.get("pa"))
                ab = stat.get("atBats") or stat.get("ab")
                hr = stat.get("homeRuns") or stat.get("hr")
                k = stat.get("strikeOuts") or stat.get("strikeouts") or stat.get("so")
                bb = (stat.get("baseOnBalls") or stat.get("walks")
                      or stat.get("bb") or stat.get("baseonballs"))
                avg = stat.get("avg") or stat.get("battingAverage")
                obp = stat.get("obp") or stat.get("onBasePercentage")
                slg = stat.get("slg") or stat.get("sluggingPercentage")
                hits = stat.get("hits") or stat.get("h")

                # Type-cast counts safely
                def _to_int(v):
                    if v is None: return None
                    try: return int(v)
                    except (TypeError, ValueError):
                        try: return int(float(v))
                        except (TypeError, ValueError): return None

                pa = _to_int(pa)
                ab = _to_int(ab)
                hr = _to_int(hr)
                k = _to_int(k)
                bb = _to_int(bb)
                hits = _to_int(hits)

                # If PA is missing but we have AB and BB, derive: PA ≈ AB + BB
                # (ignoring HBP/SF which are small).
                if (pa is None or pa == 0) and ab and bb is not None:
                    pa = ab + bb

                # If still no PA, fall back to AB for denominator (less precise)
                denom = pa if (pa and pa > 0) else ab

                if pa and pa > 0:
                    out[f"vs_{label}_pa"] = pa
                if denom and denom > 0:
                    # If HR field exists, use it. If MISSING but split has decent
                    # sample (PA>=20) AND SLG is modest (<.400), assume 0 HRs
                    # (the API often omits zero-count fields for "good" pitchers
                    # who gave up no HRs to one side). This was the Yesavage bug:
                    # 54 PA, .269 SLG, no HR field = should be 0% not NaN.
                    if hr is not None:
                        out[f"vs_{label}_hr_per_pa"] = round(hr / denom * 100, 3)
                        # v43.11: persist HR count so "3 HRs allowed to LHB in
                        # 180 PA" is visible alongside the rate. Helps spot
                        # situational vulnerability that the rate-only signal
                        # buries (e.g., 4 HRs in 35 vs-LHB PA is a real signal
                        # even though it's the same rate as 2 HRs in 17 PA).
                        out[f"vs_{label}_hr"] = hr
                    # v40: If HR field is ABSENT, leave vs_*_hr_per_pa unset.
                    # Previously fabricated 0.0 when slg<.320 — but a missing
                    # field is an ambiguous API omission, and a fake 0%
                    # propagates into _platoon_hr_flag (creating spurious 💥
                    # vulnerability flags + +4 pick_score bonus to opposing
                    # hitters) and the display. props.py separately derives
                    # a real low rate from SLG for pitcher_mult, so this
                    # doesn't hurt the projection path.
                    if k is not None:
                        out[f"vs_{label}_k_percent"] = round(k / denom * 100, 2)
                    if bb is not None:
                        out[f"vs_{label}_bb_percent"] = round(bb / denom * 100, 2)
                if avg:
                    try: out[f"vs_{label}_avg"] = float(avg)
                    except (TypeError, ValueError): pass
                if obp:
                    try: out[f"vs_{label}_obp"] = float(obp)
                    except (TypeError, ValueError): pass
                if slg:
                    try: out[f"vs_{label}_slg"] = float(slg)
                    except (TypeError, ValueError): pass
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600)
def get_pitcher_handedness_splits(season: int = CURRENT_SEASON,
                                    pitcher_ids: tuple = (),
                                    _cache_version: str = "v1") -> pd.DataFrame:
    """
    Fetch vs-LHB and vs-RHB splits from MLB Stats API for the given pitcher IDs.

    Why MLB Stats API instead of Savant: Savant's /custom endpoint doesn't
    accept hfStands param reliably; statSplits via the MLB API IS documented
    and stable.

    Pass the list of pitcher_ids from today's slate to avoid fetching
    splits for every pitcher in the league (~700 calls).

    _cache_version: bump when adding new fields to invalidate cache.
    """
    if not pitcher_ids:
        return pd.DataFrame()

    rows = []
    for pid in pitcher_ids:
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            continue
        splits = _fetch_pitcher_splits_single(pid_int, season)
        if splits:
            splits["player_id"] = pid_int
            rows.append(splits)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# HITTER splits vs LHP / vs RHP - the SINGLE biggest accuracy gap in the model
# until June 2026. Reference apps that nail platoon predictions all use these.
#
# Jo Adell case study: overall barrel% = 8.8% (looks weak) but vs-LHP barrel% =
# 31.6% (elite). Without these splits, the model uses the 8.8% for ALL matchups
# and dramatically underestimates Adell vs lefties. Same for any reverse-platoon
# or strong-platoon hitter.
# ----------------------------------------------------------------------------

def _fetch_hitter_splits_single(hitter_id: int, season: int) -> dict:
    """
    Fetch a single hitter's vs-LHP and vs-RHP splits from MLB Stats API.
    Returns dict with keys like vs_lhp_pa, vs_lhp_hr_per_pa, vs_lhp_iso, etc.

    Field naming convention: `vs_lhp_*` / `vs_rhp_*` (vs pitcher hand).
    Note this differs from pitcher splits which use `vs_lhb_*` / `vs_rhb_*`
    (vs batter hand). The codes from MLB API are the same (vl/vr) but the
    semantic is opposite depending on the player type.
    """
    out = {}
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{hitter_id}/stats"
        f"?stats=statSplits&sitCodes=vl,vr&group=hitting&season={season}"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return out
        data = r.json()
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                code = split.get("split", {}).get("code", "")
                # For HITTERS: vl = vs LHP, vr = vs RHP
                if code == "vl":
                    label = "lhp"
                elif code == "vr":
                    label = "rhp"
                else:
                    continue
                stat = split.get("stat", {})

                def _to_int(v):
                    if v is None: return None
                    try: return int(v)
                    except (TypeError, ValueError):
                        try: return int(float(v))
                        except (TypeError, ValueError): return None

                def _to_float(v):
                    if v is None: return None
                    try: return float(v)
                    except (TypeError, ValueError): return None

                pa = _to_int(stat.get("plateAppearances") or stat.get("pa"))
                ab = _to_int(stat.get("atBats") or stat.get("ab"))
                hr = _to_int(stat.get("homeRuns") or stat.get("hr"))
                k = _to_int(stat.get("strikeOuts") or stat.get("so"))
                bb = _to_int(stat.get("baseOnBalls") or stat.get("bb"))
                hits = _to_int(stat.get("hits") or stat.get("h"))
                avg = _to_float(stat.get("avg") or stat.get("battingAverage"))
                obp = _to_float(stat.get("obp") or stat.get("onBasePercentage"))
                slg = _to_float(stat.get("slg") or stat.get("sluggingPercentage"))
                ops = _to_float(stat.get("ops"))

                if pa is None and ab and bb is not None:
                    pa = ab + bb
                denom = pa if (pa and pa > 0) else ab

                if pa and pa > 0:
                    out[f"vs_{label}_pa"] = pa
                if denom and denom > 0:
                    if hr is not None:
                        out[f"vs_{label}_hr_per_pa"] = round(hr / denom * 100, 3)
                        # v43.11: persist the raw HR count too. Rate alone hides
                        # whether "3% vs LHP" is "1 HR in 33 PA" (noise) or
                        # "12 HRs in 400 PA" (real platoon signal).
                        out[f"vs_{label}_hr"] = hr
                    if k is not None:
                        out[f"vs_{label}_k_percent"] = round(k / denom * 100, 2)
                    if bb is not None:
                        out[f"vs_{label}_bb_percent"] = round(bb / denom * 100, 2)
                if avg is not None:
                    out[f"vs_{label}_avg"] = avg
                if obp is not None:
                    out[f"vs_{label}_obp"] = obp
                if slg is not None:
                    out[f"vs_{label}_slg"] = slg
                    # ISO = SLG - AVG (the math definition)
                    if avg is not None:
                        out[f"vs_{label}_iso"] = round(max(0.0, slg - avg), 3)
                if ops is not None:
                    out[f"vs_{label}_ops"] = ops
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600)
def get_hitter_handedness_splits(season: int = CURRENT_SEASON,
                                   hitter_ids: tuple = (),
                                   _cache_version: str = "v1") -> pd.DataFrame:
    """
    Fetch vs-LHP and vs-RHP splits from MLB Stats API for the given hitter IDs.

    Why this matters: a hitter's overall season stats average across BOTH
    handednesses. For platoon hitters (most of MLB) the rate vs the opposite
    arm is meaningfully different. Reverse-platoon guys like Jo Adell can have
    overall barrel% of 8% but vs-LHP barrel% of 31% — without the split,
    we project him as a weak HR threat vs lefties when he's actually elite.

    Returns DataFrame with columns:
      player_id, vs_lhp_pa, vs_lhp_avg, vs_lhp_obp, vs_lhp_slg, vs_lhp_iso,
      vs_lhp_ops, vs_lhp_hr_per_pa, vs_lhp_hr, vs_lhp_k_percent, vs_lhp_bb_percent
      (and matching vs_rhp_* columns; vs_lhp_hr added v43.11 — raw HR count)

    Pass hitter_ids from today's slate to avoid hammering the API.
    """
    if not hitter_ids:
        return pd.DataFrame()

    rows = []
    for hid in hitter_ids:
        try:
            hid_int = int(hid)
        except (TypeError, ValueError):
            continue
        splits = _fetch_hitter_splits_single(hid_int, season)
        if splits:
            splits["player_id"] = hid_int
            rows.append(splits)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Day/Night splits — hitters AND pitchers perform differently in day vs night.
# Documented effect: most batters slightly worse in day games (~1-3% lower OPS).
# Some pitchers significantly better at night under lights.
# ----------------------------------------------------------------------------

def _parse_stat_split_response(data: dict, group: str = "hitting") -> dict:
    """
    Parse MLB Stats API statSplits response into vs_day_*/vs_night_* fields.
    Handles missing fields gracefully (returns whatever's available).
    Used for both hitters and pitchers.

    Recognizes splits by code (d/n) OR by description ("Day"/"Night") since
    MLB API responses vary.
    """
    out = {}
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            split_meta = split.get("split", {})
            code = (split_meta.get("code") or "").lower()
            desc = (split_meta.get("description") or "").lower()
            # Identify day vs night by code OR description
            label = None
            if code == "d" or "day" in desc:
                label = "day"
            elif code == "n" or "night" in desc:
                label = "night"
            # vs LHB / vs RHB
            elif code == "vl" or "vs. lhb" in desc or "left" in desc:
                label = "lhb"
            elif code == "vr" or "vs. rhb" in desc or "right" in desc:
                label = "rhb"
            else:
                continue
            stat = split.get("stat", {})

            def _to_int(v):
                if v is None: return None
                try: return int(v)
                except (TypeError, ValueError):
                    try: return int(float(v))
                    except (TypeError, ValueError): return None

            pa = (stat.get("plateAppearances") or stat.get("battersFaced")
                  or stat.get("pa"))
            ab = stat.get("atBats")
            hr = stat.get("homeRuns")
            k = stat.get("strikeOuts") or stat.get("strikeouts")
            bb = stat.get("baseOnBalls") or stat.get("walks")
            avg = stat.get("avg")
            obp = stat.get("obp")
            slg = stat.get("slg")
            ops = stat.get("ops")

            pa = _to_int(pa); ab = _to_int(ab); hr = _to_int(hr)
            k = _to_int(k); bb = _to_int(bb)

            # Derive PA from AB+BB if missing
            if (pa is None or pa == 0) and ab and bb is not None:
                pa = ab + bb
            denom = pa if (pa and pa > 0) else ab

            if pa and pa > 0:
                out[f"vs_{label}_pa"] = pa
            if denom and denom > 0:
                if hr is not None:
                    out[f"vs_{label}_hr_per_pa"] = round(hr / denom * 100, 3)
                # v40: Absent HR field → leave unset (no fabricated 0.0). A
                # manufactured 0% reads as elite suppression downstream and
                # corrupts _platoon_hr_flag (which divides r_safe/l_safe with
                # min 0.001) — producing 40× ratios that trigger spurious
                # 💥 RHB-vulnerability flags and +4 pick_score bonuses.
                # Missing stays missing.
                if k is not None:
                    out[f"vs_{label}_k_percent"] = round(k / denom * 100, 2)
            if avg:
                try: out[f"vs_{label}_avg"] = float(avg)
                except (TypeError, ValueError): pass
            if obp:
                try: out[f"vs_{label}_obp"] = float(obp)
                except (TypeError, ValueError): pass
            if slg:
                try: out[f"vs_{label}_slg"] = float(slg)
                except (TypeError, ValueError): pass
            if ops:
                try: out[f"vs_{label}_ops"] = float(ops)
                except (TypeError, ValueError): pass
    return out


def _fetch_day_night_splits_single(player_id: int, season: int, group: str = "hitting") -> dict:
    """Fetch day/night splits for one player (hitter or pitcher)."""
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=statSplits&sitCodes=d,n&group={group}&season={season}"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return {}
        return _parse_stat_split_response(r.json(), group)
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def get_hitter_day_night_splits(season: int = CURRENT_SEASON,
                                  hitter_ids: tuple = (),
                                  _cache_version: str = "v2") -> pd.DataFrame:
    """
    Fetch day/night splits for the given hitter IDs.
    Only fetches for confirmed lineup spots to avoid 200+ API calls.

    _cache_version: bump when adding new fields to invalidate Streamlit cache.
    """
    if not hitter_ids:
        return pd.DataFrame()
    rows = []
    for pid in hitter_ids:
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            continue
        splits = _fetch_day_night_splits_single(pid_int, season, "hitting")
        if splits:
            splits["player_id"] = pid_int
            rows.append(splits)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def get_pitcher_day_night_splits(season: int = CURRENT_SEASON,
                                   pitcher_ids: tuple = (),
                                   _cache_version: str = "v2") -> pd.DataFrame:
    """
    Fetch day/night splits for the given pitcher IDs.

    _cache_version: bump when adding new fields to invalidate Streamlit cache.
    """
    if not pitcher_ids:
        return pd.DataFrame()
    rows = []
    for pid in pitcher_ids:
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            continue
        splits = _fetch_day_night_splits_single(pid_int, season, "pitching")
        if splits:
            splits["player_id"] = pid_int
            rows.append(splits)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def classify_game_day_night(game_time_iso: str, venue_timezone: str = None) -> str:
    """
    Classify a game as 'day' or 'night' based on its LOCAL start time.
    MLB convention: games starting BEFORE 5 PM venue-local are day games.

    venue_timezone: optional IANA timezone string (e.g. "America/Phoenix").
        If not provided, defaults to "America/New_York" which is WRONG for
        west-coast venues. Callers should always pass the venue timezone.

    Returns 'day' or 'night'. Defaults to 'night' if classification fails.
    """
    try:
        from datetime import datetime
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            ZoneInfo = None
        # Parse ISO time (e.g., "2026-05-26T19:05:00Z")
        if game_time_iso.endswith("Z"):
            dt_utc = datetime.fromisoformat(game_time_iso[:-1] + "+00:00")
        else:
            dt_utc = datetime.fromisoformat(game_time_iso)
        tz_to_use = venue_timezone or "America/New_York"
        if ZoneInfo is not None:
            try:
                dt_local = dt_utc.astimezone(ZoneInfo(tz_to_use))
            except Exception:
                # Try fallback to America/New_York if requested zone fails
                try:
                    dt_local = dt_utc.astimezone(ZoneInfo("America/New_York"))
                except Exception:
                    from datetime import timedelta
                    dt_local = dt_utc - timedelta(hours=4)
        else:
            # Fallback: assume Eastern (UTC-4 in summer, UTC-5 in winter)
            from datetime import timedelta
            dt_local = dt_utc - timedelta(hours=4)
        # 5 PM cutoff is standard for "day game" classification
        return "day" if dt_local.hour < 17 else "night"
    except Exception:
        return "night"


# Map of stadium → IANA timezone for accurate day/night classification
VENUE_TIMEZONES = {
    # Pacific
    "Oracle Park": "America/Los_Angeles",
    "Dodger Stadium": "America/Los_Angeles",
    "Petco Park": "America/Los_Angeles",
    "Angel Stadium": "America/Los_Angeles",
    "T-Mobile Park": "America/Los_Angeles",
    "Sutter Health Park": "America/Los_Angeles",  # Athletics 2025+ temporary
    # Mountain (Arizona never observes DST)
    "Chase Field": "America/Phoenix",
    "Coors Field": "America/Denver",
    "Salt River Fields": "America/Phoenix",  # spring training
    # Central
    "Globe Life Field": "America/Chicago",
    "Daikin Park": "America/Chicago",
    "Minute Maid Park": "America/Chicago",
    "Wrigley Field": "America/Chicago",
    "Guaranteed Rate Field": "America/Chicago",
    "Rate Field": "America/Chicago",
    "Target Field": "America/Chicago",
    "Kauffman Stadium": "America/Chicago",
    "Busch Stadium": "America/Chicago",
    "American Family Field": "America/Chicago",
    # Eastern (all others)
}


def get_venue_timezone(venue_name: str) -> str:
    """Look up IANA timezone for a venue. Defaults to America/New_York."""
    return VENUE_TIMEZONES.get(venue_name, "America/New_York")


# ----------------------------------------------------------------------------
# Injury / IL list — return-from-IL fatigue flag for pitchers
# Pitchers fresh off IL typically throw 30-50 fewer pitches their first start.
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_player_il_status(player_id: int, season: int = CURRENT_SEASON) -> dict:
    """
    Check if a player has recent IL transactions.
    Returns {"on_il": bool, "days_since_return": int|None, "il_count_this_season": int}.
    Uses MLB Stats API transactions endpoint.
    """
    out = {"on_il": False, "days_since_return": None, "il_count_this_season": 0}
    try:
        from datetime import datetime, timedelta
        # Get player's transactions for the season
        url = (
            f"https://statsapi.mlb.com/api/v1/transactions"
            f"?playerId={player_id}&startDate={season}-01-01&endDate={season}-12-31"
        )
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return out
        data = r.json()
        transactions = data.get("transactions", [])
        if not transactions:
            return out

        # Find most recent IL-related transaction
        # IMPORTANT: be STRICT here. Earlier versions matched things like:
        #   - "Selected from minors" (SCL) → flagged as IL
        #   - "Placed on paternity list" → flagged as IL
        #   - "Placed on bereavement list" → flagged as IL
        #   - "Placed on restricted list" → flagged as IL
        # All false positives. We now ONLY count transactions whose description
        # explicitly mentions an injured list.
        today = datetime.now().date()
        il_count = 0
        latest_return = None
        latest_il_placement = None
        for txn in transactions:
            description = (txn.get("description") or "").lower()
            txn_date_str = txn.get("date") or txn.get("effectiveDate")
            if not txn_date_str:
                continue
            try:
                txn_date = datetime.strptime(txn_date_str[:10], "%Y-%m-%d").date()
            except Exception:
                continue

            # STRICT IL test: description must mention an injured list
            # (10-day IL, 15-day IL, 60-day IL, 7-day concussion IL, etc.)
            is_il_txn = (
                "injured list" in description
                or "the il" in description
                or "10-day il" in description
                or "15-day il" in description
                or "60-day il" in description
                or "7-day il" in description
            )
            # Exclude paternity, bereavement, restricted, suspended, military lists
            is_other_list = any(kw in description for kw in (
                "paternity list", "bereavement list", "restricted list",
                "suspended list", "military list", "family medical",
            ))
            if is_other_list:
                continue  # NOT an IL transaction
            if not is_il_txn:
                continue  # not IL related at all

            # Now classify: placement vs. activation
            is_placement = (
                "placed" in description
                or "transferred to" in description
            )
            is_activation = (
                "activated" in description
                or "reinstated" in description
            )
            if is_placement:
                il_count += 1
                if latest_il_placement is None or txn_date > latest_il_placement:
                    latest_il_placement = txn_date
            elif is_activation:
                if latest_return is None or txn_date > latest_return:
                    latest_return = txn_date

        # On IL = latest placement is more recent than latest return
        if latest_il_placement and (latest_return is None or latest_il_placement > latest_return):
            out["on_il"] = True
        if latest_return:
            out["days_since_return"] = (today - latest_return).days
        out["il_count_this_season"] = il_count
        return out
    except Exception:
        return out


@st.cache_data(ttl=3600)
def get_pitchers_il_status(pitcher_ids: tuple, season: int = CURRENT_SEASON,
                             _cache_version: str = "v1") -> pd.DataFrame:
    """Bulk-fetch IL status for slate pitchers.

    _cache_version: bump when adding new fields to invalidate cache.
    """
    if not pitcher_ids:
        return pd.DataFrame()
    rows = []
    for pid in pitcher_ids:
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            continue
        status = get_player_il_status(pid_int, season)
        # Include if pitcher has ANY IL history this season (on IL, recent return,
        # or any past stints). Healthy pitchers with no IL data are excluded since
        # they don't need columns populated.
        has_data = (
            status.get("on_il")
            or status.get("days_since_return") is not None
            or (status.get("il_count_this_season") or 0) > 0
        )
        if has_data:
            status["player_id"] = pid_int
            rows.append(status)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Pitcher arsenal (pitch-mix, velo, spin, swing/miss per pitch type)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_pitcher_arsenal(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Returns one row per pitcher x pitch_type with: usage %, swstr%, hh%,
    velocity, spin rate, xwOBAcon, run value per 100, etc.
    """
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
        f"?type=pitcher&pitchType=&year={season}&team=&min=10&hand="
        "&csv=true"
    )
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "last_name, first_name" in df.columns:
        df["player_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
            if isinstance(s, str) and "," in s else s
        )
    return df


@st.cache_data(ttl=3600)
def get_pitcher_arsenal_vs_hand(season: int = CURRENT_SEASON,
                                  batter_hand: str = "R") -> pd.DataFrame:
    """
    Pitcher arsenal split by hitter handedness.

    Why this matters: pitchers throw very different mixes vs LHB vs RHB.
    A right-handed pitcher might throw 38% sliders to RHB (slider moves
    away from same-side hitter) but only 12% sliders to LHB. Same pitcher,
    completely different arsenal depending on who's at the plate. RHP
    typically throw more changeups to LHB (changeup runs away from
    opposite-hand hitter) and more sinkers to RHB.

    Without this split, we project a LHB facing a slider-heavy RHP using
    the pitcher's OVERALL slider usage — which dramatically understates
    the changeup he's actually going to see.

    Returns a DataFrame matching get_pitcher_arsenal() schema, but with
    only the rows where the pitcher faced batters of `batter_hand` ("L" or "R").
    """
    if batter_hand not in ("L", "R"):
        return pd.DataFrame()
    # Savant's pitch-arsenal-stats endpoint accepts &stand=L or &stand=R to
    # filter by batter side. (This is the SAME endpoint as get_pitcher_arsenal,
    # just with the stand param set.)
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
        f"?type=pitcher&pitchType=&year={season}&team=&min=10&hand="
        f"&stand={batter_hand}&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return df
        if "last_name, first_name" in df.columns:
            df["player_name"] = df["last_name, first_name"].apply(
                lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
                if isinstance(s, str) and "," in s else s
            )
        # Tag rows so we know which hand split they're from when merging
        df["vs_batter_hand"] = batter_hand
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def get_pitcher_full_season_from_gamelog(pitcher_id: int,
                                            season: int = CURRENT_SEASON) -> dict:
    """
    Compute season-total ERA/WHIP/K9/BB9/HR9/IP/GS by summing the pitcher's
    own game log. This is independent of the /api/v1/stats?stats=season endpoint
    that's been flaky. Same /people/{id}/stats?stats=gameLog endpoint that
    powers recent form -- known to work.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
        f"?stats=gameLog&group=pitching&season={season}&sportId=1"
    )
    splits = []
    for attempt in range(2):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [])
            if splits:
                break
        except Exception:
            if attempt < 1:
                import time
                time.sleep(0.5)
            continue
    if not splits:
        return {}
    ip_sum, er_sum, k_sum, bb_sum, hr_sum = 0.0, 0, 0, 0, 0
    h_sum = 0
    gs_sum = 0
    gp_sum = 0
    for s in splits:
        st_ = s.get("stat", {}) or {}
        ip_sum += _safe_float(st_.get("inningsPitched")) or 0
        er_sum += _safe_int(st_.get("earnedRuns")) or 0
        k_sum += _safe_int(st_.get("strikeOuts")) or 0
        bb_sum += _safe_int(st_.get("baseOnBalls")) or 0
        hr_sum += _safe_int(st_.get("homeRuns")) or 0
        h_sum += _safe_int(st_.get("hits")) or 0
        gs_sum += _safe_int(st_.get("gamesStarted")) or 0
        gp_sum += _safe_int(st_.get("gamesPlayed")) or 0
    if ip_sum == 0:
        return {}
    # WHIP = (BB + H) / IP
    whip = round((bb_sum + h_sum) / ip_sum, 2) if ip_sum > 0 else None
    return {
        "era": round(er_sum * 9 / ip_sum, 2),
        "whip": whip,
        "k9": round(k_sum * 9 / ip_sum, 2),
        "bb9": round(bb_sum * 9 / ip_sum, 2),
        "hr9": round(hr_sum * 9 / ip_sum, 2),
        "ip": round(ip_sum, 1),
        "games_started": gs_sum,
        "games_played": gp_sum,
        "strikeouts": k_sum,
        "walks": bb_sum,
        "earned_runs": er_sum,
        "source": "gamelog",  # marker so we know which source
    }


@st.cache_data(ttl=1800)
def get_pitcher_recent_form(pitcher_id: int, season: int = CURRENT_SEASON,
                              n_starts: int = 5) -> dict:
    """
    Recent form: last N starts K/9, ERA, IP.
    Returns trending up/down arrow.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
        f"?stats=gameLog&group=pitching&season={season}&sportId=1"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {}
        # Last N starts only
        starts = [s for s in splits if int(s.get("stat", {}).get("gamesStarted", 0)) > 0]
        recent = starts[-n_starts:] if len(starts) > n_starts else starts
        if not recent:
            return {}
        ip_sum, er_sum, k_sum, bb_sum, hr_sum = 0.0, 0, 0, 0, 0
        for s in recent:
            st_ = s.get("stat", {})
            ip_sum += float(st_.get("inningsPitched", 0) or 0)
            er_sum += int(st_.get("earnedRuns", 0) or 0)
            k_sum += int(st_.get("strikeOuts", 0) or 0)
            bb_sum += int(st_.get("baseOnBalls", 0) or 0)
            hr_sum += int(st_.get("homeRuns", 0) or 0)
        if ip_sum == 0:
            return {}
        return {
            "recent_starts": len(recent),
            "recent_ip": round(ip_sum, 1),
            "recent_era": round(er_sum * 9 / ip_sum, 2),
            "recent_k9": round(k_sum * 9 / ip_sum, 2),
            "recent_bb9": round(bb_sum * 9 / ip_sum, 2),
            "recent_hr9": round(hr_sum * 9 / ip_sum, 2),
            "recent_k": k_sum,
        }
    except Exception:
        return {}


@st.cache_data(ttl=1800)
def get_hitter_recent_form_trad(player_id: int, season: int = CURRENT_SEASON,
                                  n_games: int = 15,
                                  _cache_version: str = "v3") -> dict:
    """Last 15 games hitter form via game log - lightweight.

    Now also tracks HR streaks and hot/cold pattern indicators.

    _cache_version: bumped to v3 for v42c when 10-game window stats were
    added (recent_iso_10, recent_avg_10, etc). Forces Streamlit cache to
    invalidate so old cached results don't override fresh fetches.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=gameLog&group=hitting&season={season}&sportId=1"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        recent = splits[-n_games:] if len(splits) > n_games else splits
        if not recent:
            return {}
        ab, h, hr, k, bb, rbi = 0, 0, 0, 0, 0, 0
        d, t = 0, 0
        # Streak tracking - per-game HR list (most recent first)
        per_game_hr = []
        per_game_hits = []
        for s in recent:
            st_ = s.get("stat", {})
            ab += int(st_.get("atBats", 0) or 0)
            h += int(st_.get("hits", 0) or 0)
            game_hr = int(st_.get("homeRuns", 0) or 0)
            hr += game_hr
            per_game_hr.append(game_hr)
            per_game_hits.append(int(st_.get("hits", 0) or 0))
            k += int(st_.get("strikeOuts", 0) or 0)
            bb += int(st_.get("baseOnBalls", 0) or 0)
            rbi += int(st_.get("rbi", 0) or 0)
            d += int(st_.get("doubles", 0) or 0)
            t += int(st_.get("triples", 0) or 0)
        if ab == 0:
            return {}

        # v42c: Also compute last-10 stats for "ideal window" recent form
        # (community consensus: 10 days = signal sweet spot, 5 too noisy,
        # 15 too smoothed). The 15-game stats above stay for backward
        # compatibility and longer-window context; 10-game stats added below
        # for ranking sensitivity.
        last_10_splits = recent[-10:] if len(recent) > 10 else recent
        ab_10, h_10, hr_10, d_10, t_10, k_10, bb_10 = 0, 0, 0, 0, 0, 0, 0
        for s10 in last_10_splits:
            st10 = s10.get("stat", {})
            ab_10 += int(st10.get("atBats", 0) or 0)
            h_10 += int(st10.get("hits", 0) or 0)
            hr_10 += int(st10.get("homeRuns", 0) or 0)
            d_10 += int(st10.get("doubles", 0) or 0)
            t_10 += int(st10.get("triples", 0) or 0)
            k_10 += int(st10.get("strikeOuts", 0) or 0)
            bb_10 += int(st10.get("baseOnBalls", 0) or 0)
        # ISO over 10 games
        recent_iso_10 = round((d_10 + 2 * t_10 + 3 * hr_10) / ab_10, 3) if ab_10 else 0.0
        # AVG over 10
        recent_avg_10 = round(h_10 / ab_10, 3) if ab_10 else 0.0
        # K% over 10
        recent_k_pct_10 = round(k_10 / (ab_10 + bb_10) * 100, 1) if (ab_10 + bb_10) else 0.0

        # v42t: 5-game window — captures very recent hot/cold streaks.
        # User-requested signal: "hitting the ball hard and in the air over
        # past 5 games to determine if HR form is better/worse than baseline."
        # ISO captures extra-base output (the cleanest power proxy from MLB
        # Stats API game logs — Savant batted-ball stats aren't game-by-game).
        # SLG captures total bases per AB, similar but includes singles.
        # 5-game window is noisy (1 multi-HR game = huge swing) but is
        # exactly what the user wants for catching hot streaks early.
        last_5_splits = recent[-5:] if len(recent) > 5 else recent
        ab_5, h_5, hr_5, d_5, t_5, k_5, bb_5 = 0, 0, 0, 0, 0, 0, 0
        for s5 in last_5_splits:
            st5 = s5.get("stat", {})
            ab_5 += int(st5.get("atBats", 0) or 0)
            h_5 += int(st5.get("hits", 0) or 0)
            hr_5 += int(st5.get("homeRuns", 0) or 0)
            d_5 += int(st5.get("doubles", 0) or 0)
            t_5 += int(st5.get("triples", 0) or 0)
            k_5 += int(st5.get("strikeOuts", 0) or 0)
            bb_5 += int(st5.get("baseOnBalls", 0) or 0)
        # ISO over 5 games
        recent_iso_5 = round((d_5 + 2 * t_5 + 3 * hr_5) / ab_5, 3) if ab_5 else 0.0
        # SLG over 5 games (total bases / AB)
        tb_5 = h_5 + d_5 + 2 * t_5 + 3 * hr_5  # total bases
        recent_slg_5 = round(tb_5 / ab_5, 3) if ab_5 else 0.0
        # AVG over 5 (less HR-relevant but useful for display)
        recent_avg_5 = round(h_5 / ab_5, 3) if ab_5 else 0.0

        # Streak metrics - reverse so most-recent game is first
        per_game_hr_rev = list(reversed(per_game_hr))
        # Count games since last HR (None = never homered in window)
        games_since_hr = None
        for i, ghr in enumerate(per_game_hr_rev):
            if ghr > 0:
                games_since_hr = i
                break
        # Consecutive games with a HR (streak)
        consec_hr_games = 0
        for ghr in per_game_hr_rev:
            if ghr > 0:
                consec_hr_games += 1
            else:
                break
        # HRs in last 5, last 10 (rolling windows)
        hr_last_5 = sum(per_game_hr_rev[:5])
        hr_last_10 = sum(per_game_hr_rev[:10])
        # Multi-HR games in window
        multi_hr_games = sum(1 for ghr in per_game_hr if ghr >= 2)

        # NEW: RECENCY-WEIGHTED recent_hr metric
        # Last 3 games weighted 3x; games 4-7 weighted 2x; games 8-15 weighted 1x.
        # Captures hot streaks more accurately than equal-weight rolling sum.
        # Returns a "weighted HR rate" — equivalent HRs assuming the recent
        # weighting pattern continues. Useful for picking up real momentum.
        weighted_hr = 0.0
        weighted_pa = 0.0
        for i, ghr in enumerate(per_game_hr_rev):
            # Per-game PA estimate from AB+BB (close enough for weighting)
            this_ab = int(recent[len(recent) - 1 - i].get("stat", {}).get("atBats", 0) or 0)
            this_bb = int(recent[len(recent) - 1 - i].get("stat", {}).get("baseOnBalls", 0) or 0)
            game_pa = this_ab + this_bb
            if i < 3:    weight = 3.0   # last 3 games
            elif i < 7:  weight = 2.0   # games 4-7
            else:        weight = 1.0   # games 8-15
            weighted_hr += ghr * weight
            weighted_pa += game_pa * weight
        recent_hr_weighted_rate = (
            round(weighted_hr / weighted_pa * 100, 3)
            if weighted_pa > 0 else 0.0
        )

        # Streak label for the table
        if consec_hr_games >= 2:
            streak_label = f"🔥 {consec_hr_games}g HR streak"
        elif hr_last_5 >= 3:
            streak_label = f"🔥 {hr_last_5} HR/L5"
        elif hr_last_5 >= 2:
            streak_label = f"🌶️ {hr_last_5} HR/L5"
        elif multi_hr_games >= 1:
            streak_label = f"⚡ {multi_hr_games}x multi-HR/L{len(recent)}"
        elif hr_last_10 == 0 and len(recent) >= 10:
            streak_label = "❄️ no HR L10"
        else:
            streak_label = ""

        return {
            "recent_games": len(recent),
            "recent_ab": ab, "recent_h": h, "recent_hr": hr,
            "recent_k": k, "recent_bb": bb, "recent_rbi": rbi,
            "recent_avg": round(h / ab, 3),
            "recent_iso": round((d + 2 * t + 3 * hr) / ab, 3) if ab else 0.0,
            "recent_k_pct": round(k / (ab + bb) * 100, 1) if (ab + bb) else 0.0,
            "recent_ops_proxy": round((h + bb) / (ab + bb), 3) if (ab + bb) else 0.0,
            "hr_streak_games": consec_hr_games,
            "recent_hr_weighted_rate": recent_hr_weighted_rate,
            "games_since_hr": games_since_hr,
            "hr_last_5": hr_last_5,
            "hr_last_10": hr_last_10,
            # v42c: 10-game window — community-validated as ideal HR signal
            "recent_iso_10": recent_iso_10,
            "recent_avg_10": recent_avg_10,
            "recent_k_pct_10": recent_k_pct_10,
            "recent_ab_10": ab_10,
            "recent_h_10": h_10,
            "recent_hr_10": hr_10,
            # v42t: 5-game window — user-requested for catching hot streaks
            # earlier than the 10-game signal. Display-only (does NOT feed
            # into hr_form scoring — that would be re-weighting a calibration
            # we have +25.9pp evidence is working). Useful for the user to
            # visually identify trending hitters.
            "recent_iso_5": recent_iso_5,
            "recent_slg_5": recent_slg_5,
            "recent_avg_5": recent_avg_5,
            "recent_hr_5": hr_5,
            "recent_ab_5": ab_5,
            "multi_hr_games": multi_hr_games,
            "streak_label": streak_label,
        }
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Sprint Speed (affects BABIP and infield hits) - Statcast
# ----------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_sprint_speed(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Statcast sprint speed leaderboard. ft/sec, league avg ~27.
    Elite is 30+. Slow is 25-.
    """
    url = (
        "https://baseballsavant.mlb.com/leaderboard/sprint_speed"
        f"?year={season}&position=&team=&min=10&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "last_name, first_name" in df.columns:
            df["player_name"] = df["last_name, first_name"].apply(
                lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
                if isinstance(s, str) and "," in s else s
            )
        return df
    except Exception:
        return pd.DataFrame()


# ----------------------------------------------------------------------------
# Statcast Run Values per pitch type (better pitch quality metric than whiff%)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_pitch_run_values(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Run value per 100 pitches by pitcher + pitch type.
    Negative = good for pitcher. -1.5 is elite, +1.5 is brutal.
    """
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-arsenals"
        f"?year={season}&min=10&type=run_value&hand=&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "last_name, first_name" in df.columns:
            df["player_name"] = df["last_name, first_name"].apply(
                lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
                if isinstance(s, str) and "," in s else s
            )
        return df
    except Exception:
        return pd.DataFrame()


# ----------------------------------------------------------------------------
# Traditional stats from MLB Stats API (WHIP, HR/9, OBP, HR totals)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_hitter_traditional(season: int = CURRENT_SEASON, _stats_day: str = "") -> pd.DataFrame:
    """Pull season AVG, OBP, SLG, HR, RBI, R for all hitters.

    Uses playerPool=All to catch non-qualifying hitters (early-season,
    platoon, callups). Defensive row parsing.
    """
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&season={season}&sportIds=1"
        "&playerPool=All&limit=3000"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
    except Exception:
        return pd.DataFrame()

    rows = []
    for s in splits:
        try:
            p = s.get("player", {}) or {}
            st_ = s.get("stat", {}) or {}
            pid = p.get("id")
            if pid is None:
                continue
            rows.append({
                "player_id": pid,
                "player_name": p.get("fullName"),
                "avg": _safe_float(st_.get("avg")),
                "obp": _safe_float(st_.get("obp")),
                "slg": _safe_float(st_.get("slg")),
                "ops": _safe_float(st_.get("ops")),
                "home_run": _safe_int(st_.get("homeRuns")),
                "rbi": _safe_int(st_.get("rbi")),
                "runs": _safe_int(st_.get("runs")),
                "sb": _safe_int(st_.get("stolenBases")),
                "trad_pa": _safe_int(st_.get("plateAppearances")),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    return _normalize_player_df(df)



@st.cache_data(ttl=900)  # Shortened from 3600 to 15min so failures recover faster
def get_pitcher_traditional(season: int = CURRENT_SEASON, _stats_day: str = "") -> pd.DataFrame:
    """Pull season ERA, WHIP, HR/9, K/9, BB/9 for all pitchers.

    Uses playerPool=All so non-qualifying pitchers (early-season, callups,
    long relievers spot-starting) are included. Retries 3x; if all fail OR
    return empty, we DON'T cache - by raising an exception that streamlit
    won't cache, then catching it to return empty. Next refresh tries again.
    """
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=pitching&season={season}&sportIds=1"
        "&playerPool=All&limit=3000"
    )
    splits = []
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [])
            if splits:
                break
        except Exception as e:
            last_err = e
            if attempt < 2:
                import time
                time.sleep(1.5)
            continue

    # CRITICAL: if we have no data, raise so the cache doesn't store empty
    if not splits:
        raise RuntimeError(f"MLB Stats API returned no pitcher data (last error: {last_err})")

    rows = []
    for s in splits:
        try:
            p = s.get("player", {}) or {}
            st_ = s.get("stat", {}) or {}
            pid = p.get("id")
            if pid is None:
                continue
            rows.append({
                "player_id": pid,
                "player_name": p.get("fullName"),
                "era": _safe_float(st_.get("era")),
                "whip": _safe_float(st_.get("whip")),
                "hr9": _safe_float(st_.get("homeRunsPer9")),
                # API has used both field names in different versions - try both
                "bb9": _safe_float(st_.get("walksPer9Inn") or st_.get("baseOnBallsPer9Inn")),
                "k9": _safe_float(st_.get("strikeoutsPer9Inn") or st_.get("strikeOutsPer9Inn")),
                "ip": _safe_float(st_.get("inningsPitched")),
                "wins": _safe_int(st_.get("wins")),
                "losses": _safe_int(st_.get("losses")),
                "games_started": _safe_int(st_.get("gamesStarted")),
                "games_played": _safe_int(st_.get("gamesPlayed")),
                "strikeouts": _safe_int(st_.get("strikeOuts")),
                "walks": _safe_int(st_.get("baseOnBalls")),
                "earned_runs": _safe_int(st_.get("earnedRuns")),
            })
        except Exception:
            continue

    if not rows:
        raise RuntimeError("MLB Stats API returned splits but no usable rows")

    df = pd.DataFrame(rows)
    return _normalize_player_df(df)


def get_pitcher_traditional_safe(season: int = CURRENT_SEASON, _stats_day: str = "") -> pd.DataFrame:
    """Wrapper that catches the cache-bust exception and returns empty df."""
    try:
        return get_pitcher_traditional(season, _stats_day=_stats_day)
    except Exception:
        return pd.DataFrame()


# ----------------------------------------------------------------------------
# Per-pitcher fallback - guarantees data for today's starters
# ----------------------------------------------------------------------------

@st.cache_data(ttl=1800)
def get_player_season_pitching(player_id: int, season: int = CURRENT_SEASON) -> dict:
    """Fetch one pitcher's season stats directly by ID. Retries once on failure."""
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=season&group=pitching&season={season}&sportId=1"
    )
    splits = []
    for attempt in range(2):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [])
            if splits:
                break
        except Exception:
            if attempt < 1:
                import time
                time.sleep(0.5)
            continue
    if not splits:
        return {}
    try:
        st_ = splits[0].get("stat", {}) or {}
        return {
            "era": _safe_float(st_.get("era")),
            "whip": _safe_float(st_.get("whip")),
            "hr9": _safe_float(st_.get("homeRunsPer9")),
            "bb9": _safe_float(st_.get("walksPer9Inn") or st_.get("baseOnBallsPer9Inn")),
            "k9": _safe_float(st_.get("strikeoutsPer9Inn") or st_.get("strikeOutsPer9Inn")),
            "ip": _safe_float(st_.get("inningsPitched")),
            "games_started": _safe_int(st_.get("gamesStarted")),
            "games_played": _safe_int(st_.get("gamesPlayed")),
            "strikeouts": _safe_int(st_.get("strikeOuts")),
            "walks": _safe_int(st_.get("baseOnBalls")),
            "earned_runs": _safe_int(st_.get("earnedRuns")),
        }
    except Exception:
        return {}


def fill_hitter_bats(lineups: list, ids: set | None = None) -> dict:
    """
    Bulk-fetch batting handedness for all hitter IDs in the provided lineups.
    Returns {player_id: "R"/"L"/"S"} mapping.
    """
    if ids is None:
        ids = set()
        for lineup in lineups:
            for p in lineup or []:
                pid = p.get("id")
                if pid is not None:
                    try:
                        ids.add(int(pid))
                    except (ValueError, TypeError):
                        continue
    if not ids:
        return {}
    try:
        ids_str = ",".join(str(p) for p in ids)
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}"
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        out = {}
        for person in r.json().get("people", []):
            pid = person.get("id")
            side = (person.get("batSide") or {}).get("code")
            if pid and side:
                out[pid] = side
        return out
    except Exception:
        return {}


@st.cache_data(ttl=86400)
def fetch_player_debut_years(ids: tuple) -> dict:
    """
    Bulk-fetch MLB debut year for player IDs. Used for rookie detection.
    Returns {player_id: debut_year_int}. Cached 24h.
    """
    if not ids:
        return {}
    try:
        ids_str = ",".join(str(p) for p in ids)
        url = (
            f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}"
            "&hydrate=mlbDebutDate"
        )
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        out = {}
        for person in r.json().get("people", []):
            pid = person.get("id")
            debut = person.get("mlbDebutDate")
            if pid and debut:
                try:
                    out[pid] = int(debut[:4])
                except (ValueError, TypeError):
                    continue
        return out
    except Exception:
        return {}


def is_rookie(debut_year: int | None, current_year: int = CURRENT_SEASON) -> bool:
    """
    A player is a rookie if their MLB debut is this year or the prior offseason.
    Strictly: debut_year == current_year means they're a definite rookie.
    debut_year == current_year - 1 with limited PA could also be a rookie,
    but that requires PA threshold checking which we leave to the caller.
    """
    if debut_year is None:
        return False
    return debut_year == current_year


@st.cache_data(ttl=86400)  # cache for 24h - LA changes slowly
def get_hitter_launch_angle_season(player_id: int, season: int = CURRENT_SEASON) -> float | None:
    """
    Pull real season-long launch angle for ONE hitter from Statcast search.
    This is the bulletproof source - reads raw batted-ball events and
    computes the player's actual average launch angle.

    Returns None if no batted-ball data exists for this player.
    """
    from datetime import date
    end = date.today()
    start = date(season, 3, 1)  # season start (early March covers spring + reg)
    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv"
        f"?all=true&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones="
        f"&hfPull=&hfC=&hfSea={season}%7C&hfSit=&player_type=batter&hfOuts=&opponent="
        f"&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={start.isoformat()}"
        f"&game_date_lt={end.isoformat()}&batters_lookup%5B%5D={player_id}"
        f"&team=&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0"
        f"&min_results=0&group_by=name&sort_col=pitches"
        f"&player_event_sort=api_p_release_speed&sort_order=desc&min_pas=0&type=details"
    )
    try:
        # Statcast season CSVs for one player can be 5MB+, so the per-player
        # search is slow. Bump timeout to 60s. The bulk leaderboard fetch
        # in fill_hitter_la_for_slate handles 95%+ of players in one call,
        # so this slow path only runs for a handful of names per slate.
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return None
        # Only count batted balls (type == 'X')
        bbe = df[df["type"] == "X"] if "type" in df.columns else df
        if bbe.empty or "launch_angle" not in bbe.columns:
            return None
        la_series = pd.to_numeric(bbe["launch_angle"], errors="coerce")
        la_series = la_series.dropna()
        if la_series.empty:
            return None
        return round(float(la_series.mean()), 1)
    except Exception:
        return None


def fill_hitter_la_for_slate(hitter_stats: pd.DataFrame, slate: pd.DataFrame,
                                season: int = CURRENT_SEASON,
                                log_steps: list | None = None) -> pd.DataFrame:
    """
    For every hitter projected to play today, if launch_angle is missing,
    patch it in.

    STRATEGY (May 2026): try bulk leaderboard FIRST, then fall back to slow
    per-player Statcast search only for any IDs the bulk pull missed.
    Previous version only did per-player which timed out at 30s for
    150-200 slate hitters (Statcast season CSVs can be 5MB+ each).

    DIAGNOSTIC: pass a `log_steps` list to receive per-step messages about
    which bulk endpoints were tried, which succeeded, how many IDs were
    matched, and whether per-player fallback ran. Used by the LA diagnostic
    expander in app.py to make silent failures visible.

    Returns the hitter_stats df with launch_angle patched in.
    """
    def _log(msg):
        if log_steps is not None:
            log_steps.append(msg)

    if hitter_stats is None or hitter_stats.empty:
        _log("EARLY EXIT: hitter_stats is None or empty")
        return hitter_stats
    if "launch_angle" not in hitter_stats.columns:
        hitter_stats = hitter_stats.copy()
        hitter_stats["launch_angle"] = pd.NA
        _log("Created launch_angle column (was missing)")

    df = hitter_stats.copy()
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    _log(f"Input: {len(df)} hitters, {df['launch_angle'].isna().sum()} missing LA")

    # ----- STEP 1: Bulk leaderboard pull (single API hit, covers everyone) -----
    # Same endpoints as the in-pipeline fallback in get_hitter_stats.
    bulk_la_urls = [
        f"https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?type=batter&year={season}&position=&team=&min=1&csv=true",
        f"https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={season}&type=batter&filter=&min=1"
        f"&selections=pa,launch_angle,launch_speed,avg_hit_angle"
        f"&chart=false&x=pa&y=pa&r=no&csv=true",
        f"https://baseballsavant.mlb.com/leaderboard/exit-velocity"
        f"?type=batter&year={season}&min=1&csv=true",
    ]
    la_map_bulk = {}
    for i, url in enumerate(bulk_la_urls, 1):
        url_label = url.split("/leaderboard/")[1].split("?")[0]
        try:
            rr = requests.get(url, headers=HEADERS, timeout=20)
            rr.raise_for_status()
            ev_df = pd.read_csv(io.StringIO(rr.text))
            if ev_df.empty:
                _log(f"  [{i}/{len(bulk_la_urls)}] {url_label}: empty CSV")
                continue
            la_col = None
            for cand in ["launch_angle", "avg_hit_angle", "angle",
                            "avg_launch_angle", "avg_la", "la"]:
                if cand in ev_df.columns:
                    coerced = pd.to_numeric(ev_df[cand], errors="coerce")
                    if coerced.notna().any():
                        ev_df[cand] = coerced
                        la_col = cand
                        break
            id_col = None
            for cand in ["player_id", "mlb_id", "MLBAMID", "playerid", "id"]:
                if cand in ev_df.columns:
                    id_col = cand
                    break
            if la_col and id_col:
                ev_df[id_col] = pd.to_numeric(ev_df[id_col], errors="coerce").astype("Int64")
                tmp_map = dict(zip(ev_df[id_col], ev_df[la_col]))
                # Drop NaNs from the map
                la_map_bulk = {k: v for k, v in tmp_map.items()
                                if v is not None and not pd.isna(v)}
                _log(f"  [{i}/{len(bulk_la_urls)}] {url_label}: {len(la_map_bulk)} IDs (la_col={la_col}, id_col={id_col})")
                if la_map_bulk:
                    break
            else:
                avail_cols = list(ev_df.columns)[:8]
                _log(f"  [{i}/{len(bulk_la_urls)}] {url_label}: no la_col or id_col found. Got: {avail_cols}")
        except Exception as e:
            _log(f"  [{i}/{len(bulk_la_urls)}] {url_label}: EXCEPTION {type(e).__name__}: {str(e)[:120]}")
            continue

    if la_map_bulk:
        # Apply bulk values everywhere we have a missing LA
        missing_mask = df["launch_angle"].isna()
        n_before = missing_mask.sum()
        df.loc[missing_mask, "launch_angle"] = (
            df.loc[missing_mask, "player_id"].map(la_map_bulk)
        )
        n_after = df["launch_angle"].isna().sum()
        _log(f"Bulk pull patched {n_before - n_after} LA values; {n_after} still missing")
    else:
        _log("Bulk pull returned no map. ALL endpoints failed or empty.")

    # ----- STEP 2: Per-player slow fetch ONLY for slate-relevant hitters
    # who are STILL missing after the bulk pull -----
    relevant_ids = set()
    if not slate.empty:
        for _, g in slate.iterrows():
            for col in ("away_team_id", "home_team_id"):
                tid = g.get(col)
                if tid is None or pd.isna(tid):
                    continue
                try:
                    roster = get_team_roster(int(tid))
                    for p in roster:
                        pid = p.get("id")
                        if pid:
                            relevant_ids.add(int(pid))
                except Exception:
                    continue

    if not relevant_ids:
        _log("No slate-relevant IDs to fetch per-player. Done.")
        return df

    missing_la_ids = set()
    for pid in relevant_ids:
        rows = df[df["player_id"] == pid]
        if rows.empty:
            continue
        existing_la = rows.iloc[0].get("launch_angle")
        if existing_la is None or pd.isna(existing_la):
            missing_la_ids.add(pid)

    _log(f"Per-player fallback: {len(missing_la_ids)} slate hitters still missing LA")

    # Cap per-player fetches to 50 to bound worst-case time. Anyone past 50
    # gets neutral (no LA) — far better than hanging the page.
    patched = 0
    failed = 0
    for i, pid in enumerate(missing_la_ids):
        if i >= 50:
            _log(f"  Capped at 50 per-player fetches; {len(missing_la_ids) - 50} skipped")
            break
        la = get_hitter_launch_angle_season(pid, season)
        if la is not None:
            df.loc[df["player_id"] == pid, "launch_angle"] = la
            patched += 1
        else:
            failed += 1
    _log(f"Per-player fallback: patched {patched}, failed {failed}")
    final_missing = df["launch_angle"].isna().sum()
    _log(f"FINAL: {len(df) - final_missing}/{len(df)} hitters have LA ({final_missing} still missing)")

    return df


# ----------------------------------------------------------------------------
# HR PROFILE: avg exit velo on HRs + avg HR distance (June 2026)
# ----------------------------------------------------------------------------
# For each hitter, what's their average exit velocity ON HOME RUNS specifically
# (not all batted balls), and how far does that HR go on average?
#
# A player who hits 110+ mph HRs at 380 ft has a "laser" profile — line-drive
# HRs over the wall. A player who hits 100 mph HRs at 425 ft has a "moonshot"
# profile — high-trajectory HRs that clear the fence on arc.
#
# Both can be productive HR hitters but they project differently:
#   - Laser HRs are LESS affected by wind/altitude (lower hang time)
#   - Moonshot HRs are MORE affected by park dimensions and weather
#
# Data source: Savant's barrels-leaderboard endpoint, which has avg_hr_distance
# and max_hit_speed natively. We pull it once per slate (cached 1hr).
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_hitter_hr_profile(season: int = CURRENT_SEASON,
                           _cache_version: str = "v42c") -> pd.DataFrame:
    """
    Returns per-hitter HR profile metrics:
      - player_id
      - avg_hr_distance  (avg distance of THIS hitter's HRs, in feet)
      - max_hit_speed    (max exit velo of any batted ball — proxy for top-end power)
      - hr_profile       (categorical: "moonshot" / "balanced" / "laser" / None)
      - hr_profile_label (display string with emoji)

    _cache_version: bump to invalidate stale cached results. Bumped to v42c
    when the fabricated-zero fix shipped (Savant returns 0 for hitters with
    no HRs; we coerce ≤50 ft/mph to NaN).
    """
    url = (
        "https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?type=batter&year={season}&position=&team=&min=1&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return pd.DataFrame()

        # Column name candidates (Savant varies slightly across endpoints)
        id_col = None
        for cand in ("player_id", "mlb_id", "playerid", "id", "MLBAMID"):
            if cand in df.columns:
                id_col = cand
                break
        if id_col is None:
            return pd.DataFrame()

        dist_col = None
        for cand in ("avg_hr_distance", "avg_hr_dist", "hr_distance_avg"):
            if cand in df.columns:
                dist_col = cand
                break

        ev_col = None
        for cand in ("max_hit_speed", "max_ev", "max_exit_velocity"):
            if cand in df.columns:
                ev_col = cand
                break

        if not dist_col and not ev_col:
            return pd.DataFrame()

        out = pd.DataFrame()
        out["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
        if dist_col:
            _dist = pd.to_numeric(df[dist_col], errors="coerce")
            # v42b: Savant returns 0 for hitters with no HRs (no distance to
            # average). 0 ft is not a real measurement — coerce to NaN so
            # downstream classifiers see "no data" instead of an absurd value.
            # Any real HR is > 50 ft; 0 means missing.
            _dist = _dist.mask(_dist <= 50)
            out["avg_hr_distance"] = _dist.round(1)
        if ev_col:
            _ev = pd.to_numeric(df[ev_col], errors="coerce")
            # v42b: same fix — 0 mph max EV means "no batted balls tracked".
            # Any real swing > 50 mph; 0 is missing.
            _ev = _ev.mask(_ev <= 50)
            out["max_hit_speed"] = _ev.round(1)

        # Classify HR profile:
        #   moonshot — high distance (>= 415 ft avg HR distance)
        #   laser    — high EV but lower distance (max EV >= 110 AND avg_hr_dist < 400)
        #   balanced — middle ground
        #   None     — no data
        def _classify(r):
            dist = r.get("avg_hr_distance")
            ev = r.get("max_hit_speed")
            if (dist is None or pd.isna(dist)) and (ev is None or pd.isna(ev)):
                return (None, None)
            try:
                dist_f = float(dist) if (dist is not None and not pd.isna(dist)) else None
                ev_f = float(ev) if (ev is not None and not pd.isna(ev)) else None
            except (TypeError, ValueError):
                return (None, None)
            if dist_f is not None and dist_f >= 415:
                return ("moonshot", f"🚀 moonshot ({dist_f:.0f}ft)")
            if dist_f is not None and dist_f >= 405:
                return ("balanced+", f"⚖️ balanced+ ({dist_f:.0f}ft)")
            if ev_f is not None and ev_f >= 112 and (dist_f is None or dist_f < 400):
                return ("laser", f"⚡ laser ({ev_f:.0f}mph)")
            if dist_f is not None and dist_f >= 390:
                return ("balanced", f"⚖️ balanced ({dist_f:.0f}ft)")
            if dist_f is not None:
                return ("short", f"📏 short porch ({dist_f:.0f}ft)")
            return (None, None)
        classifications = out.apply(_classify, axis=1)
        out["hr_profile"] = [c[0] for c in classifications]
        out["hr_profile_label"] = [c[1] for c in classifications]
        return out
    except Exception:
        return pd.DataFrame()


def fill_pitcher_stats_for_slate(pitcher_stats: pd.DataFrame, slate: pd.DataFrame,
                                   season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    For every probable starter in the slate, if they're missing key stats
    in pitcher_stats, fetch them individually and patch them in.
    Guarantees ERA/K9/etc for every starter today (assuming they have stats).
    """
    if pitcher_stats is None or slate is None or slate.empty:
        return pitcher_stats

    df = pitcher_stats.copy() if pitcher_stats is not None and not pitcher_stats.empty else pd.DataFrame()
    if df.empty:
        df = pd.DataFrame(columns=["player_id"])

    # Ensure key columns exist
    for col in ["era", "whip", "k9", "bb9", "hr9", "ip", "games_started", "p_throws"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Get starter IDs from slate
    starter_ids = set()
    for col in ["away_pitcher_id", "home_pitcher_id"]:
        if col in slate.columns:
            for pid in slate[col].dropna().unique():
                try:
                    starter_ids.add(int(pid))
                except (ValueError, TypeError):
                    continue

    # Bulk-fetch handedness for all starters via people endpoint
    if starter_ids:
        try:
            ids_str = ",".join(str(p) for p in starter_ids)
            url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}"
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            for person in r.json().get("people", []):
                pid = person.get("id")
                hand = (person.get("pitchHand") or {}).get("code")
                if pid is None or not hand:
                    continue
                existing = df[df["player_id"] == pid] if "player_id" in df.columns else pd.DataFrame()
                if existing.empty:
                    df = pd.concat([df, pd.DataFrame([{"player_id": pid, "p_throws": hand}])],
                                      ignore_index=True)
                else:
                    df.at[existing.index[0], "p_throws"] = hand
        except Exception:
            pass

    for pid in starter_ids:
        existing = df[df["player_id"] == pid] if "player_id" in df.columns else pd.DataFrame()
        needs_fill = (
            existing.empty
            or pd.isna(existing.iloc[0].get("k9"))
            or pd.isna(existing.iloc[0].get("era"))
        )
        if not needs_fill:
            continue

        # Try the standard season endpoint first
        stats = get_player_season_pitching(pid, season)
        # Fall back to game log aggregation (independent code path - more reliable)
        if not stats or stats.get("k9") is None or stats.get("era") is None:
            gl_stats = get_pitcher_full_season_from_gamelog(pid, season)
            if gl_stats and gl_stats.get("k9") is not None:
                stats = gl_stats
        if not stats or stats.get("k9") is None:
            continue

        if existing.empty:
            new_row = {"player_id": pid}
            new_row.update(stats)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            idx = existing.index[0]
            for k, v in stats.items():
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    df.at[idx, k] = v

    return _normalize_player_df(df)


# ----------------------------------------------------------------------------
# Recent form: rolling 15-game Statcast
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_recent_form_hitter(player_id: int, season: int = CURRENT_SEASON, days: int = 15) -> dict:
    """
    Pulls last N days of statcast pitch-level data for one hitter and
    aggregates a few rolling indicators.
    """
    end = date.today()
    from datetime import timedelta
    start = end - timedelta(days=days)
    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv"
        f"?all=true&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones="
        "&hfPull=&hfC=&hfSea={s}%7C&hfSit=&player_type=batter&hfOuts=&opponent="
        "&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={st}&game_date_lt={en}"
        "&batters_lookup%5B%5D={pid}&team=&position=&hfRO=&home_road=&hfFlag="
        "&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name&sort_col=pitches"
        "&player_event_sort=api_p_release_speed&sort_order=desc&min_pas=0&type=details"
    ).format(s=season, st=start.isoformat(), en=end.isoformat(), pid=player_id)

    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return {}
        bbe = df[df["type"] == "X"] if "type" in df.columns else df
        return {
            "pa": int(df["pitch_number"].count()) if "pitch_number" in df.columns else len(df),
            "xwoba_recent": float(df["estimated_woba_using_speedangle"].mean())
                if "estimated_woba_using_speedangle" in df.columns else None,
            "barrel_pct_recent": float((bbe["launch_speed_angle"] == 6).mean() * 100)
                if "launch_speed_angle" in bbe.columns and len(bbe) > 0 else None,
            "hard_hit_recent": float((bbe["launch_speed"] >= 95).mean() * 100)
                if "launch_speed" in bbe.columns and len(bbe) > 0 else None,
            "avg_ev_recent": float(bbe["launch_speed"].mean())
                if "launch_speed" in bbe.columns and len(bbe) > 0 else None,
        }
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Team-level pitching - used as fallback when probable pitcher is TBD
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_team_pitching(team_id: int, season: int = CURRENT_SEASON) -> dict:
    """
    Get team-aggregate pitching stats. Used when the day's starter is TBD
    or unknown (e.g. bullpen game). Returns dict shaped like a single
    pitcher row so models can drop it in as a proxy.
    """
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=pitching&season={season}&sportIds=1"
        f"&teamId={team_id}"
    )
    splits = []
    for attempt in range(2):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [])
            if splits:
                break
        except Exception:
            if attempt < 1:
                import time
                time.sleep(0.5)
            continue
    if not splits:
        return {}
    for s in splits:
        try:
            team = s.get("team", {}) or {}
            if team.get("id") != team_id:
                continue
            st_ = s.get("stat", {}) or {}
            return {
                "player_id": None,
                "player_name": f"Team Avg ({team.get('abbreviation', '')})",
                "is_team_avg": True,
                "era": _safe_float(st_.get("era")),
                "whip": _safe_float(st_.get("whip")),
                "hr9": _safe_float(st_.get("homeRunsPer9")),
                "bb9": _safe_float(st_.get("walksPer9Inn") or st_.get("baseOnBallsPer9Inn")),
                "k9": _safe_float(st_.get("strikeoutsPer9Inn") or st_.get("strikeOutsPer9Inn")),
                "ip": _safe_float(st_.get("inningsPitched")),
                "earned_runs": _safe_int(st_.get("earnedRuns")),
                "home_run": _safe_int(st_.get("homeRuns")),
            }
        except Exception:
            continue
    return {}


def get_team_pitching_proxy(team_id: int, season: int = CURRENT_SEASON,
                              hr_penalty: float = 1.10) -> dict:
    """
    Wrapper that returns team pitching with a small HR-allowance penalty applied.
    TBD/bullpen games tend to allow MORE HRs than a typical starter.
    """
    proxy = get_team_pitching(team_id, season)
    if not proxy:
        return {}
    if proxy.get("hr9") is not None:
        proxy["hr9_tbd_adjusted"] = round(proxy["hr9"] * hr_penalty, 2)
    proxy["tbd_proxy"] = True
    return proxy


@st.cache_data(ttl=21600)  # 6 hour cache
def get_team_bullpen_hr9(team_id: int, season: int = CURRENT_SEASON) -> float | None:
    """
    Fetch the team's BULLPEN-ONLY HR/9 (excludes starters).

    Used for bullpen-leverage adjustment: when a starter exits early, the
    bullpen pitches the remainder. A team with a high bullpen HR/9
    (Rockies, Twins, A's, etc.) means even good starters expose hitters to
    HR-vulnerable relievers later in the game.

    Returns None if data unavailable (caller should fall back to league avg).
    """
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=pitching&season={season}&sportIds=1"
        f"&teamId={team_id}&playerPool=ALL"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
    except Exception:
        return None

    # Aggregate HR and IP across all pitchers on the team where the pitcher
    # appears as a reliever (gamesStarted < 50% of appearances).
    total_hr = 0
    total_ip = 0.0
    for s in splits:
        try:
            stat = s.get("stat", {}) or {}
            games = _safe_int(stat.get("gamesPlayed")) or 0
            starts = _safe_int(stat.get("gamesStarted")) or 0
            if games == 0:
                continue
            if starts / games >= 0.5:
                continue  # Classified as a starter
            hr = _safe_int(stat.get("homeRuns")) or 0
            ip = _safe_float(stat.get("inningsPitched")) or 0.0
            total_hr += hr
            total_ip += ip
        except Exception:
            continue

    if total_ip < 30:
        return None
    return round((total_hr * 9.0) / total_ip, 2)


@st.cache_data(ttl=3600)
def get_team_hitting_aggregates(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Returns one row per team with aggregate hitting tendencies:
      - team_id, team_abbr, k_pct, bb_pct, hr_per_pa, iso

    Iterates all 30 teams, fetching team-level hitting from MLB Stats API.
    Cached 1 hour.
    """
    # First, fetch the list of all MLB teams
    try:
        teams_url = f"https://statsapi.mlb.com/api/v1/teams?sportId=1&season={season}"
        r = requests.get(teams_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        teams_data = r.json()
    except Exception:
        return pd.DataFrame()

    rows = []
    for team in teams_data.get("teams", []):
        team_id = team.get("id")
        team_abbr = team.get("abbreviation") or team.get("teamCode")
        team_name = team.get("name")
        if not team_id:
            continue
        # Fetch this team's hitting stats
        try:
            url = (
                f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
                f"?stats=season&group=hitting&season={season}&sportIds=1"
            )
            tr = requests.get(url, headers=HEADERS, timeout=15)
            tr.raise_for_status()
            data = tr.json()
        except Exception:
            continue
        # v43.62 (reviewer suggestion): use an explicit flag for the
        # nested loops. The old pattern (`break` inner + `if rows[-1] is
        # this team: break`) is brittle if a team returns multiple split
        # blocks AND the last-pushed row happens to match this team_id.
        # An explicit flag is unambiguous.
        _team_done = False
        for split_block in data.get("stats", []):
            if _team_done:
                break
            for sp in split_block.get("splits", []):
                stat = sp.get("stat") or {}
                try:
                    pa = int(stat.get("plateAppearances") or 0)
                    k = int(stat.get("strikeOuts") or 0)
                    hr = int(stat.get("homeRuns") or 0)
                    ab = int(stat.get("atBats") or 0)
                    hits = int(stat.get("hits") or 0)
                    doubles = int(stat.get("doubles") or 0)
                    triples = int(stat.get("triples") or 0)
                    walks = int(stat.get("baseOnBalls") or 0)
                except (ValueError, TypeError):
                    continue
                if pa == 0 or ab == 0:
                    continue
                iso = ((doubles + 2*triples + 3*hr) / ab) if ab > 0 else None
                rows.append({
                    "team_id": team_id,
                    "team_abbr": team_abbr,
                    "team_name": team_name,
                    "pa": pa,
                    "k_pct": round(k / pa * 100, 1),
                    "bb_pct": round(walks / pa * 100, 1),
                    "hr_per_pa": round(hr / pa * 100, 2),
                    "iso": round(iso, 3) if iso is not None else None,
                })
                _team_done = True
                break  # Just take the first split (regular season)
    return pd.DataFrame(rows)


# =============================================================================
# REAL ROOF STATUS — pulls actual condition from MLB game feed
# =============================================================================
# MLB Stats API includes the literal roof status in gameData.weather.condition
# for indoor/retractable venues. Values seen in production:
#   - "Roof Closed"   (retractable park with roof closed for this game)
#   - "Dome"          (permanent dome, e.g., Tropicana Field)
#   - "Clear", "Partly Cloudy", "Cloudy", "Rain"   (outdoor or roof open)
# This gives us the GROUND TRUTH instead of guessing from temperature.

@st.cache_data(ttl=300)  # 5min — refresh often, roof decisions can change pre-game
def get_game_roof_status(game_pk: int) -> dict:
    """
    Return real roof status + MLB-reported weather for a game.

    Returns dict:
        {
            "roof_closed": bool | None,  # None = unknown
            "condition": str,            # raw condition string from MLB
            "temp": float | None,        # MLB-reported temp
            "wind": str,                 # MLB-reported wind ("X mph, From CF")
        }
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    out = {"roof_closed": None, "condition": "", "temp": None, "wind": ""}
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return out
        data = r.json()
        wx = data.get("gameData", {}).get("weather", {})
        condition = wx.get("condition", "") or ""
        out["condition"] = condition
        out["wind"] = wx.get("wind", "") or ""
        temp_str = wx.get("temp", "")
        try:
            out["temp"] = float(temp_str) if temp_str else None
        except (ValueError, TypeError):
            out["temp"] = None
        # Decide roof state from condition string
        cond_lower = condition.lower()
        if "roof closed" in cond_lower or "dome" in cond_lower:
            out["roof_closed"] = True
        elif "roof open" in cond_lower:
            out["roof_closed"] = False
        elif condition and any(
            outdoor_kw in cond_lower
            for outdoor_kw in ("clear", "sunny", "cloud", "rain", "overcast",
                                "wind", "partly", "drizzle", "mist", "fog",
                                "snow", "shower")
        ):
            # Any actual weather description = open-air conditions
            out["roof_closed"] = False
        # else: leave as None (unknown)
        return out
    except Exception:
        return out


# =============================================================================
# SLATE-WIDE TRANSACTIONS — surface roster moves so user knows what changed
# =============================================================================

@st.cache_data(ttl=900)  # 15min — roster moves can happen mid-day
def get_recent_transactions(days_back: int = 2, _stats_day: str = "") -> pd.DataFrame:
    """
    Pull all MLB roster transactions from the last N days.

    Returns df with columns:
        date, type_code, type_desc, player_name, player_id, from_team, to_team,
        description

    type_code legend (most important for hitter/pitcher impact):
        TR  = Traded
        SC  = Signed as free agent
        DFA = Designated for Assignment
        REL = Released
        OUT = Outrighted
        CU  = Called Up (from minors)
        SD  = Sent Down (to minors)
        SCL = Selected from minors (added to 40-man)
        IL  = Placed on Injured List
        RTN = Returned to team / activated from IL
        STA = Status change

    These directly impact:
      - Trades/signings: stats/team data may not yet reflect new team
      - DFA/release: player no longer available
      - Call-ups: new player to evaluate (low PA, insufficient sample)
      - IL moves: pitcher fresh from IL = role detection adjustment
    """
    from datetime import date, timedelta
    end_d = date.today()
    start_d = end_d - timedelta(days=days_back)
    url = (
        "https://statsapi.mlb.com/api/v1/transactions"
        f"?startDate={start_d.isoformat()}&endDate={end_d.isoformat()}"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
    except Exception:
        return pd.DataFrame()
    rows = []
    for txn in data.get("transactions", []):
        person = txn.get("person", {}) or {}
        from_team = (txn.get("fromTeam") or {}).get("name", "")
        to_team = (txn.get("toTeam") or {}).get("name", "")
        rows.append({
            "date": txn.get("date", ""),
            "type_code": txn.get("typeCode", ""),
            "type_desc": txn.get("typeTr", "") or txn.get("typeDesc", ""),
            "player_name": person.get("fullName", ""),
            "player_id": person.get("id"),
            "from_team": from_team,
            "to_team": to_team,
            "description": txn.get("description", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # Sort newest first
        df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# PITCHER PRIMARY POSITION (SP / RP / P) — MLB's official role label
# =============================================================================
# MLB Stats API returns each player's primaryPosition. For pitchers this is
# usually one of:
#   - "SP"  (Starting Pitcher)
#   - "RP"  (Relief Pitcher)
#   - "P"   (generic Pitcher - ambiguous)
#   - "TWP" (Two-Way Player, e.g., Ohtani)
# When MLB says SP, trust it. When they say RP but we have them as a probable
# starter today, that's a real signal they're being used as an opener.

@st.cache_data(ttl=3600)
def get_pitcher_primary_positions(pitcher_ids: tuple) -> dict:
    """
    Bulk-fetch primary position designations for a tuple of pitcher IDs.
    Returns {player_id: "SP" | "RP" | "P" | "TWP" | ""}.
    """
    if not pitcher_ids:
        return {}
    # MLB Stats API people endpoint supports comma-separated IDs (up to ~100)
    ids_str = ",".join(str(int(pid)) for pid in pitcher_ids if pid)
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}"
    out = {}
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return {}
        data = r.json()
        for person in data.get("people", []):
            pid = person.get("id")
            pos = (person.get("primaryPosition") or {}).get("abbreviation", "")
            if pid:
                try:
                    out[int(pid)] = pos
                except (ValueError, TypeError):
                    continue
        return out
    except Exception:
        return {}
