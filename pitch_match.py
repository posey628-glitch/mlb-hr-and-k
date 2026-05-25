"""
pitch_match.py
===============
The killer matchup signal: how well a hitter's pitch-by-pitch profile
matches up against today's pitcher's arsenal.

Logic:
  1. For each hitter, pull their xwOBA per pitch type from Savant
     (e.g. Judge has .550 xwOBA vs FF, .280 vs SL).
  2. For today's pitcher, pull their pitch-mix usage % per pitch type
     (e.g. Strider throws 55% FF, 35% SL, 10% CB).
  3. Compute Pitch Match Score = weighted sum:
       sum_over_pitches( pitcher_usage * hitter_xwoba_vs_that_pitch )
     A high value = "hitter feasts on the pitches this guy throws most."

Also exposes per-pitch-type hitter stats so we can show:
  "vs FF (.380 xwOBA, 95mph avg) — pitcher throws 52% FF"
"""

from __future__ import annotations
import io
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/csv, */*",
}

CURRENT_SEASON = datetime.now().year


@st.cache_data(ttl=3600)
def get_hitter_pitch_arsenal(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    For each batter, returns one row per pitch type they faced (≥10 PA),
    with BA, SLG, wOBA, xwOBA, whiff%, K%, hard_hit%, run_value, and avg velo.

    Columns include: player_id, player_name, pitch_name, pitch_type,
    pitches, pa, ba, slg, woba, xwoba, est_woba, whiff_percent, k_percent,
    put_away, hard_hit_percent, run_value_per_100, velocity, etc.
    """
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
        f"?type=batter&pitchType=&year={season}&team=&min=10&hand=&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "last_name, first_name" in df.columns:
            df["player_name"] = df["last_name, first_name"].apply(
                lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
                if isinstance(s, str) and "," in s else s
            )
        return df
    except Exception as e:
        st.warning(f"Could not load hitter pitch arsenal: {e}")
        return pd.DataFrame()


def pitch_match_score(
    batter_id: int,
    pitcher_arsenal: pd.DataFrame,    # rows for one pitcher
    hitter_arsenal: pd.DataFrame,     # rows for one batter
) -> dict:
    """
    Cross today's pitcher's usage against this hitter's per-pitch metrics.
    Returns:
      - pitch_match_score: overall score (xwOBA-based, 0-100, higher = better hitter matchup)
      - pitch_hr_score: HR-specific score (SLG/barrel-based, 0-100)
      - best/worst pitch breakdown
    """
    if pitcher_arsenal.empty or hitter_arsenal.empty:
        return {"pitch_match_score": None, "pitch_hr_score": None}

    # Normalize pitch name column
    p_name_col = "pitch_name" if "pitch_name" in pitcher_arsenal.columns else "pitch_type"
    h_name_col = "pitch_name" if "pitch_name" in hitter_arsenal.columns else "pitch_type"
    usage_col = "pitch_usage" if "pitch_usage" in pitcher_arsenal.columns else "usage"

    # Hitter metrics by pitch type
    # xwoba: overall production
    # slg: HR-relevant slugging power
    # hard_hit: barrel proxy when barrel not available
    h_xwoba = {}
    h_slg = {}
    h_hard_hit = {}
    h_barrel = {}
    for _, r in hitter_arsenal.iterrows():
        pitch = r[h_name_col]
        xw = r.get("est_woba", r.get("xwoba", None))
        if xw is not None and not pd.isna(xw):
            h_xwoba[pitch] = float(xw)
        slg = r.get("slg", None)
        if slg is not None and not pd.isna(slg):
            h_slg[pitch] = float(slg)
        hh = r.get("hard_hit_percent", r.get("hard_hit", None))
        if hh is not None and not pd.isna(hh):
            h_hard_hit[pitch] = float(hh)
        brl = r.get("barrel_percent", r.get("barrel_batted_rate", None))
        if brl is not None and not pd.isna(brl):
            h_barrel[pitch] = float(brl)

    # Weight by pitcher usage - compute both xwOBA-based and HR-based scores
    total_weight = 0.0
    weighted_xwoba = 0.0
    weighted_slg = 0.0
    weighted_barrel = 0.0
    has_slg = 0.0  # weight covered by SLG data
    has_barrel = 0.0
    breakdown = []
    missing_usage = 0.0
    for _, r in pitcher_arsenal.iterrows():
        usage = r.get(usage_col, 0)
        if pd.isna(usage) or usage <= 0:
            continue
        pitch = r[p_name_col]
        h_val = h_xwoba.get(pitch)
        if h_val is None:
            missing_usage += usage
            continue
        weighted_xwoba += usage * h_val
        total_weight += usage

        # SLG-by-pitch (for HR matching)
        slg_val = h_slg.get(pitch)
        if slg_val is not None:
            weighted_slg += usage * slg_val
            has_slg += usage

        # Barrel by pitch (best HR signal)
        brl_val = h_barrel.get(pitch)
        if brl_val is not None:
            weighted_barrel += usage * brl_val
            has_barrel += usage

        breakdown.append({
            "pitch": pitch,
            "pitcher_usage": float(usage),
            "hitter_xwoba_vs": float(h_val),
            "hitter_slg_vs": slg_val,
            "hitter_barrel_vs": brl_val,
            "contribution": float(usage * h_val),
        })

    if total_weight == 0:
        return {"pitch_match_score": None, "pitch_hr_score": None}

    if missing_usage > total_weight:
        return {"pitch_match_score": None, "pitch_hr_score": None}

    avg_xwoba = weighted_xwoba / total_weight
    # Convert to 0-100 score (xwOBA range ~0.250 - 0.450)
    # Convert to 0-100 score - widened from 0.250-0.450 to 0.200-0.450
    # so very bad matchups (< 0.250 xwOBA) get spread out instead of all at 0
    score = max(0, min(100, (avg_xwoba - 0.200) / 0.250 * 100))

    # HR-specific score: prefer barrel, fall back to SLG, else None
    pitch_hr_score = None
    pitch_hr_basis = None
    if has_barrel > total_weight * 0.5:
        # >50% of pitcher usage has barrel data for this hitter
        avg_barrel = weighted_barrel / has_barrel
        # Convert to 0-100 (barrel% range ~2-18%)
        pitch_hr_score = round(max(0, min(100, (avg_barrel - 2.0) / 16.0 * 100)), 1)
        pitch_hr_basis = "barrel"
    elif has_slg > total_weight * 0.5:
        avg_slg = weighted_slg / has_slg
        # Convert to 0-100 (SLG range ~.300 - .600)
        pitch_hr_score = round(max(0, min(100, (avg_slg - 0.300) / 0.300 * 100)), 1)
        pitch_hr_basis = "slg"

    # Best and worst pitch for this hitter in this matchup (by xwOBA)
    breakdown_sorted = sorted(breakdown, key=lambda x: x["hitter_xwoba_vs"], reverse=True)
    best = breakdown_sorted[0] if breakdown_sorted else None
    worst = breakdown_sorted[-1] if breakdown_sorted else None

    return {
        "pitch_match_score": round(score, 1),
        "pitch_hr_score": pitch_hr_score,
        "pitch_hr_basis": pitch_hr_basis,
        "weighted_xwoba": round(avg_xwoba, 3),
        "best_pitch": best["pitch"] if best else None,
        "best_pitch_xwoba": best["hitter_xwoba_vs"] if best else None,
        "best_pitch_usage": best["pitcher_usage"] if best else None,
        "worst_pitch": worst["pitch"] if worst else None,
        "worst_pitch_xwoba": worst["hitter_xwoba_vs"] if worst else None,
        "breakdown": breakdown,
    }


def lineup_pitch_match(
    lineup: list[dict],
    pitcher_id: int,
    hitter_arsenal_all: pd.DataFrame,
    pitcher_arsenal_all: pd.DataFrame,
) -> pd.DataFrame:
    """Run pitch_match_score for every hitter in a lineup vs one pitcher."""
    if not lineup or pitcher_id is None or pd.isna(pitcher_id):
        return pd.DataFrame()

    pid_col_p = "player_id"
    p_arsenal = pitcher_arsenal_all[
        pitcher_arsenal_all.get(pid_col_p) == pitcher_id
    ] if pid_col_p in pitcher_arsenal_all.columns else pd.DataFrame()

    rows = []
    for p in lineup:
        bid = p.get("id")
        if not bid:
            continue
        h_arsenal = hitter_arsenal_all[
            hitter_arsenal_all.get("player_id") == bid
        ] if "player_id" in hitter_arsenal_all.columns else pd.DataFrame()
        res = pitch_match_score(bid, p_arsenal, h_arsenal)
        rows.append({
            "player_id": bid,
            "player_name": p["name"],
            **{k: v for k, v in res.items() if k != "breakdown"},
        })
    return pd.DataFrame(rows)
