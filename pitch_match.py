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
    Cross today's pitcher's usage against this hitter's per-pitch xwOBA.
    Returns dict with overall score + best/worst pitch matchup.
    """
    if pitcher_arsenal.empty or hitter_arsenal.empty:
        return {"pitch_match_score": None}

    # Normalize pitch name column
    p_name_col = "pitch_name" if "pitch_name" in pitcher_arsenal.columns else "pitch_type"
    h_name_col = "pitch_name" if "pitch_name" in hitter_arsenal.columns else "pitch_type"
    usage_col = "pitch_usage" if "pitch_usage" in pitcher_arsenal.columns else "usage"

    # Hitter xwOBA by pitch type
    h_xwoba = {}
    for _, r in hitter_arsenal.iterrows():
        xw = r.get("est_woba", r.get("xwoba", None))
        if xw is not None and not pd.isna(xw):
            h_xwoba[r[h_name_col]] = float(xw)

    # Weight by pitcher usage
    total_weight = 0.0
    weighted_xwoba = 0.0
    breakdown = []
    for _, r in pitcher_arsenal.iterrows():
        usage = r.get(usage_col, 0)
        if pd.isna(usage) or usage <= 0:
            continue
        pitch = r[p_name_col]
        h_val = h_xwoba.get(pitch, 0.310)  # league avg fallback
        weighted_xwoba += usage * h_val
        total_weight += usage
        breakdown.append({
            "pitch": pitch,
            "pitcher_usage": float(usage),
            "hitter_xwoba_vs": float(h_val),
            "contribution": float(usage * h_val),
        })

    if total_weight == 0:
        return {"pitch_match_score": None}

    avg_xwoba = weighted_xwoba / total_weight
    # Convert to 0-100 score (xwOBA range ~0.250 - 0.450)
    score = max(0, min(100, (avg_xwoba - 0.250) / 0.200 * 100))

    # Best and worst pitch for this hitter in this matchup
    breakdown_sorted = sorted(breakdown, key=lambda x: x["hitter_xwoba_vs"], reverse=True)
    best = breakdown_sorted[0] if breakdown_sorted else None
    worst = breakdown_sorted[-1] if breakdown_sorted else None

    return {
        "pitch_match_score": round(score, 1),
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
