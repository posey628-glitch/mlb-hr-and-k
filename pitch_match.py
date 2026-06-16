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
    h_pa = {}  # NEW: per-pitch PA counts for sample-size shrinkage
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
        # Track PA so we can shrink small-sample per-pitch barrel rates
        pa = r.get("pa", r.get("plate_appearances", r.get("total_pitches", None)))
        if pa is not None and not pd.isna(pa):
            try:
                h_pa[pitch] = float(pa)
            except (TypeError, ValueError):
                pass

    # Weight by pitcher usage - BUT apply real-world adaptation:
    # MLB pitchers and analytics teams DO adjust pitch mix vs specific hitters.
    # If hitter has elite xwOBA vs a pitch (>.400), pitcher throws ~50% less of it.
    # If hitter is weak vs a pitch (<.250), pitcher throws ~30% more of it.
    # League average xwOBA ~.310 = neutral, no usage adjustment.
    # Reference: actual data shows pitch mix shifts ~20-40% based on hitter weakness.
    total_weight = 0.0
    weighted_xwoba = 0.0
    weighted_slg = 0.0
    weighted_barrel = 0.0
    has_slg = 0.0
    has_barrel = 0.0
    breakdown = []
    missing_usage = 0.0
    for _, r in pitcher_arsenal.iterrows():
        usage_raw = r.get(usage_col, 0)
        if pd.isna(usage_raw) or usage_raw <= 0:
            continue
        pitch = r[p_name_col]
        h_val = h_xwoba.get(pitch)
        if h_val is None:
            missing_usage += usage_raw
            continue

        # ADAPTIVE USAGE: shift toward pitches hitter is weaker against
        # h_val = hitter's xwOBA vs this pitch
        # If h_val > 0.380 (elite), pitcher avoids → 0.55× usage
        # If h_val 0.340-0.380 (good), pitcher reduces → 0.75× usage
        # If h_val 0.280-0.340 (avg), no change → 1.0× usage
        # If h_val 0.240-0.280 (weak), pitcher uses more → 1.20× usage
        # If h_val < 0.240 (very weak), pitcher throws lots → 1.35× usage
        if h_val >= 0.380:
            adapt_factor = 0.55
        elif h_val >= 0.340:
            adapt_factor = 0.75
        elif h_val >= 0.280:
            adapt_factor = 1.0
        elif h_val >= 0.240:
            adapt_factor = 1.20
        else:
            adapt_factor = 1.35
        usage = usage_raw * adapt_factor

        # v43.18 (user-reported "Abrams 100 arsenal HR"): SEPARATE barrel
        # shrinkage from xwOBA shrinkage. xwOBA stabilizes faster (prior 30
        # is fine), but barrel rates on small per-pitch samples are very
        # noisy — a hitter with 5% season barrel can show 30% on one pitch
        # in 50 PA, and at prior=30 that gets 50/80=63% weight, enough to
        # push the weighted average above the 22% ceiling and produce
        # spurious 100 scores. Doubled barrel prior to 75 PA shrinks small
        # samples more aggressively toward 7.5% league avg.
        pitch_pa = h_pa.get(pitch, 20)
        BARREL_PRIOR_PA = 75   # v43.18: was 30
        XWOBA_PRIOR_PA = 30    # unchanged — xwOBA is more stable
        shrink_weight_xwoba = pitch_pa / (pitch_pa + XWOBA_PRIOR_PA)
        shrink_weight_brl = pitch_pa / (pitch_pa + BARREL_PRIOR_PA)

        # Apply Bayesian shrinkage to xwoba (default prior 0.320 = league avg)
        h_val_shrunk = h_val * shrink_weight_xwoba + 0.320 * (1 - shrink_weight_xwoba)

        weighted_xwoba += usage * h_val_shrunk
        total_weight += usage

        slg_val = h_slg.get(pitch)
        if slg_val is not None:
            slg_shrunk = slg_val * shrink_weight_xwoba + 0.400 * (1 - shrink_weight_xwoba)
            weighted_slg += usage * slg_shrunk
            has_slg += usage

        brl_val = h_barrel.get(pitch)
        if brl_val is not None:
            # SAMPLE-SIZE SHRINKAGE with v43.18 stronger prior
            brl_shrunk = brl_val * shrink_weight_brl + 7.5 * (1 - shrink_weight_brl)
            # v43.18: per-pitch HARD CAP at 20%. Even Aaron Judge's best
            # pitch barrel is ~18%. A shrunk value above 20% is the
            # residual of an inflated small-sample observation — clamp it
            # so it can't pull the weighted average to the ceiling alone.
            brl_shrunk = min(brl_shrunk, 20.0)
            weighted_barrel += usage * brl_shrunk
            has_barrel += usage

        breakdown.append({
            "pitch": pitch,
            "pitcher_usage_raw": float(usage_raw),
            "pitcher_usage_adjusted": float(usage),
            "adapt_factor": adapt_factor,
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
        avg_barrel = weighted_barrel / has_barrel
        # Convert to 0-100 (barrel% range ~2-18%)
        # CALIBRATION FIX: raised elite ceiling from 18% to 22% so only true
        # mash-the-pitcher-on-every-pitch matchups hit 100. Schmitt at 14.9%
        # overall barrel was hitting 100 because his per-pitch barrel vs the
        # pitcher's secondary pitches was elite. Now Schmitt-tier matchups
        # land around 70-80, leaving 90+ for genuine "barrel everything" cases.
        pitch_hr_score = round(max(0, min(100, (avg_barrel - 2.0) / 20.0 * 100)), 1)
        pitch_hr_basis = "barrel"
    elif has_slg > total_weight * 0.5:
        avg_slg = weighted_slg / has_slg
        # Convert to 0-100 (SLG range ~.300 - .650)
        pitch_hr_score = round(max(0, min(100, (avg_slg - 0.300) / 0.350 * 100)), 1)
        pitch_hr_basis = "slg"

    # Best and worst pitch for this hitter in this matchup (by xwOBA)
    breakdown_sorted = sorted(breakdown, key=lambda x: x["hitter_xwoba_vs"], reverse=True)
    best = breakdown_sorted[0] if breakdown_sorted else None
    worst = breakdown_sorted[-1] if breakdown_sorted else None

    # v42s: PITCH EXPOSURE EDGE (reviewer-validated display signal)
    # Counts how well a hitter's strengths align with what the pitcher
    # actually throws a lot of. A hitter who crushes sliders is only useful
    # if the pitcher throws sliders >25% of the time.
    #   +1 for each high-usage pitch (>25%) where hitter is elite (xwOBA > .360)
    #   -1 for each high-usage pitch (>25%) where hitter is bad (xwOBA < .260)
    # Range: typically -3 to +3. Display-only, NOT used in scoring (the
    # weighted xwOBA composite above already captures this; this is just a
    # cleaner at-a-glance signal).
    exposure_edge = 0
    for row in breakdown:
        if row["pitcher_usage_raw"] > 25.0:
            if row["hitter_xwoba_vs"] > 0.360:
                exposure_edge += 1
            elif row["hitter_xwoba_vs"] < 0.260:
                exposure_edge -= 1

    # v42s: PITCH MIX VOLATILITY (reviewer-validated display signal)
    # Standard deviation of pitcher's pitch usage percentages.
    # High std (>20) = predictable pitcher who relies heavily on 1-2 pitches.
    # Low std (<10) = balanced 4-5 pitch mix, harder to sit on any one pitch.
    # Display-only — gives the user context about how reliable the matchup
    # projection is. A predictable pitcher matchup is more "knowable."
    try:
        usages = [b.get("pitcher_usage_raw", 0) for b in breakdown]
        if usages:
            import numpy as _np
            pitch_volatility = round(float(_np.std(usages)), 1)
        else:
            pitch_volatility = None
    except Exception:
        pitch_volatility = None

    return {
        "pitch_match_score": round(score, 1),
        "pitch_hr_score": pitch_hr_score,
        "pitch_hr_basis": pitch_hr_basis,
        "weighted_xwoba": round(avg_xwoba, 3),
        "best_pitch": best["pitch"] if best else None,
        "best_pitch_xwoba": best["hitter_xwoba_vs"] if best else None,
        # use adjusted (post-adapt) usage as the displayed value
        "best_pitch_usage": best.get("pitcher_usage_adjusted", best.get("pitcher_usage_raw", 0)) if best else None,
        "worst_pitch": worst["pitch"] if worst else None,
        "worst_pitch_xwoba": worst["hitter_xwoba_vs"] if worst else None,
        "pitch_exposure_edge": exposure_edge,
        "pitch_volatility": pitch_volatility,
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
