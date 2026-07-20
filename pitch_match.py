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

import numpy as np
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

    v43.29 (reviewer-validated): the pitcher_hand parameter added in v43.20
    was reverted. Savant's `&hand=` on this endpoint filters by BATTER
    handedness, not pitcher handedness — so the v43.20 attempt produced
    LHB-only data when we requested pitcher_hand="L" (rather than data
    vs LHP). Half the per-batter lookups silently returned empty.
    Going back to hand-agnostic for the hitter side; pitcher side stays
    hand-split via the separate verified endpoint.
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

        # v43.30 (reviewer-validated, CRITICAL accuracy fix): adaptive
        # usage reweighting REMOVED. The old code multiplied pitcher_usage
        # by 0.55-1.35 based on hitter xwOBA against that pitch — the
        # theory being "smart pitchers reduce what hitters crush, throw
        # more of what hitters can't hit." But:
        #   1. It directly fights the model's stated purpose. A hitter who
        #      crushes the pitcher's main pitch (xwOBA .420 on 60% FF)
        #      should get a HIGHER pitch-match score, not LOWER. The
        #      adaptation reduced that FF contribution to the weighted
        #      average, pulling the score DOWN.
        #   2. No empirical evidence pitchers adapt this predictably.
        #      Pitchers throw their best pitches more often, full stop.
        #   3. The factors were hardcoded — every pitcher got the same
        #      adaptation regardless of stuff, control, or season history.
        # Simulated case: hitter .420 xwOBA on 60% main pitch + .280 on
        # 40% secondary → raw weighted .394 vs adapted .373. The hitter
        # punishes the matchup but the score moved the wrong way.
        # Reviewer recommendation: cut it. Honest signal beats
        # principle-driven distortion.
        usage = usage_raw  # no adapt_factor anymore

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
            "pitcher_usage_adjusted": float(usage),  # v43.30: same as raw now
            "adapt_factor": 1.0,                     # v43.30: kept for back-compat
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
        # Convert to 0-100 (barrel% range ~2-20%).
        # v43.30 (reviewer-validated): divisor aligned to per-pitch cap.
        # The per-pitch barrel is hard-capped at 20% (line ~199), so
        # weighted_barrel / has_barrel can never exceed 20. Previous
        # divisor 20.0 with -2 floor gave max score (20-2)/20*100 = 90
        # — meaning the ceiling was unreachable and the top of the
        # scale was structurally compressed. The fix: divisor 18.0
        # (= cap 20 minus floor 2) so a hitter who maxes out the
        # per-pitch cap on every pitcher pitch reaches exactly 100.
        # Doesn't change Brier — display-only stretch on the high end.
        pitch_hr_score = round(max(0, min(100, (avg_barrel - 2.0) / 18.0 * 100)), 1)
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
            pitch_volatility = round(float(np.std(usages)), 1)
        else:
            pitch_volatility = None
    except Exception:
        pitch_volatility = None

    # v43.29 (user-requested): mini-arsenal string showing hitter's
    # performance against the pitcher's TOP 3 most-thrown pitches.
    # Example: "FF 52%: brl 11%/slg .520 | SL 24%: brl 5%/slg .310 | CH 14%: brl 2%/slg .240"
    # This is the per-matchup "what's this hitter going to see and how do
    # they handle it" snapshot the user asked for. Computed from the same
    # breakdown the existing best_pitch / worst_pitch uses.
    mini_arsenal = ""
    try:
        # Sort breakdown by pitcher_usage_raw desc, take top 3 with hitter data
        with_data = [
            b for b in breakdown
            if b.get("hitter_xwoba_vs") is not None
            and b.get("pitcher_usage_raw", 0) > 0
        ]
        top3 = sorted(with_data, key=lambda b: -b.get("pitcher_usage_raw", 0))[:3]
        parts = []
        for b in top3:
            pitch = b.get("pitch") or "?"
            usage = b.get("pitcher_usage_raw", 0)
            brl = b.get("hitter_barrel_vs")
            slg = b.get("hitter_slg_vs")
            xwoba = b.get("hitter_xwoba_vs")
            # Build segment with whatever stats we have
            stat_parts = []
            if brl is not None and not pd.isna(brl):
                stat_parts.append(f"brl {float(brl):.0f}%")
            if slg is not None and not pd.isna(slg):
                stat_parts.append(f"slg {float(slg):.3f}".replace("0.", "."))
            if xwoba is not None and not pd.isna(xwoba) and not stat_parts:
                # Fallback to xwoba if barrel/slg missing
                stat_parts.append(f"xwoba {float(xwoba):.3f}".replace("0.", "."))
            stats_str = "/".join(stat_parts) if stat_parts else "—"
            # v43.30 (reviewer-validated): Savant's pitch-name strings use
            # "4-Seam Fastball" (digit-dash form), not "Four-Seam Fastball".
            # The old map missed virtually every key and fell through to
            # `pitch[:3].upper()` — producing "4-S" instead of "FF". Cover
            # both forms defensively in case Savant ever normalizes.
            pitch_abbr = {
                "4-Seam Fastball": "FF", "Four-Seam Fastball": "FF",
                "Sinker": "SI", "2-Seam Fastball": "SI", "Two-Seam Fastball": "SI",
                "Cutter": "FC",
                "Slider": "SL", "Sweeper": "SW", "Slurve": "SV",
                "Curveball": "CU", "Knuckle Curve": "KC", "Slow Curve": "SC",
                "Changeup": "CH", "Splitter": "FS", "Split-Finger": "FS",
                "Forkball": "FO", "Knuckleball": "KN", "Eephus": "EP",
                "Screwball": "SC",
            }.get(pitch, pitch[:3].upper() if pitch else "?")
            # v43.30 (reviewer-validated): Savant returns pitch_usage ALREADY
            # as a percent (e.g. 20.6 for 20.6%). The previous formatter
            # multiplied by 100 a second time, producing "2060%". Drop the
            # ×100. Note: exposure_edge logic and pitch_match_score itself
            # are unaffected — the score divides by total_weight, so absolute
            # scale cancels. This was display-only nonsense.
            parts.append(f"{pitch_abbr} {usage:.0f}%: {stats_str}")
        mini_arsenal = " | ".join(parts)
    except Exception:
        mini_arsenal = ""

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
        "mini_arsenal": mini_arsenal,  # v43.29: user-requested top-3 pitch breakdown
        "breakdown": breakdown,
    }
