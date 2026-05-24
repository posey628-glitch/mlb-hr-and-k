"""
sleepers.py
============
Identifies dark-horse HR picks and grand slam candidates for today's slate.

A "sleeper HR" is a hitter whose composite HR_PROB is well above what their
public profile (HR total, ownership likelihood) would suggest. Surprise edge.

A "grand slam candidate" is a hitter whose compound P(GS) is highest:
  P(bases loaded when up) × P(HR in that PA)

Both metrics get HR_MULT (park × weather) applied at the very end so the
ranking always reflects today's specific game conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hr_probability(
    matchup_df: pd.DataFrame,
    pitcher_row: pd.Series,
    hr_mult: float,
) -> pd.DataFrame:
    """
    Add per-hitter HR probability for THIS GAME.

    Components (each percentile-ranked across the lineup, then weighted):
      - barrel_pct (hitter)       0.30
      - iso (hitter)              0.20
      - hard_hit (hitter)         0.15
      - pitcher HR/9 surrogate    0.15  (uses pitcher barrel_allowed)
      - pitcher xwOBA allowed     0.10
      - pitcher hard-hit allowed  0.10

    Final value is scaled by HR_MULT (park × weather), so a Coors day with
    20mph out wind will lift everyone, and Oracle Park on a cold night will
    suppress everyone, RELATIVE to their season-level expectation.
    """
    if matchup_df.empty:
        return matchup_df

    df = matchup_df.copy()
    p = pitcher_row if pitcher_row is not None else pd.Series(dtype=float)

    def _rank(s):
        """Percentile rank 0-100; NaN stays NaN (honest)."""
        return s.rank(pct=True) * 100

    # Build weighted score per hitter using only components with real data
    weight_sums = pd.Series(0.0, index=df.index)
    weighted_score = pd.Series(0.0, index=df.index)

    for col, weight in [("barrel_pct", 0.30), ("iso", 0.20), ("hard_hit", 0.15)]:
        if col in df.columns:
            r = _rank(df[col])
            mask = r.notna()
            weight_sums[mask] += weight
            weighted_score[mask] += r[mask] * weight

    # Pitcher context - only adds when real pitcher data exists
    p_barrel = p.get("barrel_batted_rate", np.nan)
    p_xwoba = p.get("xwoba", np.nan)
    p_hh = p.get("hard_hit_percent", np.nan)

    def _pitcher_boost(val, neutral, scale):
        if pd.isna(val):
            return None
        return float(np.clip(50 + (val - neutral) * scale, 0, 100))

    pitcher_boosts = [
        (_pitcher_boost(p_barrel, 7.5, 4), 0.15),
        (_pitcher_boost(p_xwoba, 0.310, 250), 0.10),
        (_pitcher_boost(p_hh, 38.0, 1.5), 0.10),
    ]
    for boost_val, w in pitcher_boosts:
        if boost_val is not None:
            weight_sums += w
            weighted_score += boost_val * w

    # Honest score: NaN when no real data at all, otherwise weight-scaled
    components = (weighted_score / weight_sums.replace(0, np.nan))

    # Apply today's park × weather multiplier
    df["hr_prob"] = (components * hr_mult).round(2)
    df["hr_mult_today"] = hr_mult
    return df


def find_sleepers(hr_df: pd.DataFrame, season_hr_col: str = "home_run") -> pd.DataFrame:
    """
    Flag hitters whose today HR_PROB greatly exceeds their season HR pace.

    Sleeper score = HR_PROB percentile MINUS season HR percentile.
    Real data only - NaN if either input missing.
    """
    df = hr_df.copy()
    if df.empty:
        return df

    hr_pct = df["hr_prob"].rank(pct=True) * 100 if "hr_prob" in df.columns else pd.Series([np.nan] * len(df))
    if season_hr_col in df.columns:
        season_pct = df[season_hr_col].rank(pct=True) * 100
    else:
        season_pct = pd.Series([np.nan] * len(df), index=df.index)

    df["sleeper_score"] = (hr_pct - season_pct).round(1)
    df["is_sleeper"] = df["sleeper_score"] >= 25  # arbitrary cutoff - tune
    return df


def grand_slam_probability(
    matchup_df: pd.DataFrame,
    pitcher_row: pd.Series,
    hr_mult: float,
) -> pd.DataFrame:
    """
    Compound probability: P(bases loaded for this hitter) × P(HR | PA).

    P(bases loaded) drivers:
      - Batting order position (3-6 hit with traffic most)
      - On-base ability of hitters in front (here: lineup-average OBP)
      - Pitcher WHIP / BB% (pitcher who allows traffic)

    P(HR | PA) is the same per-hitter HR rate used in hr_probability,
    then both get the park×weather multiplier.
    """
    if matchup_df.empty:
        return matchup_df

    df = matchup_df.copy()

    # 1) Batting-order traffic factor (1-9). 3-6 see most loaded bases.
    order_traffic = {1: 0.5, 2: 0.7, 3: 1.0, 4: 1.1, 5: 1.1, 6: 1.0, 7: 0.8, 8: 0.6, 9: 0.55}
    if "lineup_pos" not in df.columns:
        df["lineup_pos"] = range(1, len(df) + 1)
    df["order_traffic"] = df["lineup_pos"].map(order_traffic).fillna(0.7)

    # 2) Pitcher-traffic factor (high WHIP/BB% pitchers create loaded bases)
    p = pitcher_row if pitcher_row is not None else pd.Series(dtype=float)
    bb_pct = p.get("bb_percent")
    if bb_pct is None or pd.isna(bb_pct):
        pitcher_traffic = 1.0  # No data → neutral, not faked league avg
    else:
        pitcher_traffic = float(np.clip(1 + (bb_pct - 8.0) * 0.04, 0.7, 1.5))

    # 3) Lineup OBP context - if "obp" not in df, fallback to xwoba as proxy
    obp_col = "obp" if "obp" in df.columns else ("xwoba" if "xwoba" in df.columns else None)
    if obp_col:
        lineup_obp = df[obp_col].mean()
        league_avg = 0.320 if obp_col == "obp" else 0.310
        lineup_factor = float(np.clip(1 + (lineup_obp - league_avg) * 3, 0.7, 1.4))
    else:
        lineup_factor = 1.0

    # 4) Per-hitter HR rate (re-use hr_prob if present, else compute quickly)
    if "hr_prob" not in df.columns:
        df = hr_probability(df, pitcher_row, hr_mult)
    base_hr = df["hr_prob"] / 100.0  # normalize 0-1

    # Compound it
    df["gs_score"] = (
        base_hr * df["order_traffic"] * pitcher_traffic * lineup_factor * hr_mult * 100
    ).round(2)
    df["gs_traffic_factor"] = round(pitcher_traffic * lineup_factor, 3)

    return df.sort_values("gs_score", ascending=False)
