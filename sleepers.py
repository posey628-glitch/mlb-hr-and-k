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
    # Naming note: this is a 0-100 COMPOSITE SCORE (not a probability).
    # We keep `hr_prob` as a column alias for backward compatibility with
    # downstream code, but the canonical name is `hr_score`.
    score = (components * hr_mult).round(2)
    df["hr_score"] = score
    df["hr_prob"] = score  # alias - same value
    df["hr_mult_today"] = round(float(hr_mult), 4)  # v43.30: kill float noise
    return df


def find_sleepers(hr_df: pd.DataFrame, season_hr_col: str = "home_run",
                    min_pa: int = 100) -> pd.DataFrame:
    """
    Flag hitters whose today HR_SCORE greatly exceeds their season HR pace.

    Sleeper score = today's HR_SCORE percentile MINUS season HR percentile.
    Real data only - NaN if either input missing.

    min_pa: hitters below this threshold get NaN sleeper_score. Below ~100 PA
    the season HR count is too noisy — a player with 80 PA and 5 HR may look
    like a sleeper just because their season percentile is artificially low
    (since the percentile is based on raw HR count without accounting for PA).
    """
    df = hr_df.copy()
    if df.empty:
        return df

    # Use hr_score if present, fall back to hr_prob (legacy)
    score_col = "hr_score" if "hr_score" in df.columns else "hr_prob"
    hr_pct = df[score_col].rank(pct=True) * 100 if score_col in df.columns else pd.Series([np.nan] * len(df))
    if season_hr_col in df.columns:
        season_pct = df[season_hr_col].rank(pct=True) * 100
    else:
        season_pct = pd.Series([np.nan] * len(df), index=df.index)

    df["sleeper_score"] = (hr_pct - season_pct).round(1)
    # Suppress sleeper_score for players below the PA threshold - their
    # season HR percentile is unreliable because raw HR count is noisy at
    # small samples (a 5-HR player in 80 PA vs 600 PA looks the same here).
    if "pa" in df.columns:
        df.loc[df["pa"].isna() | (df["pa"] < min_pa), "sleeper_score"] = np.nan
    df["is_sleeper"] = df["sleeper_score"] >= 25  # arbitrary cutoff - tune
    return df
