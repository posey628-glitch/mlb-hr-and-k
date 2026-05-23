"""
sleepers.py
============
Identifies HR sleepers - hitters whose today HR_PROB greatly exceeds
their season HR pace. Big positive sleeper_score = "you wouldn't expect
this guy, but conditions favor him."

Also computes Grand Slam compound probability per hitter.

Real data only - NaN propagates when stats are missing.
No fake-50 defaults, no league-average fillna shortcuts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ===========================================================================
# HR probability per hitter
# ===========================================================================

def hr_probability(
    matchup_df: pd.DataFrame,
    pitcher_row: pd.Series | None,
    hr_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Composite HR probability score on a 0-100 percentile basis.
    Components: hitter barrel%, ISO, hard-hit%, plus pitcher allowance.
    NaN when no real component data exists.
    """
    if matchup_df is None or matchup_df.empty:
        return matchup_df

    df = matchup_df.copy()
    p = pitcher_row if pitcher_row is not None else pd.Series(dtype=float)

    def _rank(s):
        """Percentile rank 0-100; NaN stays NaN (honest)."""
        return s.rank(pct=True) * 100

    # Build weighted score using only components with real data
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

    # Honest score: NaN when no real data at all
    components = weighted_score / weight_sums.replace(0, np.nan)

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

    if "hr_prob" in df.columns:
        hr_pct = df["hr_prob"].rank(pct=True) * 100
    else:
        hr_pct = pd.Series([np.nan] * len(df), index=df.index)

    if season_hr_col in df.columns:
        season_pct = df[season_hr_col].rank(pct=True) * 100
    else:
        season_pct = pd.Series([np.nan] * len(df), index=df.index)

    df["sleeper_score"] = (hr_pct - season_pct).round(1)

    # Flag the strongest sleepers (top 25% delta) - only if real scores
    if df["sleeper_score"].notna().any():
        threshold = df["sleeper_score"].quantile(0.75)
        df["is_sleeper"] = df["sleeper_score"] >= threshold
    else:
        df["is_sleeper"] = False

    return df


# ===========================================================================
# Grand Slam probability
# ===========================================================================

def grand_slam_probability(
    matchup_df: pd.DataFrame,
    pitcher_row: pd.Series | None,
    hr_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Compound probability of a Grand Slam:
      P(GS) ≈ P(HR | PA) × P(bases loaded | this lineup position)

    Bases-loaded probability is approximated by the typical traffic ahead
    of each lineup spot, multiplied by an estimate of how much the pitcher
    walks/allows base runners.
    """
    if matchup_df is None or matchup_df.empty:
        return matchup_df

    df = matchup_df.copy()

    # Approximate "bases loaded when you bat" frequency by lineup position
    # (rough heuristic from historical play-by-play - cleanup spot sees most)
    order_traffic = {
        1: 0.005, 2: 0.012, 3: 0.025, 4: 0.040, 5: 0.035,
        6: 0.028, 7: 0.022, 8: 0.015, 9: 0.010,
    }
    if "lineup_pos" in df.columns:
        df["order_traffic"] = df["lineup_pos"].map(order_traffic)
    else:
        df["order_traffic"] = np.nan

    # Pitcher walk rate boost - high BB% pitchers create more bases-loaded situations
    p = pitcher_row if pitcher_row is not None else pd.Series(dtype=float)
    p_bb = p.get("bb_percent", np.nan)
    if pd.isna(p_bb):
        bb_factor = 1.0  # no boost when we have no data
    else:
        bb_factor = float(np.clip(p_bb / 8.0, 0.7, 1.5))  # league avg BB% ~ 8

    df["bases_loaded_prob"] = df["order_traffic"] * bb_factor

    # Per-PA HR rate from hr_prob if available, else from season HR/PA
    if "hr_prob" in df.columns and df["hr_prob"].notna().any():
        # Convert percentile-style hr_prob (0-100) to a rough per-PA rate
        hr_pa_rate = (df["hr_prob"] / 100) * 0.05  # cap at 5% per PA top of scale
    elif "home_run" in df.columns and "pa" in df.columns:
        hr_pa_rate = df["home_run"] / df["pa"].replace(0, np.nan)
    else:
        hr_pa_rate = pd.Series([np.nan] * len(df), index=df.index)

    df["hr_pa_rate"] = hr_pa_rate

    # Grand Slam = HR + bases loaded
    df["gs_score"] = (
        df["bases_loaded_prob"] * df["hr_pa_rate"] * hr_mult * 100
    ).round(3)

    return df
