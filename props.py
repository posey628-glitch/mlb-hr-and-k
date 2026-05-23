"""
props.py
=========
Calibrated HR Probability and K Projection models with over/under math.
Real data only - returns None when underlying stats are missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# HR Probability - calibrated from barrel rate + matchup + park/weather
# ----------------------------------------------------------------------------

def hr_prob_per_pa(
    hitter_row: dict,
    pitcher_row: dict | None = None,
    park_hr_factor: float = 1.0,
    weather_hr_factor: float = 1.0,
) -> float | None:
    """
    Probability of at least one HR in a single PA for this hitter.
    Returns None if we don't have the underlying stats to compute it.

    Calibration: A 10% barrel-rate hitter at a neutral park is roughly 4.5% per PA.
    Scales linearly with barrel rate, then multiplied by environmental factors.
    """
    if not hitter_row:
        return None

    barrel = hitter_row.get("barrel_pct")
    if barrel is None or pd.isna(barrel) or barrel < 0:
        return None

    # Base HR rate per PA from barrel rate
    # Empirical relationship: HR/PA ≈ barrel_rate × 0.40 (about 40% of barrels become HRs)
    base = (barrel / 100) * 0.40

    # Matchup adjustment - high-xwoba pitchers concede more HRs
    matchup_mult = 1.0
    if pitcher_row:
        p_xwoba = pitcher_row.get("xwoba")
        if p_xwoba is not None and not pd.isna(p_xwoba):
            # League avg pitcher xwoba ~0.310. Scale linearly around that.
            matchup_mult = max(0.5, min(1.6, p_xwoba / 0.310))

    # FB% boost - fly-ball hitters get more HR chances
    fb_mult = 1.0
    fb = hitter_row.get("fb_pct")
    if fb is not None and not pd.isna(fb) and fb > 0:
        # League avg FB% ~24%. Boost above, suppress below.
        fb_mult = max(0.7, min(1.4, fb / 24))

    prob = base * matchup_mult * fb_mult * park_hr_factor * weather_hr_factor
    return float(min(0.30, max(0.0, prob)))  # cap at 30% per PA (sanity bound)


def hr_prob_per_game(
    hitter_row: dict,
    pitcher_row: dict | None = None,
    park_hr_factor: float = 1.0,
    weather_hr_factor: float = 1.0,
    expected_pa: float = 4.0,
) -> float | None:
    """
    Probability of >=1 HR in a game given expected_pa plate appearances.
    Uses binomial expansion: 1 - (1 - p_per_pa)^expected_pa
    """
    prob_per_pa = hr_prob_per_pa(
        hitter_row, pitcher_row, park_hr_factor, weather_hr_factor
    )
    if prob_per_pa is None:
        return None
    return float(1 - (1 - prob_per_pa) ** expected_pa)


# ----------------------------------------------------------------------------
# K Total Projection - blend season + recent K/9 with situational factors
# ----------------------------------------------------------------------------

def k_total_projection(
    pitcher_row: dict,
    opp_lineup_k_pct: float | None,
    ump_k_factor: float = 1.0,
    catcher_framing_factor: float = 1.0,
    park_k_factor: float = 1.0,
    expected_ip: float = 5.5,
    recent_k9_weight: float = 0.35,
) -> dict:
    """
    Project pitcher K total using only real data.
    Returns dict with mean=None if insufficient data.

    Math: blended K/9 × situational multipliers × expected innings / 9.
    Sigma is 35% of mean (empirical SD for a single MLB start) with a
    floor of 1.4 K to handle low-K projections without going to zero variance.
    """
    if not pitcher_row:
        return {"mean": None}

    # Real season K/9 required
    season_k9 = pitcher_row.get("k9")
    if season_k9 is None or pd.isna(season_k9) or season_k9 == 0:
        return {"mean": None}

    recent_k9 = pitcher_row.get("recent_k9")
    if recent_k9 is not None and not pd.isna(recent_k9) and recent_k9 > 0:
        blended_k9 = recent_k9 * recent_k9_weight + season_k9 * (1 - recent_k9_weight)
    else:
        blended_k9 = season_k9  # no recent data, use season as-is

    # Opposing lineup adjustment - only if real data
    if opp_lineup_k_pct is None or pd.isna(opp_lineup_k_pct):
        lineup_adj = 1.0  # no adjustment when we don't have it
    else:
        lineup_adj = opp_lineup_k_pct / 22  # league avg ~22%

    proj_k9 = (
        blended_k9
        * lineup_adj
        * ump_k_factor
        * catcher_framing_factor
        * park_k_factor
    )

    mean = proj_k9 * expected_ip / 9
    # Empirical sigma for a single start: ~35% of mean
    # Min sigma of 1.4 K to handle low-K projections
    sigma = max(mean * 0.35, 1.4)

    def p_over(line):
        """
        P(K total > line). Lines like 5.5 mean "more than 5.5 strikeouts",
        i.e. 6 or more, so we DON'T add a continuity correction since line is
        already at the half-integer.
        """
        from math import erf, sqrt
        if sigma <= 0:
            return 0.5
        z = (line - mean) / (sigma * sqrt(2))
        return float(1 - 0.5 * (1 + erf(z)))

    return {
        "mean": round(mean, 2),
        "low": round(mean - sigma, 2),
        "high": round(mean + sigma, 2),
        "sigma": round(sigma, 2),
        "blended_k9": round(blended_k9, 2),
        "lineup_adj": round(lineup_adj, 3),
        "p_over_5.5": round(p_over(5.5), 3),
        "p_over_6.5": round(p_over(6.5), 3),
        "p_over_7.5": round(p_over(7.5), 3),
        "p_over_8.5": round(p_over(8.5), 3),
    }


def implied_prob_from_american(odds: int) -> float:
    """Convert American odds to implied probability (with vig)."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def american_from_prob(prob: float) -> int:
    """Convert decimal probability to American odds (no vig)."""
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    return int(round(100 * (1 - prob) / prob))


def find_edge(model_prob: float, book_odds: int, threshold: float = 0.02) -> dict:
    """
    Compare model probability to book implied probability.
    Returns edge dict if model > book by threshold, else empty.
    """
    if model_prob is None or pd.isna(model_prob):
        return {}
    book_prob = implied_prob_from_american(book_odds)
    edge = model_prob - book_prob
    if edge < threshold:
        return {}
    return {
        "model_prob": round(model_prob, 4),
        "book_prob": round(book_prob, 4),
        "edge": round(edge, 4),
        "fair_odds": american_from_prob(model_prob),
        "book_odds": book_odds,
    }


# ----------------------------------------------------------------------------
# Verdict tagging for a hitter's HR play
# ----------------------------------------------------------------------------

def hr_verdict(hr_game_pct: float | None, sample_size: int | None = None,
                pa_threshold: int = 80) -> str:
    """
    Categorize an HR Game% into a tier for display:
      🔥 ELITE    >= 25%
      ✅ STRONG   18-25%
      📊 SOLID    12-18%
      💤 WEAK     5-12%
      ❌ AVOID    < 5%
      ⚠️ SMALL    insufficient sample
    """
    if hr_game_pct is None or pd.isna(hr_game_pct):
        return ""
    if sample_size is not None and not pd.isna(sample_size) and sample_size < pa_threshold:
        return "⚠️ SMALL"
    if hr_game_pct >= 25:
        return "🔥 ELITE"
    if hr_game_pct >= 18:
        return "✅ STRONG"
    if hr_game_pct >= 12:
        return "📊 SOLID"
    if hr_game_pct >= 5:
        return "💤 WEAK"
    return "❌ AVOID"


def hr_signal_emoji(hr_game_pct: float | None, sample_size: int | None = None,
                     pa_threshold: int = 80) -> str:
    """Single-emoji signal column."""
    if hr_game_pct is None or pd.isna(hr_game_pct):
        return "⚪"
    if sample_size is not None and not pd.isna(sample_size) and sample_size < pa_threshold:
        return "⚪"
    if hr_game_pct >= 22:
        return "🟢"
    if hr_game_pct >= 14:
        return "🟡"
    if hr_game_pct >= 7:
        return "🟠"
    return "🔴"


def pitcher_signal_emoji(test_score: float | None, sample_size: int | None = None,
                          pa_threshold: int = 80) -> str:
    """Pitcher signal for the K/suppression matrix."""
    if test_score is None or pd.isna(test_score):
        return "⚪"
    if sample_size is not None and not pd.isna(sample_size) and sample_size < pa_threshold:
        return "⚪"
    if test_score >= 65:
        return "🟢"
    if test_score >= 45:
        return "🟡"
    if test_score >= 30:
        return "🟠"
    return "🔴"
