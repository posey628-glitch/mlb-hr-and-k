"""
props.py
=========
Converts internal model outputs into actual betting-relevant numbers:

  - HR probability per hitter (0.00 - 1.00) — directly comparable to "+450 HR" odds
  - Strikeout total projection per pitcher (with std dev range)
  - Implied odds → break-even threshold logic
  - "Edge vs market" calculations

Designed for prop betting on HR and K markets.

CALIBRATION NOTE:
  League-avg HR/PA in 2024-25 was ~3.0%. A "good" HR prop hitter sits at 5-7%.
  Aaron Judge in a great matchup ~12-15%. The model targets this range.

  League-avg K/9 is ~8.6. Strong starter K projection 6.5-9.5 over 5-6 IP.
  Strider/Skubal types project 8.5-11 in a good matchup.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


# Base HR rate per PA, MLB-wide. Used as anchor for prob calibration.
LEAGUE_HR_PER_PA = 0.030
LEAGUE_K_PER_9 = 8.6


def hr_prob_per_pa(
    hitter_row: dict,
    pitcher_row: dict,
    park_factor: float = 1.0,
    park_hand_factor: float = 1.0,
    weather_mult: float = 1.0,
    pitch_match_score: float | None = None,
    ttop_mult: float = 1.0,
    defense_factor: float = 1.0,
    min_pa: int = 100,
) -> float | None:
    """
    Returns P(HR | single PA today) using ONLY real data.
    Returns None if hitter has insufficient sample (< min_pa) or no real data.
    """
    pa = hitter_row.get("pa") if hitter_row else None
    hr = hitter_row.get("home_run") if hitter_row else None

    # Hard requirement: real PA and HR data
    if pa is None or pd.isna(pa) or hr is None or pd.isna(hr):
        return None
    if pa < min_pa:
        return None

    # Real hitter base rate
    h_base = hr / pa

    # Pitcher HR/9 adjustment - only if pitcher row has real HR/9
    p_hr9 = pitcher_row.get("hr9") if pitcher_row else None
    if p_hr9 is None or pd.isna(p_hr9) or p_hr9 == 0:
        pitcher_mult = 1.0  # No adjustment if we don't know
    else:
        p_hr_per_pa = (p_hr9 / 9) / 4.3  # ~4.3 PA per inning
        league_p_hr_per_pa = (1.20 / 9) / 4.3  # league HR/9 ~1.20
        pitcher_mult = p_hr_per_pa / league_p_hr_per_pa
        pitcher_mult = max(0.5, min(2.0, pitcher_mult))

    # Pitch match adjustment - only if we have a real score
    pm_mult = 1.0
    if pitch_match_score is not None and not pd.isna(pitch_match_score):
        pm_mult = 0.5 + (pitch_match_score / 100)
        pm_mult = max(0.6, min(1.6, pm_mult))

    prob = (
        h_base
        * pitcher_mult
        * park_factor
        * park_hand_factor
        * weather_mult
        * pm_mult
        * ttop_mult
        * defense_factor
    )
    # Realistic cap: even elite hitters in best matchups rarely exceed 10% per PA
    # Aaron Judge's career-high HR/PA is ~7.5%; with park+weather boost maybe 10%
    return float(np.clip(prob, 0.001, 0.10))


def hr_prob_full_game(prob_per_pa: float | None, expected_pa: float = 4.2) -> float | None:
    """
    P(at least 1 HR in the game) = 1 - (1 - p_pa) ^ PA.
    Returns None if input is None (no real projection possible).
    """
    if prob_per_pa is None or pd.isna(prob_per_pa):
        return None
    return float(1 - (1 - prob_per_pa) ** expected_pa)


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
    # Empirical sigma for a single start: ~35% of mean (verified vs historical data)
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
    if odds is None:
        return None
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def american_from_prob(p: float) -> int:
    """Inverse: probability → American odds (fair, no vig)."""
    if p is None or p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return int(round(-p / (1 - p) * 100))
    return int(round((1 - p) / p * 100))


def edge_vs_market(model_prob: float, market_odds: int) -> dict:
    """
    Compare model probability to a sportsbook line.

    Returns:
      market_prob   - what the book is implying (includes vig)
      fair_odds     - what odds the model thinks are fair
      edge_pct      - (model_prob - market_prob) / market_prob * 100
      kelly         - optional Kelly stake (cap at 25% for safety)
    """
    if model_prob is None or market_odds is None:
        return {}
    mp = implied_prob_from_american(market_odds)
    if mp is None:
        return {}
    edge_pct = (model_prob - mp) / mp * 100
    fair = american_from_prob(model_prob)
    # Decimal odds for Kelly
    dec = 1 + (market_odds / 100 if market_odds > 0 else 100 / -market_odds)
    b = dec - 1
    q = 1 - model_prob
    kelly_full = (b * model_prob - q) / b if b > 0 else 0
    kelly_quarter = max(0, min(0.25, kelly_full / 4))  # quarter Kelly capped
    return {
        "market_prob": round(mp, 4),
        "fair_odds": fair,
        "edge_pct": round(edge_pct, 1),
        "kelly_quarter": round(kelly_quarter, 4),
        "recommend": "✅ BET" if edge_pct > 5 else "—" if edge_pct > -3 else "❌ FADE",
    }


def verdict_color(score: float, scale: tuple = (40, 60)) -> str:
    """
    Convert any 0-100 score into a stoplight verdict.
      < scale[0]   = 🔴 Fade
      < scale[1]   = 🟡 Neutral
      ≥ scale[1]   = 🟢 Smash
    """
    if score is None or pd.isna(score):
        return "—"
    if score >= scale[1]:
        return "🟢"
    if score >= scale[0]:
        return "🟡"
    return "🔴"
