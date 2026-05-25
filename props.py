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

    # Real hitter base rate - BLENDED with barrel-based expected HR/PA
    # to reduce noise from small/lucky samples.
    #
    # xHR/PA derivation:
    #   barrel_pct = % of batted balls that are barrels (Statcast definition)
    #   BBE_rate ≈ 0.70 (typical PAs that become batted balls, not K/BB)
    #   HR_per_barrel ≈ 0.55 (Statcast research: barrels become HRs ~55% of time)
    #   xHR/PA ≈ barrel_pct × 0.70 × 0.55 = barrel_pct × 0.385
    #
    # Previously used 0.50 multiplier which overstated; calibrated to real MLB rates.
    # League-avg ~8% barrel × 0.385 ≈ 3.1% xHR/PA, matching observed ~2.8% league HR/PA.
    #
    # Blend weight grows with sample size: 100 PA → 30% observed / 70% xHR;
    # 300 PA → 60% observed / 40% xHR; 600 PA → 80% observed / 20% xHR.
    h_observed = hr / pa
    barrel_pct = hitter_row.get("barrel_pct")
    if barrel_pct is not None and not pd.isna(barrel_pct) and barrel_pct > 0:
        h_xhr = float(barrel_pct) / 100 * 0.385
        # Weight observed by sample size (asymptote at 0.80)
        observed_weight = min(0.80, 0.30 + (pa - 100) / 500 * 0.50)
        observed_weight = max(0.30, observed_weight)
        h_base = h_observed * observed_weight + h_xhr * (1 - observed_weight)
    else:
        # No barrel data - fall back to observed only
        h_base = h_observed

    # Pitcher HR/9 adjustment - prefer handedness splits if available
    p_hr9 = pitcher_row.get("hr9") if pitcher_row else None

    # NEW: Try to use vs LHB / vs RHB HR/PA splits instead of overall HR/9
    # If a RHP gives up 4.5% HR/PA to LHB but only 2.1% to RHB,
    # an LHB facing him should see the 4.5 number, not the average.
    h_bats = (hitter_row.get("bats") or "").upper()
    split_hr_per_pa = None
    if pitcher_row is not None and h_bats in ("L", "R"):
        # Switch hitters effectively bat opposite of the pitcher, so use the
        # opposite-side split if available
        col_prefix = "vs_lhb_" if h_bats == "L" else "vs_rhb_"
        split_hr = pitcher_row.get(f"{col_prefix}hr_per_pa")
        split_pa = pitcher_row.get(f"{col_prefix}pa")
        # Only use the split if we have enough sample (≥40 PA)
        if (split_hr is not None and not pd.isna(split_hr) and split_hr > 0
                and split_pa is not None and not pd.isna(split_pa) and split_pa >= 40):
            split_hr_per_pa = float(split_hr) / 100.0  # convert pct → rate

    if split_hr_per_pa is not None:
        # Sample-size shrinkage on the SPLIT itself.
        # League avg HR/PA ≈ 2.8%. A pitcher with 30 PA vs LHB has noisy splits;
        # a pitcher with 200 PA vs LHB has reliable splits.
        # Bayesian shrink: weight = pa / (pa + 80). 80 is the prior weight in PA.
        # If pa=200: 200/280 = 71% real, 29% league avg
        # If pa=40 :  40/120 = 33% real, 67% league avg
        split_pa = float(pitcher_row.get(f"{col_prefix}pa", 0) or 0)
        shrink_w = split_pa / (split_pa + 80)
        split_shrunk = split_hr_per_pa * shrink_w + 0.028 * (1 - shrink_w)
        league_p_hr_per_pa = 0.028  # ~2.8% league HR per PA
        pitcher_mult = split_shrunk / league_p_hr_per_pa
        pitcher_mult = max(0.5, min(2.0, pitcher_mult))
    elif p_hr9 is None or pd.isna(p_hr9) or p_hr9 == 0:
        pitcher_mult = 1.0  # No adjustment if we don't know
    else:
        p_hr_per_pa = (p_hr9 / 9) / 4.3  # ~4.3 PA per inning
        league_p_hr_per_pa = (1.20 / 9) / 4.3  # league HR/9 ~1.20
        pitcher_mult = p_hr_per_pa / league_p_hr_per_pa
        pitcher_mult = max(0.5, min(2.0, pitcher_mult))

    # PLATOON ADVANTAGE multiplier
    # Real MLB data: opposite-handed matchups produce ~12% more HRs than same-side
    # (LHB vs RHP: 1.07x baseline; LHB vs LHP: 0.94x; RHB vs LHP: 1.06x; RHB vs RHP: 0.96x)
    # Switch hitters get the favorable side, so always neutral or slightly +
    #
    # NOTE: If we're already using a HANDEDNESS-SPLIT pitcher_mult above,
    # we should dampen the platoon mult to avoid double-counting.
    # Real splits already encode the platoon effect in the data.
    platoon_mult = 1.0
    p_throws = (pitcher_row.get("p_throws") or pitcher_row.get("throws") or "").upper() if pitcher_row else ""
    if h_bats and p_throws and h_bats != "S":
        # If we have real splits, the data already shows the platoon effect.
        # Use a smaller residual platoon adjustment.
        if split_hr_per_pa is not None:
            # Splits used - reduce platoon impact to 1/3 to capture league avg residual
            if h_bats != p_throws:
                platoon_mult = 1.025 if h_bats == "L" else 1.020
            else:
                platoon_mult = 0.980 if h_bats == "L" else 0.985
        else:
            # No splits available - full platoon adjustment
            if h_bats != p_throws:
                platoon_mult = 1.07 if h_bats == "L" else 1.06
            else:
                platoon_mult = 0.94 if h_bats == "L" else 0.96
    # Switch hitters (S) stay at 1.0 since they hit the favorable side

    # Pitch match adjustment - only if we have a real score
    # Was: 0.6-1.6 range, too generous (gave non-power hitters huge boosts)
    # Now: 0.75-1.30 range — meaningful but doesn't substitute for raw power
    pm_mult = 1.0
    if pitch_match_score is not None and not pd.isna(pitch_match_score):
        # 50 = neutral. Each point above adds 0.6% boost; each below subtracts.
        pm_mult = 1.0 + (pitch_match_score - 50) * 0.006
        pm_mult = max(0.75, min(1.30, pm_mult))

    # Compute hitter-side context multiplier (everything BUT pitcher_mult and h_base).
    # Cap it to prevent inflation when multiple small boosts compound.
    # Example bug fix: 1.17 wind × 1.21 park × 1.30 pitch_match = 1.84x
    # compounded multiplier was pushing modest hitters to elite tier.
    # New cap: 1.35× total context (still allows great matchups to boost ~35%).
    ctx_mult_raw = (
        park_factor
        * park_hand_factor
        * weather_mult
        * pm_mult
        * ttop_mult
        * defense_factor
        * platoon_mult
    )
    ctx_mult = min(1.35, max(0.65, ctx_mult_raw))

    prob = (
        h_base
        * pitcher_mult
        * ctx_mult
    )
    # Use a SOFT squash that asymptotes at a REALISTIC ceiling.
    # Reference: Aaron Judge's actual rate of "at-least-1-HR in a game" peaked
    # at ~25% in his 62-HR 2022 season.
    #
    # NEW BEHAVIOR (May 2026 update): Loosened the squash so elite hitters
    # show meaningful differentiation. Previously all 13 elites clustered
    # at 6.0-6.2% per PA (visible as identical hr_game_pct in top picks).
    # Now:
    #   - Below 4% per PA: linear pass-through (no squash)
    #   - 4-5%: gentle squash (some compression)
    #   - 5-7%: stronger squash (most differentiation happens here)
    #   - Asymptote: 7.0% per PA (max game = 1 - 0.93^4.2 = 26.4%)
    # This gives Judge (~20% barrel + great matchup) clear separation from
    # Schwarber (~26% barrel + great matchup).
    if prob <= 0.04:
        squashed = prob
    else:
        excess = prob - 0.04
        # tanh squash: asymptotes at 0.030 (so max = 0.07 = 7.0% per PA)
        # Scale parameter widened to 0.040 so differentiation stays wider
        squashed = 0.04 + 0.030 * np.tanh(excess / 0.040)
    return float(max(0.001, squashed))


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
