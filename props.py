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
    bullpen_hr9: float | None = None,  # v37+ bullpen leverage adjustment
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

    # NEW: sample-size shrinkage on raw observed HR/PA to combat early-season noise.
    # A hitter with 5 HR in 30 PA has 16.7% observed rate — that's not predictive.
    # Shrink toward league avg using prior weight of 50 PA.
    # Below 50 PA: observed barely matters. Above 300 PA: observed is most trusted.
    h_observed_raw = hr / pa
    obs_shrink_w = pa / (pa + 50)  # 50 PA prior
    h_observed = h_observed_raw * obs_shrink_w + LEAGUE_HR_PER_PA * (1 - obs_shrink_w)

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

    # HITTER vs-HANDEDNESS SPLIT ADJUSTMENT (June 2026)
    # The h_base above uses OVERALL season stats. But platoon effects are
    # real and large — Adell case study has 8.8% overall barrel but 31.6%
    # vs-LHP barrel. Without this adjustment, a strong reverse-platoon hitter
    # facing his preferred arm gets projected at his weak overall rate.
    #
    # Strategy: if we have vs-LHP/vs-RHP HR rate from MLB Stats API,
    # compute a multiplier = (split HR/PA) / (overall HR/PA), shrunken by
    # split sample size, and apply to h_base.
    # Switch hitters bat opposite of pitcher arm → look up that side.
    p_throws_now = (pitcher_row.get("p_throws") or pitcher_row.get("throws") or "").upper() if pitcher_row else ""
    h_bats_now = (hitter_row.get("bats") or "").upper() if hitter_row else ""
    # Determine which split applies. A LHB always faces vs-RHP if pitcher is R,
    # but the hitter's split is denoted by the pitcher's hand (vs-LHP, vs-RHP).
    h_split_key = None
    if p_throws_now == "L":
        h_split_key = "lhp"
    elif p_throws_now == "R":
        h_split_key = "rhp"
    # For switch hitters: they bat opposite the pitcher, so they will face
    # the pitcher with the favorable platoon. The split lookup is the same:
    # use the pitcher's hand to find the hitter's vs-LHP or vs-RHP rate.
    # Switch hitter splits already reflect this — their vs-RHP stats are from
    # them batting left vs RHP.

    if h_split_key:
        split_hr_rate = hitter_row.get(f"vs_{h_split_key}_hr_per_pa")
        split_pa = hitter_row.get(f"vs_{h_split_key}_pa")
        if (split_hr_rate is not None and not pd.isna(split_hr_rate)
                and split_pa is not None and not pd.isna(split_pa)
                and float(split_pa) >= 30):
            try:
                split_hr_pa = float(split_hr_rate) / 100.0  # pct → rate
                # Compute multiplier vs the overall rate.
                # Use h_observed_raw (not the league-shrunk h_observed) as
                # the "overall" baseline since split stats are also observed.
                # Guard against divide-by-zero on contact hitters with 0 HR.
                overall_rate = h_observed_raw if h_observed_raw > 0.005 else 0.025
                split_mult_raw = split_hr_pa / overall_rate
                # Shrink the multiplier toward 1.0 based on split sample size.
                # 30 PA: very heavy shrink (most weight to overall);
                # 150 PA: ~50% confidence;
                # 400 PA: ~75% confidence in split.
                split_pa_f = float(split_pa)
                split_w = split_pa_f / (split_pa_f + 100)  # 100 PA prior
                # Cap shrunk multiplier in [0.5, 2.0] to prevent extreme outliers
                split_mult_shrunk = 1.0 + (split_mult_raw - 1.0) * split_w
                split_mult_shrunk = max(0.5, min(2.0, split_mult_shrunk))
                h_base = h_base * split_mult_shrunk
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    # NEW: RECENCY ADJUSTMENT - blend in recent-form hot/cold signal.
    # recent_hr_weighted_rate gives last-3 games triple-weight, weighted across
    # last 15 games. If this rate is significantly above/below season h_base,
    # nudge h_base toward it (but bounded so we don't over-react to small samples).
    #
    # CAP: ±25% deviation from h_base. A hitter on a hot streak can boost their
    # projection by at most 25%; a cold hitter can be reduced by at most 25%.
    recent_weighted = hitter_row.get("recent_hr_weighted_rate")
    if (recent_weighted is not None and not pd.isna(recent_weighted)
            and recent_weighted > 0 and h_base > 0):
        recent_rate_decimal = float(recent_weighted) / 100  # pct → rate
        # Compute hot/cold ratio capped at [0.75, 1.25]
        hot_cold_ratio = recent_rate_decimal / h_base
        hot_cold_ratio = max(0.75, min(1.25, hot_cold_ratio))
        # Apply only 20% of the deviation (lowered from 30% in May 2026).
        # Baseball research consistently shows hot/cold streaks have minimal
        # predictive value beyond 1-2 weeks. Most quantitative analysts use
        # 10-15% weight; 20% is a balanced choice that respects the signal
        # without overreacting to small samples. Result: projections more
        # stable, less reactive to short streaks.
        adjustment = 1.0 + (hot_cold_ratio - 1.0) * 0.20
        h_base = h_base * adjustment

    # NEW: DAY/NIGHT adjustment - hitters perform differently in day vs night.
    # Documented effect: most batters slightly worse in day games due to
    # shadow / sun glare / visibility. Some hitters' splits are EXTREME.
    # E.g. some batters hit 2x more HRs at night than day.
    # game_type comes from app.py based on game start time.
    game_type = hitter_row.get("game_type")  # "day" or "night"
    if game_type in ("day", "night") and h_base > 0:
        dn_pa_col = f"vs_{game_type}_pa"
        dn_hr_col = f"vs_{game_type}_hr_per_pa"
        dn_pa = hitter_row.get(dn_pa_col)
        dn_hr = hitter_row.get(dn_hr_col)
        # Need ≥40 PA in the split for reliability
        if (dn_pa is not None and not pd.isna(dn_pa) and dn_pa >= 40
                and dn_hr is not None and not pd.isna(dn_hr) and dn_hr > 0):
            dn_rate_decimal = float(dn_hr) / 100
            # Shrink toward h_base (don't fully trust split over base):
            # shrink_w = pa / (pa + 100). 100 PA prior weight.
            shrink_w = float(dn_pa) / (float(dn_pa) + 100)
            dn_shrunk = dn_rate_decimal * shrink_w + h_base * (1 - shrink_w)
            # Cap adjustment to ±15% of h_base (day/night is real but bounded)
            dn_ratio = dn_shrunk / h_base
            dn_ratio = max(0.85, min(1.15, dn_ratio))
            # Apply 50% of the adjustment (conservative)
            h_base = h_base * (1.0 + (dn_ratio - 1.0) * 0.50)

    # Pitcher HR/9 adjustment - prefer handedness splits if available
    p_hr9 = pitcher_row.get("hr9") if pitcher_row else None

    # NEW: Try to use vs LHB / vs RHB HR/PA splits instead of overall HR/9
    # If a RHP gives up 4.5% HR/PA to LHB but only 2.1% to RHB,
    # an LHB facing him should see the 4.5 number, not the average.
    h_bats = (hitter_row.get("bats") or "").upper()
    # Fetch pitcher throws early so we can do switch-hitter handling
    p_throws_early = (pitcher_row.get("p_throws") or pitcher_row.get("throws") or "").upper() if pitcher_row else ""
    split_hr_per_pa = None
    split_pa_count = 0
    split_source = None  # for debugging: "hr_per_pa" or "slg_derived"
    # Determine which side of the pitcher's splits to use.
    # - LHB → vs_lhb_ (pitcher's stats vs LHB)
    # - RHB → vs_rhb_
    # - Switch hitter: bats opposite of pitcher's throwing arm
    #     vs RHP → switch bats L → use vs_lhb_ (pitcher's LHB-facing stats)
    #     vs LHP → switch bats R → use vs_rhb_
    effective_side = None
    if h_bats == "L":
        effective_side = "L"
    elif h_bats == "R":
        effective_side = "R"
    elif h_bats == "S" and p_throws_early in ("L", "R"):
        # Switch hitter chooses opposite side from the pitcher
        effective_side = "L" if p_throws_early == "R" else "R"

    if pitcher_row is not None and effective_side in ("L", "R"):
        col_prefix = "vs_lhb_" if effective_side == "L" else "vs_rhb_"
        split_hr = pitcher_row.get(f"{col_prefix}hr_per_pa")
        split_pa = pitcher_row.get(f"{col_prefix}pa")

        # PRIMARY: direct HR/PA split if available
        if (split_hr is not None and not pd.isna(split_hr) and split_hr > 0
                and split_pa is not None and not pd.isna(split_pa) and split_pa >= 40):
            split_hr_per_pa = float(split_hr) / 100.0  # convert pct → rate
            split_pa_count = float(split_pa)
            split_source = "hr_per_pa"
        else:
            # FALLBACK: derive from SLG split (when MLB API doesn't return raw counts).
            # SLG correlates strongly with HR/PA. Empirical mapping (2023-2024 data):
            #   League avg SLG ~ .398, league avg HR/PA ~ 2.8%
            #   .350 SLG → ~2.0% HR/PA
            #   .450 SLG → ~3.6% HR/PA
            #   .500 SLG → ~4.5% HR/PA
            #   .550 SLG → ~5.5% HR/PA
            # Linear-ish fit: HR/PA% ≈ (SLG - 0.250) * 12.5
            split_slg = pitcher_row.get(f"{col_prefix}slg")
            # SLG splits don't tell us PA, so assume modest reliability (use 80 as PA)
            if split_slg is not None and not pd.isna(split_slg) and split_slg > 0:
                try:
                    slg_val = float(split_slg)
                    # Cap derivation to reasonable range
                    derived_hr_pct = max(0.5, min(7.0, (slg_val - 0.250) * 12.5))
                    split_hr_per_pa = derived_hr_pct / 100.0
                    split_pa_count = 80  # treat as moderately reliable, shrinks somewhat
                    split_source = "slg_derived"
                except (TypeError, ValueError):
                    pass

    if split_hr_per_pa is not None:
        # Sample-size shrinkage on the SPLIT itself.
        # League avg HR/PA ≈ 2.8%. A pitcher with 30 PA vs LHB has noisy splits;
        # a pitcher with 200 PA vs LHB has reliable splits.
        # Bayesian shrink: weight = pa / (pa + 80). 80 is the prior weight in PA.
        # If pa=200: 200/280 = 71% real, 29% league avg
        # If pa=40 :  40/120 = 33% real, 67% league avg
        # Use the canonical LEAGUE_HR_PER_PA constant (3.0%) for both the
        # shrinkage prior AND the denominator. Previously these were:
        #   - shrinkage toward 0.028
        #   - denominator 0.028 (splits) OR (1.30/9)/4.3 = 0.03362 (HR9)
        # That dual-baseline inconsistency caused subtle pitcher_mult drift
        # between the two paths for the same pitcher in edge cases.
        shrink_w = split_pa_count / (split_pa_count + 80)
        split_shrunk = split_hr_per_pa * shrink_w + LEAGUE_HR_PER_PA * (1 - shrink_w)
        pitcher_mult = split_shrunk / LEAGUE_HR_PER_PA
        pitcher_mult = max(0.5, min(2.0, pitcher_mult))
    elif p_hr9 is None or pd.isna(p_hr9) or p_hr9 == 0:
        pitcher_mult = 1.0  # No adjustment if we don't know
    else:
        # IP GATE — Lyon Richardson case (June 2026).
        # Without this, a pitcher with 0.2 IP and 1 HR allowed has HR/9=13.50
        # which clips to the 2.0× cap on EVERY hitter facing him. KC lineup
        # was inflated to 20-24% HR Game% because of this single bad sample.
        # Below 10 IP: treat as neutral (1.0×) — sample too noisy.
        # 10-30 IP: shrink toward 1.0 (partial credit, scales with IP).
        # 30+ IP: use raw HR/9 as before (real signal).
        # Same principle as the era_savant clip(upper=12.0) and the splits-path
        # Bayesian shrinkage — every other pitcher-rate path already guards
        # against tiny samples; HR/9 was the one hole.
        pitcher_ip_val = pitcher_row.get("ip") if pitcher_row else None
        try:
            pitcher_ip_f = (float(pitcher_ip_val)
                            if pitcher_ip_val is not None and not pd.isna(pitcher_ip_val)
                            else None)
        except (TypeError, ValueError):
            pitcher_ip_f = None

        if pitcher_ip_f is not None and pitcher_ip_f < 10.0:
            # Below 10 IP — treat as neutral. Sample too noisy.
            pitcher_mult = 1.0
        else:
            p_hr_per_pa = (p_hr9 / 9) / 4.3  # ~4.3 PA per inning

            # BULLPEN LEVERAGE BLEND (v37+)
            # In a typical game, ~70% of opposing PAs come against the
            # starter and ~30% come against relievers (after the starter
            # exits in inning 5-6). If the team's bullpen has a notably
            # high or low HR/9, the hitter's TRUE HR exposure is the
            # weighted average, not just the starter's number.
            #
            # League-avg bullpen HR/9 ≈ 1.15. We blend at 70/30 by IP share:
            #   blended_hr_per_pa = 0.7 * starter_hr_per_pa + 0.3 * bullpen_hr_per_pa
            #
            # Effect: matters most when bullpen and starter are very different.
            # Examples:
            #   - Good starter (HR/9 0.8) + bad bullpen (HR/9 1.6) →
            #     starter alone says HR rate 2.1%, blended says 2.6% (+25%)
            #   - Bad starter (HR/9 2.0) + good bullpen (HR/9 0.9) →
            #     starter alone says 5.2%, blended says 4.4% (-15%)
            if bullpen_hr9 is not None and not pd.isna(bullpen_hr9) and bullpen_hr9 > 0:
                bp_hr_per_pa = (float(bullpen_hr9) / 9) / 4.3
                p_hr_per_pa = 0.7 * p_hr_per_pa + 0.3 * bp_hr_per_pa

            raw_mult = p_hr_per_pa / LEAGUE_HR_PER_PA
            # Bayesian shrinkage toward 1.0 for IP in [10, 30].
            # At IP=10: 50% real, 50% league avg (1.0)
            # At IP=30: ~75% real, 25% league avg
            # At IP=60: ~86% real, 14% league avg
            # At IP=100: ~91% real (close to raw)
            if pitcher_ip_f is not None:
                ip_shrink_w = pitcher_ip_f / (pitcher_ip_f + 10.0)
                shrunk_mult = raw_mult * ip_shrink_w + 1.0 * (1 - ip_shrink_w)
            else:
                # No IP data — use raw with caps
                shrunk_mult = raw_mult
            pitcher_mult = max(0.5, min(2.0, shrunk_mult))

    # NEW: PITCHER DAY/NIGHT adjustment.
    # Some pitchers are dramatically better at night (or day). Apply a modest
    # multiplier on pitcher_mult based on their day/night HR rate split.
    # Capped at ±10% so it doesn't dominate.
    if game_type in ("day", "night") and pitcher_row is not None:
        p_dn_pa = pitcher_row.get(f"vs_{game_type}_pa")
        p_dn_hr = pitcher_row.get(f"vs_{game_type}_hr_per_pa")
        if (p_dn_pa is not None and not pd.isna(p_dn_pa) and p_dn_pa >= 50
                and p_dn_hr is not None and not pd.isna(p_dn_hr) and p_dn_hr > 0):
            # Compare pitcher's day/night HR rate to league average
            p_dn_rate = float(p_dn_hr) / 100  # pct → decimal
            # Shrinkage: 60 PA prior toward LEAGUE_HR_PER_PA
            shrink_w = float(p_dn_pa) / (float(p_dn_pa) + 60)
            p_dn_shrunk = p_dn_rate * shrink_w + LEAGUE_HR_PER_PA * (1 - shrink_w)
            # Ratio of pitcher's split to overall expected (using current pitcher_mult)
            # If pitcher_mult is 0.7 (good suppressor), expected rate is 0.7 × LEAGUE.
            # If day/night actual exceeds that, he's slightly worse at this time.
            expected_pitcher_rate = LEAGUE_HR_PER_PA * pitcher_mult
            if expected_pitcher_rate > 0:
                dn_ratio = p_dn_shrunk / expected_pitcher_rate
                dn_ratio = max(0.90, min(1.10, dn_ratio))
                pitcher_mult = pitcher_mult * (1.0 + (dn_ratio - 1.0) * 0.50)
                pitcher_mult = max(0.5, min(2.0, pitcher_mult))

    # GROUND BALL DAMPENER (June 2026)
    # Pitchers with high ground-ball rates physically suppress HRs because
    # ground balls don't leave the yard. The pitcher_mult derived above
    # captures HR/9 directly, but HR/9 lags real GB tendency — early-season
    # GB pitchers might have an inflated HR/9 from a few flyball outliers.
    #
    # We apply a small explicit dampener:
    #   GB% 45-50%: -2.5% pitcher_mult (above-average GB pitcher)
    #   GB% 50-55%: -5% pitcher_mult (strong GB pitcher)
    #   GB% 55%+:   -7.5% pitcher_mult (elite GB suppressor)
    # Symmetric on the flyball side:
    #   GB% 35-40%: +2.5% pitcher_mult (flyball-prone)
    #   GB% 30-35%: +5% pitcher_mult (heavy flyball pitcher)
    #   GB% <30%:   +7.5% pitcher_mult (extreme flyball, HR-prone)
    # League avg GB% ≈ 43%. Multiplier neutral at 40-45%.
    gb_pct_raw = None
    if pitcher_row is not None:
        for key in ("gb_pct", "groundballs_percent", "gb_allowed"):
            v = pitcher_row.get(key)
            if v is not None:
                try:
                    gb_pct_raw = float(v)
                    break
                except (TypeError, ValueError):
                    continue
    if gb_pct_raw is not None and not pd.isna(gb_pct_raw):
        if gb_pct_raw >= 55:
            pitcher_mult *= 0.925
        elif gb_pct_raw >= 50:
            pitcher_mult *= 0.95
        elif gb_pct_raw >= 45:
            pitcher_mult *= 0.975
        elif gb_pct_raw < 30:
            pitcher_mult *= 1.075
        elif gb_pct_raw < 35:
            pitcher_mult *= 1.05
        elif gb_pct_raw < 40:
            pitcher_mult *= 1.025
        # Re-clamp
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
    elif h_bats == "S" and p_throws in ("L", "R"):
        # Switch hitters ALWAYS bat from the favorable side, so they always get
        # the platoon advantage. If we already pulled the opposite-side split
        # above (the typical case now), the data encodes it — use a small residual.
        # If no splits available, give the full opposite-side bonus.
        if split_hr_per_pa is not None:
            # Residual after splits already applied (smaller)
            platoon_mult = 1.022  # avg of L/R favorable values
        else:
            # No splits — give the full opposite-side bonus
            platoon_mult = 1.065  # avg of L/R favorable values

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
    # at ~25% in his 62-HR 2022 season. Schwarber/Olson 2024 hit 24.4%/24.3%.
    #
    # CALIBRATION (May 2026): loosened from 0.030/0.040 → 0.032/0.045 to bring
    # elite hitter game% up to 24-26% range matching real MLB data. Previous
    # caps had Schwarber at ~22.6% game (real 24.4%) — systematically 1-2pp low.
    #
    # Behavior:
    #   - Below 4% per PA: linear pass-through (no squash)
    #   - 4-5%: gentle squash (some compression)
    #   - 5-7%: stronger squash (most differentiation happens here)
    #   - Theoretical asymptote: 7.2% per PA (tanh→1.0 in the limit)
    #   - PRACTICAL ceiling: ~6.7% per PA in realistic input range.
    #     Per-game max: 1 - (1-0.067)^4.2 ≈ 25.4% — matches Schwarber-tier real rate.
    if prob <= 0.04:
        squashed = prob
    else:
        excess = prob - 0.04
        # Tanh squash: theoretical asymptote at 0.032 (theoretical max = 7.2% per PA).
        # Scale parameter 0.045 (was 0.040) widens the differentiation band, so
        # 6%/PA input → 5.5% out (was 5.4%), 8%/PA → 6.4% (was 6.3%), 12%/PA → 6.9%.
        squashed = 0.04 + 0.032 * np.tanh(excess / 0.045)
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
        # 2024-2025 league-avg K rate is ~22.5%. Using 22 here gave every K
        # projection a +2.3% upward bias for league-average lineups.
        lineup_adj = opp_lineup_k_pct / 22.5

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

        HYBRID: for low-K projections (mean < 5, typically openers/swing-men),
        the Normal approximation diverges from the true discrete distribution
        by 3-5 percentage points. Use Poisson for these. At K≥5 the Normal
        approximation is accurate enough for prop pricing.
        """
        from math import erf, sqrt
        if mean < 5:
            # Poisson is the right discrete distribution for rare-event counts.
            # P(K > line) = P(K >= ceil(line)) since K is integer.
            from math import factorial, exp
            k_min = int(line + 0.999)  # ceil for half-integer lines (5.5 → 6)
            if mean <= 0:
                return 0.0
            # Compute P(K < k_min) = sum_{i=0}^{k_min-1} e^-λ λ^i / i!
            cum = 0.0
            for i in range(k_min):
                cum += exp(-mean) * (mean ** i) / factorial(i)
            return max(0.0, min(1.0, 1.0 - cum))
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
