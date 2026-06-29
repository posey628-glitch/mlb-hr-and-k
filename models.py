"""
models.py
==========
Composite scoring + full column coverage matching the Kasper-style dashboard.

Composites (mirrors screenshots exactly):
  - Matchup Score
  - Test Score (matchup discounted for sample size)
  - Ceiling
  - Zone Fit
  - HR Form (with directional arrow ↑ / → / ↓)
  - kHR (expected K rate vs this matchup)
  - "Likely HR%" (estimated per-PA HR probability today)

All scores 0-100, percentile-ranked across the slate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SCORING_WEIGHTS = {
    # v43.23 (audit-driven, user-validated rebalance):
    # Audit showed barrel_pct had ~25% effective weight across pick_score
    # because it was the anchor in nearly every component. Other strong
    # predictors (pulled_brl_pct corr 0.737, avg_ev 0.605, fb_pct 0.461)
    # were underweighted at 3-5% effective. K% was barely used despite
    # being a real HR-rate dampener. Rebalanced to flatten dominance.
    "matchup": {
        # Pitcher-vs-hitter composite. Barrel% reduced (it's already heavily
        # weighted in power_score, hr_form, and pitch_hr_score). K%_inv and
        # pitcher_barrel_allowed strengthened — strikeouts cap HR opportunity
        # per game, and pitcher barrel allowed is the most direct measure
        # of pitcher HR vulnerability.
        "xwoba": 0.18,                   # was 0.20
        "iso": 0.10,                     # was 0.12
        "barrel_pct": 0.08,              # was 0.13 — reduce duplication
        "hard_hit": 0.10,                # was 0.08
        "k_pct_inv": 0.12,               # was 0.10 — K% caps HR opportunity
        "sweet_spot_pct": 0.05,
        "pitcher_xwoba": 0.15,
        "pitcher_k_inv": 0.10,
        "pitcher_barrel_allowed": 0.12,  # was 0.07 — direct HR vuln signal
    },
    "hr_form": {
        # Recent-form composite. v43.23 audit revealed barrel_pct/iso at
        # 22%+19% = 41% of this single composite was excessive — and this
        # composite is ALSO weighted at 13% of pick_score. So barrel% in
        # ps_form alone was 0.13 × 0.22 = ~3% of pick_score before counting
        # any other path. Reduced both, lifted pulled_brl%, avg_ev, fb_pct
        # to reflect their independent correlations.
        "barrel_pct": 0.15,        # was 0.22
        "iso": 0.14,               # was 0.19
        "hard_hit": 0.13,
        "avg_ev": 0.14,            # was 0.11 — corr 0.605, underweighted
        "fb_pct": 0.13,            # was 0.11 — corr 0.461, underweighted
        "pulled_brl_pct": 0.14,    # was 0.08 — corr 0.737, badly underweighted
        "recent_iso_10": 0.10,
        "gb_pct": 0.07,            # neg-weighted
        "blast_pct": 0.05,         # auto-skipped if bat tracking off
    },
    "ceiling": {
        # Ceiling/upside composite — kept barrel-heavy because ceiling is
        # specifically about top-end power outcomes (not balanced HR rate).
        # This composite is read only by zone_fit / boom audit, not pick_score.
        "iso": 0.25,
        "barrel_pct": 0.25,
        "pulled_brl_pct": 0.15,
        "xwoba": 0.15,
        "hard_hit": 0.10,
        "pitcher_barrel_allowed": 0.10,
    },
}


def _safe_pct_rank(s: pd.Series) -> pd.Series:
    """
    Percentile rank 0-100. NaNs stay NaN — no fake-50 defaults.
    Composite scores will then also be NaN for players with missing data,
    which is honest.
    """
    return (s.rank(pct=True) * 100)


def _score_from_weights(df: pd.DataFrame, weights: dict, neg: tuple = ()) -> pd.Series:
    """
    Weighted average of percentile-ranked columns, RENORMALIZING when data is sparse.

    - Rows with NO data in any weighted column → NaN (honest)
    - Rows with partial data → weighted_sum / (sum of weights with data)
    - Columns in `neg` are inverted (lower raw = higher score)

    v43.64 (reviewer doc fix #5): the previous docstring said "score scaled
    by weights of columns that DO have data," which implied a SCALING (the
    score gets smaller with less data). What it actually does is
    RENORMALIZE — a hitter with only one component present gets that
    component's full percentile as their score, identical to a hitter at
    that percentile across ALL components. That's a real asymmetry: a
    hitter we only have ONE signal on isn't penalized for having less
    data, they just get treated as "league average on the missing axes."
    The v43.18 blast-median-imputation fix patched ONE symptom of this
    (rows with missing blast got score = 0 for that axis because of the
    `.fillna(0)` above, which DID effectively scale them down) — but only
    for blast specifically. Other sparse columns still behave as
    documented here. Acknowledge the property; don't pretend it's scaling.
    """
    contributions = []   # list of (weight_series, ranked_series) tuples
    for col, w in weights.items():
        if col not in df.columns:
            continue
        ranked = _safe_pct_rank(df[col])
        if col in neg:
            ranked = 100 - ranked
        # Per-row weight: 0 where data is missing, w where present
        weight_present = ranked.notna().astype(float) * w
        contributions.append((weight_present, ranked.fillna(0)))

    if not contributions:
        return pd.Series([np.nan] * len(df), index=df.index)

    total_weight = sum(wp for wp, _ in contributions)
    weighted_sum = sum(wp * r for wp, r in contributions)

    # Where total_weight is 0 (no real data at all), score is NaN
    score = weighted_sum / total_weight.replace(0, np.nan)
    return score.round(2)


def _form_arrow(recent: float, season: float, threshold: float = 0.10) -> str:
    """Compare recent rolling form vs season baseline."""
    if pd.isna(recent) or pd.isna(season) or season == 0:
        return "→"
    diff = (recent - season) / abs(season)
    if diff > threshold:
        return "↑"
    if diff < -threshold:
        return "↓"
    return "→"


def _classify_role(row, min_ip: float, min_gs: int, full_ip: float) -> str:
    """Module-level role classifier — single source of truth for both
    build_pitcher_slate() and recompute_pitcher_roles().

    Previously this 60-line function was duplicated as a nested function
    inside both call sites, which caused logic drift over time.

    Inputs (from row):
      ip, games_started, games_played, is_rookie, primary_position (MLB SP/RP),
      days_since_return, il_count_this_season

    Returns one of:
      ✓ / 🌱 ✓ / 🌱 NEW STARTER / 🏥 RETURNING / 🔄 SWING / 🚨 OPENER /
      ⚠️ LOW IP / ❔ NO DATA
    """
    ip = row.get("ip")
    gs = row.get("games_started")
    gp = row.get("games_played")
    _rookie_raw = row.get("is_rookie", False)
    try:
        is_rookie = bool(_rookie_raw) if (_rookie_raw is not None and not pd.isna(_rookie_raw)) else False
    except (TypeError, ValueError):
        is_rookie = False
    # MLB's official primaryPosition: "SP" / "RP" / "P" / "TWP" / ""
    _pos_raw = row.get("primary_position")
    mlb_position = "" if (_pos_raw is None or pd.isna(_pos_raw)) else str(_pos_raw).upper()
    # IL info (NA-safe)
    days_since_return = row.get("days_since_return")
    if pd.isna(days_since_return):
        days_since_return = None
    _il_raw = row.get("il_count_this_season", 0)
    try:
        il_count = 0 if pd.isna(_il_raw) else (int(_il_raw) if _il_raw else 0)
    except (TypeError, ValueError):
        il_count = 0
    is_returning_starter = (
        days_since_return is not None
        and 0 <= days_since_return <= 30
    ) or (il_count >= 1)

    # NO DATA case
    if (ip is None or pd.isna(ip)) and (gs is None or pd.isna(gs)):
        return "❔ NO DATA"

    rookie_prefix = "🌱 " if is_rookie else ""
    gs_n = float(gs) if (gs is not None and not pd.isna(gs)) else 0
    gp_n = float(gp) if (gp is not None and not pd.isna(gp)) else 0
    ip_n = float(ip) if (ip is not None and not pd.isna(ip)) else 0

    # Case 1: MLB explicitly says this pitcher is an RP — opener role
    if mlb_position == "RP":
        return rookie_prefix + "🚨 OPENER"

    # Case 2: MLB explicitly says SP — trust that, refine for IL/rookie status
    if mlb_position == "SP":
        if is_returning_starter and ip_n < min_ip * 0.8:
            return rookie_prefix + "🏥 RETURNING"
        if is_rookie and gs_n < min_gs:
            return "🌱 NEW STARTER"
        return rookie_prefix + "✓"

    # Case 3: No MLB SP/RP designation — use heuristics
    # SWING man: started SOME games but mostly relief, AND average start length
    # is short. The previous check (gp > gs*1.5 AND IP/outing < 3) misclassified
    # starters like Ben Brown who had a piggyback appearance or two early in
    # the season but still go 5+ IP in their actual starts.
    #
    # Now we require ALL of:
    #   - gp >= gs * 2.0 (much more relief than starts — was 1.5)
    #   - at least 3 relief appearances (gp - gs >= 3)
    #   - IP/outing < 2.8 (short outings — was <3.0)
    #   - AND IP/GS < 4.0 (their actual starts are also short, not just relief)
    # The IP/GS check is the key safeguard against Ben Brown false positives.
    if gs_n > 0 and gp_n >= gs_n * 2.0 and (gp_n - gs_n) >= 3:
        ip_per_outing = ip_n / gp_n if gp_n > 0 else 5.0
        # Approximate IP-per-start: assume relief outings are ~1.0 IP, the
        # rest are starts. If that "implied start IP" is still short, it's
        # a true swingman; if it's 4+, he's a starter who did some relief.
        relief_ip_est = (gp_n - gs_n) * 1.0
        start_ip_est = max(0, ip_n - relief_ip_est)
        ip_per_start_est = start_ip_est / gs_n if gs_n > 0 else 0
        if ip_per_outing < 2.8 and ip_per_start_est < 4.0:
            return rookie_prefix + "🔄 SWING"

    # Zero starts but has appearances → opener
    if gs_n == 0 and gp_n > 0:
        return rookie_prefix + "🚨 OPENER"
    # Zero games entirely → debut
    if gs_n == 0 and gp_n == 0:
        return "🌱 NEW STARTER"

    # Has starts. Refine:
    if is_returning_starter and ip_n < min_ip * 0.8:
        return rookie_prefix + "🏥 RETURNING"
    if is_rookie or gs_n < min_gs:
        return "🌱 NEW STARTER"
    if ip_n < full_ip * 0.6:
        return rookie_prefix + "⚠️ LOW IP"
    return rookie_prefix + "✓" if rookie_prefix else "✓"


# ============================================================================
# v43.63 (test-harness support): module-level column-coverage constants
# ----------------------------------------------------------------------------
# These were locals inside build_matchup_table — fine for runtime, but
# unreachable from the pytest harness that wants to assert "every column
# in this list survives the function on synthetic input." Lifting to module
# level makes them accessible to tests AND preserves all runtime behavior
# (build_matchup_table now references the module-level names).
# ============================================================================
KNOWN_CONSUMED_COLUMNS = [
    # HR-criteria checklist inputs (add_hr_criteria)
    "pull_pct", "avg_ev", "barrel_pct", "ideal_attack_angle_pct",
    # v43.66 researcher framework — Must-Have + Nuclear input columns.
    # If any of these get dropped by the display_cols whitelist, the
    # corresponding criterion silently goes dark on every row.
    "pulled_brl_pct", "pull_air_pct", "hard_hit", "avg_dist",
    "fb_pct", "blast_pct", "iso", "slg", "gb_pct",
    "barrel_count", "home_run", "near_hr_est",
    # Comprehensive HR composite (compute_comprehensive_hr_composite)
    "hr_game_pct", "max_hit_speed",
    "matchup_opp", "pitch_hr_score",
    "recent_hr_weighted_rate", "hr_streak_games", "hr_form",
    "env_boost", "discipline_score",
    # Handedness override inputs (apply_handedness_overrides)
    "vs_lhp_pa", "vs_lhp_barrel_pct", "vs_lhp_hard_hit",
    "vs_lhp_xwoba", "vs_lhp_avg_ev", "vs_lhp_iso",
    "vs_rhp_pa", "vs_rhp_barrel_pct", "vs_rhp_hard_hit",
    "vs_rhp_xwoba", "vs_rhp_avg_ev", "vs_rhp_iso",
    # Total bases (props.total_bases_per_pa) — reads slg/xslg
    "xslg",
    # Grade context platoon annotations (grade_context block)
    "bats", "opp_pitcher_throws", "platoon_hitter_flag",
]

ALWAYS_EXPECTED_COLUMNS = {
    # HR-criteria inputs that should always be in hitter_stats
    "pull_pct": "build_matchup_table rename of pull_percent",
    "avg_ev": "Savant hitter_stats — required for grade caps",
    "barrel_pct": "Savant hitter_stats — required for composite",
    # v43.58: blast_pct is the primary swing-quality signal we expect now
    # (replaces ideal_attack_angle_pct which Savant doesn't expose in the
    # current endpoint). HR Criteria #4 falls back to blast_pct.
    "blast_pct": (
        "bat tracking fetch (📡 toggle in sidebar). If the toggle is OFF, "
        "turn it on. If the toggle is ON, check the 'Bat tracking fetch' "
        "status line above this warning. With blast_pct missing, HR "
        "Criteria #4 will be '·' for every hitter (no fallback signal)."
    ),
}


def build_matchup_table(
    lineup: list[dict],
    pitcher_row: pd.Series | None,
    hitter_stats: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    recent_form_dict: dict | None = None,  # {player_id: {recent_iso, recent_avg, ...}}
    pitcher_arsenal_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the matchup table with ALL columns from the screenshots.

    v43.62 (reviewer doc fix #1.8) — IMPORTANT scope note:
    This function builds ONE lineup's table at a time (typically 9 starters,
    plus separately the bench frame). Any column produced by ranking inside
    this function — `matchup`, `ceiling`, `hr_form` — is therefore a
    **per-lineup percentile** (rank within those 9 rows), NOT a slate-wide
    quantity. Comparing those values across games will mislead: the #1 hitter
    in a weak lineup looks the same as the #1 in a stacked lineup.

    Use absolute columns (`matchup_opp`, `hr_score`, `hr_game_pct`) for
    cross-game comparisons. `pick_score` was rebuilt in v41 specifically
    to use absolute `matchup_opp` — keep new cross-game features on that
    path, not the per-lineup ranks.
    """
    if not lineup:
        return pd.DataFrame()

    ids = [p["id"] for p in lineup if p.get("id")]
    h = hitter_stats[hitter_stats["player_id"].isin(ids)].copy() if "player_id" in hitter_stats.columns else hitter_stats.copy()

    rows = []
    for i, p in enumerate(lineup, start=1):
        match = h[h["player_id"] == p["id"]] if "player_id" in h.columns else pd.DataFrame()
        row = match.iloc[0].to_dict() if len(match) else {"player_id": p.get("id")}
        row["player_name"] = p["name"]
        is_fill = bool(p.get("is_roster_fill", False))
        # Mark whether this row came from a real lineup or roster-padding fallback.
        # Used downstream so we don't apply lineup-spot PA scaling to fake positions.
        row["is_roster_fill"] = is_fill
        # If roster-fill, set lineup_pos to NaN so the column displays "—" rather
        # than a misleading synthetic position number. Real-lineup rows get 1-9.
        row["lineup_pos"] = i if not is_fill else np.nan
        row["position"] = p.get("position", "")
        row["bats"] = p.get("bats", "")
        # Inject recent form if provided
        if recent_form_dict and p.get("id") in recent_form_dict:
            for k, v in recent_form_dict[p["id"]].items():
                row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)

    # Pitcher context columns (same value for whole lineup)
    if pitcher_row is not None and not pitcher_row.empty:
        df["pitcher_xwoba"] = pitcher_row.get("xwoba", np.nan)
        df["pitcher_k_pct"] = pitcher_row.get("k_percent", np.nan)
        df["pitcher_k_inv"] = -1 * df["pitcher_k_pct"]
        df["pitcher_barrel_allowed"] = pitcher_row.get("barrel_batted_rate", np.nan)
        df["pitcher_hr"] = pitcher_row.get("home_run", np.nan)
        df["pitcher_whiff"] = pitcher_row.get("whiff_percent", np.nan)
        # v43: pitcher zone% — needed for plate_discipline_flag
        df["pitcher_zone_pct"] = pitcher_row.get("zone_percent", np.nan)
    else:
        df["pitcher_xwoba"] = np.nan
        df["pitcher_k_inv"] = np.nan
        df["pitcher_barrel_allowed"] = np.nan
        df["pitcher_zone_pct"] = np.nan

    # Normalize column names to consistent shorts
    # COALESCE LA SOURCES BEFORE RENAME
    # The rename below maps multiple Savant column names to "la". If two map
    # to the same target, pandas creates duplicate columns and the dedupe
    # at the bottom keeps whichever appears first — which may be the empty
    # one. Pre-emptively pick the most-populated LA candidate and drop the
    # rest, so the survivor carries real data through the rename.
    la_candidates = ["launch_angle", "launch_angle_avg", "avg_hit_angle",
                       "la_avg", "la"]
    la_present = [c for c in la_candidates if c in df.columns]
    if len(la_present) > 1:
        # Score each by populated count; keep the winner
        best_col = None
        best_count = -1
        for c in la_present:
            coerced = pd.to_numeric(df[c], errors="coerce")
            n = int(coerced.notna().sum())
            if n > best_count:
                best_count = n
                best_col = c
        if best_col is not None:
            # Drop the others so the rename has a single source for "la"
            to_drop = [c for c in la_present if c != best_col]
            df = df.drop(columns=to_drop)

    # v43.14 (reviewer-validated, latent risk fix): same coalesce for avg_ev.
    # Three source columns (`avg_best_speed`, `exit_velocity_avg`,
    # `launch_speed`) map to `avg_ev` in the rename. After the rename, three
    # columns share that name; the `~columns.duplicated()` dedupe at line ~312
    # keeps the FIRST one — which could be the least-populated. `la` got the
    # explicit pre-coalesce above; `avg_ev` was relying on column ordering
    # luck. Apply the same pattern so the winner is data-driven, not
    # order-driven. Not biting today (avg_ev coverage 88%, matches la), but
    # latent — the day Savant changes column order we'd silently lose 87%
    # of our exit-velo data.
    ev_candidates = ["avg_best_speed", "exit_velocity_avg", "launch_speed",
                       "avg_ev"]
    ev_present = [c for c in ev_candidates if c in df.columns]
    if len(ev_present) > 1:
        best_col = None
        best_count = -1
        for c in ev_present:
            coerced = pd.to_numeric(df[c], errors="coerce")
            n = int(coerced.notna().sum())
            if n > best_count:
                best_count = n
                best_col = c
        if best_col is not None:
            to_drop = [c for c in ev_present if c != best_col]
            df = df.drop(columns=to_drop)

    rename = {
        "barrel_batted_rate": "barrel_pct",
        "hard_hit_percent": "hard_hit",
        "k_percent": "k_pct",
        "bb_percent": "bb_pct",
        "avg_best_speed": "avg_ev",
        "exit_velocity_avg": "avg_ev",
        "launch_speed": "avg_ev",
        "sweet_spot_percent": "sweet_spot_pct",
        "flyballs_percent": "fb_pct",
        "groundballs_percent": "gb_pct",
        "linedrives_percent": "ld_pct",
        "whiff_percent": "whiff_pct",
        # v43.66 (researcher framework): Avg Distance (ft) — used by
        # Must-Have ≥315 ft / Nuclear ≥330 ft thresholds. Native Savant
        # name is `avg_hit_distance`; normalized to `avg_dist` to match
        # this codebase's percentage-shorthand convention (`pull_pct`,
        # `hard_hit`, etc.).
        "avg_hit_distance": "avg_dist",
        # v43.66: raw barrel COUNT (not %). Used to derive near_hr_est =
        # max(0, barrels - home_run) for the Nuclear ≥3 "Near HR" threshold.
        # The researcher's definition of "Near HR" isn't documented; we
        # use "barrels that didn't leave the yard" as the operational
        # proxy. Flagged in tooltip.
        "barrels": "barrel_count",
        # Savant uses several names for launch angle
        "launch_angle": "la",
        "launch_angle_avg": "la",
        "avg_hit_angle": "la",
        "la_avg": "la",
        # Savant uses isolated_power for ISO in some exports
        "isolated_power": "iso",
        "iso_power": "iso",
        "pull_air_percent": "pull_air_pct",
        # v43.51 (reviewer-validated): missing rename caused 3 features to be
        # silently inert. add_hr_criteria's pull≥40 check, the mechanics tier
        # of comprehensive grade, and the mechanical-fail cap (pull<35 AND
        # ev<88) all read pull_pct, but build_matchup_table only produced
        # pull_percent. The v43.39 fix added pull_percent to display_cols
        # whitelist but didn't add the rename, so pull_pct stayed None
        # everywhere downstream. Adding the rename here fixes all three.
        "pull_percent": "pull_pct",
    }
    df = df.rename(columns=rename)

    # Drop dupes from the rename collision
    if isinstance(df.columns, pd.Index) and df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # If ISO is STILL missing, derive from SLG - AVG
    if "iso" not in df.columns or df["iso"].isna().all():
        slg_col = next((c for c in ["slg", "xslg"] if c in df.columns), None)
        avg_col = next((c for c in ["batting_avg", "avg", "ba", "xba"] if c in df.columns), None)
        if slg_col and avg_col:
            # ISO = SLG - AVG. Always non-negative in real samples but can be
            # negative in tiny samples where AVG > SLG due to all singles.
            df["iso"] = (df[slg_col] - df[avg_col]).clip(lower=0).round(3)

    # v42d/e: Derive pull_air_pct when Savant doesn't populate it.
    # User's data coverage audit showed pull_air_percent comes back 0%
    # populated from Savant's custom leaderboard endpoint. Best derivation:
    # pull_percent × fb_pct. This is APPROXIMATE — assumes pull rate on flies
    # matches overall pull rate (slightly under-estimates extreme pull
    # hitters) but the relative ordering is preserved.
    #
    # v42e: derive when MORE THAN 50% missing, not just all-missing. Savant
    # may return the column present with a few populated rows but mostly
    # NaN — in that case we still want the derivation to fire so the
    # column has data for everyone.
    _need_derive = (
        "pull_air_pct" not in df.columns
        or df["pull_air_pct"].isna().mean() > 0.5
    )
    if _need_derive:
        if "pull_percent" in df.columns and "fb_pct" in df.columns:
            df["pull_air_pct"] = (df["pull_percent"] * df["fb_pct"] / 100).round(2)
        elif "pull_pct" in df.columns and "fb_pct" in df.columns:
            df["pull_air_pct"] = (df["pull_pct"] * df["fb_pct"] / 100).round(2)

    # If LA is still missing entirely but we have any launch_speed_angle data,
    # leave it blank (no derivation - it's a real measurement, not an identity)

    df["k_pct_inv"] = -df["k_pct"] if "k_pct" in df.columns else np.nan

    # Composite scores
    df["matchup"] = _score_from_weights(df, SCORING_WEIGHTS["matchup"])
    # v43.16: gb_pct is inverted via `neg` — high GB rate REDUCES hr_form,
    # matching the user's intent ("HR form should... avoid high ground ball
    # rate"). The percentile rank is flipped: a 60% GB hitter ranks at the
    # 5th percentile post-inversion, contributing minimally to hr_form even
    # though their raw GB% is high.
    df["hr_form"] = _score_from_weights(
        df, SCORING_WEIGHTS["hr_form"], neg=("gb_pct",)
    )
    df["ceiling"] = _score_from_weights(df, SCORING_WEIGHTS["ceiling"])

    # COLD-STREAK CEILING on hr_form (June 2026)
    # The hr_form score above is built from SEASON-LONG power metrics
    # (barrel%, iso, hard_hit, etc.) — none of which decay when a hitter
    # goes cold. Result: an elite power hitter who hasn't homered in 13
    # games still scores 90+ on hr_form, which then inflates pick_score and
    # crowds out genuinely-hot hitters from the picks.
    #
    # Carroll case (June 2026): games_since_hr=13, hr_last_5=0, but hr_form=92.78
    # → ranked #3 in picks, squeezing Kurtz (hr_last_5=2, games_since_hr=1)
    # out of picks AND honorable mentions entirely.
    #
    # Fix: apply a games_since_hr ceiling. "Hot form" should require recent
    # HRs. Cold streaks cap the form score regardless of underlying power.
    if "hr_form" in df.columns and "games_since_hr" in df.columns:
        def _apply_cold_ceiling(row):
            form = row.get("hr_form")
            if form is None or pd.isna(form):
                return form
            gsh = row.get("games_since_hr")
            if gsh is None or pd.isna(gsh):
                return form  # No data — don't penalize
            try:
                gsh_n = float(gsh)
            except (TypeError, ValueError):
                return form
            # Ceiling schedule:
            #   < 5 games:   no cap (hot or recent enough)
            #   5-6 games:   75 ceiling (cooling off)
            #   7-9 games:   60 ceiling (notable cold streak)
            #   10+ games:   40 ceiling (clearly cold, can't be "hot form")
            if gsh_n >= 10:
                ceiling = 40.0
            elif gsh_n >= 7:
                ceiling = 60.0
            elif gsh_n >= 5:
                ceiling = 75.0
            else:
                return form  # No cap
            return min(float(form), ceiling)
        df["hr_form"] = df.apply(_apply_cold_ceiling, axis=1)

    # Test Score = matchup × PA sample weight, BLENDED with recent form so it's
    # meaningfully different from matchup (which is mostly season stats).
    # 70% matchup + 30% hr_form weighted by PA reliability.
    if "pa" in df.columns:
        pa_factor = (df["pa"] / 150.0).clip(0.5, 1.0)
        # Blend matchup with hr_form (recent trend) - this is what makes test_score
        # different from matchup. Players hot recently get boosted vs season-only.
        if "hr_form" in df.columns:
            blended = df["matchup"] * 0.70 + df["hr_form"] * 0.30
            df["test_score"] = (blended * pa_factor).round(2)
        else:
            df["test_score"] = (df["matchup"] * pa_factor).round(2)
    else:
        df["test_score"] = df["matchup"]

    # Zone Fit: only if real data exists - no fake xwoba defaults
    if "xwobacon" in df.columns and df["xwobacon"].notna().any():
        base = _safe_pct_rank(df["xwobacon"]) / 100
        if "pitcher_xwoba" in df.columns and not df["pitcher_xwoba"].isna().all():
            p_factor = ((df["pitcher_xwoba"] - 0.250) / 0.150).clip(0, 1)
            df["zone_fit"] = (base * 0.5 + p_factor * 0.5).round(3)
        else:
            df["zone_fit"] = base.round(3)
    else:
        df["zone_fit"] = np.nan

    # kHR - real data only. NaN K% → NaN kHR
    if "k_pct" in df.columns and "pitcher_k_pct" in df.columns:
        df["k_combined"] = (df["k_pct"] + df["pitcher_k_pct"]) / 2
        df["kHR"] = (100 - _safe_pct_rank(df["k_combined"])).round(2)
    else:
        df["kHR"] = np.nan

    # HR Form arrow
    if "recent_iso" in df.columns and "iso" in df.columns:
        df["hr_form_arrow"] = df.apply(
            lambda r: _form_arrow(r.get("recent_iso"), r.get("iso")), axis=1
        )
        df["hr_form_label"] = df.apply(
            lambda r: f"{r['hr_form']:.0f}% {r['hr_form_arrow']}"
            if pd.notna(r.get("hr_form")) else "—", axis=1
        )
    else:
        df["hr_form_arrow"] = "→"
        df["hr_form_label"] = df["hr_form"].apply(
            lambda x: f"{x:.0f}% →" if pd.notna(x) else "—"
        )

    # "Likely HR%" - only if real data
    if "barrel_pct" in df.columns and "fb_pct" in df.columns:
        df["likely_hr_pct"] = ((df["barrel_pct"] * df["fb_pct"] / 100) * 0.75).round(2)
    elif "barrel_pct" in df.columns:
        df["likely_hr_pct"] = (df["barrel_pct"] * 0.35).round(2)
    else:
        df["likely_hr_pct"] = np.nan

    # Drop synthetic pitches/bip estimates — they were 100% derived, not real

    # Final column order matching the screenshots
    # CONTACT PROFILE FLAG
    # Contact hitters (Arraez, Kwan, Tucker, classic high-OBP guys) hit so
    # few HRs that they always show low HR Game%. The model is correct; the
    # issue is user expectations. Flag them explicitly so users know not to
    # use them as HR plays. Threshold: ISO < 0.120 AND barrel% < 5 AND
    # K% < 16 — Arraez profile (ISO ~0.05, barrel ~0%, K% ~4%).
    # Less strict: ISO < 0.140 AND barrel% < 7 AND K% < 18 — Tucker tier.
    def _contact_flag(row):
        try:
            iso = float(row.get("iso")) if not pd.isna(row.get("iso")) else None
            barrel = float(row.get("barrel_pct")) if not pd.isna(row.get("barrel_pct")) else None
            k_pct = float(row.get("k_pct")) if not pd.isna(row.get("k_pct")) else None
        except (TypeError, ValueError):
            return ""
        if iso is None or barrel is None or k_pct is None:
            return ""
        # Extreme contact (Arraez tier)
        if iso < 0.120 and barrel < 5.0 and k_pct < 16.0:
            return "🎯 contact, not HR"
        # Solid contact, low power (Kwan/Hoerner tier)
        if iso < 0.140 and barrel < 7.0 and k_pct < 18.0:
            return "🎯 low-power profile"
        return ""
    df["contact_flag"] = df.apply(_contact_flag, axis=1)

    # GROUND BALL TYPE FLAG (v42o) — at-a-glance identification of hitters
    # whose batted-ball profile makes HRs structurally unlikely. Display-only,
    # does NOT change scoring (barrel% already filters for HR-relevant contact).
    #
    # Three tiers:
    #   🦗 EXTREME GB — gb_pct >= 55 (top ~10% of GB rate; HRs structurally rare)
    #   🦗 GB-leaning — gb_pct >= 48 AND pull_pct < 38 (high-GB AND oppo-prone)
    #   (empty)       — everyone else
    #
    # Note: A high-GB hitter with high barrel% (rare but possible — Bregman
    # at times) won't fire the EXTREME flag because barrel% is still doing
    # the right thing. The flag fires for the structural profile, not the
    # absolute HR projection.
    def _gb_type_flag(row):
        try:
            gb = float(row.get("gb_pct")) if not pd.isna(row.get("gb_pct")) else None
            # v43.51: was reading pull_percent — that's now renamed to pull_pct
            # at line 357. Use pull_pct with fallback to pull_percent for safety
            # in case any code path bypasses the rename.
            _pull_raw = row.get("pull_pct") if not pd.isna(row.get("pull_pct", None)) else row.get("pull_percent")
            pull = float(_pull_raw) if _pull_raw is not None and not pd.isna(_pull_raw) else None
        except (TypeError, ValueError):
            return ""
        if gb is None:
            return ""
        if gb >= 55.0:
            return "🦗 extreme GB"
        if gb >= 48.0 and pull is not None and pull < 38.0:
            return "🦗 GB-leaning"
        return ""
    df["gb_type_flag"] = df.apply(_gb_type_flag, axis=1)

    # 5-GAME FORM TREND FLAG (v42t)
    # User-requested: compare 5-game ISO/SLG to season baseline to answer
    # "is this hitter's HR form better or worse than their season pace?"
    # Display-only flag; does NOT feed into scoring.
    #
    # Tiers:
    #   🔥 hot streak    — recent_iso_5 ≥ season iso × 1.40 AND meaningful AB
    #   📈 trending up   — recent_iso_5 ≥ season iso × 1.15 AND meaningful AB
    #   ❄️ cold streak   — recent_iso_5 ≤ season iso × 0.50 AND meaningful AB
    #   📉 trending down — recent_iso_5 ≤ season iso × 0.75 AND meaningful AB
    #   (empty)          — within ±25% of baseline, normal variance
    #
    # Requires at least 10 ABs in the 5-game window to avoid noise from
    # platoon/injury players who barely played.
    def _form_trend_flag(row):
        try:
            iso5 = row.get("recent_iso_5")
            ab5 = row.get("recent_ab_5")
            iso_season = row.get("iso")
            if (iso5 is None or pd.isna(iso5) or ab5 is None or pd.isna(ab5)
                or iso_season is None or pd.isna(iso_season)):
                return ""
            iso5 = float(iso5); ab5 = float(ab5); iso_season = float(iso_season)
            if ab5 < 10 or iso_season <= 0.05:
                return ""  # not enough recent AB, or weak-baseline hitter (avoid divide-by-tiny)
            ratio = iso5 / iso_season
            if ratio >= 1.40:
                return f"🔥 hot ({iso5:.3f} vs {iso_season:.3f})"
            if ratio >= 1.15:
                return f"📈 trending up ({iso5:.3f} vs {iso_season:.3f})"
            if ratio <= 0.50:
                return f"❄️ cold ({iso5:.3f} vs {iso_season:.3f})"
            if ratio <= 0.75:
                return f"📉 trending down ({iso5:.3f} vs {iso_season:.3f})"
            return ""
        except (TypeError, ValueError):
            return ""
    df["form_trend_flag"] = df.apply(_form_trend_flag, axis=1)

    # PLATE DISCIPLINE MISMATCH FLAG (v43)
    # User-requested signal: compare batter zone approach to pitcher zone approach.
    # Uses data we already pull — pitcher zone% (how often they throw strikes)
    # vs hitter z_swing% (swings on in-zone pitches) and oz_swing% (chase rate).
    #
    # This is NOT the literal "per-zone HR rate" zone fit (which requires Savant
    # heart/shadow/chase/waste tier data we don't yet fetch). It IS the
    # high-value subset of that signal: who's likely to put bat on ball in
    # exploitable spots.
    #
    # Categories:
    #   💥 ATTACK + AGGRESSIVE — pitcher attacks zone AND hitter swings at zone:
    #         pitcher_zone_pct >= 50 AND z_swing_pct >= 70 = HR risk for pitcher
    #         (lots of bat-on-ball in the heart of the plate)
    #   🎯 ATTACK + PATIENT — pitcher attacks zone, hitter waits on it:
    #         pitcher_zone_pct >= 50 AND z_swing_pct < 60 = neutral
    #         (pitcher gets ahead but hitter is selective)
    #   🆓 WILD + CHASER — wild pitcher meets free swinger:
    #         pitcher_zone_pct < 45 AND oz_swing_pct >= 32 = HR risk
    #         (pitcher misses zone, hitter expands → mistakes get punished)
    #   ⚠️ WILD + DISCIPLINED — wild pitcher meets patient hitter:
    #         pitcher_zone_pct < 45 AND oz_swing_pct < 25 = walks not HRs
    #         (pitcher can't find zone, hitter takes the BB)
    #   (empty) — no notable mismatch
    #
    # Display-only flag. Does NOT feed into pick_score (preserving the
    # +25.9pp calibration we already measured).
    def _plate_discipline_flag(row):
        try:
            p_zone = row.get("pitcher_zone_pct")
            h_zswing = row.get("z_swing_percent")
            h_ozswing = row.get("oz_swing_percent")
            if p_zone is None or pd.isna(p_zone):
                return ""
            p_zone = float(p_zone)
            # ATTACK pitcher (≥50% in zone) cases
            if p_zone >= 50.0:
                if h_zswing is not None and not pd.isna(h_zswing):
                    z = float(h_zswing)
                    if z >= 70.0:
                        return "💥 attack × aggressive"
                    if z < 60.0:
                        return "🎯 attack × patient"
            # WILD pitcher (<45% in zone) cases
            if p_zone < 45.0:
                if h_ozswing is not None and not pd.isna(h_ozswing):
                    o = float(h_ozswing)
                    if o >= 32.0:
                        return "🆓 wild × chaser"
                    if o < 25.0:
                        return "⚠️ wild × disciplined"
            return ""
        except (TypeError, ValueError):
            return ""
    df["plate_discipline_flag"] = df.apply(_plate_discipline_flag, axis=1)

    # ZONE FIT FLAG (v43 experimental)
    # Uses Savant heart/shadow/chase/waste tier data IF the fetchers
    # successfully returned data. If empty (URL pattern wrong, schema
    # mismatch, etc.), this flag stays blank for everyone and
    # plate_discipline_flag carries the workload.
    #
    # Logic: weighted sum of (pitcher's tier distribution × hitter's wOBA in tier).
    # A pitcher who throws lots of heart-of-plate pitches vs a hitter who
    # crushes heart-of-plate = HR risk.
    #
    # Thresholds will be CALIBRATED FROM REAL DATA before going to scoring.
    # For now it's a labeled display with the raw composite value so the
    # user can verify the signal looks sensible.
    def _zone_fit_flag(row):
        try:
            # Need both pitcher tier % and hitter tier wOBA
            p_heart = row.get("pitcher_heart_pct")
            h_heart = row.get("hitter_heart_woba")
            if p_heart is None or pd.isna(p_heart):
                return ""
            if h_heart is None or pd.isna(h_heart):
                return ""
            # Compute weighted xwOBA = sum(pitcher_tier_pct × hitter_tier_woba)
            tiers = ("heart", "shadow", "chase", "waste")
            total_pct = 0.0
            weighted = 0.0
            for t in tiers:
                p_pct = row.get(f"pitcher_{t}_pct")
                h_woba = row.get(f"hitter_{t}_woba")
                if (p_pct is None or pd.isna(p_pct)
                    or h_woba is None or pd.isna(h_woba)):
                    continue
                p_pct = float(p_pct); h_woba = float(h_woba)
                weighted += p_pct * h_woba
                total_pct += p_pct
            if total_pct < 80.0:  # need ≥80% coverage of pitch distribution
                return ""
            avg_woba = weighted / total_pct
            # v43.1: thresholds calibrated to realistic composite values.
            # Tier-weighted composites get diluted by chase/waste (where every
            # hitter performs poorly). Elite matchups produce composites
            # around 0.350-0.370, not 0.420. See data_fetcher.zone_fit_score
            # for the same recalibration.
            if avg_woba >= 0.370:
                return f"💣 elite zone fit ({avg_woba:.3f})"
            if avg_woba >= 0.340:
                return f"🎯 strong zone fit ({avg_woba:.3f})"
            if avg_woba <= 0.260:
                return f"🛡️ poor zone fit ({avg_woba:.3f})"
            return ""
        except (TypeError, ValueError):
            return ""
    df["zone_fit_flag"] = df.apply(_zone_fit_flag, axis=1)

    # SPLIT CONFIDENCE FLAG (June 2026)
    # When a hitter's projection is being driven by a vs-LHP or vs-RHP split
    # with a small PA sample, the user has no way to know that. A hitter with
    # 22.77% HR Game% based on 35 PA vs LHP is much shakier than one based
    # on 300 PA. Flag them so speculative plays are visible.
    #
    # Thresholds are SEASON-AWARE (v39i fix). In April most hitters have
    # tiny samples and flagging everything would be useless noise. In August
    # most hitters have meaningful samples and the bar should be higher.
    # See _season_thresholds() for the per-month calibration.
    #
    # v43.19 (reviewer-noted limitation): _season_thresholds() is called
    # without a date, so it keys off TODAY's calendar rather than the
    # slate's date. For live use this is correct. For backtest where
    # someone evaluates an August slate while running the app in October,
    # the thresholds will reflect October's calibration not August's.
    # Tracked as a follow-up — fix is to thread slate_date through the
    # build_matchup_table signature.
    _thresh = _season_thresholds()
    _split_thin = _thresh.get("split_thin", 40)
    _split_small = _thresh.get("split_small", 70)

    p_throws_raw = None
    if pitcher_row is not None:
        try:
            p_throws_raw = (pitcher_row.get("p_throws") or pitcher_row.get("throws") or "")
            p_throws_raw = str(p_throws_raw).upper() if p_throws_raw else None
        except Exception:
            p_throws_raw = None

    def _split_confidence(row):
        if not p_throws_raw or p_throws_raw not in ("L", "R"):
            return ""
        split_key = "lhp" if p_throws_raw == "L" else "rhp"
        pa_val = row.get(f"vs_{split_key}_pa")
        if pa_val is None or pd.isna(pa_val):
            return ""  # No split data — handled separately
        try:
            pa_f = float(pa_val)
        except (TypeError, ValueError):
            return ""
        if pa_f < _split_thin:
            return "⚠️ thin split"
        if pa_f < _split_small:
            return "📊 small split"
        return ""
    df["split_confidence"] = df.apply(_split_confidence, axis=1)

    # v43.4: HANDEDNESS DIVERGENCE FLAG (reviewer-validated concern).
    # The grade and pick_score use SEASON-OVERALL barrel%, hard_hit%, EV,
    # ISO etc. for power_score and hr_form. Only HR outcome rate
    # (vs_lhp_hr_per_pa / vs_rhp_hr_per_pa) is handedness-aware in the
    # actual scoring. So a hitter who's bad against one side but good
    # overall would still get an inflated grade against that side.
    #
    # We don't pull handedness-specific barrel/hard_hit/EV (those are
    # Statcast/Savant only — MLB Stats API gives basic splits only). But
    # we DO have handedness-specific ISO. So we compare today's-side ISO
    # to season-overall ISO and flag when they diverge meaningfully.
    # This is a HONEST WARNING that the grade may not reflect tonight's
    # actual matchup quality:
    #   ⚠️ reverse split    : today-side ISO is ≥60 pts BELOW overall
    #                          (grade likely overstated — fade)
    #   💪 favored split    : today-side ISO is ≥60 pts ABOVE overall
    #                          (grade likely understated — back)
    # 60-pt ISO threshold = 1 std-dev of typical platoon splits (~50-70 pts).
    # Requires ≥30 PA on the relevant side to avoid noise.
    def _handedness_divergence(row):
        if not p_throws_raw or p_throws_raw not in ("L", "R"):
            return ""
        split_key = "lhp" if p_throws_raw == "L" else "rhp"
        # Skip switch hitters — vs_lhp/vs_rhp data already reflects which
        # side they batted, so any divergence here is meaningful as-is.
        # (Their grade is already side-correct via the HR rate split.)
        season_iso = row.get("iso")
        split_iso = row.get(f"vs_{split_key}_iso")
        split_pa = row.get(f"vs_{split_key}_pa")
        if (season_iso is None or pd.isna(season_iso)
            or split_iso is None or pd.isna(split_iso)
            or split_pa is None or pd.isna(split_pa)):
            return ""
        try:
            season_iso_f = float(season_iso)
            split_iso_f = float(split_iso)
            split_pa_f = float(split_pa)
        except (TypeError, ValueError):
            return ""
        if split_pa_f < 30:
            return ""  # too noisy to trust the divergence
        diff = split_iso_f - season_iso_f
        if diff <= -0.060:
            return f"⚠️ reverse split ({split_iso_f:.3f} vs {season_iso_f:.3f})"
        if diff >= 0.060:
            return f"💪 favored split ({split_iso_f:.3f} vs {season_iso_f:.3f})"
        return ""
    df["handedness_divergence"] = df.apply(_handedness_divergence, axis=1)

    display_cols = [
        "player_id", "player_name", "lineup_pos", "position", "bats",
        "is_roster_fill",  # CRITICAL: flag for whether lineup_pos is real or fill
        "contact_flag",  # 🎯 contact profile (set expectations)
        "gb_type_flag",  # 🦗 GB-leaning / extreme GB (structural HR floor)
        "form_trend_flag",  # 🔥/📈/❄️/📉 5-game ISO vs season baseline
        "plate_discipline_flag",  # 💥/🎯/🆓/⚠️ pitcher zone% × hitter z/oz swing%
        "zone_fit_flag",  # 💣/🎯/🛡️ Savant heart/shadow/chase/waste tier fit (v43, experimental)
        "pitcher_zone_pct",  # surfaced for the plate_discipline flag context
        "split_confidence",  # ⚠️ thin split / 📊 small split (caution flag)
        "handedness_divergence",  # v43.4: ⚠️/💪 ISO divergence vs overall (grade caveat)
        # Composites (matching screenshot order)
        "matchup", "test_score", "ceiling", "zone_fit",
        "hr_form", "hr_form_label", "hr_form_arrow", "kHR",
        # Pitches / BIP / ISO / xwOBA family
        "pitches", "bip", "iso", "xwoba", "xwobacon",
        # Quality of contact
        "barrel_pct", "pulled_brl_pct", "hard_hit", "sweet_spot_pct",
        "fb_pct", "gb_pct", "ld_pct",
        "la", "avg_ev",
        # v43.66 (researcher framework): new inputs the Must-Have / Nuclear
        # checklists read. avg_dist is fetched directly from Savant; barrel_count
        # is the raw count (not %); near_hr_est is derived in app.py as
        # max(0, barrel_count - home_run). All three MUST survive display_cols
        # or the new criteria silently die — the v43.5/v43.51-class bug.
        "avg_dist", "barrel_count", "near_hr_est",
        # v43.17: bat tracking columns (Blast %, bat speed, etc.) — only
        # populated when sidebar opt-in toggle is enabled AND Savant
        # fetch succeeds. Otherwise NaN, _score_from_weights handles that
        # gracefully and they appear empty in the export.
        "blast_pct", "bat_speed", "fast_swing_pct", "squared_up_pct",
        # v43.51 (reviewer-validated): IAA was double-filtered — dropped in
        # the app.py merge AND missing from display_cols. Now wired in both
        # places. If Savant still doesn't return it, the column is just NaN
        # everywhere and the HR Criteria checklist shows '·' for everyone
        # (graceful degradation), not silently dropping it before it can be
        # checked.
        "ideal_attack_angle_pct",
        # v42k: pull_air_pct (derived from pull_percent × fb_pct when Savant's
        # native column is empty). Was being computed in _normalize_player_df
        # but dropped here, so never reached combined_all — the H2H comparison
        # tool and other consumers always saw it missing.
        "pull_air_pct",
        # v43.39 (reviewer-validated): raw pull_pct/pull_percent was being
        # filtered out, so add_hr_criteria's pull≥40 check and
        # compute_comprehensive_hr_grade's mechanical-fail cap couldn't
        # see it. Pull_air_pct (pull × FB / 100) is on a ~10-20 scale,
        # not 40+. Adding both canonical names so whichever Savant returns
        # survives into the matchup_df.
        "pull_pct", "pull_percent",
        # Plate discipline
        "k_pct", "bb_pct", "whiff_pct", "swing_percent",
        # Rates
        "obp", "slg", "ops", "babip",
        # v43.52 (assertion-caught): xslg is consumed by total_bases_per_pa
        # in props.py — it's blended with slg (xslg weighted higher, 60/40)
        # for the expected-bases projection. Without xslg in this whitelist,
        # total_bases was falling back to slg-only, losing the predictive
        # xSLG signal. The v43.51 column-coverage assertion caught this on
        # the next run — exactly the bug class it was built to catch.
        "xslg",
        # v43.29 (reviewer-validated CRITICAL fix): xBA + actual BA were
        # being filtered out by this display_cols whitelist before reaching
        # hit_prob_per_pa, so hit_game_pct fell back to the 0.250 default
        # constant for every hitter — user-visible symptom: "every Red Sox
        # player is B+." Same display_cols-drops-needed-columns failure
        # pattern as the vs-LHP splits we fixed earlier. Adding aliases too
        # for defensive resilience.
        "xba", "ba", "avg", "batting_avg",
        # Counts
        "pa", "home_run", "recent_hr", "recent_iso", "recent_avg",
        "recent_hr_weighted_rate",
        "streak_label", "hr_streak_games", "hr_last_5", "hr_last_10", "games_since_hr",
        # v42c: 10-game window stats (community-validated ideal HR signal)
        "recent_iso_10", "recent_avg_10", "recent_k_pct_10",
        "recent_ab_10", "recent_h_10", "recent_hr_10",
        # v42t: 5-game window — hot/cold streak detection
        "recent_iso_5", "recent_slg_5", "recent_avg_5",
        "recent_hr_5", "recent_ab_5",
        # Day/night splits (vs day games, vs night games)
        "vs_day_pa", "vs_day_avg", "vs_day_obp", "vs_day_slg", "vs_day_ops",
        "vs_day_hr_per_pa", "vs_day_k_percent",
        "vs_night_pa", "vs_night_avg", "vs_night_obp", "vs_night_slg", "vs_night_ops",
        "vs_night_hr_per_pa", "vs_night_k_percent",
        # HITTER vs-LHP / vs-RHP splits — CRITICAL: must be in display_cols
        # whitelist or they get dropped before reaching hr_prob_per_pa().
        # Was the silent bug that prevented the entire hitter-split system
        # added in v20 from ever actually firing on projections.
        # v43.11: vs_X_hr (raw count) added alongside rate so users can see
        # "12 HRs in 280 vs-RHP PA" not just "4.3% HR rate."
        "vs_lhp_pa", "vs_lhp_avg", "vs_lhp_obp", "vs_lhp_slg",
        "vs_lhp_iso", "vs_lhp_ops", "vs_lhp_hr", "vs_lhp_hr_per_pa",
        "vs_lhp_k_percent", "vs_lhp_bb_percent",
        "vs_rhp_pa", "vs_rhp_avg", "vs_rhp_obp", "vs_rhp_slg",
        "vs_rhp_iso", "vs_rhp_ops", "vs_rhp_hr", "vs_rhp_hr_per_pa",
        "vs_rhp_k_percent", "vs_rhp_bb_percent",
        # v43.51 (reviewer-validated): the Savant-sourced handedness contact-
        # quality columns were missing from this whitelist, so
        # apply_handedness_overrides could only blend ISO (which comes from
        # the MLB Stats API split path). The four other stats it tries to
        # blend (barrel_pct, hard_hit, xwoba, avg_ev) were silently being
        # the season-overall values, NOT the handedness-specific ones. The
        # v43.5 "central handedness fix" was 4/5 inert. Adding them now.
        "vs_lhp_barrel_pct", "vs_lhp_hard_hit", "vs_lhp_xwoba", "vs_lhp_avg_ev",
        "vs_rhp_barrel_pct", "vs_rhp_hard_hit", "vs_rhp_xwoba", "vs_rhp_avg_ev",
        # HR PROFILE — avg HR distance + max exit velo (June 2026)
        "avg_hr_distance", "max_hit_speed",
        "hr_profile", "hr_profile_label",
        # Lift Score (v38) — contact-quality + air-ball + pitcher FB tendency
        "lift_score",
        # Today's HR projection
        "likely_hr_pct",
    ]
    keep = [c for c in display_cols if c in df.columns]

    # ========================================================================
    # v43.51 (reviewer-validated structural defense): COLUMN-COVERAGE ASSERTION
    # ========================================================================
    # The single most recurrent bug class in this codebase: a column is
    # produced upstream under one name, then renamed/dropped/whitelisted
    # away in build_matchup_table, so downstream scoring functions read
    # None and silently degrade. v43.5 (handedness Statcast overrides),
    # v43.36 (HR criteria pull threshold), v43.5 (pull_pct rename), and
    # IAA exclusion were all the same shape.
    #
    # This assertion catches the bug class STRUCTURALLY at build time. It
    # checks each column that downstream scoring depends on, and if any
    # known-consumed column was produced upstream but then dropped here,
    # surfaces a warning in session_state for the diagnostic panel. It
    # never crashes — only reports — so a missing column is information,
    # not a regression.
    #
    # NOTE: this catches DROPPED columns (column existed in df before the
    # whitelist filter). It also catches NEVER-PRODUCED columns (column
    # missing from df entirely) for a small subset of "always-expected"
    # columns — see ALWAYS_EXPECTED_COLUMNS below. v43.54 (reviewer fix):
    # added ALWAYS_EXPECTED_COLUMNS to catch the IAA-never-fetched case
    # which the produced-then-dropped check couldn't catch.
    # v43.63: KNOWN_CONSUMED_COLUMNS + ALWAYS_EXPECTED_COLUMNS are now
    # module-level (so the pytest harness can verify column coverage at
    # commit time, not just at runtime).
    # ========================================================================
    try:
        # Find columns that existed in df BEFORE the whitelist but didn't
        # make the whitelist cut. If any are known-consumed downstream,
        # surface them.
        produced_but_dropped = [
            c for c in KNOWN_CONSUMED_COLUMNS
            if c in df.columns and c not in keep
        ]
        # v43.54: find columns that were never produced upstream at all
        never_produced = [
            (c, reason) for c, reason in ALWAYS_EXPECTED_COLUMNS.items()
            if c not in df.columns
        ]
        if produced_but_dropped or never_produced:
            import streamlit as _st
            try:
                _warn = _st.session_state.setdefault(
                    "_column_coverage_warnings", []
                )
                if produced_but_dropped:
                    _msg = (
                        "build_matchup_table dropped known-consumed columns "
                        "from display_cols: " + ", ".join(produced_but_dropped)
                    )
                    if _msg not in _warn:
                        _warn.append(_msg)
                for col, reason in never_produced:
                    _msg = (
                        f"Always-expected column '{col}' was NEVER PRODUCED "
                        f"upstream. Cause: {reason}"
                    )
                    if _msg not in _warn:
                        _warn.append(_msg)
            except Exception:
                pass
    except Exception:
        pass

    return df[keep]


def add_power_score(
    matchup_df: pd.DataFrame,
    park_mult: float = 1.0,
    weather_mult: float = 1.0,
    pitcher_hr9: float | None = None,
    pitcher_barrel_allowed: float | None = None,
) -> pd.DataFrame:
    """
    Compute Power Score (0-99) for HR likelihood.

    Uses ABSOLUTE scoring based on real MLB thresholds (not per-lineup
    percentile ranks), so two great hitters on the same team don't both
    score 99 just because one is slightly better than the other.

    Each component is scored 0-100 based on where it falls between known
    league min/max thresholds. Then weighted-averaged. Then multiplied by
    environmental factors (park, weather, opposing pitcher).
    """
    if matchup_df is None or matchup_df.empty:
        return matchup_df
    df = matchup_df.copy()

    # Absolute thresholds: (column, weight, poor_value, elite_value)
    # Poor → 0, Elite → 100, linear in between.
    # "Elite" should be a near-ceiling that even MLB's best rarely hit.
    # We set elite ABOVE the actual top of the league so even Judge tops out
    # around 75-80, leaving headroom for env factors.
    #
    # WEIGHTING NOTE (June 2026): `pulled_brl_pct` has the highest single-
    # variable correlation with HR Game% in our dataset (0.806) — higher than
    # barrel_pct itself (because only PULLED barrels become HRs at
    # meaningful rates). Bumped weight 0.07→0.11 and reduced slg 0.07→0.03
    # to reflect the data's actual signal strength.
    #
    # v38 REWEIGHT (June 2026): Reviewer found fb_pct correlation 0.468 with
    # HR Game% — same as ISO (0.20 weight) — but fb_pct was only weighted
    # 0.07. SLG is largely redundant with iso and barrel_pct. Bumped fb_pct
    # 0.07→0.13, dropped slg 0.03→0.00, gave the freed 0.03 to pulled_brl_pct
    # (which has the highest single correlation 0.806). Total still sums to
    # the same 0.96. The LA-weight slot deliberately stays open since
    # sweet_spot_pct now carries that signal more cleanly.
    # meaningful rates). Bumped weight 0.07→0.11 and reduced slg 0.07→0.03
    # to reflect the data's actual signal strength.
    #
    # v38d FULL REBALANCE (June 2026): Reviewer correlation analysis across
    # all power components produced data-driven weights. Key insight: hard_hit
    # (0.590 corr) and avg_ev (0.605 corr) were underweighted at 0.10 each
    # given their signal. fb_pct was first bumped 0.07→0.13 in v38b, then
    # trimmed to 0.10 in v38d to diversify across multiple high-correlation
    # signals rather than overweight any single one. sweet_spot_pct was
    # found to be noisier than expected (0.257) and trimmed 0.05→0.03. LA
    # re-added at small 0.02 (noisy but non-zero signal).
    #
    # Specs total weights sum to 0.98 (LA is added separately at 0.04 via
    # compute_row, bringing the conceptual total to 1.02). Output is
    # normalized by total_weight at the end, so the absolute sum doesn't
    # affect scores — the RELATIVE weights are what matter. Earlier comment
    # claimed "sum to exactly 1.00" which was inaccurate (reviewer-flagged
    # v43.15). LA effectively carries slightly more relative weight than
    # the bare 0.04 number suggests; budget that when hand-tuning.
    # v42d: ROLLED BACK pull_air_pct weighting. User's data coverage audit
    # showed pull_air_percent is 0% populated from Savant's leaderboard
    # endpoint — so the 5% weight slot we added in v42b was silently doing
    # nothing for every hitter. Restored the original weights until we
    # find a Savant endpoint that actually returns pull_air data.
    # (Possible alternative: Savant Statcast Search detail-level export.
    #  Held until we can test.)
    specs = [
        # v43.23 (audit-driven rebalance): barrel_pct was 25% of power_score
        # AND power_score is 20% of pick_score AND barrel_pct appears in
        # every other component. Effective weight ~25% of total pick_score
        # on one input. Reduced barrel here; lifted pulled_brl_pct (corr
        # 0.737 was badly underweighted at 0.11), avg_ev (corr 0.605),
        # fb_pct (corr 0.461). After rebalance, no single input drives
        # more than ~12% of power_score and the underused-but-real signals
        # carry the weight their correlations earn.
        ("barrel_pct",     0.18, 4.0, 22.0),    # was 0.25 — anchor reduced
        ("pulled_brl_pct", 0.15, 2.0, 18.0),    # was 0.11 — corr 0.737
        ("iso",            0.16, 0.100, 0.350), # was 0.18
        ("avg_ev",         0.13, 85.0, 95.0),   # was 0.12 — corr 0.605
        ("hard_hit",       0.13, 30.0, 60.0),   # was 0.13
        ("fb_pct",         0.12, 18.0, 50.0),   # was 0.10 — corr 0.461
        ("recent_iso",     0.08, 0.080, 0.380), # was 0.06
        ("sweet_spot_pct", 0.03, 10.0, 40.0),   # noisier, kept small
        # v42q BUGFIX: removed `("la", 0.02, 4.0, 22.0)` — it was being
        # double-counted alongside the dedicated target-16° block at line ~707.
        # The two scoring models actively disagreed: specs treated higher LA
        # as monotonically better (cap at 22°), while the dedicated block
        # penalized LA > 16°. Net effect was a 0.06 muddled weight pulling
        # in two directions. The dedicated target-16° model is correct
        # (peak HR LA is ~16-18°) and remains at 0.04 weight.
        # pull_air_pct removed v42d (data 0% populated)
    ]

    def absolute_score(val, poor, elite):
        if pd.isna(val):
            return np.nan
        if poor == elite:
            return 50.0
        scaled = (val - poor) / (elite - poor) * 100
        return float(max(0, min(100, scaled)))

    # Compute weighted absolute score per row
    def compute_row(row):
        total_score = 0.0
        total_weight = 0.0
        for col, w, poor, elite in specs:
            if col not in df.columns:
                continue
            val = row.get(col)
            score = absolute_score(val, poor, elite)
            if pd.notna(score):
                total_score += score * w
                total_weight += w
        # Launch angle — fixed target (June 2026).
        # OLD: target=28°, weight=0.08. That was wrong because 28° is the peak
        # angle for an INDIVIDUAL HR-hit ball, but `la` is season-average
        # `avg_launch_angle` across ALL batted balls including grounders.
        # League avg `avg_launch_angle` is ~12°; elite power hitters sit at
        # 16-20° (Judge 16, Schwarber 19, Stanton 18). Targeting 28° meant the
        # formula penalized every hitter including the elites.
        # FIX: target=16°, wider tolerance (slope 3 instead of 5 → 33° away
        # = 0). Weight reduced from 0.08 to 0.04 since sweet_spot_pct now
        # covers part of this signal.
        if "la" in df.columns:
            la = row.get("la")
            if pd.notna(la):
                la_dist = abs(float(la) - 16.0)
                la_score = max(0, 100 - la_dist * 3)
                total_score += la_score * 0.04
                total_weight += 0.04
        if total_weight == 0:
            return np.nan
        max_weight = sum(w for _, w, _, _ in specs) + 0.04
        completeness = total_weight / max_weight
        # Hard threshold: if we have <60% of components, return NaN
        # (don't fake a score from sparse data)
        if completeness < 0.60:
            return np.nan
        return total_score / total_weight

    base_power = df.apply(compute_row, axis=1)

    # ---- Environmental multipliers ----
    # Use SOFTER combination: take average of factors rather than full multiply,
    # to prevent stacking pile-ups at the 99 cap. A 1.50 cap on full multiply
    # was still too generous when 4 factors all push the same direction.
    env_factors = [float(park_mult), float(weather_mult)]
    if pitcher_hr9 is not None and not pd.isna(pitcher_hr9) and pitcher_hr9 > 0:
        p_factor = pitcher_hr9 / 1.20
        env_factors.append(max(0.75, min(1.25, p_factor)))
    if pitcher_barrel_allowed is not None and not pd.isna(pitcher_barrel_allowed):
        b_factor = pitcher_barrel_allowed / 7.5
        env_factors.append(max(0.85, min(1.15, b_factor)))

    # Geometric mean is gentler than full multiplication for stacked factors
    import math
    log_sum = sum(math.log(f) for f in env_factors)
    env_mult = math.exp(log_sum / len(env_factors)) ** 1.5  # Power 1.5 keeps some kick
    env_mult = max(0.65, min(1.30, env_mult))

    df["power_score"] = (base_power * env_mult).clip(0, 99).round(1)
    df["env_mult"] = round(env_mult, 3)

    # MATCHUP OPPORTUNITY SCORE - separate from Power Score.
    # This is a "the situation favors a HR today regardless of who's batting"
    # score. Captures contact hitters in great matchups (bad pitcher, hot park,
    # wind out) who would normally fly under the radar.
    #
    # Higher weight on env, lower on raw hitter power. So Horwitz (low barrel)
    # facing a 6 ERA pitcher in Coors with wind blowing out gets a high
    # opportunity score even though his power_score is low.
    if base_power is not None:
        # Take hitter contribution AT 50% (everyone gets some baseline), env at full
        opportunity = (50 + (base_power - 50) * 0.40) * env_mult
        df["matchup_opp"] = opportunity.clip(0, 99).round(1)
    else:
        df["matchup_opp"] = np.nan

    return df


def add_lift_score(
    matchup_df: pd.DataFrame,
    pitcher_gb_pct: float | None = None,
) -> pd.DataFrame:
    """
    Compute Lift Score (0-100) — the contact-quality-meets-air-ball signal.

    Validated against 2026 slate data: composite has 0.672 correlation with
    HR Game%, higher than any single component including hard_hit (0.648).

    Components:
      - hard_hit_pct  (35%): how often the hitter makes 95+ mph contact
      - fb_pct        (25%): how often that contact goes in the air
      - sweet_spot_pct (25%): how often launch angle is in the 16-32° HR zone
      - pitcher FB tendency (15%): bonus for FB-prone pitchers (gb_pct < league)

    Why it matters: a 17% barrel hitter who pounds it into the ground (Cruz)
    still can't homer. Distinguishes Schwarber/Alvarez/Ohtani lift profile
    from groundball-heavy contact profiles even when raw contact quality
    looks similar.

    Where it differs from power_score: power_score weights barrel%, ISO, and
    pulled_brl_pct heavily — measuring whether the hitter has POWER. Lift Score
    measures whether the hitter ELEVATES that power. Two different signals;
    both useful.
    """
    if matchup_df is None or matchup_df.empty:
        return matchup_df
    df = matchup_df.copy()

    def _to_pctile(val, poor, elite):
        """Linear interpolation from poor→0 to elite→100, clamped."""
        if val is None or pd.isna(val):
            return None
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        pct = (v - poor) / (elite - poor) * 100
        return max(0, min(100, pct))

    # Component pctile thresholds based on MLB distributions:
    # hard_hit: typical range 25-55% (Judge ~58%, weak hitters ~25%)
    # fb_pct:   typical range 18-50% (heavy GB hitters 18%, Schwarber ~50%)
    # sweet_spot_pct: typical range 25-42% (poor 25%, elite 40%+)
    def _row_lift(row):
        hh = _to_pctile(row.get("hard_hit"), 25, 55)
        fb = _to_pctile(row.get("fb_pct"), 18, 50)
        ss = _to_pctile(row.get("sweet_spot_pct"), 25, 42)

        # Need at least 2 of 3 hitter components to score
        present = sum(1 for x in [hh, fb, ss] if x is not None)
        if present < 2:
            return None

        # Weight the present components proportionally
        weights = {"hh": 0.35, "fb": 0.25, "ss": 0.25}
        total_w = 0.0
        weighted_sum = 0.0
        if hh is not None:
            weighted_sum += hh * weights["hh"]
            total_w += weights["hh"]
        if fb is not None:
            weighted_sum += fb * weights["fb"]
            total_w += weights["fb"]
        if ss is not None:
            weighted_sum += ss * weights["ss"]
            total_w += weights["ss"]
        hitter_part = weighted_sum / total_w  # normalized to 0-100

        # Pitcher FB bonus (15% of total).
        # League avg GB% ≈ 43%. FB-prone pitchers (GB < 38%) give up more lift.
        # Formula: 50 + (43 - pitcher_gb) * 2, capped 0-100
        # Examples:
        #   pitcher_gb = 30 (extreme FB) → 50 + 26 = 76
        #   pitcher_gb = 43 (avg)        → 50 (neutral)
        #   pitcher_gb = 55 (Webb tier)  → 50 - 24 = 26
        if pitcher_gb_pct is not None and not pd.isna(pitcher_gb_pct):
            try:
                pgb = float(pitcher_gb_pct)
                pitcher_part = max(0, min(100, 50 + (43 - pgb) * 2))
                lift = hitter_part * 0.85 + pitcher_part * 0.15
            except (TypeError, ValueError):
                lift = hitter_part
        else:
            lift = hitter_part

        return round(lift, 1)

    df["lift_score"] = df.apply(_row_lift, axis=1)
    return df


def add_discipline_score(matchup_df: pd.DataFrame) -> pd.DataFrame:
    """
    v43.23 — Plate Discipline composite (0-100).

    User-validated audit finding: K% was barely used in scoring (only
    k_pct_inv at ~2% effective weight in matchup_opp). But high-K hitters
    homer LESS PER GAME even when their HR/PA looks elite, because they
    strike out 30%+ of plate appearances and never put the ball in play.
    A Joey Gallo-shape (16% barrel, 30% K) actually has fewer total HR
    chances than a Yordan-shape (14% barrel, 17% K) over a season.

    Components:
      - K% (65%): inverted — lower K = better. Elite ≤14%, poor ≥28%.
      - BB% (35%): higher BB = better. Elite ≥14%, poor ≤4%. BB% matters
        because pitchers around the strike zone with worse stuff = more
        elevatable pitches, AND walked guys still get later PAs.

    Auto-skips if both columns absent. Returns NaN for rows missing both
    so _score_from_weights renormalizes correctly.
    """
    if matchup_df is None or matchup_df.empty:
        return matchup_df
    df = matchup_df.copy()

    def _scale(val, poor, elite):
        if val is None or pd.isna(val):
            return None
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        return max(0, min(100, (v - poor) / (elite - poor) * 100))

    def _row(row):
        # v43.25 (reviewer-validated, CRITICAL bug fix): build_matchup_table
        # renames k_percent → k_pct and bb_percent → bb_pct at line ~337
        # BEFORE this function runs (it's called on the matchup_df output).
        # The v43.23 implementation read the pre-rename names, so every row
        # returned None and discipline_score was NaN for the entire export.
        # The "ps_discipline" column then never even got created because
        # the pick_score guard checks `.notna().any()`. Net effect: my v43.23
        # rebalance trimmed every other component to "make room" for
        # discipline, but discipline was inert — so the trim shipped without
        # the addition. Read both names defensively for safety.
        k_raw = row.get("k_pct")
        if k_raw is None or pd.isna(k_raw):
            k_raw = row.get("k_percent")
        k_score = _scale(k_raw, 28, 14) if (k_raw is not None and not pd.isna(k_raw)) else None
        # BB% — higher is better. Scale: poor=4%, elite=14%
        bb_raw = row.get("bb_pct")
        if bb_raw is None or pd.isna(bb_raw):
            bb_raw = row.get("bb_percent")
        bb_score = _scale(bb_raw, 4, 14) if (bb_raw is not None and not pd.isna(bb_raw)) else None

        if k_score is None and bb_score is None:
            return None
        # Weighted blend, renormalize if one component missing
        weights = {"k": 0.65, "bb": 0.35}
        total_w = 0.0
        weighted = 0.0
        if k_score is not None:
            weighted += k_score * weights["k"]
            total_w += weights["k"]
        if bb_score is not None:
            weighted += bb_score * weights["bb"]
            total_w += weights["bb"]
        return round(weighted / total_w, 1)

    df["discipline_score"] = df.apply(_row, axis=1)
    return df


# ============================================================================
# v43.36 — HR PROFILE CRITERIA (user-requested framework)
# ============================================================================
# Concrete thresholds from a real data source: a hitter is HR-likely if they
# clear these four bars (over the last two weeks / season-aware sample):
#
#   1. Pull % ≥ 40       — 66% of HRs are pulled; hitters who pull a lot
#                          have HR profiles. (Outliers: Ohtani, Wood etc.
#                          who hit power to all fields — handled separately
#                          via raw barrel/EV strength.)
#   2. Avg EV ≥ 90 mph   — harder ball, further travel; below 90 caps HR
#                          ceiling regardless of swing path.
#   3. Barrel % ≥ 10     — 80-86% of HRs ARE barreled (Statcast). A hitter
#                          who barrels ≥10% of contact has elite HR rate.
#   4. Ideal Attack Angle ≥ 60% — % of swings in the 5-20° launch zone.
#                          League avg ~50-55%; 60%+ means consistent
#                          power-friendly swing path (Statcast bat tracking,
#                          may be empty if Savant endpoint drift).
#
# This is a DISPLAY checklist — does NOT modify pick_score or grades.
# Hitters can be sorted/filtered by how many criteria they meet (0-4).
# ============================================================================

HR_CRITERIA_THRESHOLDS = {
    "pull_pct": 40.0,
    "avg_ev": 90.0,
    "barrel_pct": 10.0,
    "ideal_attack_angle_pct": 60.0,
    # v43.58: blast_pct fallback for HR Criteria #4. Savant's June 2026
    # bat-tracking endpoint doesn't expose ideal_attack_angle_pct directly
    # (verified: 18 cols returned, none are IAA). blast_pct measures swings
    # that are BOTH fast AND squared-up — the closest available signal
    # for "elite swing execution" that we already fetch. League ~3-4%,
    # elite ~7%+. Criterion #4 falls back to this when IAA is missing.
    "blast_pct": 5.0,
}


def add_hr_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """Mark each hitter against the 4-point HR-profile checklist.

    Adds the following columns:
      - hr_crit_pull        bool|None — meets pull_pct ≥ 40
      - hr_crit_ev          bool|None — meets avg_ev ≥ 90
      - hr_crit_barrel      bool|None — meets barrel_pct ≥ 10
      - hr_crit_iaa         bool|None — meets ideal_attack_angle_pct ≥ 60
                                         OR blast_pct ≥ 5 (v43.58 fallback)
      - hr_criteria_met     int       — count of criteria met (0-4)
      - hr_criteria_total   int       — count of criteria with data (0-4)
      - hr_criteria_label   str       — visual: "✓✓✓✗" or "3/4"
      - hr_profile_grade    str       — A+ (4/4) / A (3/4) / B (2/4) / C (1/4) / F (0)
                                        also "—" if no data on any criterion

    None values for individual criteria indicate "data not available"
    (vs False meaning "data available, threshold not met"). This matters
    because ideal_attack_angle data may not be present for every hitter
    if Savant's bat-tracking endpoint isn't returning that field.

    v43.58 (production data-driven): Savant's current bat-tracking endpoint
    doesn't expose IAA — criterion #4 was permanently '·' for every hitter.
    Now falls back to blast_pct (which Savant DOES return) when IAA is
    missing. blast_pct measures "fast AND squared-up" swings — same
    semantic family as IAA (good swing execution), different metric.
    """
    if df is None or df.empty:
        return df

    def _check(val, threshold):
        if val is None or pd.isna(val):
            return None
        try:
            return float(val) >= threshold
        except (TypeError, ValueError):
            return None

    # Pull % — prefer pull_pct (raw % of contact pulled). Fall back to
    # pull_air_pct as a proxy if pull_pct missing. They're different but
    # both correlate with HR-profile pull tendency.
    pull_col = "pull_pct" if "pull_pct" in df.columns else (
        "pull_air_pct" if "pull_air_pct" in df.columns else None
    )

    def _row_criteria(row):
        c1 = _check(row.get(pull_col) if pull_col else None,
                    HR_CRITERIA_THRESHOLDS["pull_pct"])
        c2 = _check(row.get("avg_ev"),
                    HR_CRITERIA_THRESHOLDS["avg_ev"])
        c3 = _check(row.get("barrel_pct"),
                    HR_CRITERIA_THRESHOLDS["barrel_pct"])
        # v43.58: criterion #4 — IAA primary, blast_pct fallback
        iaa_val = row.get("ideal_attack_angle_pct")
        if iaa_val is not None and not pd.isna(iaa_val):
            c4 = _check(iaa_val,
                        HR_CRITERIA_THRESHOLDS["ideal_attack_angle_pct"])
        else:
            # Fall back to blast_pct (closest available signal)
            c4 = _check(row.get("blast_pct"),
                        HR_CRITERIA_THRESHOLDS["blast_pct"])
        return c1, c2, c3, c4

    crit_lists = {"pull": [], "ev": [], "barrel": [], "iaa": []}
    for _, row in df.iterrows():
        c1, c2, c3, c4 = _row_criteria(row)
        crit_lists["pull"].append(c1)
        crit_lists["ev"].append(c2)
        crit_lists["barrel"].append(c3)
        crit_lists["iaa"].append(c4)

    df["hr_crit_pull"] = crit_lists["pull"]
    df["hr_crit_ev"] = crit_lists["ev"]
    df["hr_crit_barrel"] = crit_lists["barrel"]
    df["hr_crit_iaa"] = crit_lists["iaa"]

    def _summarize(p, e, b, i):
        crits = [p, e, b, i]
        met = sum(1 for c in crits if c is True)
        total = sum(1 for c in crits if c is not None)
        if total == 0:
            return 0, 0, "—", "—"
        # Visual label — ✓ met, ✗ not met, · no data
        symbols = []
        for c in crits:
            if c is True:
                symbols.append("✓")
            elif c is False:
                symbols.append("✗")
            else:
                symbols.append("·")
        label = "".join(symbols) + f" {met}/{total}"

        # Grade based on FRACTION met (not raw count) since IAA may be missing
        # for everyone — don't penalize for unavailable data
        if total == 0:
            grade = "—"
        else:
            frac = met / total
            if frac >= 0.99: grade = "A+"
            elif frac >= 0.75: grade = "A"
            elif frac >= 0.50: grade = "B"
            elif frac >= 0.25: grade = "C"
            else: grade = "F"
        return met, total, label, grade

    summaries = [_summarize(p, e, b, i) for p, e, b, i in zip(
        crit_lists["pull"], crit_lists["ev"],
        crit_lists["barrel"], crit_lists["iaa"]
    )]
    df["hr_criteria_met"] = [s[0] for s in summaries]
    df["hr_criteria_total"] = [s[1] for s in summaries]
    df["hr_criteria_label"] = [s[2] for s in summaries]
    df["hr_profile_grade"] = [s[3] for s in summaries]

    # v43.39 (reviewer-validated CRITICAL fix): missing return df. Without
    # it, add_hr_criteria(df) returned None, and call sites doing
    # `away_matchup = add_hr_criteria(away_matchup)` overwrote the
    # DataFrame with None. Next access (e.g. `matchup_df.empty`) crashed
    # with AttributeError on NoneType.
    return df


# ============================================================================
# v43.66 — RESEARCHER'S FRAMEWORK (Must-Have + Nuclear filters)
# ----------------------------------------------------------------------------
# A trusted external researcher's HR-prediction framework, added as a
# SECONDARY checkpoint alongside DingerMaven's existing HR Score / Grade /
# Pick Score. The researcher's lens is pure batted-ball PROFILE — no
# matchup, park, weather, or pitcher quality. Useful for comparing
# DingerMaven's matchup-aware ranking against a profile-only sanity check.
#
# Two tiers:
#   - MUST-HAVE (10 thresholds): "hitters making authoritative contact,
#     elevating the ball, pulling it in the air, producing the exact
#     batted-ball profile that leads to home runs"
#   - NUCLEAR (14 thresholds, stricter overlap): "only the best 3-8 HR
#     plays" per slate
#
# Both run alongside the existing 4-point hr_criteria — neither replaces
# it. The user sees BOTH lenses on each row.
#
# Data availability notes:
#   - avg_dist: added to Statcast fetcher in v43.66 (was not previously fetched)
#   - near_hr: not directly fetched. Approximated as `max(0, barrel_count - home_run)`
#     i.e. "barrels that didn't leave the yard." Computed in app.py once
#     barrel_count is available on the hitter frame.
# ============================================================================
HR_MUST_HAVE_THRESHOLDS = {
    "barrel_pct":    15.0,    # ≥15% barrels
    "pulled_brl_pct": 10.0,   # ≥10% pulled barrels
    "pull_air_pct":  35.0,    # ≥35% pull-in-air
    "hard_hit":      50.0,    # ≥50% hard hit
    "avg_dist":     315.0,    # ≥315 ft avg distance
    "avg_ev":        92.0,    # ≥92 mph EV
    "iso":            0.250,  # ≥.250 ISO
    "fb_pct":        35.0,    # ≥35% fly ball
    "pull_pct":      40.0,    # ≥40% pull
    "blast_pct":     12.0,    # ≥12% blast (0-100 scale after v43.65)
}

# Some criteria have an UPPER bound (GB% ≤35). Mark direction explicitly.
# Default "≥" for entries not in this set.
_NUCLEAR_DIRECTION_LE = {"gb_pct"}

HR_NUCLEAR_THRESHOLDS = {
    "home_run":       2.0,     # ≥2 HR (season count)
    "near_hr_est":    3.0,     # ≥3 near-HR (approximated)
    "barrel_pct":     18.0,
    "pulled_brl_pct": 15.0,
    "pull_air_pct":   40.0,
    "hard_hit":       55.0,
    "avg_dist":      330.0,
    "avg_ev":         94.0,
    "iso":             0.300,
    "slg":             0.600,
    "blast_pct":      15.0,
    "fb_pct":         40.0,
    "pull_pct":       45.0,
    "gb_pct":         35.0,    # ≤35 (see _NUCLEAR_DIRECTION_LE)
}


def _check_threshold(val, threshold, direction="ge"):
    """Evaluate a single threshold. Returns True / False / None (no data).

    direction: "ge" (≥, default) or "le" (≤)
    """
    if val is None or pd.isna(val):
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if direction == "le":
        return v <= threshold
    return v >= threshold


def _build_label(per_criterion_results, met, total):
    """Build the visual label '✓✓✗✓✗··✓✓ 6/10' from per-criterion bools."""
    symbols = []
    for c in per_criterion_results:
        if c is True:
            symbols.append("✓")
        elif c is False:
            symbols.append("✗")
        else:
            symbols.append("·")
    return "".join(symbols) + f" {met}/{total}"


def add_must_have_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """Mark each hitter against the researcher's 10-point MUST-HAVE checklist.

    Adds columns:
      - must_have_<metric>  bool|None for each of the 10 metrics
                            (must_have_barrel, must_have_pullbrl, etc.)
      - must_have_met       int 0-10  count of thresholds passed
      - must_have_total     int 0-10  count of evaluatable thresholds (with data)
      - must_have_label     str       e.g. "✓✓✗✓✓✓✗··✓ 6/8"
      - must_have_pass      bool      True if 10/10 (or all-available passed)

    None values indicate "data not available" — distinguished from False
    ("data available, threshold not met"). avg_dist will be None for
    hitters where the Statcast fetch didn't return it.

    Threshold source: external researcher's framework, June 2026.
    """
    if df is None or df.empty:
        return df

    # Map threshold-dict keys to the friendly per-criterion column names
    # the user sees on the dataframe / export.
    _MH_LABELS = {
        "barrel_pct":     "must_have_barrel",
        "pulled_brl_pct": "must_have_pullbrl",
        "pull_air_pct":   "must_have_pullair",
        "hard_hit":       "must_have_hh",
        "avg_dist":       "must_have_dist",
        "avg_ev":         "must_have_ev",
        "iso":            "must_have_iso",
        "fb_pct":         "must_have_fb",
        "pull_pct":       "must_have_pull",
        "blast_pct":      "must_have_blast",
    }

    # Per-criterion eval
    per_crit_cols = {label: [] for label in _MH_LABELS.values()}
    met_list, total_list, label_list, pass_list = [], [], [], []

    # Stable ordering for the label string
    _MH_ORDER = list(_MH_LABELS.keys())

    for _, row in df.iterrows():
        row_results = []
        for metric in _MH_ORDER:
            val = row.get(metric)
            threshold = HR_MUST_HAVE_THRESHOLDS[metric]
            result = _check_threshold(val, threshold, direction="ge")
            row_results.append(result)
            per_crit_cols[_MH_LABELS[metric]].append(result)

        met = sum(1 for r in row_results if r is True)
        total = sum(1 for r in row_results if r is not None)
        met_list.append(met)
        total_list.append(total)
        if total == 0:
            label_list.append("—")
            pass_list.append(False)
        else:
            label_list.append(_build_label(row_results, met, total))
            # "Pass" = all evaluatable criteria met (handles missing data
            # by not penalizing hitters with one missing metric)
            pass_list.append(met == total and total >= 8)

    for col, vals in per_crit_cols.items():
        df[col] = vals
    df["must_have_met"] = met_list
    df["must_have_total"] = total_list
    df["must_have_label"] = label_list
    df["must_have_pass"] = pass_list
    return df


def add_nuclear_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """Mark each hitter against the researcher's 14-point NUCLEAR checklist.

    Designed to surface "only the best 3-8 HR plays" — much stricter than
    Must-Have. Common to see 0-2 hitters pass all 14 on a typical slate.
    A 12+/14 partial pass is the practical "very close to nuclear" tier.

    Adds columns:
      - nuclear_<metric>    bool|None for each of the 14 metrics
      - nuclear_met         int 0-14  thresholds passed
      - nuclear_total       int 0-14  evaluatable thresholds
      - nuclear_label       str       e.g. "✓✓✓✓✗✓···✓✓✓✓✓ 11/12"
      - nuclear_grade       str       NUCLEAR (14/14) / STRONG (≥12) / NEAR (≥10) / —

    Threshold source: external researcher's framework, June 2026.
    """
    if df is None or df.empty:
        return df

    _NUC_LABELS = {
        "home_run":       "nuclear_hr",
        "near_hr_est":    "nuclear_nearhr",
        "barrel_pct":     "nuclear_barrel",
        "pulled_brl_pct": "nuclear_pullbrl",
        "pull_air_pct":   "nuclear_pullair",
        "hard_hit":       "nuclear_hh",
        "avg_dist":       "nuclear_dist",
        "avg_ev":         "nuclear_ev",
        "iso":            "nuclear_iso",
        "slg":            "nuclear_slg",
        "blast_pct":      "nuclear_blast",
        "fb_pct":         "nuclear_fb",
        "pull_pct":       "nuclear_pull",
        "gb_pct":         "nuclear_gb",
    }
    _NUC_ORDER = list(_NUC_LABELS.keys())

    per_crit_cols = {label: [] for label in _NUC_LABELS.values()}
    met_list, total_list, label_list, grade_list = [], [], [], []

    for _, row in df.iterrows():
        row_results = []
        for metric in _NUC_ORDER:
            val = row.get(metric)
            threshold = HR_NUCLEAR_THRESHOLDS[metric]
            direction = "le" if metric in _NUCLEAR_DIRECTION_LE else "ge"
            result = _check_threshold(val, threshold, direction=direction)
            row_results.append(result)
            per_crit_cols[_NUC_LABELS[metric]].append(result)

        met = sum(1 for r in row_results if r is True)
        total = sum(1 for r in row_results if r is not None)
        met_list.append(met)
        total_list.append(total)
        if total == 0:
            label_list.append("—")
            grade_list.append("—")
        else:
            label_list.append(_build_label(row_results, met, total))
            # Tiers based on raw met count (not fraction) since the researcher's
            # framework treats this as a counting metric, not a percentage
            if met == 14:
                grade_list.append("☢️ NUCLEAR")
            elif met >= 12:
                grade_list.append("💥 STRONG")
            elif met >= 10:
                grade_list.append("🎯 NEAR")
            else:
                grade_list.append("—")

    for col, vals in per_crit_cols.items():
        df[col] = vals
    df["nuclear_met"] = met_list
    df["nuclear_total"] = total_list
    df["nuclear_label"] = label_list
    df["nuclear_grade"] = grade_list
    return df


# ============================================================================
# v43.37 — COMPREHENSIVE HR GRADE (user-requested rebuild)
# ============================================================================
# Replaces the old single-input hr_grade (which just thresholded hr_game_pct
# with two cap layers) with a TIER-WEIGHTED COMPOSITE incorporating every
# signal we have at honest weights.
#
# A+ now means "everything aligns" — not just "high HR%."
# F means "structural failure across multiple dimensions."
#
# v43.64 (reviewer doc fix #16): now 7 tiers totaling 98% (renormalized to
# 100% by weight_total at runtime). Context tier (was 2%) was removed in
# v43.61 because it read ps_bonus_lineup / ps_bonus_recent_hr which aren't
# attached to combined_picks at the time this runs. Those signals still
# count via pick_score's ps_bonus_* path — they're not lost, just no longer
# double-counted into the comprehensive grade.
#
# 7 tiers (sum 0.98, renormalized):
#
#   30% — HR Probability (calibrated hr_game_pct: park × weather × pitcher
#         × platoon × ttop × pitch_match × defense × bullpen × day_night)
#   25% — Power Signals (barrel%, pulled_brl%, avg_ev, iso, max_hit_speed)
#   12% — Swing Mechanics (pull%, fb_pct, ideal_attack_angle_pct / blast_pct)
#   12% — Matchup Quality (matchup_opp, pitch_hr_score)
#   10% — Form / Recency (recent_hr_weighted_rate, hr_streak_games, hr_form)
#   6%  — Environment (env_boost — small weight to avoid double-count with
#         hr_game_pct which already includes env)
#   3%  — Plate Discipline (discipline_score)
#  (2% context tier removed v43.61 — see GRADE_TIER_WEIGHTS comment)
#
# Composite range: 0-100 (weighted average of percentile ranks across tiers).
# Grade thresholds:
#   A+ ≥ 80  (top ~5% of slate — everything aligns)
#   A  ≥ 70
#   B+ ≥ 60
#   B  ≥ 50
#   C+ ≥ 40
#   C  ≥ 30
#   D  ≥ 20
#   F  < 20
#
# Cap layers (applied AFTER threshold):
#   Mechanical fail (pull<35 AND ev<88):       max grade B
#   Hostile env (env_boost<0.85):              one tier down
#   Same-side platoon (LvL/RvR with no break): one tier down
#   Insufficient sample (PA<25):               returns "—"
# ============================================================================

# v43.61 (reviewer-validated, deferred from prior review #7): the "context"
# tier was inactive in production because compute_comprehensive_hr_composite
# runs on combined_picks BEFORE ps_bonus_lineup / ps_bonus_recent_hr are
# attached (they're only added to the filtered q frame later). So
# has_context was always False and the 2% tier weight contributed 0 to every
# hitter. Removing the tier entirely is cleaner than reordering the pipeline
# (which has cascading downstream effects). The remaining 7 tiers sum to
# 0.98; the existing weight_total normalization in compute_comprehensive_hr_composite
# scales output to 0-100 regardless. The "context" signals (confirmed lineup,
# recent HR) are STILL captured in pick_score via ps_bonus_* — they're not
# lost, just no longer double-counted into the comprehensive grade.
GRADE_TIER_WEIGHTS = {
    "hr_prob":     0.30,
    "power":       0.25,
    "mechanics":   0.12,
    "matchup":     0.12,
    "form":        0.10,
    "env":         0.06,
    "discipline":  0.03,
}

GRADE_TIER_COLUMNS = {
    "hr_prob":    ["hr_game_pct"],
    "power":      ["barrel_pct", "pulled_brl_pct", "avg_ev", "iso", "max_hit_speed"],
    "mechanics":  ["pull_pct", "fb_pct", "ideal_attack_angle_pct"],
    "matchup":    ["matchup_opp", "pitch_hr_score"],
    "form":       ["recent_hr_weighted_rate", "hr_streak_games", "hr_form"],
    "env":        ["env_boost"],
    "discipline": ["discipline_score"],
}


def compute_comprehensive_hr_composite(slate_df: pd.DataFrame) -> pd.Series:
    """Compute the 0-100 weighted composite score for HR grade.

    Slate-wide percentile ranks each input within each tier, averages
    inputs within tier, then weights tiers per GRADE_TIER_WEIGHTS.
    Renormalizes when a tier has no data (so a hitter missing IAA isn't
    penalized for the swing-mechanics tier — they're scored on pull% and
    fb_pct alone).

    Returns a Series indexed by slate_df's index. NaN where no tier had
    any data.
    """
    if slate_df is None or slate_df.empty:
        return pd.Series(dtype=float)

    composite = pd.Series(0.0, index=slate_df.index)
    weight_total = pd.Series(0.0, index=slate_df.index)

    for tier_name, weight in GRADE_TIER_WEIGHTS.items():
        cols = GRADE_TIER_COLUMNS.get(tier_name, [])
        available = [c for c in cols if c in slate_df.columns]
        if not available:
            continue

        # Compute percentile rank for each column in tier
        ranks = []
        for c in available:
            col = pd.to_numeric(slate_df[c], errors="coerce")
            if col.isna().all():
                continue
            r = col.rank(pct=True, na_option="keep") * 100
            ranks.append(r)

        if not ranks:
            continue

        # Average across columns in the tier (skipping NaN per-row)
        tier_df = pd.concat(ranks, axis=1)
        tier_score = tier_df.mean(axis=1, skipna=True)
        # Row mask: only contributes weight where tier produced a value
        mask = tier_score.notna()
        composite = composite.add(tier_score.fillna(0) * weight, fill_value=0)
        weight_total = weight_total + mask.astype(float) * weight

    # v43.61: Context tier (binary signals from ps_bonus_*) removed.
    # See GRADE_TIER_WEIGHTS docstring above. Bonuses still flow through
    # pick_score; just no longer double-counted into the comprehensive grade.

    # Final composite: normalize by total weight that contributed
    final = composite / weight_total.replace(0, np.nan)
    return final.round(1)


def rescale_composite_to_slate(composite_series: pd.Series) -> pd.Series:
    """v43.41: Rescale composite scores so the slate distribution spans 10-95.

    v43.42 (user feedback): top cap is 95, NOT 100. "HR Score 100" was
    being read as "100% chance of HR" / "guaranteed yard" — which is wrong
    (HR Score is a slate-relative composite, not a probability). Capping at
    95 makes the elite tier read as "near-elite, not certain."

    Why: the raw composite (weighted average of percentile ranks) regresses
    to the mean. Even elite hitters score ~70-75 because they're not 99th
    percentile in every tier. Result: users see mostly 45-60 and can't
    differentiate. Rescaling makes the SPREAD interpretable — best play of
    the slate reads ~95, worst ~10, median ~50.

    Method: linear scale so slate's 5th pct → 10, 95th pct → 95.
    Anchoring on the tails (not min/max) avoids one extreme outlier
    dominating the rescale. Final clip is 0-95 (not 0-100).

    Returns:
      Series of rescaled scores (0-95), same index as input.
      Returns input unchanged if too few values to compute percentiles.
    """
    if composite_series is None or composite_series.empty:
        return composite_series
    valid = composite_series.dropna()
    if len(valid) < 10:
        return composite_series

    p5 = valid.quantile(0.05)
    p95 = valid.quantile(0.95)
    if p95 - p5 < 1e-6:
        return pd.Series(50.0, index=composite_series.index, dtype=float)

    rescaled = 10.0 + (composite_series - p5) * (95.0 - 10.0) / (p95 - p5)
    # v43.42: cap at 95, not 100 — avoid "100 = guaranteed" misread.
    # v43.65 (reviewer-validated fix #17): the previous hard clip at 95
    # created an N-way tie at the top — production export showed 30 hitters
    # all reading hr_score == 95.0, which kills sortability/discriminability
    # in EXACTLY the region that matters most (the top picks of the slate).
    #
    # Fix: above 95, apply a soft asymptotic compression that preserves
    # ordering. Below 95, behavior is unchanged. Above 95, the displayed
    # value follows a tanh-like curve that maps the linear-rescale tail
    # into [95, 99.5] — never reaches 100 (preserving the "not a probability"
    # signal) but spreads the top hitters by ~0.1-0.3 each so they sort.
    #
    # Math: for v > 95, displayed = 95 + 4.5 * tanh((v - 95) / 8).
    #   v = 95   → 95.0
    #   v = 100  → ~97.3
    #   v = 105  → ~98.4 (matches old "hard clipped" hitter)
    #   v = 120  → ~99.4 (extreme outliers approach 99.5 asymptote)
    # The /8 scale was chosen so a typical above-p95 spread (5-15 points
    # of raw composite) produces a 1-4 point displayed spread — enough to
    # rank, not enough to look like a different scale than below 95.
    above_cap = rescaled > 95.0
    if above_cap.any():
        excess = rescaled[above_cap] - 95.0
        rescaled.loc[above_cap] = 95.0 + 4.5 * np.tanh(excess / 8.0)
    rescaled = rescaled.clip(0, 99.5)
    return rescaled.round(1)


def hr_score_signal(hr_score):
    """v43.41: Color emoji from HR Score band.
    v43.42: bands adjusted for 0-95 cap.
    v43.65: range is now 0-99.5 due to soft tail compression above 95.
            Band thresholds unchanged — same green/yellow/orange/red boundaries.
    Bands aligned to slate-rescaled distribution:
      🟢 70+ (top ~25% of slate — strong play)
      🟡 50-69 (above median)
      🟠 25-49 (below median)
      🔴 <25 (bottom quartile)
      ⚪ no data
    """
    if hr_score is None or pd.isna(hr_score):
        return "⚪"
    if hr_score >= 70: return "🟢"
    if hr_score >= 50: return "🟡"
    if hr_score >= 25: return "🟠"
    return "🔴"


def comprehensive_hr_grade(composite, pull_pct=None, avg_ev=None,
                             env_mult=None, same_side_platoon=False,
                             sample_size=None, pa_threshold=25,
                             barrel_pct=None, iso=None):
    """Convert composite (0-100) to letter grade with cap layers.

    Composite thresholds:
      A+ ≥ 80  (top ~5% of slate — everything aligns)
      A  ≥ 70
      B+ ≥ 60
      B  ≥ 50
      C+ ≥ 40
      C  ≥ 30
      D  ≥ 20
      F  < 20

    Cap layers (v43.54 — reviewer-validated non-stacking):
      Mechanical fail (pull_pct<35 AND avg_ev<88):  capped at B
      Hostile env (env_mult<0.85):                  one tier down
      Same-side platoon:                            one tier down
      Absolute power floor for A+/A:                requires real elite power
      Sample<25 PA:                                 returns "—"

    v43.54 #8 (reviewer fix): caps no longer STACK. Previously, all three caps
    were applied sequentially — a hitter at A+ with all three flags could
    drop A+ → B → C+ → C+ (effectively 3 tiers). Now each cap independently
    proposes a result and we take the WORST single cap (max 2 tiers down).
    The original docs said "one tier down" — triple-stacking was a bug, not
    intent.

    v43.54 #9 (reviewer fix): the v43.51 switch from grade_composite to
    hr_score (slate-rescaled) made A+/A purely slate-relative — every slate
    produces ~5% A+ regardless of absolute quality. Added an ABSOLUTE POWER
    FLOOR: A+/A also requires real elite contact (barrel_pct ≥ 10% OR
    avg_ev ≥ 90 OR iso ≥ .200). Without that, demote one tier. Mech-fail
    is still its own separate (harsher) check. This keeps "A+ means
    something" while preserving the slate-relative ranking.
    """
    if composite is None or pd.isna(composite):
        return "—"
    if sample_size is not None and not pd.isna(sample_size):
        try:
            if float(sample_size) < pa_threshold:
                return "—"
        except (TypeError, ValueError):
            pass

    if composite >= 80: raw = "A+"
    elif composite >= 70: raw = "A"
    elif composite >= 60: raw = "B+"
    elif composite >= 50: raw = "B"
    elif composite >= 40: raw = "C+"
    elif composite >= 30: raw = "C"
    elif composite >= 20: raw = "D"
    else: raw = "F"

    # ------------------------------------------------------------------
    # v43.54: Compute each cap independently — non-stacking
    # ------------------------------------------------------------------
    # Each cap returns a CANDIDATE grade. We then take the WORST candidate
    # (the most restrictive) as the final grade — preserving "one tier
    # down per flag" semantics WITHOUT compounding across flags.
    # ------------------------------------------------------------------
    candidates = [raw]  # base case: no cap

    # 1. Mechanical fail cap: BOTH pull<35 AND ev<88 → caps to B
    try:
        pp = float(pull_pct) if pull_pct is not None and not pd.isna(pull_pct) else None
        ev = float(avg_ev) if avg_ev is not None and not pd.isna(avg_ev) else None
        if pp is not None and ev is not None and pp < 35 and ev < 88:
            _CAP_MECH = {"A+": "B", "A": "B", "B+": "B"}
            candidates.append(_CAP_MECH.get(raw, raw))
    except (TypeError, ValueError):
        pass

    # 2. Hostile env cap: env_mult < 0.85 → one tier down
    try:
        if env_mult is not None and not pd.isna(env_mult):
            env_f = float(env_mult)
            if env_f < 0.85:
                _CAP_ENV = {"A+": "A", "A": "B+", "B+": "B", "B": "C+"}
                candidates.append(_CAP_ENV.get(raw, raw))
    except (TypeError, ValueError):
        pass

    # 3. Same-side platoon cap: one tier down
    if same_side_platoon:
        _CAP_PLAT = {"A+": "A", "A": "B+", "B+": "B"}
        candidates.append(_CAP_PLAT.get(raw, raw))

    # 4. v43.54 #9: Absolute power floor for A+/A
    #    The base grade is slate-relative now (via hr_score rescale). A+/A
    #    grades should ALSO require real power markers. If neither barrel%
    #    nor avg_ev nor ISO crosses the absolute threshold, demote one tier.
    if raw in ("A+", "A"):
        has_elite_power = False
        # v43.64 (reviewer fix #15): track whether ANY input was parseable.
        # Old behavior: a single parse failure set has_elite_power = True
        # for the whole block, defeating the floor's intent. New behavior:
        # only fail open if ALL THREE inputs are unparseable (genuinely no
        # data); if ANY input parses, evaluate normally. This preserves
        # "be defensive when truly no data" while not letting one bad
        # value bypass the floor for a hitter who has real power signals
        # on the other two inputs.
        any_parseable = False
        try:
            if barrel_pct is not None and not pd.isna(barrel_pct):
                try:
                    bp = float(barrel_pct)
                    any_parseable = True
                    if bp >= 10.0:
                        has_elite_power = True
                except (TypeError, ValueError):
                    pass
            if not has_elite_power and avg_ev is not None and not pd.isna(avg_ev):
                try:
                    ev = float(avg_ev)
                    any_parseable = True
                    if ev >= 90.0:
                        has_elite_power = True
                except (TypeError, ValueError):
                    pass
            if not has_elite_power and iso is not None and not pd.isna(iso):
                try:
                    iso_f = float(iso)
                    any_parseable = True
                    if iso_f >= 0.200:
                        has_elite_power = True
                except (TypeError, ValueError):
                    pass
        except Exception:
            any_parseable = False
        # If genuinely no data on any of barrel/EV/ISO, default has_elite_power
        # to True (don't penalize a hitter we can't evaluate). Otherwise
        # apply the floor as designed.
        if not any_parseable:
            has_elite_power = True
        if not has_elite_power:
            _CAP_POWER = {"A+": "A", "A": "B+"}
            candidates.append(_CAP_POWER.get(raw, raw))

    # Take the WORST candidate (most restrictive grade)
    _GRADE_ORDER = ["A+", "A", "B+", "B", "C+", "C", "D", "F"]
    _grade_rank = {g: i for i, g in enumerate(_GRADE_ORDER)}
    worst = max(candidates, key=lambda g: _grade_rank.get(g, 999))
    return worst


def _season_phase(slate_date=None) -> str:
    """v43.18 (reviewer-validated): shared season-phase helper. Both
    pa_threshold_for_date (app.py) and _season_thresholds (models.py) used
    to independently parse `month` from the date and switch on it. If MLB
    ever shifts to a March opener, both would need updating. This helper
    is the single source of truth — both consumers now derive their
    specific thresholds from one bucket.

    v43.19 (reviewer-validated crash fix): added 'september' and 'october'
    as distinct phases. Previously this collapsed everything ≥September
    into 'late', which left _season_thresholds with unreachable Sept/Oct
    threshold dicts AND a NameError crash for any postseason date
    (the elif branches referenced an undefined `month` variable). Crash
    was dormant only because it's June.

    Returns one of: 'early', 'may', 'june', 'july', 'august',
                    'september', 'october', 'offseason'
    """
    import datetime
    if slate_date is None:
        slate_date = datetime.date.today()
    try:
        month = slate_date.month
    except AttributeError:
        try:
            slate_date = datetime.datetime.strptime(
                str(slate_date)[:10], "%Y-%m-%d"
            ).date()
            month = slate_date.month
        except Exception:
            month = 6  # default mid-season
    if month <= 4:
        return "early"
    if month == 5:
        return "may"
    if month == 6:
        return "june"
    if month == 7:
        return "july"
    if month == 8:
        return "august"
    if month == 9:
        return "september"
    if month == 10:
        return "october"
    return "offseason"


def _season_thresholds(slate_date=None):
    """
    Return season-aware thresholds. Earlier in season = lower bars.
    Returns dict with:
      full_ip / min_ip — pitcher IP reliability bars
      full_gs / min_gs — pitcher games started bars
      split_thin / split_small — split-sample PA thresholds for the
        ⚠️ thin / 📊 small split flags. Lower in April, higher by August
        when most players have accumulated meaningful platoon samples.

    v43.18: uses shared _season_phase to coordinate with app.pa_threshold_for_date
    v43.19 (reviewer-validated crash fix): table-based lookup replaces
    the if/elif chain that had two zombie branches referencing a
    nonexistent `month` variable left over from the v43.18 refactor.
    Would crash for any September/October date.
    """
    phase = _season_phase(slate_date)
    TABLE = {
        "early":     {"full_ip": 15,  "min_ip": 5,  "full_gs": 3,  "min_gs": 1,
                      "split_thin": 20, "split_small": 40},
        "may":       {"full_ip": 30,  "min_ip": 10, "full_gs": 5,  "min_gs": 2,
                      "split_thin": 30, "split_small": 55},
        "june":      {"full_ip": 50,  "min_ip": 15, "full_gs": 8,  "min_gs": 3,
                      "split_thin": 35, "split_small": 65},
        "july":      {"full_ip": 70,  "min_ip": 20, "full_gs": 10, "min_gs": 4,
                      "split_thin": 40, "split_small": 70},
        "august":    {"full_ip": 90,  "min_ip": 25, "full_gs": 14, "min_gs": 5,
                      "split_thin": 45, "split_small": 80},
        "september": {"full_ip": 110, "min_ip": 30, "full_gs": 18, "min_gs": 6,
                      "split_thin": 50, "split_small": 90},
        "october":   {"full_ip": 130, "min_ip": 30, "full_gs": 20, "min_gs": 6,
                      "split_thin": 50, "split_small": 90},
        "offseason": {"full_ip": 30,  "min_ip": 10, "full_gs": 5,  "min_gs": 2,
                      "split_thin": 30, "split_small": 60},
    }
    return TABLE.get(phase, TABLE["june"])


def build_pitcher_slate(
    slate: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    pitcher_recent: dict | None = None,
    slate_date=None,
    team_hit_map: dict | None = None,
) -> pd.DataFrame:
    """One row per starting pitcher with composite scores + recent form."""
    thresholds = _season_thresholds(slate_date)
    pitchers = []
    for _, g in slate.iterrows():
        for side in ("away", "home"):
            pid = g[f"{side}_pitcher_id"]
            if pid is None or pd.isna(pid):
                continue
            row = pitcher_stats[pitcher_stats["player_id"] == pid]
            opp_abbr = g[f"{'home' if side == 'away' else 'away'}_team_abbr"]
            base = {
                "pitcher_id": pid,
                "pitcher_name": g[f"{side}_pitcher"],
                "team": g[f"{side}_team_abbr"],
                "opp": opp_abbr,
                "home_away": "@" if side == "away" else "vs",
                "game_pk": g["gamePk"],
            }
            # Opposing team aggregates - lookup by abbr first, then by team_id
            opp_team_id = g.get(f"{'home' if side == 'away' else 'away'}_team_id")
            opp_agg = (team_hit_map or {}).get(opp_abbr) or {}
            if not opp_agg and opp_team_id is not None and not pd.isna(opp_team_id):
                try:
                    opp_agg = (team_hit_map or {}).get(int(opp_team_id)) or {}
                except (ValueError, TypeError):
                    opp_agg = {}
            base["opp_k_pct"] = opp_agg.get("k_pct")
            base["opp_hr_per_pa"] = opp_agg.get("hr_per_pa")
            base["opp_iso"] = opp_agg.get("iso")

            # Park HR factor at this venue (for hr_suppress adjustment)
            try:
                from park_factors import get_park
                venue = g.get("venue", "")
                park_info = get_park(venue) if venue else {}
                hr_factor = park_info.get("hr_factor", 100)
                base["park_hr_factor"] = round(float(hr_factor) / 100, 3)
            except Exception:
                base["park_hr_factor"] = 1.0
            if len(row) > 0:
                r = row.iloc[0].to_dict()
                base.update({
                    "throws": r.get("p_throws"),
                    "pa": r.get("pa"),
                    "xwoba_allowed": r.get("xwoba"),
                    "k_pct": r.get("k_percent"),
                    "bb_pct": r.get("bb_percent"),
                    "barrel_allowed": r.get("barrel_batted_rate"),
                    "hard_hit_allowed": r.get("hard_hit_percent"),
                    "whiff_pct": r.get("whiff_percent"),
                    "csw_pct": r.get("csw_percent"),
                    "zone_pct": r.get("zone_percent"),
                    "fb_allowed": r.get("flyballs_percent"),
                    "gb_allowed": r.get("groundballs_percent"),
                    "hr_allowed": r.get("home_run"),
                    "era": r.get("era"),
                    "whip": r.get("whip"),
                    "hr9": r.get("hr9"),
                    "k9": r.get("k9"),
                    "bb9": r.get("bb9"),
                    "ip": r.get("ip"),
                    "games_started": r.get("games_started"),
                    "games_played": r.get("games_played"),
                    "is_rookie": bool(r.get("is_rookie", False)),
                    "debut_year": r.get("debut_year"),
                })
                # Copy ALL vs_lhb_* / vs_rhb_* handedness split columns + day/night
                # splits + IL status. These get dropped if not explicitly copied
                # (build_pitcher_slate only copies known fields by default).
                for k, v in r.items():
                    if isinstance(k, str) and (
                        k.startswith("vs_lhb_") or k.startswith("vs_rhb_")
                        or k.startswith("vs_day_") or k.startswith("vs_night_")
                        or k in ("on_il", "days_since_return", "il_count_this_season")
                    ):
                        base[k] = v
            if pitcher_recent and pid in pitcher_recent:
                base.update(pitcher_recent[pid])
            pitchers.append(base)

    df = pd.DataFrame(pitchers)
    if df.empty:
        return df

    # ------------------------------------------------------------------
    # UNIVERSAL ERA + HR/9 CAP (v39e)
    # ------------------------------------------------------------------
    # Previously the cap at 12.0 only applied to era_savant (the path
    # where ERA is derived from earned_runs / ip_savant). When ERA flowed
    # in directly from MLB Stats API (the normal path), extreme outliers
    # like Kade Morris (ERA 20.25 in 4 IP) flowed through uncapped.
    #
    # That's OK for percentile-rank usage (pct_rank bounds outliers
    # automatically), but it's NOT OK for:
    #   - Threshold logic (era >= 5.00 → EXPLOIT+) — fine
    #   - ERA-driven projections (era × multiplier) — uncapped ERA can compound
    #   - Display sanity (ERA of 20+ looks broken to users)
    #
    # Cap at 12.0 matches the existing era_savant cap. Same logic for HR/9:
    # extreme small-sample values get capped at 6.0.
    if "era" in df.columns:
        df["era"] = pd.to_numeric(df["era"], errors="coerce").clip(upper=12.0)
    if "hr9" in df.columns:
        df["hr9"] = pd.to_numeric(df["hr9"], errors="coerce").clip(upper=6.0)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # IP fallback - if MLB Stats API failed, estimate from Statcast PA
    # Statcast 'pa' is plate appearances faced by the pitcher
    # ~4.3 PA per inning is league average
    # ------------------------------------------------------------------
    have_real_ip = "ip" in df.columns and df["ip"].notna().any()
    if not have_real_ip and "pa" in df.columns and df["pa"].notna().any():
        df["ip"] = (df["pa"] / 4.3).round(1)
        df["ip_estimated"] = True
    else:
        df["ip_estimated"] = False
        if "ip" not in df.columns:
            df["ip"] = np.nan

    # ------------------------------------------------------------------
    # K/9 fallback - derive from k_pct if missing
    # Math: K/9 = (K%/100) × (PA per IP) × 9 = K_pct × 4.3/100 × 9
    # ------------------------------------------------------------------
    have_real_k9 = "k9" in df.columns and df["k9"].notna().any()
    if not have_real_k9 and "k_pct" in df.columns and df["k_pct"].notna().any():
        df["k9"] = (df["k_pct"] * 4.3 * 9 / 100).round(2)
        df["k9_estimated"] = True
    else:
        df["k9_estimated"] = False
        if "k9" not in df.columns:
            df["k9"] = np.nan

    # ------------------------------------------------------------------
    # IP PER OUTING - better signal than IP/GS since it uses games_played
    # ------------------------------------------------------------------
    def _ip_per_outing(row):
        ip = row.get("ip")
        gp = row.get("games_played")
        gs = row.get("games_started")
        if ip is None or pd.isna(ip):
            return None
        denom = gp if (gp is not None and not pd.isna(gp) and gp > 0) else gs
        if denom is None or pd.isna(denom) or denom == 0:
            return None
        return round(float(ip) / float(denom), 2)

    df["ip_per_outing"] = df.apply(_ip_per_outing, axis=1)

    # ------------------------------------------------------------------
    # SAMPLE NOISE FLAG - catches stat inconsistencies that suggest the
    # pitcher's season numbers are misleading due to small sample.
    # ------------------------------------------------------------------
    def _sample_noise(row):
        ip = row.get("ip")
        era = row.get("era")
        whip = row.get("whip")
        ip_n = float(ip) if (ip is not None and not pd.isna(ip)) else None
        era_n = float(era) if (era is not None and not pd.isna(era)) else None
        whip_n = float(whip) if (whip is not None and not pd.isna(whip)) else None

        # Pattern 1: classic mismatch — high ERA but low WHIP (HR-heavy bad luck)
        if (ip_n is not None and ip_n < 20
                and era_n is not None and era_n > 5.0
                and whip_n is not None and whip_n < 1.1):
            return True

        # Pattern 2: small-sample extreme on EITHER end. Tyler Phillips (May
        # 2026 case study) had 33 IP and a 1.07 ERA which was flagged as
        # EXPLOIT-able by the grade system but the underlying sample was too
        # small to be predictive. The earlier filter required ERA>5.0 so
        # ultra-low-ERA small-sample pitchers slipped through.
        # Now flag: under 40 IP AND ERA outside (2.00, 6.00). The 40 IP
        # threshold corresponds to ~7 starts which is around when ERA starts
        # to stabilize for starters; under that, extreme ERAs are usually
        # sample noise rather than real signal.
        if (ip_n is not None and ip_n < 40
                and era_n is not None and (era_n < 2.00 or era_n > 6.00)):
            return True

        # Pattern 3: barrel rate suspiciously zero in tiny sample
        barrel = row.get("barrel_allowed")
        if (ip_n is not None and ip_n < 10
                and barrel is not None and not pd.isna(barrel)
                and float(barrel) == 0):
            return True

        return False

    df["sample_noise"] = df.apply(_sample_noise, axis=1)

    # ------------------------------------------------------------------
    # RELIABILITY FACTOR - season-aware thresholds + graceful no-data fallback
    # ------------------------------------------------------------------
    full_ip = thresholds["full_ip"]
    min_ip = thresholds["min_ip"]
    full_gs = thresholds["full_gs"]
    min_gs = thresholds["min_gs"]

    def _reliability(row):
        ip = row.get("ip")
        gs = row.get("games_started")
        gp = row.get("games_played")
        # No IP data at all = use Statcast K% as a weak signal
        if ip is None or pd.isna(ip):
            # Don't auto-penalize if we just can't see the data
            return 0.7
        # Explicit zero starts = reliever
        if gs is not None and not pd.isna(gs) and gs == 0:
            return 0.4
        # Bulk reliever: games_played >> games_started AND low IP per OUTING
        # (a real starter with 8 GS and 12 GP because of an IL stint shouldn't
        # be downgraded — but a swing-man with 6 GS and 17 GP and 1.6 IP/outing
        # is clearly relief-heavy. Use ip/gp not ip/gs, since IP includes
        # relief outings too.)
        bulk_relief = False
        if (gp is not None and gs is not None
                and not pd.isna(gp) and not pd.isna(gs)
                and gs > 0 and gp > gs * 1.5):
            ip_per_outing = ip / gp if gp > 0 else 0
            # True bulk reliever role: <3.0 IP per outing
            if ip_per_outing < 3.0:
                bulk_relief = True
        base = 0.65 if bulk_relief else 1.0
        ip_factor = min(1.0, ip / full_ip)
        if gs is not None and not pd.isna(gs) and gs > 0:
            start_factor = min(1.0, gs / full_gs)
        else:
            start_factor = ip_factor
        return round(max(0.3, base * (ip_factor * 0.5 + start_factor * 0.5)), 2)

    df["reliability"] = df.apply(_reliability, axis=1)

    # ------------------------------------------------------------------
    # ROLE FLAG - uses module-level _classify_role (shared with recompute)
    # ------------------------------------------------------------------
    df["role"] = df.apply(
        lambda r: _classify_role(r, min_ip=min_ip, min_gs=min_gs, full_ip=full_ip),
        axis=1,
    )

    # ------------------------------------------------------------------
    # COMPOSITE SCORE COMPONENTS - real data only, no fake-50 defaults
    # ------------------------------------------------------------------
    df["k_score"] = _safe_pct_rank(df["k_pct"]) if "k_pct" in df.columns else np.nan
    df["whiff_score"] = _safe_pct_rank(df["whiff_pct"]) if "whiff_pct" in df.columns else np.nan
    # Suppression: lower xwOBA = better
    if "xwoba_allowed" in df.columns and df["xwoba_allowed"].notna().any():
        df["suppress_score"] = 100 - _safe_pct_rank(df["xwoba_allowed"])
    else:
        df["suppress_score"] = np.nan
    # ERA component: lower ERA = better
    if "era" in df.columns and df["era"].notna().any():
        df["era_score"] = 100 - _safe_pct_rank(df["era"])
    else:
        df["era_score"] = np.nan

    # Recent form blend: shift weight onto current performance
    # BUT cap recent_k9 contribution - relievers can K 2 in 1 IP (k9=18) which
    # isn't sustainable over a multi-inning outing. Cap recent at 1.4x season K9.
    if "recent_k9" in df.columns and df["recent_k9"].notna().any() and "k9" in df.columns:
        season_k9 = df["k9"].fillna(0)
        # Cap recent K9 at 1.4 × season K9 (prevents tiny-sample explosions)
        recent_capped = df["recent_k9"].fillna(season_k9).clip(upper=season_k9 * 1.4)
        # If season K9 itself is 0/missing (no real data), don't blend
        blended_k9 = recent_capped * 0.35 + season_k9 * 0.65
        df["blended_k9"] = blended_k9.round(2)
        df["k_score_blended"] = _safe_pct_rank(blended_k9)
    else:
        df["k_score_blended"] = df["k_score"]

    # ------------------------------------------------------------------
    # TEST SCORE — composite K-suppression score across the slate.
    # Honest weights (no double-counting): K-blended captures the K rate signal
    # via blend of K% and K/9, so we don't need the standalone base k_score too.
    # Weights:
    #   K-blended       35%  (blend of K% and K/9, the primary K signal)
    #   Whiff%          20%  (called/swinging strike rate, K-leading indicator)
    #   xwOBA suppress  25%  (overall stuff quality vs all hitters)
    #   ERA             20%  (results-based check on the above)
    # Total = 100%. Effective K-focused weight is 35% + ~10% (whiff overlap) = ~45%
    # which honestly reflects that this is a strikeout-leaning score.
    def _composite_test(row):
        parts = []
        max_weight = 0.35 + 0.20 + 0.25 + 0.20  # = 1.0
        if pd.notna(row.get("k_score_blended")):
            parts.append((row["k_score_blended"], 0.35))
        if pd.notna(row.get("whiff_score")):
            parts.append((row["whiff_score"], 0.20))
        if pd.notna(row.get("suppress_score")):
            parts.append((row["suppress_score"], 0.25))
        if pd.notna(row.get("era_score")):
            parts.append((row["era_score"], 0.20))
        if not parts:
            return np.nan
        total_w = sum(w for _, w in parts)
        raw = sum(v * w for v, w in parts) / total_w
        # Data completeness penalty: only count fully when we have all 4 components.
        # If only 2 of 4 weights are present, score gets penalized so sparse data
        # doesn't masquerade as a high-confidence score.
        completeness = total_w / max_weight
        if completeness < 0.5:
            penalty = 0.80
        elif completeness < 0.7:
            penalty = 0.88
        elif completeness < 0.85:
            penalty = 0.95
        else:
            penalty = 1.0
        return raw * penalty

    raw_test = df.apply(_composite_test, axis=1)
    # Apply reliability multiplier - relievers get scaled down
    # Cap at 95 so the score visually never claims "100% sure"
    df["test_score"] = (raw_test * df["reliability"]).clip(upper=95).round(2)

    # kHR: K-focused composite (was misleadingly named - this is a K rating)
    def _composite_khr(row):
        parts = []
        max_weight = 0.50 + 0.20 + 0.30  # = 1.0
        if pd.notna(row.get("k_score_blended")):
            parts.append((row["k_score_blended"], 0.50))
        if pd.notna(row.get("k_score")):
            parts.append((row["k_score"], 0.20))
        if pd.notna(row.get("whiff_score")):
            parts.append((row["whiff_score"], 0.30))
        if not parts:
            return np.nan
        total_w = sum(w for _, w in parts)
        raw = sum(v * w for v, w in parts) / total_w
        completeness = total_w / max_weight
        if completeness < 0.5:
            penalty = 0.80
        elif completeness < 0.7:
            penalty = 0.88
        else:
            penalty = 1.0
        return raw * penalty

    raw_khr = df.apply(_composite_khr, axis=1)
    df["kHR"] = (raw_khr * df["reliability"]).clip(upper=95).round(2)

    # ------------------------------------------------------------------
    # HR_SUPPRESS - separate score for how well pitcher prevents HRs
    # Uses: low barrel% allowed, low xwOBA allowed, low HR/9, AND
    # batted-ball mix (low FB% allowed / high GB% = HR suppression)
    # ------------------------------------------------------------------
    def _hr_suppress(row):
        parts = []
        # Barrel% allowed: lower = better (invert percentile)
        if pd.notna(row.get("barrel_allowed")):
            barrel_pct = row.get("_barrel_pct_rank", np.nan)
            if pd.notna(barrel_pct):
                parts.append((100 - barrel_pct, 0.32))
        # xwOBA allowed: lower = better
        if pd.notna(row.get("suppress_score")):
            parts.append((row["suppress_score"], 0.28))
        # HR/9: lower = better
        if pd.notna(row.get("_hr9_pct_rank")):
            parts.append((100 - row["_hr9_pct_rank"], 0.20))
        # FB allowed: lower = better (fewer flyballs → fewer HRs)
        if pd.notna(row.get("_fb_allowed_rank")):
            parts.append((100 - row["_fb_allowed_rank"], 0.12))
        # GB allowed: higher = better (groundball pitchers suppress HRs)
        if pd.notna(row.get("_gb_allowed_rank")):
            parts.append((row["_gb_allowed_rank"], 0.08))
        if not parts:
            return np.nan
        total_w = sum(w for _, w in parts)
        raw = sum(v * w for v, w in parts) / total_w
        # Completeness penalty if not all components present (max total w = 1.0)
        completeness = total_w / 1.0
        penalty = 0.85 if completeness < 0.6 else 0.95 if completeness < 0.85 else 1.0
        return raw * penalty

    # Precompute pct ranks for barrel, hr9, fb_allowed, gb_allowed
    if "barrel_allowed" in df.columns:
        df["_barrel_pct_rank"] = _safe_pct_rank(df["barrel_allowed"])
    if "hr9" in df.columns and df["hr9"].notna().any():
        df["_hr9_pct_rank"] = _safe_pct_rank(df["hr9"])
    if "fb_allowed" in df.columns and df["fb_allowed"].notna().any():
        df["_fb_allowed_rank"] = _safe_pct_rank(df["fb_allowed"])
    if "gb_allowed" in df.columns and df["gb_allowed"].notna().any():
        df["_gb_allowed_rank"] = _safe_pct_rank(df["gb_allowed"])

    raw_hr_supp = df.apply(_hr_suppress, axis=1)

    # Opponent HR factor: if facing a high-HR team, suppress score should drop
    # (pitcher is in a tougher spot). League-avg HR/PA ~ 2.8%.
    # Each 1pp above league avg = -5% to suppress score; capped at ±15%.
    def _opp_hr_mult(row):
        opp_hr = row.get("opp_hr_per_pa")
        if opp_hr is None or pd.isna(opp_hr):
            return 1.0
        mult = 1 - (float(opp_hr) - 2.8) * 0.05
        return max(0.85, min(1.15, mult))
    df["opp_hr_mult"] = df.apply(_opp_hr_mult, axis=1)

    # Park HR factor: pitcher in Coors has tougher suppression environment
    # park_hr_factor of 1.21 (Coors) → suppress drops by ~15%
    # park_hr_factor of 0.88 (Oracle) → suppress boosts by ~10%
    def _park_suppress_mult(row):
        pf = row.get("park_hr_factor")
        if pf is None or pd.isna(pf):
            return 1.0
        # Inverse relationship: higher park HR factor = harder to suppress
        # Use 2 - pf (so 1.20 park → 0.80 mult, 0.90 park → 1.10 mult)
        mult = 2 - float(pf)
        return max(0.80, min(1.15, mult))
    df["park_suppress_mult"] = df.apply(_park_suppress_mult, axis=1)

    df["hr_suppress"] = (
        raw_hr_supp * df["reliability"] * df["opp_hr_mult"] * df["park_suppress_mult"]
    ).clip(upper=95).round(2)

    # Clean up internal helper columns
    df = df.drop(columns=[c for c in [
        "_barrel_pct_rank", "_hr9_pct_rank",
        "_fb_allowed_rank", "_gb_allowed_rank"
    ] if c in df.columns])

    # ------------------------------------------------------------------
    # PROJ K - reliability-adjusted, uses blended K/9 when available
    # NOW ALSO accounts for opposing team's actual K%:
    #   - If opp K% > league avg (22%), more strikeouts expected → boost
    #   - If opp K% < league avg, fewer strikeouts → reduce
    # ------------------------------------------------------------------
    if "k9" in df.columns:
        effective_k9 = df.get("blended_k9", df["k9"]).fillna(df["k9"])
        def _expected_ip(row):
            ip = row.get("ip")
            gs = row.get("games_started")
            gp = row.get("games_played")
            role = row.get("role", "")

            # Role-based defaults take PRIORITY over IP-based ones.
            # Jax case: 5 GS / 16 GP / 27.2 IP → if we just check ip/gs we miss
            # that he's a swing-man pitching ~1.7 IP per outing.
            # NOTE: "🔄 BULK" and "🚨 RELIEVER" labels are deprecated — app.py
            # normalizes them to "🚨 OPENER" before _expected_ip runs, so we
            # only need to check for OPENER here. Kept RELIEVER as a defensive
            # fallback in case role classification ran without normalization.
            if "🚨 OPENER" in str(role) or "🚨 RELIEVER" in str(role):
                # Opener typically goes 1-2 IP then a bulk reliever takes over
                base_ip = 1.5
            elif "🌱 NEW STARTER" in str(role):
                # Rookie/recent recall — short leash, expect 4-5 IP
                base_ip = 4.5
            elif "🏥 RETURNING" in str(role):
                # Returning from IL — abundance of caution, 4-5 IP
                base_ip = 4.5
            elif "🔄 SWING" in str(role):
                # For swing-man, base on actual ip_per_outing capped at 3.5
                if (ip is not None and not pd.isna(ip) and gp is not None
                        and not pd.isna(gp) and gp > 0):
                    ip_per_outing = ip / gp
                    base_ip = min(3.5, max(1.5, ip_per_outing))
                else:
                    base_ip = 2.5
            elif gs is not None and not pd.isna(gs) and gs == 0:
                base_ip = 1.5
            elif ip is None or pd.isna(ip) or ip < 10:
                base_ip = 2.0
            elif gs is not None and not pd.isna(gs) and gs < 5:
                base_ip = 3.5
            elif ip is not None and not pd.isna(ip) and ip < 25:
                base_ip = 4.0
            else:
                base_ip = 5.5
            # IL fatigue adjustment - fresh from IL = fewer innings expected.
            days_since = row.get("days_since_return")
            if (days_since is not None and not pd.isna(days_since)
                    and 0 <= days_since <= 7):
                if days_since <= 3:
                    base_ip = min(base_ip, 3.5)
                else:
                    base_ip = base_ip * 0.85
            # HIGH WORKLOAD ADJUSTMENT: pitchers who averaged 105+ pitches in
            # their recent starts are more likely to be pulled early in the
            # next start (manager wants to manage cumulative workload). Real
            # research: starters with 110+ avg pitch counts in prior 3 starts
            # go ~0.5 IP shorter on average in the next outing.
            avg_pitches = row.get("avg_recent_pitches")
            if (avg_pitches is not None and not pd.isna(avg_pitches)
                    and avg_pitches > 105):
                # 105-110 → mild dampener; 110+ → stronger
                if avg_pitches > 110:
                    base_ip = base_ip * 0.88
                else:
                    base_ip = base_ip * 0.94
            return base_ip
        df["expected_ip"] = df.apply(_expected_ip, axis=1)

        # Opponent K% adjustment (capped at ±15% to prevent over-correction)
        def _opp_k_mult(row):
            opp_k = row.get("opp_k_pct")
            if opp_k is None or pd.isna(opp_k):
                return 1.0
            # League avg K% ~ 22.5%. Every 1pp deviation → 4.5% K change.
            mult = 1 + (float(opp_k) - 22.5) * 0.045
            return max(0.85, min(1.15, mult))
        df["opp_k_mult"] = df.apply(_opp_k_mult, axis=1)

        df["proj_k"] = (effective_k9 * df["expected_ip"] / 9 * df["opp_k_mult"]).round(1)

    # Form arrow: recent ERA vs season ERA (negate so "improvement" = lower ERA = up arrow)
    if "recent_era" in df.columns and "era" in df.columns:
        def _safe_form(r):
            recent = r.get("recent_era")
            season = r.get("era")
            if recent is None or pd.isna(recent) or season is None or pd.isna(season):
                return "→"
            try:
                return _form_arrow(-float(recent), -float(season))
            except (TypeError, ValueError):
                return "→"
        df["form_arrow"] = df.apply(_safe_form, axis=1)
    else:
        df["form_arrow"] = "→"

    return df.sort_values("test_score", ascending=False, na_position="last").reset_index(drop=True)


def recompute_pitcher_roles(p_slate: pd.DataFrame, slate_date=None) -> pd.DataFrame:
    """
    Re-run role classification on an existing p_slate (after adding primary_position).

    Why this exists:
    build_pitcher_slate() computes roles INSIDE its body. By the time app.py
    fetches MLB primaryPosition and adds it to p_slate, the role column has
    already been set without that signal. This function lets us redo role
    classification with the now-complete data.

    Usage:
        p_slate = build_pitcher_slate(...)
        p_slate["primary_position"] = fetch_positions(...)
        p_slate = recompute_pitcher_roles(p_slate, slate_date=today)
    """
    if p_slate is None or p_slate.empty:
        return p_slate
    thresholds = _season_thresholds(slate_date)
    min_ip = thresholds["min_ip"]
    min_gs = thresholds["min_gs"]
    full_ip = thresholds["full_ip"]

    p_slate = p_slate.copy()
    p_slate["role"] = p_slate.apply(
        lambda r: _classify_role(r, min_ip=min_ip, min_gs=min_gs, full_ip=full_ip),
        axis=1,
    )
    return p_slate
