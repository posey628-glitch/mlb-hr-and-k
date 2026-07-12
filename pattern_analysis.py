"""
Pattern analysis from accumulated snapshot/outcome data.

The goal: ingest every prior snapshot+outcome pairing and surface PATTERNS
that suggest where the model is over- or under-predicting. This is the
foundation for model improvement — humans (or future automation) read the
patterns and adjust thresholds/weights accordingly.

What this provides:
  - cohort_analysis(): how did hitters meeting threshold X actually perform?
  - feature_correlation(): which projection features actually correlate
    with outcomes?
  - calibration_drift(): how does Brier score change over accumulated slates?
  - researcher_framework_backtest(): did Must-Have / Nuclear passers actually
    homer more than the rest?
  - prop_accuracy(): for each prop (HR, hits, 2+ bases, K), how accurate
    were our projections?

What this DOES NOT do:
  - Mutate model weights automatically (deliberate — would break easily)
  - Train ML models (out of scope)
  - Replace human judgment (it surfaces info to inform judgment)

Sample size warnings:
  Reliable signal needs ~20-30+ slates of data. Below that, patterns are
  noise. Functions return both the metric AND the sample size so callers
  can warn users when N is too small to trust.

v43.70 — first ship, foundational infrastructure. Can be extended as
the dataset accumulates.
"""
from __future__ import annotations
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Snapshot+outcome ingestion
# ============================================================================
def merge_snapshots_with_outcomes(snapshots: dict) -> pd.DataFrame:
    """Flatten all snapshots+outcomes into one long DataFrame.

    Each row = one (snapshot_date, player_id) pair with BOTH the projections
    and the actual outcome columns. Used by every downstream analysis.

    Args:
        snapshots: dict of {snapshot_key: payload} from gist storage

    Returns:
        DataFrame with columns: snapshot_date, player_id, player_name +
        all projection columns + all outcome columns (hr, hits, total_bases,
        homered, got_hit, etc.). Empty DataFrame if no snapshots have outcomes.
    """
    if not snapshots:
        return pd.DataFrame()

    rows = []
    # v43.95: iterate in sorted key order so the dedupe below keeps the
    # LATEST hourly snapshot of each date (freshest lineups/projections).
    for snap_key, payload in sorted(snapshots.items()):
        if not isinstance(payload, dict):
            continue
        # Only use snapshots that HAVE outcome data attached
        hitter_outcomes = payload.get("hitter_outcomes") or {}
        if not hitter_outcomes:
            continue
        snapshot_date = str(snap_key).split("T")[0]

        # The hitter projections live in payload["hitters_compact"] (v43.88
        # column-oriented), payload["picks"], or payload["hitters"] (legacy
        # records) depending on snapshot version. Decode locally — this
        # module deliberately doesn't import backtest.
        picks = []
        _compact = payload.get("hitters_compact")
        if isinstance(_compact, dict) and _compact.get("columns"):
            _cols = _compact["columns"]
            picks = [dict(zip(_cols, r)) for r in (_compact.get("rows") or [])]
        if not picks:
            picks = payload.get("picks") or payload.get("hitters") or []
        if not picks:
            continue

        for pick in picks:
            if not isinstance(pick, dict):
                continue
            pid = pick.get("player_id")
            if not pid:
                continue
            # v44.59 (code review #7): coerce the pid ONCE so a float-
            # deserialized id ("592450.0") or a non-numeric id can't crash the
            # merge (int(pid) used to raise) or silently miss (str(pid) of a
            # float never matched string keys). Try int form, then string form.
            outcome = {}
            try:
                _pid_i = int(float(pid))
                outcome = (hitter_outcomes.get(str(_pid_i))
                           or hitter_outcomes.get(_pid_i) or {})
            except (TypeError, ValueError):
                outcome = hitter_outcomes.get(str(pid)) or {}
            if not outcome:
                continue  # snapshot has outcomes but not for this player

            # Build one combined row
            row = {"snapshot_date": snapshot_date, "player_id": pid,
                   "_snap_key": str(snap_key)}
            # Projection columns (whitelist what's useful for analysis)
            # v43.88 (review finding): this whitelist predated v43.83's
            # snapshot expansion, so only 12 of the 24 HR_CANDIDATE_FEATURES
            # made it into the merged frame — daily correlations silently
            # tracked half the intended predictor set. Now includes the
            # full candidate set plus pick_score decomposition context.
            for col in [
                "player_name", "team", "hr_score", "hr_game_pct", "pick_score",
                "grade", "barrel_pct", "iso", "avg_ev", "blast_pct", "blast_pct_real",
                "pull_pct", "pull_air_pct", "pulled_brl_pct", "hard_hit",
                "fb_pct", "gb_pct", "ld_pct",
                "must_have_met", "must_have_total", "must_have_pass",
                "nuclear_met", "nuclear_total", "nuclear_grade",
                "hit_game_pct", "tb_game_pct", "expected_total_bases", "tb_pa",
                "hr_pa_pct", "power_score",
                "xwoba", "xslg", "slg", "obp", "ops",
                "k_pct", "bb_pct", "whiff_pct",
                "discipline_score", "lift_score", "matchup_opp",
                "recent_hr", "recent_hr_weighted_rate",
                "pitch_hr_score", "pitch_match_score",
                "env_boost", "opp_pitcher_xwoba", "dinger_score", "power_composite", "barrel_matchup_score", "two_way_matchup_score",
                "is_moonshot_target", "is_laser_target",
                "sleeper_score", "lineup_pos", "is_roster_fill",
            ]:
                if col in pick:
                    row[col] = pick[col]
            # Outcome columns
            for col in [
                "hr", "h", "ab", "k", "bb", "rbi", "doubles", "triples",
                "total_bases", "runs",
                "homered", "got_hit", "got_2plus_bases", "got_3plus_bases",
            ]:
                if col in outcome:
                    row[col] = outcome[col]
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # v44.63 (user: imputed bat-tracking shouldn't skew data/correlations).
    # For GRADING, use the real (un-imputed) blast_pct. blast_pct_real is NaN
    # for hitters without bat-tracking, so those rows drop out of the blast_pct
    # correlation instead of contributing a fabricated median that dilutes the
    # signal and can misrank an elite hitter. If blast_pct_real isn't present
    # (older snapshots pre-v44.63), keep the existing (imputed) blast_pct.
    if "blast_pct_real" in df.columns:
        df["blast_pct"] = df["blast_pct_real"]
    # v43.95 (diagnostic-exposed): the same player-game appeared once PER
    # HOURLY SNAPSHOT of a date (6 snapshots of 2 slates → 1116 "rows" from
    # ~470 unique player-games). That inflated n ~2.4×, overweighted
    # whichever dates had more snapshots in every correlation, and
    # overstated significance. One outcome per (date, player) is the
    # statistical truth — keep the latest snapshot's projection of each.
    if not df.empty and "player_id" in df.columns:
        df = (
            df.sort_values("_snap_key")
              .drop_duplicates(subset=["snapshot_date", "player_id"], keep="last")
              .drop(columns=["_snap_key"])
              .reset_index(drop=True)
        )
    return df


# ============================================================================
# Cohort analysis — predicted threshold vs actual outcome
# ============================================================================
def cohort_analysis(
    merged_df: pd.DataFrame,
    threshold_col: str,
    threshold_value: float,
    outcome_col: str = "homered",
    direction: str = "ge",
) -> dict:
    """Compute: among hitters passing a threshold, what % actually delivered?

    Example: cohort_analysis(df, "barrel_pct", 15.0, "homered", "ge")
      → {"in_cohort": 247, "out_cohort": 1842, "in_rate": 0.084,
         "out_rate": 0.047, "lift": 1.79, "n_total": 2089}

    Args:
        merged_df: output of merge_snapshots_with_outcomes
        threshold_col: column to filter on (e.g., "barrel_pct")
        threshold_value: threshold value
        outcome_col: binary outcome column (homered, got_hit, etc.)
        direction: "ge" (>=) or "le" (<=)

    Returns:
        dict with cohort sizes, hit rates, and lift ratio
    """
    if merged_df.empty or threshold_col not in merged_df.columns:
        return {"error": "no data", "n_total": 0}
    if outcome_col not in merged_df.columns:
        return {"error": f"no {outcome_col} column", "n_total": 0}

    df = merged_df.dropna(subset=[threshold_col, outcome_col])
    n_total = len(df)
    if n_total == 0:
        return {"error": "all NaN", "n_total": 0}

    if direction == "ge":
        in_cohort = df[df[threshold_col] >= threshold_value]
    else:
        in_cohort = df[df[threshold_col] <= threshold_value]
    out_cohort = df[~df.index.isin(in_cohort.index)]

    in_rate = in_cohort[outcome_col].mean() if len(in_cohort) else 0.0
    out_rate = out_cohort[outcome_col].mean() if len(out_cohort) else 0.0
    lift = in_rate / out_rate if out_rate > 0 else float("inf")

    return {
        "threshold_col": threshold_col,
        "threshold_value": threshold_value,
        "direction": direction,
        "outcome_col": outcome_col,
        "in_cohort": len(in_cohort),
        "out_cohort": len(out_cohort),
        "n_total": n_total,
        "in_rate": float(in_rate),
        "out_rate": float(out_rate),
        "lift": float(lift) if lift != float("inf") else None,
        # Sample size signal
        "reliable": len(in_cohort) >= 50 and len(out_cohort) >= 50,
    }


# ============================================================================
# Feature-outcome correlation
# ============================================================================
def feature_correlation(
    merged_df: pd.DataFrame,
    feature_cols: list,
    outcome_col: str = "homered",
) -> pd.DataFrame:
    """Per feature: correlation with the binary outcome.

    Higher (positive) correlation = feature predicts the outcome well.
    Near zero = no relationship in accumulated data.
    Negative = feature predicts the OPPOSITE.

    Args:
        merged_df: output of merge_snapshots_with_outcomes
        feature_cols: list of column names to analyze
        outcome_col: binary outcome

    Returns:
        DataFrame with columns: feature, corr, n_pairs, abs_corr (sorted desc)
    """
    if merged_df.empty or outcome_col not in merged_df.columns:
        return pd.DataFrame()

    rows = []
    for feat in feature_cols:
        if feat not in merged_df.columns:
            continue
        valid = merged_df[[feat, outcome_col]].dropna()
        if len(valid) < 20:
            continue
        # Point-biserial correlation (Pearson with binary outcome)
        try:
            corr = float(valid[feat].astype(float).corr(
                valid[outcome_col].astype(float)
            ))
        except Exception:
            continue
        rows.append({
            "feature": feat,
            "corr": corr,
            "n_pairs": len(valid),
            "abs_corr": abs(corr),
        })
    out = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)
    return out


# ============================================================================
# Calibration drift over time
# ============================================================================
def calibration_drift(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Per snapshot date: Brier score, predicted vs actual HR rate.

    Brier score is the gold-standard calibration metric — lower is better.
    Drift over time tells us if the model's calibration is degrading or
    improving as the season progresses.

    Args:
        merged_df: output of merge_snapshots_with_outcomes

    Returns:
        DataFrame with one row per snapshot_date showing:
          - n_hitters, predicted_hr_rate, actual_hr_rate, brier_score
    """
    if merged_df.empty or "homered" not in merged_df.columns:
        return pd.DataFrame()
    if "hr_game_pct" not in merged_df.columns:
        return pd.DataFrame()

    rows = []
    for date, group in merged_df.groupby("snapshot_date"):
        valid = group.dropna(subset=["hr_game_pct", "homered"])
        if len(valid) < 10:
            continue
        # hr_game_pct is 0-100 %; convert to probability
        pred = valid["hr_game_pct"].astype(float) / 100.0
        actual = valid["homered"].astype(float)
        brier = float(((pred - actual) ** 2).mean())
        rows.append({
            "snapshot_date": date,
            "n_hitters": len(valid),
            "n_homered": int(actual.sum()),
            "predicted_rate": float(pred.mean()),
            "actual_rate": float(actual.mean()),
            "brier": brier,
        })
    return pd.DataFrame(rows).sort_values("snapshot_date")


# ============================================================================
# Researcher's framework backtest — do Must-Have/Nuclear passers homer more?
# ============================================================================
def researcher_framework_backtest(merged_df: pd.DataFrame) -> dict:
    """Compare HR rate of Must-Have passers, Nuclear-tier hitters, and the rest.

    The critical question this answers: does the researcher's framework
    actually identify hitters who homer at a higher rate? If yes, the
    framework is a real signal worth keeping. If no, it's a vanity metric
    and we should consider dropping the thresholds or recalibrating.

    Args:
        merged_df: output of merge_snapshots_with_outcomes

    Returns:
        dict with HR rates per cohort + sample sizes
    """
    if merged_df.empty or "homered" not in merged_df.columns:
        return {"error": "no data"}

    result = {"slate_average_hr_rate": float(merged_df["homered"].mean()),
              "n_total": len(merged_df)}

    # Must-Have cohort. v44.48: the binary must_have_pass (pass ALL 9 strict
    # criteria) fires for ~0 hitters, so its "lift" was always n=0 — useless.
    # Instead, cohort on the COUNT: hitters clearing 7+ of the 9 power criteria
    # ("Must-Have-strong") vs the rest. That's a meaningful, populated tier and
    # the count already correlates well as a feature (Section G). Falls back to
    # the binary pass if the count column isn't present.
    if "must_have_met" in merged_df.columns:
        mh = merged_df.dropna(subset=["must_have_met", "homered"])
        _mh_count = pd.to_numeric(mh["must_have_met"], errors="coerce").fillna(0)
        # v44.61: a FIXED 7/9 cutoff produced n=3 (0.7%) — too small to measure.
        # Pick the threshold that puts roughly the top 8-12% of players in-tier,
        # so the cohort is always populated enough to have a meaningful lift. We
        # walk down from the max count until the in-tier group is at least ~8%
        # of players (floor of 20 for stability), but never below 5 criteria
        # (below that it's no longer a "strong profile"). This is data-driven:
        # if a slate is weak, the bar auto-lowers; if stacked, it stays high.
        _target_n = max(20, int(0.08 * len(mh)))
        _mh_thresh = 7
        for _t in (7, 6, 5):
            if (_mh_count >= _t).sum() >= _target_n:
                _mh_thresh = _t
                break
            _mh_thresh = _t  # keep walking down to the floor (5)
        _mh_mask = _mh_count >= _mh_thresh
        mh_pass = mh[_mh_mask]
        mh_fail = mh[~_mh_mask]
        result["must_have"] = {
            "n_pass": len(mh_pass),
            "n_fail": len(mh_fail),
            "pass_hr_rate": float(mh_pass["homered"].mean()) if len(mh_pass) else None,
            "fail_hr_rate": float(mh_fail["homered"].mean()) if len(mh_fail) else None,
            "lift": (
                float(mh_pass["homered"].mean() / mh_fail["homered"].mean())
                if len(mh_pass) and len(mh_fail) and mh_fail["homered"].mean() > 0
                else None
            ),
            "reliable": len(mh_pass) >= 30 and len(mh_fail) >= 30,
            "threshold_note": f"{_mh_thresh}+ of 9 power criteria (adaptive)",
        }
    elif "must_have_pass" in merged_df.columns:
        mh = merged_df.dropna(subset=["must_have_pass", "homered"])
        _mh_mask = mh["must_have_pass"].fillna(False).astype(bool)
        mh_pass = mh[_mh_mask]
        mh_fail = mh[~_mh_mask]
        result["must_have"] = {
            "n_pass": len(mh_pass),
            "n_fail": len(mh_fail),
            "pass_hr_rate": float(mh_pass["homered"].mean()) if len(mh_pass) else None,
            "fail_hr_rate": float(mh_fail["homered"].mean()) if len(mh_fail) else None,
            "lift": (
                float(mh_pass["homered"].mean() / mh_fail["homered"].mean())
                if len(mh_pass) and len(mh_fail) and mh_fail["homered"].mean() > 0
                else None
            ),
            "reliable": len(mh_pass) >= 30 and len(mh_fail) >= 30,
        }

    # Nuclear cohort. v44.48: same issue as Must-Have — the elite grades
    # (NUCLEAR/STRONG/NEAR need ~10-12 of 14 boxes) fire for ~1 hitter, so the
    # lift was n=1 and uninformative. Cohort on the nuclear_met COUNT instead:
    # hitters clearing 9+ of the criteria vs the rest. nuclear_met is the
    # strongest RISING feature in Section G (+0.065 trend), so this surfaces a
    # real, populated signal. Falls back to the grade tiers if count absent.
    if "nuclear_met" in merged_df.columns:
        nuc = merged_df.dropna(subset=["nuclear_met", "homered"])
        _nuc_count = pd.to_numeric(nuc["nuclear_met"], errors="coerce").fillna(0)
        # v44.61: fixed 9/12 gave n=0. Adaptive cutoff targeting the top ~8%
        # (floor 15), walking down from 9 to a floor of 6 so the cohort is
        # always measurable. nuclear_met is the strongest RISING feature in
        # Section G, so a populated version of this is worth surfacing.
        _nuc_target = max(15, int(0.08 * len(nuc)))
        _nuc_thresh = 9
        for _t in (9, 8, 7, 6):
            if (_nuc_count >= _t).sum() >= _nuc_target:
                _nuc_thresh = _t
                break
            _nuc_thresh = _t
        _nuc_mask = _nuc_count >= _nuc_thresh
        nuc_in = nuc[_nuc_mask]
        nuc_out = nuc[~_nuc_mask]
        result["nuclear"] = {
            "n_in": len(nuc_in),
            "n_out": len(nuc_out),
            "in_hr_rate": float(nuc_in["homered"].mean()) if len(nuc_in) else None,
            "out_hr_rate": float(nuc_out["homered"].mean()) if len(nuc_out) else None,
            "lift": (
                float(nuc_in["homered"].mean() / nuc_out["homered"].mean())
                if len(nuc_in) and len(nuc_out) and nuc_out["homered"].mean() > 0
                else None
            ),
            "reliable": len(nuc_in) >= 30 and len(nuc_out) >= 30,
            "threshold_note": f"{_nuc_thresh}+ nuclear criteria (adaptive)",
        }
    elif "nuclear_grade" in merged_df.columns:
        nuc = merged_df.dropna(subset=["nuclear_grade", "homered"])
        elite_grades = ["☢️ NUCLEAR", "💥 STRONG", "🎯 NEAR"]
        nuc_in = nuc[nuc["nuclear_grade"].isin(elite_grades)]
        nuc_out = nuc[~nuc["nuclear_grade"].isin(elite_grades)]
        result["nuclear"] = {
            "n_in": len(nuc_in),
            "n_out": len(nuc_out),
            "in_hr_rate": float(nuc_in["homered"].mean()) if len(nuc_in) else None,
            "out_hr_rate": float(nuc_out["homered"].mean()) if len(nuc_out) else None,
            "lift": (
                float(nuc_in["homered"].mean() / nuc_out["homered"].mean())
                if len(nuc_in) and len(nuc_out) and nuc_out["homered"].mean() > 0
                else None
            ),
            "reliable": len(nuc_in) >= 30 and len(nuc_out) >= 30,
        }

    # v44.51 (user-requested): grade the ELITE CONVERGENCE cohort — hitters
    # who clear high bars across ALL custom metrics at once. This tests whether
    # full multi-model agreement actually produces a higher HR rate (the whole
    # premise of the section). Thresholds mirror the display section.
    _ec_cols = ["hr_score", "dinger_score", "power_composite",
                "barrel_matchup_score", "two_way_matchup_score"]
    if all(c in merged_df.columns for c in ["hr_score", "homered"]):
        ec = merged_df.dropna(subset=["hr_score", "homered"]).copy()
        _ec_mask = pd.to_numeric(ec["hr_score"], errors="coerce") >= 80
        for _c, _thr in [("dinger_score", 80), ("power_composite", 78),
                         ("barrel_matchup_score", 80), ("two_way_matchup_score", 75)]:
            if _c in ec.columns:
                _ec_mask &= pd.to_numeric(ec[_c], errors="coerce").fillna(0) >= _thr
        ec_in = ec[_ec_mask.fillna(False)]
        ec_out = ec[~_ec_mask.fillna(False)]
        result["elite_convergence"] = {
            "n_in": len(ec_in),
            "n_out": len(ec_out),
            "in_hr_rate": float(ec_in["homered"].mean()) if len(ec_in) else None,
            "out_hr_rate": float(ec_out["homered"].mean()) if len(ec_out) else None,
            "lift": (
                float(ec_in["homered"].mean() / ec_out["homered"].mean())
                if len(ec_in) and len(ec_out) and ec_out["homered"].mean() > 0
                else None
            ),
            "reliable": len(ec_in) >= 20,  # elite cohort is small by design
            "threshold_note": "HR≥80, Dinger≥80, Combo≥78, BrlMatch≥80, TwoWay≥75",
        }

    # v44.67 (user: pattern-analyze day/night + home/away). Segment the slate
    # HR rate by game context to reveal whether these situational factors carry
    # real signal worth adding to the model. Reported as split rates + lift.
    def _segment(col, in_label, out_label):
        if col not in merged_df.columns:
            return None
        seg = merged_df.dropna(subset=[col, "homered"]).copy()
        if seg.empty:
            return None
        # coerce to boolean-ish
        _truthy = seg[col].apply(
            lambda v: True if v in (True, "True", "true", 1, "1") else
            (False if v in (False, "False", "false", 0, "0") else None)
        )
        seg = seg[_truthy.notna()]
        if seg.empty:
            return None
        grp_in = seg[_truthy[_truthy.notna()] == True]
        grp_out = seg[_truthy[_truthy.notna()] == False]
        if not len(grp_in) or not len(grp_out):
            return None
        r_in = float(grp_in["homered"].mean())
        r_out = float(grp_out["homered"].mean())
        return {
            "in_label": in_label, "out_label": out_label,
            "in_rate": r_in, "out_rate": r_out,
            "n_in": len(grp_in), "n_out": len(grp_out),
            "lift": (r_in / r_out) if r_out > 0 else None,
            "reliable": len(grp_in) >= 30 and len(grp_out) >= 30,
        }

    _home_seg = _segment("is_home", "Home", "Away")
    if _home_seg:
        result["context_home_away"] = _home_seg
    _day_seg = _segment("is_day_game", "Day game", "Night game")
    if _day_seg:
        result["context_day_night"] = _day_seg

    # v44.74 (user: do individual matchups — pitcher, park, weather — actually
    # move HR outcomes?). Measure matchup quality two ways:
    #   1. Overall: do hitters in a favorable environment (env_boost high) or
    #      facing an exploitable pitcher homer more than those who aren't?
    #   2. INTERACTION: does a good matchup lift HR rate MORE for high-profile
    #      hitters than for low-profile ones? This is the real question — a
    #      great park should help a masher more than a slap hitter.
    def _matchup_seg(col, threshold, in_label, out_label, ge=True):
        if col not in merged_df.columns:
            return None
        seg = merged_df.dropna(subset=[col, "homered"]).copy()
        _val = pd.to_numeric(seg[col], errors="coerce")
        seg = seg[_val.notna()]
        if seg.empty:
            return None
        _val = pd.to_numeric(seg[col], errors="coerce")
        _mask = (_val >= threshold) if ge else (_val <= threshold)
        grp_in, grp_out = seg[_mask], seg[~_mask]
        if not len(grp_in) or not len(grp_out):
            return None
        r_in, r_out = float(grp_in["homered"].mean()), float(grp_out["homered"].mean())
        return {
            "in_label": in_label, "out_label": out_label,
            "in_rate": r_in, "out_rate": r_out,
            "n_in": len(grp_in), "n_out": len(grp_out),
            "lift": (r_in / r_out) if r_out > 0 else None,
            "reliable": len(grp_in) >= 30 and len(grp_out) >= 30,
        }

    # 1. Favorable environment (park × weather × pull-wind ≥ 1.05)
    _env_seg = _matchup_seg("env_boost", 1.05, "Favorable env (≥1.05)", "Neutral/poor env")
    if _env_seg:
        result["matchup_env"] = _env_seg

    # 2. Exploitable pitcher (grade EXPLOIT / EXPLOIT+)
    if "opp_pitcher_grade" in merged_df.columns:
        seg = merged_df.dropna(subset=["opp_pitcher_grade", "homered"]).copy()
        _exploit = seg["opp_pitcher_grade"].astype(str).str.contains("EXPLOIT", na=False)
        grp_in, grp_out = seg[_exploit], seg[~_exploit]
        if len(grp_in) and len(grp_out):
            r_in, r_out = float(grp_in["homered"].mean()), float(grp_out["homered"].mean())
            result["matchup_pitcher"] = {
                "in_label": "Vs exploitable pitcher", "out_label": "Vs tough/neutral pitcher",
                "in_rate": r_in, "out_rate": r_out,
                "n_in": len(grp_in), "n_out": len(grp_out),
                "lift": (r_in / r_out) if r_out > 0 else None,
                "reliable": len(grp_in) >= 30 and len(grp_out) >= 30,
            }

    # 3. INTERACTION: profile × matchup. Split hitters into high-profile
    # (must_have_met >= adaptive bar) vs rest, THEN within each, compare
    # favorable-env vs not. If the env lift is bigger for high-profile hitters,
    # that's evidence matchup and profile COMPOUND — the thing the user suspects.
    if ("must_have_met" in merged_df.columns and "env_boost" in merged_df.columns):
        inter = merged_df.dropna(subset=["must_have_met", "env_boost", "homered"]).copy()
        if not inter.empty:
            _mh = pd.to_numeric(inter["must_have_met"], errors="coerce").fillna(0)
            _env = pd.to_numeric(inter["env_boost"], errors="coerce").fillna(1.0)
            # high profile = top third by must_have_met (data-adaptive)
            _hi_bar = _mh.quantile(0.66) if _mh.nunique() > 2 else _mh.median()
            _hi = _mh >= _hi_bar
            _fav = _env >= 1.05
            def _rate(mask):
                _s = inter[mask]
                return (float(_s["homered"].mean()), len(_s)) if len(_s) else (None, 0)
            hi_fav_r, hi_fav_n = _rate(_hi & _fav)
            hi_unf_r, hi_unf_n = _rate(_hi & ~_fav)
            lo_fav_r, lo_fav_n = _rate(~_hi & _fav)
            lo_unf_r, lo_unf_n = _rate(~_hi & ~_fav)
            if all(x is not None for x in [hi_fav_r, hi_unf_r, lo_fav_r, lo_unf_r]):
                result["matchup_interaction"] = {
                    "hi_profile_fav_env": {"rate": hi_fav_r, "n": hi_fav_n},
                    "hi_profile_poor_env": {"rate": hi_unf_r, "n": hi_unf_n},
                    "lo_profile_fav_env": {"rate": lo_fav_r, "n": lo_fav_n},
                    "lo_profile_poor_env": {"rate": lo_unf_r, "n": lo_unf_n},
                    "hi_env_lift": (hi_fav_r / hi_unf_r) if hi_unf_r and hi_unf_r > 0 else None,
                    "lo_env_lift": (lo_fav_r / lo_unf_r) if lo_unf_r and lo_unf_r > 0 else None,
                    "reliable": min(hi_fav_n, hi_unf_n, lo_fav_n, lo_unf_n) >= 20,
                }

    return result


def per_player_patterns(merged_df: pd.DataFrame,
                        min_games: int = 8,
                        max_players: int = 40) -> dict:
    """v44.70 (user: patterns fit certain players and not others — league-wide
    correlation averages over real individual variation).

    For each hitter with enough graded games, measure how their ACTUAL HR rate
    compares to what the model PROJECTED for them (hr_game_pct). This surfaces
    the players the model systematically over- or under-rates — the ones where
    the league-wide pattern doesn't fit.

    Why min_games matters: a per-player rate from 3 games is noise. We only
    report players at/above min_games so the signal is real. As history grows,
    more players cross the threshold and this becomes richer.

    Returns:
      {
        "players": [ {player, n, actual_hr_rate, proj_hr_rate, edge, ...}... ],
        "n_qualified": int, "min_games": int,
      }
    where edge = actual - projected (positive = model UNDER-rates this player,
    negative = model OVER-rates them). Sorted by |edge| so the biggest model
    mismatches surface first — those are the players whose individual pattern
    diverges most from the league-wide model.
    """
    out = {"players": [], "n_qualified": 0, "min_games": min_games}
    if merged_df is None or merged_df.empty:
        return out
    if "player_id" not in merged_df.columns or "homered" not in merged_df.columns:
        return out

    df = merged_df.dropna(subset=["homered"]).copy()
    if df.empty:
        return out

    # projected per-game HR probability, if we snapshotted it
    _has_proj = "hr_game_pct" in df.columns

    rows = []
    for pid, grp in df.groupby("player_id"):
        n = len(grp)
        if n < min_games:
            continue
        actual = float(grp["homered"].mean())
        name = ""
        if "player_name" in grp.columns and grp["player_name"].notna().any():
            name = str(grp["player_name"].dropna().iloc[0])
        rec = {
            "player_id": pid,
            "player": name or str(pid),
            "n": n,
            "actual_hr_rate": round(actual, 4),
        }
        if _has_proj:
            _proj = pd.to_numeric(grp["hr_game_pct"], errors="coerce").mean()
            # hr_game_pct is stored 0-100; convert to 0-1 rate for comparison
            _proj_rate = float(_proj) / 100.0 if _proj == _proj else None
            if _proj_rate is not None:
                rec["proj_hr_rate"] = round(_proj_rate, 4)
                rec["edge"] = round(actual - _proj_rate, 4)
        rows.append(rec)

    # sort by |edge| (biggest model mismatch first); fall back to actual rate
    if rows and "edge" in rows[0]:
        rows.sort(key=lambda r: abs(r.get("edge", 0)), reverse=True)
    else:
        rows.sort(key=lambda r: r.get("actual_hr_rate", 0), reverse=True)

    out["players"] = rows[:max_players]
    out["n_qualified"] = len(rows)
    return out


# ============================================================================
# Prop-level accuracy summary
# ============================================================================
def prop_accuracy_summary(merged_df: pd.DataFrame) -> dict:
    """For each prop type, compute prediction accuracy across accumulated slates.

    Returns per-prop metrics: predicted rate, actual rate, calibration error,
    Brier score, sample size.
    """
    if merged_df.empty:
        return {}

    result = {}
    # v43.79 (auditor-found): dropped "Total Bases (≥2)" from this backtest.
    # tb_game_pct is EXPECTED TOTAL BASES (a count 0-4+), not a probability
    # in [0,100]%. Dividing by 100 and comparing to a binary (got_2plus_bases)
    # produced a garbage Brier score that would mislead threshold tuning.
    # A proper Total Bases backtest needs either:
    #   (a) a tb_prob_2plus column in snapshots (which we don't have yet), or
    #   (b) comparing the raw expected count to the actual total_bases integer
    # Neither is a probability-vs-binary Brier calc. Removing the misleading
    # entry entirely rather than shipping wrong numbers.
    prop_specs = [
        ("HR", "hr_game_pct", "homered"),
        ("Hit (≥1)", "hit_game_pct", "got_hit"),
    ]
    for label, pred_col, actual_col in prop_specs:
        if pred_col not in merged_df.columns or actual_col not in merged_df.columns:
            continue
        valid = merged_df.dropna(subset=[pred_col, actual_col])
        if len(valid) < 30:
            continue
        try:
            # Both HR and Hit predictors are proper 0-100 percentages
            pred = valid[pred_col].astype(float) / 100.0
            pred = pred.clip(0, 1)  # safety against outliers
            actual = valid[actual_col].astype(float)
            brier = float(((pred - actual) ** 2).mean())
            result[label] = {
                "n": len(valid),
                "predicted_rate": float(pred.mean()),
                "actual_rate": float(actual.mean()),
                "calibration_error": float(pred.mean() - actual.mean()),
                "brier": brier,
                "reliable": len(valid) >= 200,
            }
        except Exception as e:
            logger.warning(f"prop_accuracy_summary failed for {label}: {e}")

    return result


def power_target_accuracy(merged_df: pd.DataFrame) -> dict:
    """v44.18: how often did the Moonshot / Laser targets actually homer?

    The per-game Moonshot (400+ ft) and Laser (105+ mph) picks are tagged
    upstream (is_moonshot_target / is_laser_target) and snapshotted, so once
    outcomes attach we can grade them. Returns for each: the tagged targets'
    HR rate vs the field (non-target) HR rate, plus lift and sample size.
    A lift > 1.0 means the target picks genuinely find the HR hitters.
    """
    if merged_df.empty or "homered" not in merged_df.columns:
        return {}
    out = {}
    for label, tag in [("Moonshot", "is_moonshot_target"),
                       ("Laser", "is_laser_target")]:
        if tag not in merged_df.columns:
            continue
        valid = merged_df.dropna(subset=["homered"]).copy()
        valid[tag] = pd.to_numeric(valid[tag], errors="coerce").fillna(0)
        targets = valid[valid[tag] == 1]
        field = valid[valid[tag] == 0]
        if len(targets) < 5:  # need a few slates of picks
            continue
        t_rate = float(targets["homered"].mean())
        f_rate = float(field["homered"].mean()) if len(field) else 0.0
        out[label] = {
            "n_targets": len(targets),
            "target_hr_rate": t_rate,
            "field_hr_rate": f_rate,
            "lift": (t_rate / f_rate) if f_rate > 0 else None,
        }
    return out


def custom_metric_scorecard(merged_df: pd.DataFrame) -> "pd.DataFrame":
    """v44.32: head-to-head report card for the app's SCORING metrics.

    Shows how well each composite score (hr_score, pick_score, dinger_score,
    power_composite) actually predicts HRs across accumulated slates — via
    point-biserial correlation with the `homered` outcome, plus the top-decile
    hit rate (did the hitters this metric ranked in its top 10% actually homer
    more?). This is how we tell whether the CUSTOM metrics are earning their
    keep and whether reweights (like the v44.24 Dinger tune) are helping.

    Returns a DataFrame sorted by correlation, empty if not enough data.
    """
    if merged_df is None or merged_df.empty or "homered" not in merged_df.columns:
        return pd.DataFrame()
    _metrics = [
        ("hr_score", "HR Score"),
        ("pick_score", "Pick Score"),
        ("dinger_score", "Dinger Score"),
        ("power_composite", "HR+Dinger Combo"),
        ("barrel_matchup_score", "Barrel Matchup"),
        ("two_way_matchup_score", "Two-Way Matchup"),
    ]
    _out = merged_df.dropna(subset=["homered"]).copy()
    if len(_out) < 20:
        return pd.DataFrame()
    _y = pd.to_numeric(_out["homered"], errors="coerce")
    rows = []
    for _col, _label in _metrics:
        if _col not in _out.columns:
            continue
        _x = pd.to_numeric(_out[_col], errors="coerce")
        _pair = pd.concat([_x, _y], axis=1).dropna()
        if len(_pair) < 20 or _pair.iloc[:, 0].nunique() < 3:
            continue
        _corr = float(_pair.iloc[:, 0].corr(_pair.iloc[:, 1]))
        # top-decile hit rate: HR rate among this metric's top 10% of hitters
        _n_top = max(1, int(len(_pair) * 0.10))
        _top = _pair.nlargest(_n_top, _pair.columns[0])
        _top_rate = float(_top.iloc[:, 1].mean())
        _base_rate = float(_pair.iloc[:, 1].mean())
        rows.append({
            "metric": _label,
            "corr_with_HR": round(_corr, 4),
            "top10pct_HR_rate": round(_top_rate, 4),
            "slate_HR_rate": round(_base_rate, 4),
            "lift": round(_top_rate / _base_rate, 2) if _base_rate > 0 else None,
            "n": len(_pair),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("corr_with_HR", ascending=False)


# ============================================================================
# Threshold sweep — find the threshold for a feature that maximizes lift
# ============================================================================
def threshold_sweep(
    merged_df: pd.DataFrame,
    feature_col: str,
    outcome_col: str = "homered",
    candidate_values: Optional[list] = None,
) -> pd.DataFrame:
    """Sweep candidate threshold values to find the one with the best lift.

    Useful for "should the Must-Have barrel% threshold be 12, 15, or 18?"
    Shows the predicted-outcome lift at each candidate level.

    Args:
        merged_df: output of merge_snapshots_with_outcomes
        feature_col: feature to sweep
        outcome_col: binary outcome
        candidate_values: list of threshold values to try. If None, uses
            slate percentiles 50/60/70/80/90.

    Returns:
        DataFrame: threshold, n_pass, pass_rate, fail_rate, lift
    """
    if merged_df.empty or feature_col not in merged_df.columns:
        return pd.DataFrame()
    if outcome_col not in merged_df.columns:
        return pd.DataFrame()

    df = merged_df.dropna(subset=[feature_col, outcome_col])
    if len(df) < 50:
        return pd.DataFrame()

    if candidate_values is None:
        candidate_values = [
            float(df[feature_col].quantile(q))
            for q in [0.5, 0.6, 0.7, 0.8, 0.9]
        ]
    rows = []
    for val in candidate_values:
        in_c = df[df[feature_col] >= val]
        out_c = df[df[feature_col] < val]
        if len(in_c) < 20 or len(out_c) < 20:
            continue
        in_rate = float(in_c[outcome_col].mean())
        out_rate = float(out_c[outcome_col].mean())
        lift = in_rate / out_rate if out_rate > 0 else None
        rows.append({
            "threshold": round(val, 3),
            "n_pass": len(in_c),
            "pass_rate": in_rate,
            "fail_rate": out_rate,
            "lift": lift,
        })
    return pd.DataFrame(rows)


# ============================================================================
# v43.83 — Rolling feature importance + adaptive composite score
# ============================================================================
# The user-requested "look at results every day and find a pattern" loop.
# Runs daily on app load: computes correlations, tracks which features
# stayed predictive vs decayed, and generates an adaptive_score using the
# top-N predictors weighted by their rolling correlation strength.
#
# What this is NOT: it does not modify hr_score or pick_score. It builds
# a PARALLEL adaptive_score that the user can compare to see whether a
# data-driven weighting outperforms the current fixed-weights model.
# Auto-modifying the shipped model would need months of validation.

# The canonical candidate feature set for HR prediction. Ordered by
# theoretical importance (barrel/EV are direct HR precursors, discipline
# and matchup effects are indirect).
HR_CANDIDATE_FEATURES = [
    "barrel_pct", "iso", "avg_ev", "hard_hit", "blast_pct",
    "pull_pct", "pull_air_pct", "pulled_brl_pct",
    "fb_pct", "xslg", "slg", "xwoba",
    "hr_score", "hr_game_pct", "hr_pa_pct",
    "must_have_met", "nuclear_met",
    "matchup_opp", "power_score", "pitch_hr_score",
    "lift_score", "discipline_score",
    "recent_hr_weighted_rate",
    "env_boost", "dinger_score", "power_composite", "barrel_matchup_score", "two_way_matchup_score",
    # v44.76 (user: does sleeper_score actually affect HR probability? — and
    # audit whether every pick_score component earns its weight). Track the
    # remaining pick_score inputs so their real HR correlation is visible. If
    # sleeper_score's correlation is ~0 or negative, that's evidence its 5%
    # weight should shrink or go to zero.
    "sleeper_score", "convergence_count",
]


def compute_daily_correlations(merged_df: pd.DataFrame,
                                outcome_col: str = "homered") -> dict:
    """Compute Pearson correlations between all candidate features and outcome.

    Called by the daily auto-runner. Returns a dict suitable for stashing
    in the correlations history.

    Args:
        merged_df: output of merge_snapshots_with_outcomes
        outcome_col: binary outcome ("homered", "got_hit", etc.)

    Returns:
        {"date_computed": str, "n_samples": int, "outcome": str,
         "correlations": {feature: {"corr": float, "n": int}}}
    """
    from datetime import date
    result = {
        "date_computed": str(date.today()),
        "n_samples": len(merged_df),
        "outcome": outcome_col,
        "correlations": {},
    }
    if merged_df.empty or outcome_col not in merged_df.columns:
        return result

    for feat in HR_CANDIDATE_FEATURES:
        if feat not in merged_df.columns:
            continue
        valid = merged_df[[feat, outcome_col]].dropna()
        if len(valid) < 30:
            continue
        try:
            corr = float(valid[feat].astype(float).corr(
                valid[outcome_col].astype(float)
            ))
            if pd.notna(corr):
                result["correlations"][feat] = {
                    "corr": round(corr, 4),
                    "n": int(len(valid)),
                }
        except Exception:
            continue
    return result


def rolling_feature_importance(correlation_history: list,
                                lookback_days: int = 14) -> pd.DataFrame:
    """Roll up the last N days of correlations into a feature-importance table.

    Shows: which features are consistently predictive (high avg corr, low std),
    which are noisy (high std), which are decaying (recent < older).

    Args:
        correlation_history: list of dicts from compute_daily_correlations,
            each stored with a snapshot date
        lookback_days: window size

    Returns:
        DataFrame with columns: feature, avg_corr, recent_corr, older_corr,
        trend, std, n_days, reliability
    """
    if not correlation_history:
        return pd.DataFrame()

    # Take last N entries
    recent = correlation_history[-lookback_days:]
    if not recent:
        return pd.DataFrame()

    # Collect corr values per feature across days
    per_feat = {}
    for entry in recent:
        corrs = entry.get("correlations", {})
        for feat, obj in corrs.items():
            per_feat.setdefault(feat, []).append(obj.get("corr", 0.0))

    rows = []
    for feat, corrs in per_feat.items():
        if len(corrs) < 3:  # need at least 3 days
            continue
        arr = pd.Series(corrs)
        avg_corr = float(arr.mean())
        std = float(arr.std(ddof=0))
        # Trend: compare recent third to older third. Requires ≥6 days —
        # below that there aren't enough points to split into thirds, so
        # trend is UNDEFINED, not zero.
        # v44.07 (user spotted every trend showing +0.000 at 5 days): the
        # old code hard-coded trend=0.0 below 6 days, which rendered as a
        # real-looking "no movement" reading when the truth is "not computed
        # yet." Now flag it so the UI can show n/a instead of a fake zero.
        n = len(corrs)
        if n >= 6:
            recent_third = float(pd.Series(corrs[-max(2, n//3):]).mean())
            older_third = float(pd.Series(corrs[:max(2, n//3)]).mean())
            trend = round(recent_third - older_third, 4)
            trend_valid = True
        else:
            recent_third = avg_corr
            older_third = avg_corr
            trend = None
            trend_valid = False
        rows.append({
            "feature": feat,
            "avg_corr": round(avg_corr, 4),
            "recent_corr": round(recent_third, 4),
            "older_corr": round(older_third, 4),
            "trend": trend,
            "trend_valid": trend_valid,
            "std": round(std, 4),
            "n_days": n,
            # Reliability: high avg, low std → high reliability
            "reliability": round(
                abs(avg_corr) / (std + 0.01) if std >= 0 else 0.0, 2
            ),
        })
    # v43.87: guard against empty rows. pd.DataFrame([]).sort_values("avg_corr")
    # raises KeyError because the empty df has no columns. Return an empty df
    # with the proper columns so callers (Section G) can safely check .empty
    # and column presence.
    if not rows:
        return pd.DataFrame(columns=[
            "feature", "avg_corr", "recent_corr", "older_corr",
            "trend", "trend_valid", "std", "n_days", "reliability",
        ])
    return pd.DataFrame(rows).sort_values(
        "avg_corr", key=abs, ascending=False
    )


def propose_dinger_weights(importance_df: pd.DataFrame,
                           current_weights: dict,
                           weight_sum: float = 10.0,
                           min_days: int = 5) -> dict:
    """v44.68 (user endgame: synthesize our OWN properly-weighted metric).

    Takes accumulated feature importance and proposes what the Dinger power
    weights WOULD be if derived purely from the data — weighting each feature
    by |correlation| × reliability (predictive AND stable), normalized to the
    same total as the current shipped weights so they're directly comparable.

    Returns a dict with, per feature: current weight, proposed weight, the
    delta, and the evidence (corr, reliability, days). This is a RECOMMENDATION
    for the user to evaluate — it does NOT auto-change the shipped model. The
    point is to make "what does the data say the weights should be?" a concrete,
    comparable answer instead of a guess.

    Guardrails:
      - only features with >= min_days of history and |corr| >= 0.03 count
      - a feature the data has no signal on keeps its current weight (we don't
        zero out a stat just because it's new/thin — that would be overfitting
        to a short window)
      - proposed weights are normalized to weight_sum so scale matches current
    """
    result = {"features": {}, "reliable": False, "note": ""}
    if importance_df is None or importance_df.empty:
        result["note"] = "No feature-importance history yet — need graded slates."
        return result

    imp = importance_df.set_index("feature") if "feature" in importance_df.columns else importance_df
    # Build a raw score per CURRENT feature: |corr| × reliability.
    raw = {}
    _min_days_seen = 999
    for feat, cur_w in current_weights.items():
        if feat in imp.index:
            row = imp.loc[feat]
            _corr = abs(float(row.get("avg_corr", 0.0)))
            _reliab = float(row.get("reliability", 1.0))
            _days = int(row.get("n_days", 0))
            _min_days_seen = min(_min_days_seen, _days)
            if _days >= min_days and _corr >= 0.03:
                # reliability modulates but doesn't dominate (0.5–1.5 band)
                _reliab_mult = 0.5 + min(max(_reliab, 0.0), 3.0) / 3.0
                raw[feat] = _corr * _reliab_mult
            else:
                # thin evidence → fall back to current weight's implied share
                raw[feat] = None
        else:
            raw[feat] = None

    # For features with no usable evidence, keep their CURRENT share so we don't
    # overfit. Normalize the evidence-backed features across the remaining budget.
    _cur_total = float(sum(current_weights.values())) or 1.0
    _evidence_feats = {f: v for f, v in raw.items() if v is not None}
    _kept_feats = {f: current_weights[f] for f in raw if raw[f] is None}
    _kept_budget = sum(_kept_feats.values()) / _cur_total * weight_sum
    _evidence_budget = weight_sum - _kept_budget
    _evidence_raw_total = float(sum(_evidence_feats.values())) or 1.0

    for feat, cur_w in current_weights.items():
        if feat in _evidence_feats:
            proposed = _evidence_budget * (_evidence_feats[feat] / _evidence_raw_total)
            ev = imp.loc[feat]
            evidence = {
                "corr": round(float(ev.get("avg_corr", 0.0)), 4),
                "reliability": round(float(ev.get("reliability", 0.0)), 2),
                "days": int(ev.get("n_days", 0)),
            }
        else:
            # kept at current share (rescaled to weight_sum)
            proposed = cur_w / _cur_total * weight_sum
            evidence = {"corr": None, "reliability": None, "days": 0,
                        "note": "thin evidence — kept current"}
        result["features"][feat] = {
            "current": round(float(cur_w), 2),
            "proposed": round(float(proposed), 2),
            "delta": round(float(proposed) - float(cur_w), 2),
            "evidence": evidence,
        }

    # reliable only if every evidence feature has enough days
    result["reliable"] = (_min_days_seen >= 15 and len(_evidence_feats) >= 3)
    result["note"] = (
        f"Proposed from {len(_evidence_feats)} evidence-backed feature(s); "
        f"{len(_kept_feats)} kept at current (thin data). "
        + ("Enough history to take seriously." if result["reliable"]
           else "Still early — treat as directional, not final.")
    )
    return result


def compute_adaptive_score(current_slate: pd.DataFrame,
                            importance_df: pd.DataFrame,
                            top_n_features: int = 5) -> pd.Series:
    """Build a data-driven composite score from top-N features.

    Each feature's contribution is weighted by its rolling correlation
    strength. Only features with |avg_corr| >= 0.05 and n_days >= 5 are
    included — otherwise the score degrades to noise.

    Args:
        current_slate: DataFrame with today's hitters (must contain features)
        importance_df: output of rolling_feature_importance
        top_n_features: how many predictors to combine

    Returns:
        pd.Series indexed to current_slate, values 0-100 (percentile rank)
        of the weighted combination. NaN if not enough reliable features.
    """
    if current_slate.empty or importance_df.empty:
        return pd.Series(dtype=float, index=current_slate.index)

    # Filter to reliable predictors
    reliable = importance_df[
        (importance_df["avg_corr"].abs() >= 0.05)
        & (importance_df["n_days"] >= 5)
    ].head(top_n_features)
    if reliable.empty:
        return pd.Series(dtype=float, index=current_slate.index)

    # Compute weighted percentile sum
    parts = []
    weights = []
    for _, row in reliable.iterrows():
        feat = row["feature"]
        if feat not in current_slate.columns:
            continue
        pct = current_slate[feat].rank(pct=True) * 100.0
        # Sign the contribution by correlation direction
        # (positive corr → higher feature = higher score;
        #  negative corr → higher feature = lower score)
        signed = pct if row["avg_corr"] >= 0 else (100.0 - pct)
        parts.append(signed)
        # v44.49 (user: improve as data accumulates). Weight by correlation
        # AND reliability, not correlation alone. A feature that's predictive
        # AND stable day-to-day (high reliab) should count more than one with
        # the same correlation but that's noisy. reliab is a 0-3ish stability
        # score from rolling_feature_importance; normalize to a ~0.5-1.5
        # multiplier so it modulates rather than dominates the corr weight.
        _corr_w = abs(float(row["avg_corr"]))
        _reliab = float(row.get("reliability", 1.0)) if "reliability" in row.index else 1.0
        # map reliab (typically 1.0-2.9) to a 0.5-1.5 multiplier around 1.0
        _reliab_mult = 0.5 + min(max(_reliab, 0.0), 3.0) / 3.0
        weights.append(_corr_w * _reliab_mult)

    if not parts:
        return pd.Series(dtype=float, index=current_slate.index)

    # Per-row NaN-tolerant weighted average (same pattern as pick_score fix)
    import numpy as np
    weights_arr = np.array(weights)
    parts_df = pd.concat(
        [p.rename(str(i)) for i, p in enumerate(parts)], axis=1
    )
    present_mask = parts_df.notna().astype(float).values
    row_weight_totals = present_mask @ weights_arr
    safe_totals = np.where(row_weight_totals > 0, row_weight_totals, np.nan)
    parts_filled = parts_df.fillna(0).values
    weighted_sum = parts_filled @ weights_arr
    return pd.Series(weighted_sum / safe_totals, index=current_slate.index)
