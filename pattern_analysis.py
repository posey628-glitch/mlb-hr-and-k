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
    for snap_key, payload in snapshots.items():
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
            outcome = hitter_outcomes.get(str(pid)) or hitter_outcomes.get(int(pid)) or {}
            if not outcome:
                continue  # snapshot has outcomes but not for this player

            # Build one combined row
            row = {"snapshot_date": snapshot_date, "player_id": pid}
            # Projection columns (whitelist what's useful for analysis)
            # v43.88 (review finding): this whitelist predated v43.83's
            # snapshot expansion, so only 12 of the 24 HR_CANDIDATE_FEATURES
            # made it into the merged frame — daily correlations silently
            # tracked half the intended predictor set. Now includes the
            # full candidate set plus pick_score decomposition context.
            for col in [
                "player_name", "team", "hr_score", "hr_game_pct", "pick_score",
                "grade", "barrel_pct", "iso", "avg_ev", "blast_pct",
                "pull_pct", "pull_air_pct", "pulled_brl_pct", "hard_hit",
                "fb_pct", "gb_pct", "ld_pct",
                "must_have_met", "must_have_total", "must_have_pass",
                "nuclear_met", "nuclear_total", "nuclear_grade",
                "hit_game_pct", "tb_game_pct",
                "hr_pa_pct", "power_score",
                "xwoba", "xslg", "slg", "obp", "ops",
                "k_pct", "bb_pct", "whiff_pct",
                "discipline_score", "lift_score", "matchup_opp",
                "recent_hr", "recent_hr_weighted_rate",
                "pitch_hr_score", "pitch_match_score",
                "env_boost", "opp_pitcher_xwoba",
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
    return pd.DataFrame(rows)


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

    # Must-Have pass cohort
    if "must_have_pass" in merged_df.columns:
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

    # Nuclear tier cohort (NEAR / STRONG / NUCLEAR vs rest)
    if "nuclear_grade" in merged_df.columns:
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

    return result


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
    "env_boost",
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
        # Trend: compare last third to first third
        n = len(corrs)
        if n >= 6:
            recent_third = float(pd.Series(corrs[-max(2, n//3):]).mean())
            older_third = float(pd.Series(corrs[:max(2, n//3)]).mean())
            trend = recent_third - older_third
        else:
            recent_third = avg_corr
            older_third = avg_corr
            trend = 0.0
        rows.append({
            "feature": feat,
            "avg_corr": round(avg_corr, 4),
            "recent_corr": round(recent_third, 4),
            "older_corr": round(older_third, 4),
            "trend": round(trend, 4),
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
            "trend", "std", "n_days", "reliability",
        ])
    return pd.DataFrame(rows).sort_values(
        "avg_corr", key=abs, ascending=False
    )


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
        weights.append(abs(float(row["avg_corr"])))

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
