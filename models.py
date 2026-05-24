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
    "matchup": {
        "xwoba": 0.20,
        "iso": 0.12,
        "barrel_pct": 0.13,
        "hard_hit": 0.08,
        "k_pct_inv": 0.10,
        "sweet_spot_pct": 0.05,
        "pitcher_xwoba": 0.15,
        "pitcher_k_inv": 0.10,
        "pitcher_barrel_allowed": 0.07,
    },
    "hr_form": {
        "barrel_pct": 0.30,
        "iso": 0.25,
        "hard_hit": 0.15,
        "avg_ev": 0.10,
        "fb_pct": 0.10,
        "pulled_brl_pct": 0.10,
    },
    "ceiling": {
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
    Weighted sum of percentile-ranked columns.
    - Rows with NO data in any weighted column → NaN (honest)
    - Rows with partial data → score scaled by weights of columns that DO have data
    - Columns in `neg` are inverted (lower raw = higher score)
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
        row["lineup_pos"] = i
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
    else:
        df["pitcher_xwoba"] = np.nan
        df["pitcher_k_inv"] = np.nan
        df["pitcher_barrel_allowed"] = np.nan

    # Normalize column names to consistent shorts
    rename = {
        "barrel_batted_rate": "barrel_pct",
        "hard_hit_percent": "hard_hit",
        "k_percent": "k_pct",
        "bb_percent": "bb_pct",
        "avg_best_speed": "avg_ev",
        "sweet_spot_percent": "sweet_spot_pct",
        "flyballs_percent": "fb_pct",
        "groundballs_percent": "gb_pct",
        "linedrives_percent": "ld_pct",
        "whiff_percent": "whiff_pct",
        # Savant uses launch_angle_avg in CSV exports
        "launch_angle": "la",
        "launch_angle_avg": "la",
        "avg_hit_angle": "la",  # another variant
        # Savant uses isolated_power for ISO in CSV exports
        "isolated_power": "iso",
        "pull_air_percent": "pull_air_pct",
    }
    df = df.rename(columns=rename)

    # If we now have multiple "la" columns from the rename, dedupe
    if isinstance(df.columns, pd.Index) and df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # Derive ISO from xslg-xba if still missing (math identity from real data)
    if "iso" not in df.columns or df["iso"].isna().all():
        slg_col = next((c for c in ["slg", "xslg"] if c in df.columns), None)
        avg_col = next((c for c in ["batting_avg", "avg", "xba"] if c in df.columns), None)
        if slg_col and avg_col:
            df["iso"] = (df[slg_col] - df[avg_col]).round(3)

    df["k_pct_inv"] = -df["k_pct"] if "k_pct" in df.columns else np.nan

    # Composite scores
    df["matchup"] = _score_from_weights(df, SCORING_WEIGHTS["matchup"])
    df["hr_form"] = _score_from_weights(df, SCORING_WEIGHTS["hr_form"])
    df["ceiling"] = _score_from_weights(df, SCORING_WEIGHTS["ceiling"])

    # Test Score = matchup × (PA sample weight). NaN matchup → NaN Test.
    if "pa" in df.columns:
        pa_factor = (df["pa"] / 150.0).clip(0.5, 1.0)
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
    display_cols = [
        "player_id", "player_name", "lineup_pos", "position", "bats",
        # Composites (matching screenshot order)
        "matchup", "test_score", "ceiling", "zone_fit",
        "hr_form", "hr_form_label", "hr_form_arrow", "kHR",
        # Pitches / BIP / ISO / xwOBA family
        "pitches", "bip", "iso", "xwoba", "xwobacon",
        # Quality of contact
        "barrel_pct", "pulled_brl_pct", "hard_hit", "sweet_spot_pct",
        "fb_pct", "gb_pct", "ld_pct",
        "la", "avg_ev",
        # Plate discipline
        "k_pct", "bb_pct", "whiff_pct", "swing_percent",
        # Rates
        "obp", "slg", "ops", "babip",
        # Counts
        "pa", "home_run", "recent_hr", "recent_iso", "recent_avg",
        "streak_label", "hr_streak_games", "hr_last_5", "games_since_hr",
        # Today's HR projection
        "likely_hr_pct",
    ]
    keep = [c for c in display_cols if c in df.columns]
    return df[keep]


def _season_thresholds(slate_date=None):
    """
    Return season-aware thresholds. Earlier in season = lower bars.
    Returns dict with: full_ip (IP for full reliability), reliever_ip (below = likely reliever),
    full_gs (GS for full starter), swing_gs (below = swing).
    """
    import datetime
    if slate_date is None:
        slate_date = datetime.date.today()
    try:
        month = slate_date.month
    except AttributeError:
        # If slate_date is a string, try to parse
        try:
            slate_date = datetime.datetime.strptime(str(slate_date)[:10], "%Y-%m-%d").date()
            month = slate_date.month
        except Exception:
            month = 6  # default mid-season
    # Map month to expected starter IP at that point in season
    # Season runs roughly Apr-Oct (months 3-10)
    if month <= 4:        # April: ~5 starts in
        return {"full_ip": 15, "min_ip": 5, "full_gs": 3, "min_gs": 1}
    elif month == 5:      # May: ~9 starts in
        return {"full_ip": 30, "min_ip": 10, "full_gs": 5, "min_gs": 2}
    elif month == 6:      # June: ~13 starts in
        return {"full_ip": 50, "min_ip": 15, "full_gs": 8, "min_gs": 3}
    elif month == 7:      # July: ~17 starts in
        return {"full_ip": 70, "min_ip": 20, "full_gs": 10, "min_gs": 4}
    elif month == 8:      # August: ~22 starts in
        return {"full_ip": 90, "min_ip": 25, "full_gs": 14, "min_gs": 5}
    elif month == 9:      # September: ~28 starts in
        return {"full_ip": 110, "min_ip": 30, "full_gs": 18, "min_gs": 6}
    elif month == 10:     # October
        return {"full_ip": 130, "min_ip": 30, "full_gs": 20, "min_gs": 6}
    else:                 # Offseason / spring
        return {"full_ip": 30, "min_ip": 10, "full_gs": 5, "min_gs": 2}


def build_pitcher_slate(
    slate: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    pitcher_recent: dict | None = None,
    slate_date=None,
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
            base = {
                "pitcher_id": pid,
                "pitcher_name": g[f"{side}_pitcher"],
                "team": g[f"{side}_team_abbr"],
                "opp": g[f"{'home' if side == 'away' else 'away'}_team_abbr"],
                "home_away": "@" if side == "away" else "vs",
                "game_pk": g["gamePk"],
            }
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
                })
            if pitcher_recent and pid in pitcher_recent:
                base.update(pitcher_recent[pid])
            pitchers.append(base)

    df = pd.DataFrame(pitchers)
    if df.empty:
        return df

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
    # SAMPLE NOISE FLAG - catches stat inconsistencies (high ERA + low WHIP, etc)
    # ------------------------------------------------------------------
    def _sample_noise(row):
        ip = row.get("ip")
        era = row.get("era")
        whip = row.get("whip")
        if (ip is not None and not pd.isna(ip) and ip < 20
                and era is not None and not pd.isna(era) and era > 5.0
                and whip is not None and not pd.isna(whip) and whip < 1.1):
            return True
        barrel = row.get("barrel_allowed")
        if (ip is not None and not pd.isna(ip) and ip < 10
                and barrel is not None and not pd.isna(barrel) and barrel == 0):
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
        # Bulk reliever: games_played >> games_started
        if (gp is not None and gs is not None
                and not pd.isna(gp) and not pd.isna(gs)
                and gs > 0 and gp > gs * 1.5):
            base = 0.65
        else:
            base = 1.0
        ip_factor = min(1.0, ip / full_ip)
        if gs is not None and not pd.isna(gs) and gs > 0:
            start_factor = min(1.0, gs / full_gs)
        else:
            start_factor = ip_factor
        return round(max(0.3, base * (ip_factor * 0.5 + start_factor * 0.5)), 2)

    df["reliability"] = df.apply(_reliability, axis=1)

    # ------------------------------------------------------------------
    # ROLE FLAG - season-aware, never auto-defaults to RELIEVER without evidence
    # ------------------------------------------------------------------
    def _role_flag(row):
        ip = row.get("ip")
        gs = row.get("games_started")
        gp = row.get("games_played")
        # If we have no IP and no GS data at all, say so honestly
        if (ip is None or pd.isna(ip)) and (gs is None or pd.isna(gs)):
            return "❔ NO DATA"
        # Explicit zero starts:
        #   - low IP → pure RELIEVER
        #   - meaningful IP (25+) → BULK relief role (long reliever, sometimes opens)
        if gs is not None and not pd.isna(gs) and gs == 0:
            if ip is not None and not pd.isna(ip) and ip >= 25:
                return "🔄 BULK"  # long reliever / opener candidate
            return "🚨 RELIEVER"
        # Very low IP when we DO have IP data
        if ip is not None and not pd.isna(ip) and ip < min_ip:
            return "🚨 RELIEVER"
        # Bulk reliever between starts
        if (gp is not None and gs is not None
                and not pd.isna(gp) and not pd.isna(gs)
                and gs > 0 and gp > gs * 1.5):
            return "🔄 SWING"
        # Low IP but not crazy low
        if ip is not None and not pd.isna(ip) and ip < full_ip * 0.6:
            return "⚠️ LOW IP"
        # Few starts
        if gs is not None and not pd.isna(gs) and gs < min_gs:
            return "🔄 SWING"
        return "✓"

    df["role"] = df.apply(_role_flag, axis=1)

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
    # TEST SCORE - rebalanced. K is now 35% (was 40%), with new ERA component
    # ------------------------------------------------------------------
    # Weights: K-blended 30%, Whiff 20%, Suppress (xwOBA) 25%, ERA 15%, base K 10%
    # Total weight = 1.0 if all components present; less means data is incomplete
    def _composite_test(row):
        parts = []
        max_weight = 0.30 + 0.20 + 0.25 + 0.15 + 0.10  # = 1.0
        if pd.notna(row.get("k_score_blended")):
            parts.append((row["k_score_blended"], 0.30))
        if pd.notna(row.get("whiff_score")):
            parts.append((row["whiff_score"], 0.20))
        if pd.notna(row.get("suppress_score")):
            parts.append((row["suppress_score"], 0.25))
        if pd.notna(row.get("era_score")):
            parts.append((row["era_score"], 0.15))
        if pd.notna(row.get("k_score")):
            parts.append((row["k_score"], 0.10))
        if not parts:
            return np.nan
        total_w = sum(w for _, w in parts)
        raw = sum(v * w for v, w in parts) / total_w
        # Data completeness penalty: only count fully when we have all 5 components
        # If only 3 of 5 weights are present, score gets multiplied by 0.85
        # If only 2 of 5, by 0.75. This penalizes inflated scores from sparse data.
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
    # Uses: low barrel% allowed, low xwOBA allowed, low HR/9 (when available)
    # ------------------------------------------------------------------
    def _hr_suppress(row):
        parts = []
        max_weight = 0
        # Barrel% allowed: lower = better (invert percentile)
        if pd.notna(row.get("barrel_allowed")):
            barrel_pct = row.get("_barrel_pct_rank", np.nan)
            if pd.notna(barrel_pct):
                parts.append((100 - barrel_pct, 0.40))
                max_weight += 0.40
        # xwOBA allowed: lower = better (already inverted as suppress_score)
        if pd.notna(row.get("suppress_score")):
            parts.append((row["suppress_score"], 0.35))
            max_weight += 0.35
        # HR/9: lower = better
        if pd.notna(row.get("_hr9_pct_rank")):
            parts.append((100 - row["_hr9_pct_rank"], 0.25))
            max_weight += 0.25
        if not parts or max_weight == 0:
            return np.nan
        total_w = sum(w for _, w in parts)
        raw = sum(v * w for v, w in parts) / total_w
        completeness = total_w / 1.0
        penalty = 0.85 if completeness < 0.6 else 0.95 if completeness < 0.85 else 1.0
        return raw * penalty

    # Precompute pct ranks for barrel and hr9
    if "barrel_allowed" in df.columns:
        df["_barrel_pct_rank"] = _safe_pct_rank(df["barrel_allowed"])
    if "hr9" in df.columns and df["hr9"].notna().any():
        df["_hr9_pct_rank"] = _safe_pct_rank(df["hr9"])

    raw_hr_supp = df.apply(_hr_suppress, axis=1)
    df["hr_suppress"] = (raw_hr_supp * df["reliability"]).clip(upper=95).round(2)

    # Clean up internal helper columns
    df = df.drop(columns=[c for c in ["_barrel_pct_rank", "_hr9_pct_rank"] if c in df.columns])

    # ------------------------------------------------------------------
    # PROJ K - reliability-adjusted, uses blended K/9 when available
    # ------------------------------------------------------------------
    # Expected IP varies by role - relievers/openers won't go deep
    if "k9" in df.columns:
        effective_k9 = df.get("blended_k9", df["k9"]).fillna(df["k9"])
        def _expected_ip(row):
            ip = row.get("ip")
            gs = row.get("games_started")
            # Hard reliever: explicit GS=0, OR very low IP
            if gs is not None and not pd.isna(gs) and gs == 0:
                return 1.5  # opener / bulk-relief: at most a couple innings
            if ip is None or pd.isna(ip) or ip < 10:
                return 2.0
            # Swing/spot starter: limited usage
            if gs is not None and not pd.isna(gs) and gs < 5:
                return 3.5
            if ip < 25:
                return 4.0
            # Established starter
            return 5.5
        df["expected_ip"] = df.apply(_expected_ip, axis=1)
        df["proj_k"] = (effective_k9 * df["expected_ip"] / 9).round(1)

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
