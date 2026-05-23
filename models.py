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
        # Today's HR projection
        "likely_hr_pct",
    ]
    keep = [c for c in display_cols if c in df.columns]
    return df[keep]


def build_pitcher_slate(
    slate: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    pitcher_recent: dict | None = None,
) -> pd.DataFrame:
    """One row per starting pitcher with composite scores + recent form."""
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
    # IP PER OUTING - better signal than IP/GS since it uses games_played
    # ------------------------------------------------------------------
    def _ip_per_outing(row):
        ip = row.get("ip")
        gp = row.get("games_played")
        gs = row.get("games_started")
        # Prefer games_played; fall back to games_started; else IP/6 guess
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
        # ERA-WHIP inconsistency at low sample = 1 bad inning skewing ERA
        if (ip is not None and not pd.isna(ip) and ip < 20
                and era is not None and not pd.isna(era) and era > 5.0
                and whip is not None and not pd.isna(whip) and whip < 1.1):
            return True
        # Statcast zeros at very low sample
        barrel = row.get("barrel_allowed")
        if (ip is not None and not pd.isna(ip) and ip < 10
                and barrel is not None and not pd.isna(barrel) and barrel == 0):
            return True
        return False

    df["sample_noise"] = df.apply(_sample_noise, axis=1)

    # ------------------------------------------------------------------
    # RELIABILITY FACTOR - penalize low-IP / non-starter pitchers
    # Now uses ip_per_outing to detect bulk relievers masquerading as starters
    # ------------------------------------------------------------------
    def _reliability(row):
        ip = row.get("ip")
        gs = row.get("games_started")
        gp = row.get("games_played")
        if ip is None or pd.isna(ip):
            return 0.5  # no data = neutral, not crash-low
        # Hard cap: explicit zero starts = reliever
        if gs is not None and not pd.isna(gs) and gs == 0:
            return 0.4
        # If pitcher has played more games than started (relief between starts),
        # they're a swing role - downgrade
        if (gp is not None and gs is not None
                and not pd.isna(gp) and not pd.isna(gs)
                and gs > 0 and gp > gs * 1.5):
            base = 0.65
        else:
            base = 1.0
        ip_factor = min(1.0, ip / 30.0)
        if gs is not None and not pd.isna(gs) and gs > 0:
            start_factor = min(1.0, gs / 6.0)
        else:
            # Unknown gs - rely on IP alone
            start_factor = ip_factor
        return round(max(0.3, base * (ip_factor * 0.5 + start_factor * 0.5)), 2)

    df["reliability"] = df.apply(_reliability, axis=1)

    # ------------------------------------------------------------------
    # ROLE FLAG - now uses games_played vs games_started to spot bulk relievers
    # ------------------------------------------------------------------
    def _role_flag(row):
        ip = row.get("ip")
        gs = row.get("games_started")
        gp = row.get("games_played")
        # Hard reliever flag if we explicitly know GS=0 (real zero, not None)
        if gs is not None and not pd.isna(gs) and gs == 0:
            return "🚨 RELIEVER"
        # If IP is missing or very low, only flag as RELIEVER if we ALSO have
        # evidence (e.g. gs=0 above, or very obvious low IP). If gs is None
        # entirely, fall back to IP-based logic but don't auto-flag reliever.
        if ip is None or pd.isna(ip):
            return "⚠️ NO DATA"
        if ip < 10:
            return "🚨 RELIEVER"
        # Bulk reliever: games_played >> games_started (only if both known)
        if (gp is not None and gs is not None
                and not pd.isna(gp) and not pd.isna(gs)
                and gs > 0 and gp > gs * 1.5):
            return "🔄 SWING"
        if ip < 25:
            return "⚠️ LOW IP"
        # If gs is known and low, swing role
        if gs is not None and not pd.isna(gs) and gs < 5:
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
    def _composite_test(row):
        parts = []
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
        return sum(v * w for v, w in parts) / total_w

    raw_test = df.apply(_composite_test, axis=1)
    # Apply reliability multiplier - relievers get scaled down
    df["test_score"] = (raw_test * df["reliability"]).round(2)

    # kHR: still K-focused but apply reliability too
    def _composite_khr(row):
        parts = []
        if pd.notna(row.get("k_score_blended")):
            parts.append((row["k_score_blended"], 0.50))
        if pd.notna(row.get("k_score")):
            parts.append((row["k_score"], 0.20))
        if pd.notna(row.get("whiff_score")):
            parts.append((row["whiff_score"], 0.30))
        if not parts:
            return np.nan
        total_w = sum(w for _, w in parts)
        return sum(v * w for v, w in parts) / total_w

    raw_khr = df.apply(_composite_khr, axis=1)
    df["kHR"] = (raw_khr * df["reliability"]).round(2)

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
