"""
models.py
==========
Composite scoring engine for hitter matchups and pitcher slate.

Real data only - NaN propagates honestly when stats are missing.
No fake-50 defaults, no league-average fillna shortcuts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ===========================================================================
# Scoring weights - tuned to roughly match Kasper-style outputs
# ===========================================================================

SCORING_WEIGHTS = {
    "matchup": {
        "iso": 0.15,
        "xwoba": 0.15,
        "barrel_pct": 0.20,
        "hard_hit": 0.10,
        "sweet_spot_pct": 0.05,
        "fb_pct": 0.10,
        "xwobacon": 0.15,
        "pitcher_xwoba": 0.10,  # higher = better for hitter
    },
    "hr_form": {
        "recent_iso": 0.30,
        "recent_hr": 0.25,
        "barrel_pct": 0.20,
        "fb_pct": 0.15,
        "xwobacon": 0.10,
    },
    "ceiling": {
        "barrel_pct": 0.30,
        "hard_hit": 0.20,
        "xwobacon": 0.20,
        "iso": 0.15,
        "avg_best_speed": 0.15,
    },
}

NEG_COLS = ("k_pct", "whiff_pct")  # columns where lower is better for the hitter


def _safe_pct_rank(s: pd.Series) -> pd.Series:
    """
    Percentile rank 0-100. NaNs stay NaN - no fake-50 defaults.
    Composite scores will also be NaN for players with missing data,
    which is honest.
    """
    return s.rank(pct=True) * 100


def _score_from_weights(df: pd.DataFrame, weights: dict, neg: tuple = NEG_COLS) -> pd.Series:
    """
    Weighted sum of percentile-ranked columns.
    - Rows with NO data in any weighted column → NaN
    - Rows with partial data → score scaled by weights of columns that DO have data
    - Columns in `neg` are inverted (lower raw = higher score)
    """
    contributions = []
    for col, w in weights.items():
        if col not in df.columns:
            continue
        ranked = _safe_pct_rank(df[col])
        if col in neg:
            ranked = 100 - ranked
        weight_present = ranked.notna().astype(float) * w
        contributions.append((weight_present, ranked.fillna(0)))

    if not contributions:
        return pd.Series([np.nan] * len(df), index=df.index)

    total_weight = sum(wp for wp, _ in contributions)
    weighted_sum = sum(wp * r for wp, r in contributions)

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


# ===========================================================================
# Hitter matchup table
# ===========================================================================

def build_matchup_table(
    lineup: list,
    pitcher_row: pd.Series | None,
    hitter_stats: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    recent_form_dict: dict | None = None,
) -> pd.DataFrame:
    """Build per-hitter matchup table for one team facing today's pitcher."""
    if not lineup:
        return pd.DataFrame()

    rows = []
    for i, p in enumerate(lineup, start=1):
        pid = p.get("id")
        if pid is None:
            continue
        h = hitter_stats[hitter_stats["player_id"] == pid]
        base = {
            "player_id": pid,
            "player_name": p.get("name"),
            "lineup_pos": i,
            "position": p.get("position"),
            "bats": p.get("bats", ""),
        }
        if len(h) > 0:
            r = h.iloc[0].to_dict()
            base.update({
                "pa": r.get("pa"),
                "iso": r.get("iso"),
                "xwoba": r.get("xwoba"),
                "xwobacon": r.get("xwobacon"),
                "xba": r.get("xba"),
                "xslg": r.get("xslg"),
                "barrel_pct": r.get("barrel_batted_rate"),
                "hard_hit": r.get("hard_hit_percent"),
                "sweet_spot_pct": r.get("sweet_spot_percent"),
                "fb_pct": r.get("flyballs_percent"),
                "la": r.get("launch_angle"),
                "ev": r.get("launch_speed"),
                "best_speed": r.get("avg_best_speed"),
                "k_pct": r.get("k_percent"),
                "bb_pct": r.get("bb_percent"),
                "whiff_pct": r.get("whiff_percent"),
                "home_run": r.get("home_run"),
                "pulled_brl_pct": r.get("pulled_brl_pct"),
                "pull_pct": r.get("pull_percent"),
                "oppo_pct": r.get("opposite_percent"),
                "pull_air_pct": r.get("pull_air_percent"),
                "oppo_air_pct": r.get("opposite_air_percent"),
                "sprint_speed": r.get("sprint_speed"),
            })
        if recent_form_dict and pid in recent_form_dict:
            base.update(recent_form_dict[pid])
        rows.append(base)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Attach pitcher context (same value for every hitter in the lineup)
    if pitcher_row is not None and len(pitcher_row):
        df["pitcher_xwoba"] = pitcher_row.get("xwoba")
        df["pitcher_k_pct"] = pitcher_row.get("k_percent")
        df["pitcher_whiff_pct"] = pitcher_row.get("whiff_percent")
        df["pitcher_barrel_allowed"] = pitcher_row.get("barrel_batted_rate")
        df["pitcher_hr"] = pitcher_row.get("home_run")
        df["pitcher_throws"] = pitcher_row.get("p_throws", "")

    # Composite scores (real data only - NaN propagates)
    df["matchup"] = _score_from_weights(df, SCORING_WEIGHTS["matchup"])
    df["hr_form"] = _score_from_weights(df, SCORING_WEIGHTS["hr_form"])
    df["ceiling"] = _score_from_weights(df, SCORING_WEIGHTS["ceiling"])

    # Test Score = matchup × (PA sample weight). NaN matchup → NaN Test.
    if "pa" in df.columns:
        pa_factor = (df["pa"] / 150.0).clip(0.5, 1.0)
        df["test_score"] = (df["matchup"] * pa_factor).round(2)
    else:
        df["test_score"] = df["matchup"]

    # Zone Fit: only if real data exists
    if "xwobacon" in df.columns and df["xwobacon"].notna().any():
        base = _safe_pct_rank(df["xwobacon"]) / 100
        if "pitcher_xwoba" in df.columns and not df["pitcher_xwoba"].isna().all():
            p_factor = ((df["pitcher_xwoba"] - 0.250) / 0.150).clip(0, 1)
            df["zone_fit"] = (base * 0.5 + p_factor * 0.5).round(3)
        else:
            df["zone_fit"] = base.round(3)
    else:
        df["zone_fit"] = np.nan

    # kHR - real data only
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

    # Final display column order
    display_cols = [
        "player_id", "player_name", "lineup_pos", "position", "bats",
        "matchup", "test_score", "ceiling", "zone_fit",
        "hr_form", "hr_form_label", "hr_form_arrow", "kHR", "likely_hr_pct",
        "pa", "iso", "xwoba", "xwobacon", "xba", "xslg",
        "barrel_pct", "hard_hit", "sweet_spot_pct", "fb_pct",
        "la", "ev", "best_speed",
        "k_pct", "bb_pct", "whiff_pct",
        "home_run", "pulled_brl_pct", "pull_pct", "oppo_pct",
        "pull_air_pct", "oppo_air_pct", "sprint_speed",
        "recent_hr", "recent_iso", "recent_avg", "recent_k_pct",
        "pitcher_xwoba", "pitcher_k_pct", "pitcher_whiff_pct",
        "pitcher_barrel_allowed", "pitcher_hr", "pitcher_throws",
    ]
    keep = [c for c in display_cols if c in df.columns]
    return df[keep]


# ===========================================================================
# Pitcher slate
# ===========================================================================

def build_pitcher_slate(
    slate: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    pitcher_recent: dict | None = None,
) -> pd.DataFrame:
    """One row per starting pitcher with composite scores + recent form."""
    pitchers = []
    for _, g in slate.iterrows():
        for side in ("away", "home"):
            pid = g.get(f"{side}_pitcher_id")
            if pid is None or pd.isna(pid):
                continue
            row = pitcher_stats[pitcher_stats["player_id"] == pid]
            base = {
                "pitcher_id": pid,
                "pitcher_name": g.get(f"{side}_pitcher"),
                "team": g.get(f"{side}_team_abbr"),
                "opp": g.get(f"{'home' if side == 'away' else 'away'}_team_abbr"),
                "home_away": "@" if side == "away" else "vs",
                "game_pk": g.get("gamePk"),
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
                })
            if pitcher_recent and pid in pitcher_recent:
                base.update(pitcher_recent[pid])
            pitchers.append(base)

    df = pd.DataFrame(pitchers)
    if df.empty:
        return df

    # Composite scores - NaN propagates honestly
    if "k_pct" in df.columns:
        df["k_score"] = _safe_pct_rank(df["k_pct"])
    else:
        df["k_score"] = np.nan
    if "whiff_pct" in df.columns:
        df["whiff_score"] = _safe_pct_rank(df["whiff_pct"])
    else:
        df["whiff_score"] = np.nan
    if "xwoba_allowed" in df.columns:
        df["suppress_score"] = 100 - _safe_pct_rank(df["xwoba_allowed"])
    else:
        df["suppress_score"] = np.nan

    # Weight-scaled composite (rows missing all three → NaN, partial → scaled)
    def _composite(row, components):
        present = [(v, w) for v, w in components if pd.notna(v)]
        if not present:
            return np.nan
        total_w = sum(w for _, w in present)
        return sum(v * w for v, w in present) / total_w

    df["test_score"] = df.apply(lambda r: _composite(r, [
        (r["k_score"], 0.4), (r["whiff_score"], 0.3), (r["suppress_score"], 0.3),
    ]), axis=1).round(2)

    df["kHR"] = df.apply(lambda r: _composite(r, [
        (r["k_score"], 0.7), (r["whiff_score"], 0.3),
    ]), axis=1).round(2)

    # Estimate expected Ks today: K9 × 5.5 IP avg start. Only if real K/9.
    if "k9" in df.columns:
        df["proj_k"] = (df["k9"] * 5.5 / 9).round(1)
    else:
        df["proj_k"] = np.nan

    # Form arrow: recent ERA vs season ERA
    if "recent_era" in df.columns and "era" in df.columns:
        df["form_arrow"] = df.apply(
            lambda r: _form_arrow(-r.get("recent_era", 0), -r.get("era", 0))
            if pd.notna(r.get("recent_era")) else "→", axis=1
        )
    else:
        df["form_arrow"] = "→"

    return df
