"""
models.py
==========
Composite scoring engine for hitter matchups and pitcher slate.
Real data only - NaN propagates honestly when stats are missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SCORING_WEIGHTS = {
    "matchup": {
        "iso": 0.15,
        "xwoba": 0.15,
        "barrel_pct": 0.20,
        "hard_hit": 0.10,
        "sweet_spot_pct": 0.05,
        "fb_pct": 0.10,
        "xwobacon": 0.15,
        "pitcher_xwoba": 0.10,
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
        "avg_ev": 0.15,
    },
}

NEG_COLS = ("k_pct", "whiff_pct")


def _safe_pct_rank(s: pd.Series) -> pd.Series:
    """Percentile rank 0-100. NaNs stay NaN."""
    return s.rank(pct=True) * 100


def _score_from_weights(df: pd.DataFrame, weights: dict, neg: tuple = NEG_COLS) -> pd.Series:
    """Weighted sum of percentile-ranked columns. Honest NaN when no data."""
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
    if pd.isna(recent) or pd.isna(season) or season == 0:
        return "→"
    diff = (recent - season) / abs(season)
    if diff > threshold:
        return "↑"
    if diff < -threshold:
        return "↓"
    return "→"


def build_matchup_table(
    lineup: list,
    pitcher_row: pd.Series | None,
    hitter_stats: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    recent_form_dict: dict | None = None,
    pitcher_arsenal_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the matchup table from lineup + season stats."""
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
        if recent_form_dict and p.get("id") in recent_form_dict:
            for k, v in recent_form_dict[p["id"]].items():
                row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)

    # Pitcher context (same value for whole lineup)
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

    # Normalize column names - handle Savant's column-name inconsistencies
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
        # Launch angle - Savant uses several variants
        "launch_angle": "la",
        "launch_angle_avg": "la",
        "avg_hit_angle": "la",
        # ISO - Savant uses isolated_power in CSV exports
        "isolated_power": "iso",
        "pull_air_percent": "pull_air_pct",
    }
    df = df.rename(columns=rename)

    # Dedupe if rename created duplicates
    if isinstance(df.columns, pd.Index) and df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # Derive ISO from SLG-AVG (or xSLG-xBA) if still missing
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

    if "pa" in df.columns:
        pa_factor = (df["pa"] / 150.0).clip(0.5, 1.0)
        df["test_score"] = (df["matchup"] * pa_factor).round(2)
    else:
        df["test_score"] = df["matchup"]

    # Zone Fit
    if "xwobacon" in df.columns and df["xwobacon"].notna().any():
        base = _safe_pct_rank(df["xwobacon"]) / 100
        if "pitcher_xwoba" in df.columns and not df["pitcher_xwoba"].isna().all():
            p_factor = ((df["pitcher_xwoba"] - 0.250) / 0.150).clip(0, 1)
            df["zone_fit"] = (base * 0.5 + p_factor * 0.5).round(3)
        else:
            df["zone_fit"] = base.round(3)
    else:
        df["zone_fit"] = np.nan

    # kHR
    if "k_pct" in df.columns and "pitcher_k_pct" in df.columns:
        df["k_combined"] = (df["k_pct"] + df["pitcher_k_pct"]) / 2
        df["kHR"] = (100 - _safe_pct_rank(df["k_combined"])).round(2)
    else:
        df["kHR"] = np.nan

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

    if "barrel_pct" in df.columns and "fb_pct" in df.columns:
        df["likely_hr_pct"] = ((df["barrel_pct"] * df["fb_pct"] / 100) * 0.75).round(2)
    elif "barrel_pct" in df.columns:
        df["likely_hr_pct"] = (df["barrel_pct"] * 0.35).round(2)
    else:
        df["likely_hr_pct"] = np.nan

    display_cols = [
        "player_id", "player_name", "lineup_pos", "position", "bats",
        "matchup", "test_score", "ceiling", "zone_fit",
        "hr_form", "hr_form_label", "hr_form_arrow", "kHR", "likely_hr_pct",
        "pa", "iso", "xwoba", "xwobacon", "xba", "xslg",
        "barrel_pct", "hard_hit", "sweet_spot_pct", "fb_pct",
        "la", "ev", "avg_ev",
        "k_pct", "bb_pct", "whiff_pct",
        "home_run", "pulled_brl_pct", "pull_pct", "oppo_pct",
        "pull_air_pct", "oppo_air_pct", "sprint_speed",
        "recent_hr", "recent_iso", "recent_avg", "recent_k_pct",
        "pitcher_xwoba", "pitcher_k_pct", "pitcher_whiff",
        "pitcher_barrel_allowed", "pitcher_hr",
    ]
    keep = [c for c in display_cols if c in df.columns]
    return df[keep]


def build_pitcher_slate(
    slate: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    pitcher_recent: dict | None = None,
) -> pd.DataFrame:
    """One row per starting pitcher with composite scores."""
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

    if "k9" in df.columns:
        df["proj_k"] = (df["k9"] * 5.5 / 9).round(1)
    else:
        df["proj_k"] = np.nan

    if "recent_era" in df.columns and "era" in df.columns:
        df["form_arrow"] = df.apply(
            lambda r: _form_arrow(-r.get("recent_era", 0), -r.get("era", 0))
            if pd.notna(r.get("recent_era")) else "→", axis=1
        )
    else:
        df["form_arrow"] = "→"

    return df
