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
    return (s.rank(pct=True) * 100).fillna(50.0)


def _score_from_weights(df: pd.DataFrame, weights: dict, neg: tuple = ()) -> pd.Series:
    total = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col not in df.columns:
            continue
        ranked = _safe_pct_rank(df[col])
        if col in neg:
            ranked = 100 - ranked
        total = total + (ranked * w)
    return total.round(2)


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
        "launch_angle": "la",
        "pull_air_percent": "pull_air_pct",
    }
    df = df.rename(columns=rename)
    df["k_pct_inv"] = -df["k_pct"] if "k_pct" in df.columns else np.nan

    # Composite scores
    df["matchup"] = _score_from_weights(df, SCORING_WEIGHTS["matchup"])
    df["hr_form"] = _score_from_weights(df, SCORING_WEIGHTS["hr_form"])
    df["ceiling"] = _score_from_weights(df, SCORING_WEIGHTS["ceiling"])

    # Test Score - matchup discounted for sample size
    pa = df["pa"] if "pa" in df.columns else pd.Series([100] * len(df))
    pa_factor = np.clip(pa.fillna(0) / 150.0, 0.5, 1.0)
    df["test_score"] = (df["matchup"] * pa_factor).round(2)

    # Zone Fit: heuristic combining hitter xwOBAcon vs pitcher xwOBA allowed,
    # plus pitch-mix match. Will be ~0.0 - 0.3 range like screenshots.
    if "xwobacon" in df.columns:
        base = _safe_pct_rank(df["xwobacon"]) / 100  # 0-1
        if "pitcher_xwoba" in df.columns and not df["pitcher_xwoba"].isna().all():
            p_factor = (df["pitcher_xwoba"].fillna(0.310) - 0.250) / 0.150
            p_factor = p_factor.clip(0, 1)
            df["zone_fit"] = (base * 0.5 + p_factor * 0.5).round(3)
        else:
            df["zone_fit"] = base.round(3)
    else:
        df["zone_fit"] = np.nan

    # kHR - hitter's expected K rate against this pitcher today
    # ((hitter K%) + (pitcher K%)) / 2, percentile-ranked inverse (lower K = higher score)
    if "k_pct" in df.columns and "pitcher_k_pct" in df.columns:
        df["k_combined"] = (df["k_pct"].fillna(22) + df["pitcher_k_pct"].fillna(22)) / 2
        df["kHR"] = (100 - _safe_pct_rank(df["k_combined"])).round(2)
    else:
        df["kHR"] = np.nan

    # HR Form arrow (compare recent_iso to season iso, e.g.)
    if "recent_iso" in df.columns and "iso" in df.columns:
        df["hr_form_arrow"] = df.apply(
            lambda r: _form_arrow(r.get("recent_iso"), r.get("iso")), axis=1
        )
        # Combined HR_Form_pct: keep score 0-100 + arrow
        df["hr_form_label"] = df.apply(
            lambda r: f"{r['hr_form']:.0f}% {r['hr_form_arrow']}", axis=1
        )
    else:
        df["hr_form_arrow"] = "→"
        df["hr_form_label"] = df["hr_form"].apply(lambda x: f"{x:.0f}% →" if pd.notna(x) else "—")

    # "Likely HR%" - quick per-PA HR rate estimate (used in screenshot's "Likely" column)
    if "barrel_pct" in df.columns and "fb_pct" in df.columns:
        # ~ (BBE rate) * (barrel rate) * (HR per barrel ~ 75%) but normalize
        df["likely_hr_pct"] = (
            (df["barrel_pct"].fillna(7) * df["fb_pct"].fillna(35) / 100) * 0.75
        ).round(2)
    elif "barrel_pct" in df.columns:
        df["likely_hr_pct"] = (df["barrel_pct"].fillna(7) * 0.35).round(2)
    else:
        df["likely_hr_pct"] = np.nan

    # Pitches & BIP estimates per PA (for table display - rough season totals)
    if "pa" in df.columns:
        df["pitches"] = (df["pa"] * 3.9).fillna(0).astype(int)  # MLB avg
        df["bip"] = (df["pa"] * 0.65).fillna(0).astype(int)

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
                })
            if pitcher_recent and pid in pitcher_recent:
                base.update(pitcher_recent[pid])
            pitchers.append(base)

    df = pd.DataFrame(pitchers)
    if df.empty:
        return df

    df["k_score"] = _safe_pct_rank(df.get("k_pct", pd.Series([50.0] * len(df))))
    df["whiff_score"] = _safe_pct_rank(df.get("whiff_pct", pd.Series([50.0] * len(df))))
    df["suppress_score"] = 100 - _safe_pct_rank(
        df.get("xwoba_allowed", pd.Series([0.32] * len(df)))
    )
    df["test_score"] = (
        df["k_score"] * 0.4 + df["whiff_score"] * 0.3 + df["suppress_score"] * 0.3
    ).round(2)
    df["kHR"] = (df["k_score"] * 0.7 + df["whiff_score"] * 0.3).round(2)

    # Estimate expected Ks today: K9 × 5.5 IP avg start
    if "k9" in df.columns:
        df["proj_k"] = (df["k9"].fillna(8) * 5.5 / 9).round(1)

    # Form arrow: recent ERA vs season ERA
    if "recent_era" in df.columns and "era" in df.columns:
        df["form_arrow"] = df.apply(
            lambda r: _form_arrow(-r.get("recent_era", 0), -r.get("era", 0))
            if pd.notna(r.get("recent_era")) else "→", axis=1
        )
    else:
        df["form_arrow"] = "→"

    return df.sort_values("test_score", ascending=False).reset_index(drop=True)
