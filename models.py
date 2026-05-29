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
        "exit_velocity_avg": "avg_ev",
        "launch_speed": "avg_ev",
        "sweet_spot_percent": "sweet_spot_pct",
        "flyballs_percent": "fb_pct",
        "groundballs_percent": "gb_pct",
        "linedrives_percent": "ld_pct",
        "whiff_percent": "whiff_pct",
        # Savant uses several names for launch angle
        "launch_angle": "la",
        "launch_angle_avg": "la",
        "avg_hit_angle": "la",
        "la_avg": "la",
        # Savant uses isolated_power for ISO in some exports
        "isolated_power": "iso",
        "iso_power": "iso",
        "pull_air_percent": "pull_air_pct",
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

    # If LA is still missing entirely but we have any launch_speed_angle data,
    # leave it blank (no derivation - it's a real measurement, not an identity)

    df["k_pct_inv"] = -df["k_pct"] if "k_pct" in df.columns else np.nan

    # Composite scores
    df["matchup"] = _score_from_weights(df, SCORING_WEIGHTS["matchup"])
    df["hr_form"] = _score_from_weights(df, SCORING_WEIGHTS["hr_form"])
    df["ceiling"] = _score_from_weights(df, SCORING_WEIGHTS["ceiling"])

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
    display_cols = [
        "player_id", "player_name", "lineup_pos", "position", "bats",
        "is_roster_fill",  # CRITICAL: flag for whether lineup_pos is real or fill
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
        "recent_hr_weighted_rate",
        "streak_label", "hr_streak_games", "hr_last_5", "games_since_hr",
        # Day/night splits (vs day games, vs night games)
        "vs_day_pa", "vs_day_avg", "vs_day_obp", "vs_day_slg", "vs_day_ops",
        "vs_day_hr_per_pa", "vs_day_k_percent",
        "vs_night_pa", "vs_night_avg", "vs_night_obp", "vs_night_slg", "vs_night_ops",
        "vs_night_hr_per_pa", "vs_night_k_percent",
        # Today's HR projection
        "likely_hr_pct",
    ]
    keep = [c for c in display_cols if c in df.columns]
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
    specs = [
        ("barrel_pct",     0.25, 2.0,  25.0),    # Elite raised: top is ~22%, so 25 = nobody hits 100
        ("iso",            0.20, 0.080, 0.350),  # Top is ~.330
        ("hard_hit",       0.10, 25.0, 65.0),    # Top is ~62
        ("avg_ev",         0.10, 86.0, 98.0),    # Top is ~96.5
        ("fb_pct",         0.08, 18.0, 50.0),    # 50 is rare
        ("pulled_brl_pct", 0.07, 0.5,  9.0),     # Top ~8
        ("slg",            0.07, 0.330, 0.620),  # Top ~.610
        ("recent_iso",     0.05, 0.080, 0.350),
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
        # Launch angle - special "sweet spot" scoring (peak ~ 28°)
        if "la" in df.columns:
            la = row.get("la")
            if pd.notna(la):
                la_dist = abs(float(la) - 28.0)
                la_score = max(0, 100 - la_dist * 5)  # 0° away = 100; 20° away = 0
                total_score += la_score * 0.08
                total_weight += 0.08
        if total_weight == 0:
            return np.nan
        max_weight = sum(w for _, w, _, _ in specs) + 0.08
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
    # ROLE FLAG - season-aware, never auto-defaults to RELIEVER without evidence
    # ------------------------------------------------------------------
    def _role_flag(row):
        ip = row.get("ip")
        gs = row.get("games_started")
        gp = row.get("games_played")
        _rookie_raw = row.get("is_rookie", False)
        try:
            is_rookie = bool(_rookie_raw) if (_rookie_raw is not None and not pd.isna(_rookie_raw)) else False
        except (TypeError, ValueError):
            is_rookie = False
        # MLB's official primaryPosition: "SP" / "RP" / "P" / "TWP" / ""
        # When MLB says SP, trust it as a starter regardless of low season IP.
        mlb_position = (row.get("primary_position") or "").upper()
        # IL info — be defensive about pandas NA. `or 0` raises NAType.__bool__
        # error when the field is pd.NA, so we explicitly check for NA first.
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

        # Safe numeric values
        gs_n = float(gs) if (gs is not None and not pd.isna(gs)) else 0
        gp_n = float(gp) if (gp is not None and not pd.isna(gp)) else 0
        ip_n = float(ip) if (ip is not None and not pd.isna(ip)) else 0

        # CRITICAL: Every pitcher in p_slate IS a probable starter for tonight.
        # So we never need to flag them as a "RELIEVER" in the everyday sense.
        # The role classification here is about EXPECTED WORKLOAD tonight:
        #   ✓             = established starter, expect full game (5-6+ IP)
        #   🌱 NEW STARTER = rookie/recent recall with few starts (3-4 IP expected)
        #   🏥 RETURNING   = recently back from IL (4-5 IP, abundance of caution)
        #   🔄 SWING       = legit swing-man (3 IP expected, bullpen game)
        #   🚨 OPENER      = MLB designates RP but he's starting tonight (1-2 IP)
        #   ⚠️ LOW IP      = below-expected season IP, recent struggles or short leash

        # Case 1: MLB explicitly says this guy is an RP (not just generic P)
        # AND he's the probable starter → opener situation
        if mlb_position == "RP":
            return rookie_prefix + "🚨 OPENER"

        # Case 2: MLB explicitly says SP → trust that. He's a starter.
        # Just refine if returning from IL or rookie with limited starts.
        if mlb_position == "SP":
            if is_returning_starter and ip_n < min_ip * 0.8:
                return rookie_prefix + "🏥 RETURNING"
            if is_rookie and gs_n < min_gs:
                return "🌱 NEW STARTER"
            return rookie_prefix + "✓"

        # Case 3: No MLB SP/RP designation (just "P" or empty) — use heuristics.

        # SWING man: started SOME games but mostly relief (gs > 0 AND gp > gs * 1.5)
        # AND short outings (<3 IP per outing on average). This catches true
        # swing-men. Pure starters always have gp ≈ gs.
        if gs_n > 0 and gp_n > gs_n * 1.5:
            ip_per_outing = ip_n / gp_n if gp_n > 0 else 5.0
            if ip_per_outing < 3.0:
                return rookie_prefix + "🔄 SWING"

        # ZERO career starts this season AND has appeared in games → opener/RP role
        if gs_n == 0 and gp_n > 0:
            return rookie_prefix + "🚨 OPENER"

        # ZERO games at all → MLB debut tonight
        if gs_n == 0 and gp_n == 0:
            return "🌱 NEW STARTER"

        # HAS STARTS — they're a starter. Refine further:
        # - Returning from IL with low IP → 🏥 RETURNING
        # - Rookie or low-GS player → 🌱 NEW STARTER (Coleman Crow case)
        # - Low IP without IL → ⚠️ LOW IP
        # - Otherwise → ✓
        if is_returning_starter and ip_n < min_ip * 0.8:
            return rookie_prefix + "🏥 RETURNING"
        if is_rookie or gs_n < min_gs:
            # Rookie with any starts = NEW STARTER (not RELIEVER!)
            # Non-rookie with less than min_gs starts = also limited workload
            return "🌱 NEW STARTER"
        if ip_n < full_ip * 0.6:
            return rookie_prefix + "⚠️ LOW IP"
        return rookie_prefix + "✓" if rookie_prefix else "✓"

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
            if "🚨 OPENER" in str(role) or "🚨 RELIEVER" in str(role):
                # Opener typically goes 1-2 IP then a bulk reliever takes over
                base_ip = 1.5
            elif "🌱 NEW STARTER" in str(role):
                # Rookie/recent recall — short leash, expect 4-5 IP
                base_ip = 4.5
            elif "🏥 RETURNING" in str(role):
                # Returning from IL — abundance of caution, 4-5 IP
                base_ip = 4.5
            elif "🔄 BULK" in str(role):
                base_ip = 3.0
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
    full_gs = thresholds["full_gs"]

    def _role_flag(row):
        ip = row.get("ip")
        gs = row.get("games_started")
        gp = row.get("games_played")
        _rookie_raw = row.get("is_rookie", False)
        try:
            is_rookie = bool(_rookie_raw) if (_rookie_raw is not None and not pd.isna(_rookie_raw)) else False
        except (TypeError, ValueError):
            is_rookie = False
        mlb_position = (row.get("primary_position") or "").upper()
        # NA-safe IL info handling. `or 0` raises NAType.__bool__ on pd.NA.
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
        if (ip is None or pd.isna(ip)) and (gs is None or pd.isna(gs)):
            return "❔ NO DATA"
        rookie_prefix = "🌱 " if is_rookie else ""
        gs_n = float(gs) if (gs is not None and not pd.isna(gs)) else 0
        gp_n = float(gp) if (gp is not None and not pd.isna(gp)) else 0
        ip_n = float(ip) if (ip is not None and not pd.isna(ip)) else 0
        if mlb_position == "RP":
            return rookie_prefix + "🚨 OPENER"
        if mlb_position == "SP":
            if is_returning_starter and ip_n < min_ip * 0.8:
                return rookie_prefix + "🏥 RETURNING"
            if is_rookie and gs_n < min_gs:
                return "🌱 NEW STARTER"
            return rookie_prefix + "✓"
        if gs_n > 0 and gp_n > gs_n * 1.5:
            ip_per_outing = ip_n / gp_n if gp_n > 0 else 5.0
            if ip_per_outing < 3.0:
                return rookie_prefix + "🔄 SWING"
        if gs_n == 0 and gp_n > 0:
            return rookie_prefix + "🚨 OPENER"
        if gs_n == 0 and gp_n == 0:
            return "🌱 NEW STARTER"
        if is_returning_starter and ip_n < min_ip * 0.8:
            return rookie_prefix + "🏥 RETURNING"
        if is_rookie or gs_n < min_gs:
            return "🌱 NEW STARTER"
        if ip_n < full_ip * 0.6:
            return rookie_prefix + "⚠️ LOW IP"
        return rookie_prefix + "✓" if rookie_prefix else "✓"

    p_slate = p_slate.copy()
    p_slate["role"] = p_slate.apply(_role_flag, axis=1)
    return p_slate
