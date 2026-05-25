"""
app.py
=======
Posey MLB HR & K Data dashboard - Streamlit main entry.
"""

from __future__ import annotations

import os
import re
import math
import json
import datetime as dt
from datetime import date, datetime, timedelta
from typing import Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

# Core imports - make each one defensive so a single missing function
# doesn't kill the whole app
try:
    from data_fetcher import get_slate
except ImportError:
    def get_slate(*a, **k): return pd.DataFrame()

try:
    from data_fetcher import get_lineup
except ImportError:
    def get_lineup(*a, **k): return []

try:
    from data_fetcher import get_team_roster
except ImportError:
    def get_team_roster(*a, **k): return []

try:
    from data_fetcher import get_hitter_stats
except ImportError:
    def get_hitter_stats(*a, **k): return pd.DataFrame()

try:
    from data_fetcher import get_pitcher_stats
except ImportError:
    def get_pitcher_stats(*a, **k): return pd.DataFrame()

try:
    from data_fetcher import get_pitcher_arsenal
except ImportError:
    def get_pitcher_arsenal(*a, **k): return pd.DataFrame()

try:
    from data_fetcher import get_pitcher_recent_form
except ImportError:
    def get_pitcher_recent_form(*a, **k): return {}

try:
    from data_fetcher import get_hitter_recent_form_trad
except ImportError:
    def get_hitter_recent_form_trad(*a, **k): return {}

try:
    from data_fetcher import get_sprint_speed
except ImportError:
    def get_sprint_speed(*a, **k): return pd.DataFrame()

# Optional newer functions added in updated data_fetcher.py
try:
    from data_fetcher import get_hitter_traditional, get_pitcher_traditional_safe as get_pitcher_traditional
    HAVE_TRADITIONAL = True
except ImportError:
    try:
        from data_fetcher import get_hitter_traditional, get_pitcher_traditional
        HAVE_TRADITIONAL = True
    except ImportError:
        HAVE_TRADITIONAL = False
        def get_hitter_traditional(*a, **k): return pd.DataFrame()
        def get_pitcher_traditional(*a, **k): return pd.DataFrame()

try:
    from data_fetcher import fill_pitcher_stats_for_slate, fill_hitter_bats
    HAVE_FILL_HELPERS = True
except ImportError:
    HAVE_FILL_HELPERS = False
    def fill_pitcher_stats_for_slate(p, s, season=None): return p
    def fill_hitter_bats(lineups, ids=None): return {}
from models import build_matchup_table, build_pitcher_slate
from sleepers import hr_probability, find_sleepers, grand_slam_probability
# Props - core functions are required, verdict_color is optional
from props import hr_prob_per_pa, k_total_projection

# hr_prob_full_game may not exist in older props.py versions
try:
    from props import hr_prob_full_game
except ImportError:
    def hr_prob_full_game(prob_per_pa, expected_pa=4.2):
        """Fallback: P(>=1 HR in expected_pa) from per-PA prob."""
        if prob_per_pa is None:
            return None
        try:
            return float(1 - (1 - prob_per_pa) ** expected_pa)
        except Exception:
            return None
from park_factors import get_park, PARKS

# Optional newer helper
try:
    from park_factors import park_k_factor
except ImportError:
    def park_k_factor(venue_name): return 1.0
from weather import fetch_weather, hr_multiplier

try:
    from splits import bvp_for_lineup, hitter_vs_similar
    # Aliased for older app code that used the wrong names
    get_career_bvp_aggregate = bvp_for_lineup
    get_similar_arsenal_aggregate = hitter_vs_similar
    HAVE_SPLITS = True
except Exception:
    HAVE_SPLITS = False
    def get_career_bvp_aggregate(*a, **k): return None
    def get_similar_arsenal_aggregate(*a, **k): return None

try:
    from pitch_match import pitch_match_score, get_hitter_pitch_arsenal
    # Lazy-load hitter arsenal df once for the wrapper
    _hitter_arsenal_cache = {"df": None}
    def _get_hitter_arsenal():
        if _hitter_arsenal_cache["df"] is None:
            try:
                _hitter_arsenal_cache["df"] = get_hitter_pitch_arsenal()
            except Exception:
                _hitter_arsenal_cache["df"] = pd.DataFrame()
        return _hitter_arsenal_cache["df"]

    # Wrapper bridging old call signature in app.py to real function
    def pitch_match_score_for_hitter(batter_id, pitcher_row, pitcher_arsenal_df):
        """
        Old call: (batter_id, pitcher_row_dict, full_pitcher_arsenal_df)
        New call: (batter_id, single_pitcher_arsenal_df, single_hitter_arsenal_df)
        """
        try:
            # Filter the big pitcher arsenal df to just this pitcher
            p_id = pitcher_row.get("player_id") if hasattr(pitcher_row, "get") else None
            if p_id is None or pitcher_arsenal_df is None or pitcher_arsenal_df.empty:
                return {"pitch_match_score": None}
            this_pitcher_arsenal = pitcher_arsenal_df[
                pitcher_arsenal_df["player_id"] == p_id
            ] if "player_id" in pitcher_arsenal_df.columns else pd.DataFrame()
            if this_pitcher_arsenal.empty:
                return {"pitch_match_score": None}
            # Filter hitter arsenal to this batter
            hit_arsenal_df = _get_hitter_arsenal()
            this_hitter_arsenal = hit_arsenal_df[
                hit_arsenal_df["player_id"] == batter_id
            ] if not hit_arsenal_df.empty and "player_id" in hit_arsenal_df.columns else pd.DataFrame()
            return pitch_match_score(batter_id, this_pitcher_arsenal, this_hitter_arsenal)
        except Exception:
            return {"pitch_match_score": None}
    HAVE_PITCH_MATCH = True
except Exception:
    HAVE_PITCH_MATCH = False
    def pitch_match_score_for_hitter(*a, **k): return {"pitch_match_score": None}

try:
    from game_context import get_umpire_for_game, get_catcher_framing, get_vegas_totals
    # Aliases for app code using old names
    def get_catcher_framing_for_game(*a, **k):
        # Old call signature was (game_pk); new is (season).
        # The framing data doesn't actually vary game-by-game, so just return season data.
        try:
            return get_catcher_framing()
        except Exception:
            return {}
    def get_vegas_for_game(game_pk, *a, **k):
        try:
            from datetime import date
            totals = get_vegas_totals(date.today().isoformat())
            if totals is None or (hasattr(totals, "empty") and totals.empty):
                return None
            # Try to find a row matching this game
            if "gamePk" in totals.columns:
                m = totals[totals["gamePk"] == game_pk]
                return m.iloc[0].to_dict() if len(m) else None
            return None
        except Exception:
            return None
    HAVE_GAME_CONTEXT = True
except Exception:
    HAVE_GAME_CONTEXT = False
    def get_umpire_for_game(*a, **k): return {}
    def get_catcher_framing_for_game(*a, **k): return {}
    def get_vegas_for_game(*a, **k): return None


# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Posey MLB HR & K Data",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


def safe_int(val) -> Optional[int]:
    if val is None or pd.isna(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def safe_float(val) -> Optional[float]:
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def pa_threshold_for_date(d: date) -> int:
    """PA threshold scales with how deep into the season we are."""
    month = d.month
    if month <= 4:
        return 40
    if month == 5:
        return 80
    if month == 6:
        return 120
    if month == 7:
        return 160
    return 200


def hr_verdict(hr_game_pct, sample_size=None, pa_threshold=80):
    """Inline tier for HR Game% display."""
    if hr_game_pct is None or pd.isna(hr_game_pct):
        return ""
    if sample_size is not None and not pd.isna(sample_size) and sample_size < pa_threshold:
        return "⚠️ SMALL"
    if hr_game_pct >= 25:
        return "🔥 ELITE"
    if hr_game_pct >= 18:
        return "✅ STRONG"
    if hr_game_pct >= 12:
        return "📊 SOLID"
    if hr_game_pct >= 5:
        return "💤 WEAK"
    return "❌ AVOID"


def hr_signal_emoji(hr_game_pct, sample_size=None, pa_threshold=80):
    """Single emoji for the Signal column - hitters."""
    if hr_game_pct is None or pd.isna(hr_game_pct):
        return "⚪"
    if sample_size is not None and not pd.isna(sample_size) and sample_size < pa_threshold:
        return "⚪"
    if hr_game_pct >= 22:
        return "🟢"
    if hr_game_pct >= 14:
        return "🟡"
    if hr_game_pct >= 7:
        return "🟠"
    return "🔴"


def pitcher_signal_emoji(test_score, sample_size=None, pa_threshold=80):
    """Single emoji for the Signal column - pitchers."""
    if test_score is None or pd.isna(test_score):
        return "⚪"
    if sample_size is not None and not pd.isna(sample_size) and sample_size < pa_threshold:
        return "⚪"
    if test_score >= 65:
        return "🟢"
    if test_score >= 45:
        return "🟡"
    if test_score >= 30:
        return "🟠"
    return "🔴"



# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚾ Posey MLB Props")
    selected_date = st.date_input("Slate date", value=date.today())

    st.subheader("Data sources")
    use_recent_form = st.checkbox("Recent form (L15)", value=True)
    use_pitch_match = st.checkbox("Pitch match score", value=HAVE_PITCH_MATCH)
    use_bvp = st.checkbox("Batter vs Pitcher history", value=HAVE_SPLITS)
    use_weather = st.checkbox("Weather + park factors", value=True)
    use_ump = st.checkbox("Umpire + catcher framing", value=HAVE_GAME_CONTEXT)
    use_vegas = st.checkbox("Vegas totals", value=HAVE_GAME_CONTEXT)
    # Sprint speed has no impact on HR or K projections - removed from UI.
    # If you want speed-based metrics (steals, infield hits) in the future, re-enable.
    use_sprint_speed = False

    st.subheader("Display")
    show_diagnostic = st.checkbox("Show data diagnostic", value=False)
    show_legend = st.checkbox("Show legend / glossary", value=True)

    st.divider()
    if st.button("🔄 Force refresh all data", help="Clears the data cache and re-fetches from APIs. Use if you see stale or missing data."):
        st.cache_data.clear()
        st.rerun()
    st.caption("Real data only - empty cells mean we couldn't fetch that stat.")

    st.subheader("PA threshold")
    _auto_pa = pa_threshold_for_date(selected_date)
    pa_mode = st.radio(
        "Minimum PA to show in main table",
        ["Auto (season-aware)", "Custom"],
        index=0,
        help=(
            "Auto scales with time of season: Apr 40 / May 80 / Jun 120 / "
            "Jul 160 / Aug+ 200. Hitters below threshold show in 'Insufficient "
            "Sample' section. Use Custom to include more players (call-ups, "
            "platoon guys) you want to project HRs for."
        ),
    )
    if pa_mode == "Custom":
        INSUFFICIENT_PA_THRESHOLD = st.number_input(
            "Custom PA threshold", min_value=1, max_value=400,
            value=_auto_pa, step=10,
        )
    else:
        INSUFFICIENT_PA_THRESHOLD = _auto_pa
        st.caption(f"Auto threshold for {selected_date.strftime('%b %Y')}: **{_auto_pa} PA**")

    st.subheader("📈 Backtest")
    try:
        from backtest import list_snapshots
        existing = list_snapshots()
        if existing:
            st.caption(f"Snapshots saved: {len(existing)} (most recent: {existing[-1]})")
        else:
            st.caption("No snapshots yet. Save today's projections via button below the slate.")
    except Exception:
        st.caption("Backtest module unavailable")
    show_backtest = st.checkbox("Show backtest panel", value=False,
                                  help="See accuracy of past projections vs actual outcomes.")


# ============================================================================
# DATA LOAD
# ============================================================================

with st.spinner("Loading slate and stats..."):
    slate = get_slate(selected_date.isoformat())

if slate.empty:
    st.warning(f"No MLB games found on {selected_date}. Try another date.")
    st.stop()

# Statcast pulls
hitter_stats = get_hitter_stats() if not slate.empty else pd.DataFrame()
pitcher_stats = get_pitcher_stats() if not slate.empty else pd.DataFrame()
pitcher_arsenal_all = get_pitcher_arsenal() if not slate.empty else pd.DataFrame()

# Traditional stats
hitter_trad = get_hitter_traditional() if not slate.empty else pd.DataFrame()
pitcher_trad = get_pitcher_traditional() if not slate.empty else pd.DataFrame()

# Merge traditional stats - force player_id types to match
# CRITICAL: use combine_first style merge so we never overwrite Savant values
# with trad-API NaN. Only fill in where Savant data is missing.
if not hitter_trad.empty and "player_id" in hitter_stats.columns:
    hitter_stats["player_id"] = pd.to_numeric(hitter_stats["player_id"], errors="coerce").astype("Int64")
    hitter_trad["player_id"] = pd.to_numeric(hitter_trad["player_id"], errors="coerce").astype("Int64")
    drop = [c for c in ["player_name"] if c in hitter_trad.columns]
    # Use _trad suffix and then manually coalesce afterward
    hitter_stats = hitter_stats.merge(
        hitter_trad.drop(columns=drop, errors="ignore"),
        on="player_id", how="left", suffixes=("", "_trad"),
    )
    # Coalesce: keep Savant value, fall back to trad if Savant is NaN
    for col in ["obp", "slg", "ops", "avg", "home_run", "rbi", "runs", "sb"]:
        trad_col = f"{col}_trad"
        if trad_col in hitter_stats.columns:
            if col in hitter_stats.columns:
                hitter_stats[col] = hitter_stats[col].fillna(hitter_stats[trad_col])
            else:
                hitter_stats[col] = hitter_stats[trad_col]
            hitter_stats = hitter_stats.drop(columns=[trad_col])

if not pitcher_trad.empty and "player_id" in pitcher_stats.columns:
    pitcher_stats["player_id"] = pd.to_numeric(pitcher_stats["player_id"], errors="coerce").astype("Int64")
    pitcher_trad["player_id"] = pd.to_numeric(pitcher_trad["player_id"], errors="coerce").astype("Int64")
    drop = [c for c in ["player_name"] if c in pitcher_trad.columns]
    pitcher_stats = pitcher_stats.merge(
        pitcher_trad.drop(columns=drop, errors="ignore"),
        on="player_id", how="left", suffixes=("", "_trad"),
    )
    # Coalesce era, whip, ip, hr9, k9 from trad if Savant didn't provide
    for col in ["era", "whip", "ip", "hr9", "k9", "bb9"]:
        trad_col = f"{col}_trad"
        if trad_col in pitcher_stats.columns:
            if col in pitcher_stats.columns:
                pitcher_stats[col] = pitcher_stats[col].fillna(pitcher_stats[trad_col])
            else:
                pitcher_stats[col] = pitcher_stats[trad_col]
            pitcher_stats = pitcher_stats.drop(columns=[trad_col])

# If OBP/SLG still missing after both sources, derive from Statcast components:
# OPS we have; OBP ≈ (hits + walks) / PA fallback; SLG = ISO + AVG
if "slg" in hitter_stats.columns and hitter_stats["slg"].isna().all():
    if "iso" in hitter_stats.columns and "avg" in hitter_stats.columns:
        hitter_stats["slg"] = (hitter_stats["iso"] + hitter_stats["avg"]).round(3)
if "obp" in hitter_stats.columns and hitter_stats["obp"].isna().all():
    # No clean derivation available - leave NaN, will auto-hide
    pass

# Backstop: derive ISO from SLG - AVG if Statcast didn't supply it
if "iso" not in hitter_stats.columns or hitter_stats["iso"].isna().all():
    if "slg" in hitter_stats.columns and "avg" in hitter_stats.columns:
        hitter_stats["iso"] = (hitter_stats["slg"] - hitter_stats["avg"]).round(3)

# Per-pitcher fallback: fetch stats individually for any starter still missing data
with st.spinner("Filling in missing pitcher data..."):
    try:
        pitcher_stats = fill_pitcher_stats_for_slate(pitcher_stats, slate)
    except Exception as e:
        st.warning(f"Per-pitcher fill skipped: {e}")

# Per-hitter launch angle fill: pulls REAL LA from Statcast search for any
# hitter on today's slate missing it. Cached 24h per player so the cost is
# only paid once per day.
if not hitter_stats.empty:
    needs_la = ("launch_angle" not in hitter_stats.columns
                or hitter_stats["launch_angle"].isna().all())
    if needs_la:
        try:
            from data_fetcher import fill_hitter_la_for_slate
            with st.spinner("Fetching real launch angles from Statcast (one-time, cached 24h)..."):
                hitter_stats = fill_hitter_la_for_slate(hitter_stats, slate)
        except Exception as e:
            st.warning(f"LA fill skipped: {e}")

# Last-resort: if IP is still completely missing for everyone, estimate from
# Statcast PA so the model has *something* to work with (≈4.3 PA per IP)
if not pitcher_stats.empty:
    if "ip" not in pitcher_stats.columns or pitcher_stats["ip"].isna().all():
        if "pa" in pitcher_stats.columns and pitcher_stats["pa"].notna().any():
            pitcher_stats["ip"] = (pitcher_stats["pa"] / 4.3).round(1)
    # Also if k9 is missing entirely, derive from k_percent
    if "k9" not in pitcher_stats.columns or pitcher_stats["k9"].isna().all():
        if "k_percent" in pitcher_stats.columns and pitcher_stats["k_percent"].notna().any():
            pitcher_stats["k9"] = (pitcher_stats["k_percent"] * 4.3 * 9 / 100).round(2)

# Rookie identification - fetch MLB debut year for all relevant player IDs
# A "rookie" is anyone whose debut year matches the current season.
try:
    from data_fetcher import fetch_player_debut_years
    from datetime import date as _date
    cur_year = _date.today().year
    rookie_ids = set()
    if not pitcher_stats.empty and "player_id" in pitcher_stats.columns:
        pids = tuple(int(x) for x in pitcher_stats["player_id"].dropna().astype(int).tolist())
        if pids:
            debuts = fetch_player_debut_years(pids)
            pitcher_stats["debut_year"] = pitcher_stats["player_id"].map(debuts)
            pitcher_stats["is_rookie"] = pitcher_stats["debut_year"] == cur_year
            rookie_ids.update({pid for pid, yr in debuts.items() if yr == cur_year})
    if not hitter_stats.empty and "player_id" in hitter_stats.columns:
        h_pids = tuple(int(x) for x in hitter_stats["player_id"].dropna().astype(int).tolist())
        if h_pids:
            h_debuts = fetch_player_debut_years(h_pids)
            hitter_stats["debut_year"] = hitter_stats["player_id"].map(h_debuts)
            hitter_stats["is_rookie"] = hitter_stats["debut_year"] == cur_year
            rookie_ids.update({pid for pid, yr in h_debuts.items() if yr == cur_year})
except Exception:
    pass

# Sprint speed
if use_sprint_speed:
    try:
        sprint_df = get_sprint_speed()
        if not sprint_df.empty and "player_id" in sprint_df.columns and "player_id" in hitter_stats.columns:
            sprint_cols = [c for c in ["player_id", "sprint_speed", "hp_to_1b"]
                            if c in sprint_df.columns]
            hitter_stats = hitter_stats.merge(
                sprint_df[sprint_cols], on="player_id", how="left", suffixes=("", "_sp"),
            )
    except Exception:
        pass


# ============================================================================
# HEADER + DATA AVAILABILITY
# ============================================================================

st.title(f"⚾ Posey MLB HR & K Data — {selected_date.strftime('%A, %B %d, %Y')}")

n_games = len(slate)
n_pitchers = pitcher_stats["k9"].notna().sum() if "k9" in pitcher_stats.columns else 0
n_hitters = hitter_stats["xwoba"].notna().sum() if "xwoba" in hitter_stats.columns else 0
hdr_col1, hdr_col2, hdr_col3, hdr_col4 = st.columns(4)
hdr_col1.metric("Games", n_games)
hdr_col2.metric("Pitchers w/ data", int(n_pitchers))
hdr_col3.metric("Hitters w/ data", int(n_hitters))
hdr_col4.metric("PA threshold", INSUFFICIENT_PA_THRESHOLD)

if show_backtest:
    with st.expander("📈 Backtest — projection accuracy vs actual outcomes", expanded=True):
        try:
            from backtest import (
                list_snapshots, load_snapshot,
                fetch_hitter_outcomes, fetch_pitcher_outcomes,
                evaluate_hitter_projections, evaluate_pitcher_projections,
            )
            snaps = list_snapshots()
            if not snaps:
                st.info(
                    "No snapshots saved yet. Below the slate is a 💾 Save Snapshot "
                    "button — click it after lineups lock. Come back tomorrow to "
                    "see how the predictions performed."
                )
            else:
                snap_choice = st.selectbox(
                    "Evaluate snapshot from:",
                    options=snaps,
                    index=len(snaps) - 1,
                    help="Pick a date to compare projections vs what actually happened.",
                )
                if st.button("🔍 Evaluate this snapshot"):
                    with st.spinner("Fetching actual outcomes..."):
                        snapshot = load_snapshot(snap_choice)
                        if not snapshot:
                            st.error(f"Could not load snapshot for {snap_choice}")
                        else:
                            hitter_actuals = fetch_hitter_outcomes(snap_choice)
                            pitcher_actuals = fetch_pitcher_outcomes(snap_choice)

                            h_metrics = evaluate_hitter_projections(snapshot, hitter_actuals)
                            p_metrics = evaluate_pitcher_projections(snapshot, pitcher_actuals)

                            st.markdown(f"### Results for {snap_choice}")

                            if h_metrics and not h_metrics.get("error"):
                                st.markdown("#### 🎯 Hitter projection accuracy")
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Hitters tracked", h_metrics.get("hitters_who_played", 0))
                                m2.metric("Total HRs", h_metrics.get("total_actual_hrs", 0))
                                m3.metric("Slate HR rate",
                                            f"{h_metrics.get('actual_hr_rate_pct', 0)}%")
                                t10_hr = h_metrics.get("top10_hr_hit_rate", 0)
                                m4.metric("Top-10 HR pick hit rate", f"{t10_hr}%")

                                # Show calibration table
                                bands = h_metrics.get("hr_pct_bands", [])
                                if bands:
                                    st.markdown("**HR Game% calibration** (predicted vs actual hit rate by band)")
                                    bdf = pd.DataFrame(bands)
                                    st.dataframe(bdf, hide_index=True, use_container_width=True,
                                                column_config={
                                                    "band": "Predicted band",
                                                    "n": "N hitters",
                                                    "predicted_rate": st.column_config.NumberColumn("Avg predicted", format="%.1f%%"),
                                                    "actual_rate": st.column_config.NumberColumn("Actual HR rate", format="%.1f%%"),
                                                    "calibration_error": st.column_config.NumberColumn("Error", format="%.1f"),
                                                })
                                    st.caption(
                                        "Calibration error near 0 = predictions match reality. "
                                        "Positive = model under-predicted (homered more than expected). "
                                        "Negative = model over-predicted."
                                    )

                                # Power Score bands
                                ps_bands = h_metrics.get("power_score_bands", [])
                                if ps_bands:
                                    st.markdown("**Power Score band accuracy**")
                                    psdf = pd.DataFrame(ps_bands)
                                    st.dataframe(psdf, hide_index=True, use_container_width=True)

                                # Top 10 detail
                                t10 = h_metrics.get("top10_hr_predictions", [])
                                if t10:
                                    st.markdown(f"**Top 10 HR predictions:** {h_metrics.get('top10_hr_hit_rate')}% hit rate")
                                    t10df = pd.DataFrame(t10)
                                    t10df["result"] = t10df["homered"].apply(
                                        lambda x: "✅ HR" if x else "—"
                                    )
                                    st.dataframe(t10df[["name", "predicted_hr_pct", "result"]],
                                                 hide_index=True, use_container_width=True)

                                # Sleeper accuracy
                                t10s = h_metrics.get("top10_sleeper_predictions", [])
                                if t10s:
                                    st.markdown(f"**Top 10 Sleeper picks:** {h_metrics.get('top10_sleeper_hit_rate')}% hit rate")
                                    t10sdf = pd.DataFrame(t10s)
                                    t10sdf["result"] = t10sdf["homered"].apply(
                                        lambda x: "✅ HR" if x else "—"
                                    )
                                    st.dataframe(t10sdf[["name", "sleeper_score", "result"]],
                                                 hide_index=True, use_container_width=True)

                            if p_metrics and not p_metrics.get("error"):
                                st.markdown("#### ⚾ Pitcher K projection accuracy")
                                pm1, pm2, pm3 = st.columns(3)
                                pm1.metric("Pitchers tracked", p_metrics.get("total_pitchers_matched", 0))
                                pm2.metric("K RMSE", f"{p_metrics.get('k_projection_rmse', 0):.2f}")
                                pm3.metric("K bias", f"{p_metrics.get('k_projection_bias', 0):+.2f}",
                                            help="Positive = model under-projected. Negative = over-projected.")

                                detail = p_metrics.get("k_projections_detail", [])
                                if detail:
                                    pdf = pd.DataFrame(detail)
                                    st.dataframe(pdf, hide_index=True, use_container_width=True,
                                                column_config={
                                                    "name": "Pitcher",
                                                    "projected_k": st.column_config.NumberColumn("Proj K", format="%.1f"),
                                                    "actual_k": "Actual K",
                                                    "actual_ip": st.column_config.NumberColumn("IP", format="%.1f"),
                                                    "diff": st.column_config.NumberColumn("Δ", format="%+.1f"),
                                                })
                            if not h_metrics and not p_metrics:
                                st.warning("Snapshot loaded but no actual outcomes matched. Possible if games haven't been played yet.")
        except Exception as e:
            st.error(f"Backtest panel error: {e}")

if show_diagnostic:
    with st.expander("🔬 Data diagnostic — column completeness"):
        if not hitter_stats.empty:
            st.markdown("**Hitter columns (% populated):**")
            avail = pd.DataFrame([
                {"column": c, "populated": int(hitter_stats[c].notna().sum()),
                 "pct": f"{hitter_stats[c].notna().sum() / len(hitter_stats) * 100:.0f}%"}
                for c in hitter_stats.columns
            ])
            st.dataframe(avail, hide_index=True, use_container_width=True)
        if not pitcher_stats.empty:
            st.markdown("**Pitcher columns (% populated):**")
            avail = pd.DataFrame([
                {"column": c, "populated": int(pitcher_stats[c].notna().sum()),
                 "pct": f"{pitcher_stats[c].notna().sum() / len(pitcher_stats) * 100:.0f}%"}
                for c in pitcher_stats.columns
            ])
            st.dataframe(avail, hide_index=True, use_container_width=True)

    # Live network probe - this is the critical diagnostic
    with st.expander("🌐 Live API connectivity probe (click to run)"):
        if st.button("Run probe"):
            import requests
            tests = [
                ("MLB Stats API schedule",
                 "https://statsapi.mlb.com/api/v1/schedule?sportId=1"),
                ("MLB Stats API season pitching (the failing one)",
                 "https://statsapi.mlb.com/api/v1/stats?stats=season&group=pitching&season=2025&sportIds=1&limit=10"),
                ("MLB Stats API single player (Skenes 694973)",
                 "https://statsapi.mlb.com/api/v1/people/694973/stats?stats=season&group=pitching&season=2025&sportId=1"),
                ("MLB Stats API people endpoint",
                 "https://statsapi.mlb.com/api/v1/people?personIds=694973"),
                ("Baseball Savant (control - we know this works)",
                 "https://baseballsavant.mlb.com/leaderboard/custom?year=2025&type=pitcher&filter=&min=q&selections=pa&csv=true"),
            ]
            results = []
            for name, url in tests:
                try:
                    r = requests.get(url, timeout=10,
                        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/csv, */*"})
                    status = r.status_code
                    size = len(r.content)
                    has_content = "yes" if size > 200 else f"empty ({size}b)"
                    results.append({"endpoint": name, "status": status, "size_bytes": size, "has_content": has_content})
                except Exception as e:
                    results.append({"endpoint": name, "status": "ERROR", "size_bytes": 0,
                                      "has_content": str(e)[:80]})
            st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
            st.caption(
                "If statsapi.mlb.com endpoints show errors/empty but Baseball Savant works, "
                "MLB Stats API is being blocked from Streamlit Cloud's IP range or "
                "rate-limited. If everything fails, it's a network issue. If everything "
                "works, the failure is happening elsewhere in the pipeline."
            )

    with st.expander("🎯 LA-specific diagnostic (click to find which Savant URL returns launch angle)"):
        st.caption(
            "Tests 5 different Savant endpoints and shows EXACTLY which columns "
            "each returns. We're looking for a 'launch_angle' or similar column "
            "with real numeric values."
        )
        if st.button("Probe for LA"):
            import requests
            la_tests = [
                ("Custom leaderboard - basic LA selection",
                 "https://baseballsavant.mlb.com/leaderboard/custom"
                 "?year=2025&type=batter&filter=&min=1"
                 "&selections=launch_angle,launch_speed,avg_hit_angle"
                 "&chart=false&x=pa&y=pa&r=no&csv=true"),
                ("Statcast leaderboard endpoint",
                 "https://baseballsavant.mlb.com/leaderboard/statcast"
                 "?type=batter&year=2025&position=&team=&min=1&csv=true"),
                ("Exit velocity & barrels leaderboard",
                 "https://baseballsavant.mlb.com/leaderboard/exit-velocity"
                 "?type=batter&year=2025&min=1&csv=true"),
                ("Statcast search - one player (Aaron Judge 592450)",
                 "https://baseballsavant.mlb.com/statcast_search/csv"
                 "?all=true&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&stadium=&hfBBL="
                 "&hfNewZones=&hfPull=&hfC=&hfSea=2025%7C&hfSit=&player_type=batter"
                 "&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA="
                 "&game_date_gt=2025-03-01&game_date_lt=2025-10-31"
                 "&batters_lookup%5B%5D=592450&team=&position=&hfRO="
                 "&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0"
                 "&min_results=0&group_by=name&sort_col=pitches"
                 "&player_event_sort=api_p_release_speed&sort_order=desc"
                 "&min_pas=0&type=details"),
                ("Player page CSV (Aaron Judge)",
                 "https://baseballsavant.mlb.com/savant-player/aaron-judge-592450"
                 "?stats=statcast-r-hitting-mlb"),
            ]
            for name, url in la_tests:
                st.markdown(f"**{name}**")
                try:
                    r = requests.get(url, timeout=20,
                        headers={"User-Agent": "Mozilla/5.0",
                                  "Accept": "application/json, text/csv, */*"})
                    st.write(f"Status: {r.status_code}, Size: {len(r.content)} bytes")
                    if r.status_code == 200 and len(r.content) > 200:
                        # Try parsing as CSV
                        try:
                            df_p = pd.read_csv(io.StringIO(r.text))
                            la_like = [c for c in df_p.columns
                                       if any(k in c.lower() for k in ["launch", "angle", "_la", "la_"])]
                            st.write(f"Rows: {len(df_p)}, Total columns: {len(df_p.columns)}")
                            st.write(f"**LA-like columns found:** `{la_like}`")
                            if la_like:
                                # Show sample values
                                sample = df_p[la_like].head(3)
                                st.dataframe(sample)
                                # Show stats - are these populated?
                                for col in la_like:
                                    coerced = pd.to_numeric(df_p[col], errors="coerce")
                                    n_pop = coerced.notna().sum()
                                    st.write(f"  `{col}`: {n_pop}/{len(df_p)} populated, "
                                             f"mean={coerced.mean():.2f}" if n_pop else
                                             f"  `{col}`: 0 populated")
                            else:
                                # Show first 10 column names so we can spot it
                                st.write(f"First 10 cols: {list(df_p.columns[:10])}")
                                st.write(f"All cols matching 'a': {[c for c in df_p.columns if 'a' in c.lower()][:15]}")
                        except Exception as parse_err:
                            st.write(f"Couldn't parse CSV: {parse_err}")
                            # Show first 500 chars of response
                            st.code(r.text[:500])
                    else:
                        st.write(f"❌ Failed or empty response")
                except Exception as e:
                    st.write(f"❌ Error: {e}")
                st.markdown("---")

if show_legend:
    with st.expander("📖 Legend & glossary"):
        leg1, leg2, leg3 = st.columns(3)
        with leg1:
            st.markdown("**Signal column (hitters)**")
            st.markdown(
                "🟢 Strong HR play (HR Game% ≥ 22%)\n\n"
                "🟡 Decent play (14-22%)\n\n"
                "🟠 Below average (7-14%)\n\n"
                "🔴 Avoid (< 7%)\n\n"
                "⚪ Insufficient sample"
            )
            st.markdown("**Role flags (pitchers)**")
            st.markdown(
                "✓ Established starter\n\n"
                "🔄 SWING — swing-man / spot starter / bulk relief between starts\n\n"
                "⚠️ LOW IP — under 25 IP this season\n\n"
                "🚨 RELIEVER — pure reliever / opener day\n\n"
                "📉 SAMPLE NOISE — stat inconsistency (low IP)"
            )
        with leg2:
            st.markdown("**Signal column (pitchers)**")
            st.markdown(
                "🟢 Strong K play (Test ≥ 65)\n\n"
                "🟡 Decent K (45-65)\n\n"
                "🟠 Average (30-45)\n\n"
                "🔴 Avoid (< 30)\n\n"
                "⚪ Insufficient data"
            )
            st.markdown("**Pitcher Test Score formula**")
            st.markdown(
                "Blended K/9 (30%) + Whiff% (20%) + xwOBA suppression (25%) "
                "+ ERA (15%) + base K% (10%), then × **reliability factor**. "
                "Relievers get 0.4 cap, low-IP/swing get scaled multiplier."
            )
        with leg3:
            st.markdown("**Key metrics**")
            st.markdown(
                "**HR Game%** = Probability of ≥1 HR this game (calibrated from barrel rate, pitcher, park, weather)\n\n"
                "**HR PA%** = Per-PA HR probability\n\n"
                "**Matchup** = 0-100 composite (xwoba, barrel, ISO, opp pitcher quality)\n\n"
                "**Test (hitter)** = Matchup × PA sample weight\n\n"
                "**Test (pitcher)** = K/Whiff/xwOBA/ERA composite × reliability\n\n"
                "**Conf** = Pitcher reliability multiplier (0.3 = unreliable, 1.0 = full)\n\n"
                "**Proj K** = Blended K/9 × expected IP (scaled by role)\n\n"
                "**Pitch Match** = Hitter pitch-specific xwOBA vs pitcher arsenal\n\n"
                "**Sleeper** = HR-prob percentile MINUS season-HR percentile\n\n"
                "**Pick Score** = Daily top-pick composite (all factors)"
            )

        # Concrete scale guide - what's "good" vs "bad" for each metric
        st.markdown("---")
        st.markdown("**📏 What's Good vs Bad — Real MLB Scales**")
        scale_col1, scale_col2 = st.columns(2)
        with scale_col1:
            st.markdown(
                "**HITTER metrics** (higher = better unless noted)\n\n"
                "| Metric | Poor | Avg | Good | Elite |\n"
                "|---|---|---|---|---|\n"
                "| Barrel% | <4 | 6-9 | 10-14 | 15+ |\n"
                "| ISO | <.110 | .140-.180 | .200-.260 | .280+ |\n"
                "| xwOBA | <.290 | .310-.340 | .350-.380 | .400+ |\n"
                "| Hard-Hit% | <30 | 35-45 | 48-55 | 58+ |\n"
                "| Exit Velo | <87 | 89-91 | 92-94 | 95+ |\n"
                "| HR Game% | <5% | 8-12% | 15-20% | 22-25% (~max) |\n"
                "| Power Score | <30 | 40-55 | 60-75 | 85-99 |\n"
                "| K% ⬇️ | 30+ | 22-28 | 17-22 | <17 |\n"
            )
        with scale_col2:
            st.markdown(
                "**PITCHER metrics** (HR-suppression: higher = better)\n\n"
                "| Metric | Poor | Avg | Good | Elite |\n"
                "|---|---|---|---|---|\n"
                "| ERA ⬇️ | 5.0+ | 4.0-4.5 | 3.0-3.7 | <2.7 |\n"
                "| WHIP ⬇️ | 1.45+ | 1.20-1.35 | 1.05-1.20 | <1.05 |\n"
                "| K/9 | <7 | 8.0-9.0 | 9.5-11 | 12+ |\n"
                "| HR/9 ⬇️ | 1.8+ | 1.1-1.4 | 0.7-1.0 | <0.7 |\n"
                "| xwOBA Allowed ⬇️ | .340+ | .310-.330 | .280-.305 | <.275 |\n"
                "| Barrel Allowed% ⬇️ | 10+ | 7-9 | 5-6 | <5 |\n"
                "| Test Score | <30 | 40-55 | 60-75 | 80+ |\n"
                "| HR Suppress | <50 | 55-70 | 75-85 | 90+ |\n"
                "| **Opp K%** | 18-20 | 21-23 | 24-26 | 27+ |\n"
                "| **Opp HR%** ⬇️ | 4.0+ | 2.5-3.5 | 1.8-2.4 | <1.8 |\n"
            )
        st.caption(
            "⬇️ = lower is better. The color coding in the tables uses these "
            "thresholds (red = poor, yellow = avg, green = elite). Opp K% and "
            "Opp HR% are colored from the PITCHER's perspective: high Opp K% "
            "is green (he'll get more strikeouts) and high Opp HR% is red "
            "(he's facing a power lineup)."
        )

st.divider()


# ============================================================================
# PITCHER SLATE OVERVIEW
# ============================================================================

st.subheader("🥎 Pitcher Slate Overview")
st.caption(
    "Role-aware scoring: relievers and short-sample pitchers get reliability-adjusted "
    "Test/kHR/Proj K so opener days don't dominate the rankings. "
    "Look for **🚨 RELIEVER** and **⚠️ LOW IP** flags — those scores are intentionally scaled down."
)

pitcher_recent_map = {}
if use_recent_form and not slate.empty:
    pids = set()
    for _, g in slate.iterrows():
        for col in ("away_pitcher_id", "home_pitcher_id"):
            pid = safe_int(g.get(col))
            if pid is not None:
                pids.add(pid)
    with st.spinner(f"Pulling recent form for {len(pids)} pitchers..."):
        for pid in pids:
            try:
                pitcher_recent_map[pid] = get_pitcher_recent_form(pid) or {}
            except Exception:
                pitcher_recent_map[pid] = {}

# Team hitting aggregates - lets pitcher K projections account for opponent K%
try:
    from data_fetcher import get_team_hitting_aggregates
    team_hit_df = get_team_hitting_aggregates()
    team_hit_map = {}
    if not team_hit_df.empty:
        # Index by both team_abbr and team_id for robust lookup
        for _, row in team_hit_df.iterrows():
            row_d = row.to_dict()
            if pd.notna(row.get("team_abbr")):
                team_hit_map[row["team_abbr"]] = row_d
            if pd.notna(row.get("team_id")):
                team_hit_map[int(row["team_id"])] = row_d
except Exception as e:
    team_hit_map = {}

try:
    p_slate = build_pitcher_slate(slate, pitcher_stats, pitcher_recent=pitcher_recent_map,
                                    slate_date=selected_date, team_hit_map=team_hit_map)
except TypeError:
    # Older models.py doesn't accept team_hit_map - call without it
    try:
        p_slate = build_pitcher_slate(slate, pitcher_stats, pitcher_recent=pitcher_recent_map,
                                        slate_date=selected_date)
    except TypeError:
        p_slate = build_pitcher_slate(slate, pitcher_stats, pitcher_recent=pitcher_recent_map)

# ----- DATA AVAILABILITY WARNING -----
# If MLB Stats API data is mostly missing, warn the user explicitly so they
# don't think the model is broken when it's actually a data fetch issue.
if not p_slate.empty:
    has_real_ip = "ip" in p_slate.columns and p_slate["ip"].notna().any()
    has_real_era = "era" in p_slate.columns and p_slate["era"].notna().any()
    has_real_k9 = "k9" in p_slate.columns and p_slate["k9"].notna().any()
    estimated_ip = bool(p_slate.get("ip_estimated", pd.Series([False])).any()) if "ip_estimated" in p_slate.columns else False
    estimated_k9 = bool(p_slate.get("k9_estimated", pd.Series([False])).any()) if "k9_estimated" in p_slate.columns else False

    issues = []
    if not has_real_era:
        issues.append("ERA / WHIP / HR/9 (MLB Stats API)")
    if estimated_ip:
        issues.append("IP (estimated from Statcast PA)")
    if estimated_k9:
        issues.append("K/9 (estimated from K%)")

    if issues:
        st.warning(
            "**⚠️ Some pitcher data couldn't be fetched from MLB Stats API.** "
            "Missing or estimated: " + ", ".join(issues) +
            ". Scores may be less accurate this load — refresh the page to retry, "
            "or it may resolve next refresh if MLB's API was rate-limiting."
        )

if not p_slate.empty:
    # Add signal emoji
    p_slate["alert"] = p_slate.apply(
        lambda r: pitcher_signal_emoji(r.get("test_score"), r.get("pa"),
                                          INSUFFICIENT_PA_THRESHOLD),
        axis=1,
    )

    # Add a visible warning flag column combining role + sample_noise
    if not p_slate.empty:
        def _warn_flag(r):
            flags = []
            if r.get("sample_noise"):
                flags.append("📉")
            return " ".join(flags) if flags else ""
        p_slate["warn"] = p_slate.apply(_warn_flag, axis=1)

    show_cols = [c for c in [
        "alert", "role", "warn", "pitcher_name", "team", "home_away", "opp", "throws",
        "test_score", "kHR", "hr_suppress", "proj_k", "form_arrow",
        "era", "whip", "k9", "bb9", "hr9",
        "ip", "games_started", "games_played", "ip_per_outing",
        "k_pct", "whiff_pct",
        "xwoba_allowed", "barrel_allowed",
        "opp_k_pct", "opp_hr_per_pa",
        "recent_era", "recent_k9", "days_rest", "avg_recent_pitches",
        "reliability",
    ] if c in p_slate.columns]

    # Auto-hide empty columns
    show_cols = [c for c in show_cols if p_slate[c].notna().any()]

    col_config = {
        "alert": st.column_config.TextColumn("Signal", width="small"),
        "role": st.column_config.TextColumn(
            "Role", width="small",
            help="✓ established starter · ⚠️ LOW IP · 🔄 SWING · 🚨 RELIEVER (test score scaled down)",
        ),
        "warn": st.column_config.TextColumn(
            "Flag", width="small",
            help="📉 = sample noise (ERA-WHIP mismatch or zero Statcast values at low IP)",
        ),
        "pitcher_name": st.column_config.TextColumn("Pitcher"),
        "team": st.column_config.TextColumn("Tm", width="small"),
        "home_away": st.column_config.TextColumn("", width="small"),
        "opp": st.column_config.TextColumn("Opp", width="small"),
        "throws": st.column_config.TextColumn("T", width="small"),
        "test_score": st.column_config.NumberColumn(
            "Test", format="%.1f",
            help="Percentile rank vs other pitchers on TODAY's slate (0-95). "
                 "Composite of K-blended (30%) + Whiff (20%) + xwOBA suppression (25%) "
                 "+ ERA (15%) + base K% (10%), × reliability factor × data-completeness penalty. "
                 "A score of 80+ means top of today's slate, not a probability.",
        ),
        "kHR": st.column_config.NumberColumn(
            "K Rating", format="%.1f",
            help="Strikeout-focused percentile rank vs today's slate (0-95). "
                 "Composite of K%, Whiff%, blended K/9 × reliability. "
                 "HIGH = top K play. Does NOT measure HR allowance — see HR Supp for that.",
        ),
        "hr_suppress": st.column_config.NumberColumn(
            "HR Supp", format="%.1f",
            help="HR-suppression percentile rank vs today's slate (0-95). "
                 "Composite of low barrel% allowed, low xwOBA allowed, low HR/9 × reliability. "
                 "HIGH = pitcher is hard to homer off of.",
        ),
        "proj_k": st.column_config.NumberColumn(
            "Proj K", format="%.1f",
            help="Baseline expected strikeouts: blended K/9 × expected IP. "
                 "Does NOT include opponent lineup adjustment — see the per-game "
                 "K Projections tab for opp-adjusted line probabilities.",
        ),
        "form_arrow": st.column_config.TextColumn("Trend", width="small"),
        "era": st.column_config.NumberColumn("ERA", format="%.2f"),
        "whip": st.column_config.NumberColumn("WHIP", format="%.2f"),
        "k9": st.column_config.NumberColumn("K/9", format="%.2f"),
        "bb9": st.column_config.NumberColumn("BB/9", format="%.2f"),
        "hr9": st.column_config.NumberColumn("HR/9", format="%.2f"),
        "ip": st.column_config.NumberColumn("IP", format="%.1f"),
        "games_started": st.column_config.NumberColumn("GS", format="%d", width="small"),
        "games_played": st.column_config.NumberColumn(
            "G", format="%d", width="small",
            help="Total games appeared in. If G > GS by a lot, pitcher has been used in relief between starts.",
        ),
        "ip_per_outing": st.column_config.NumberColumn(
            "IP/Out", format="%.1f", width="small",
            help="Average innings per appearance. Below 5 = short outings (opener / pulled early / reliever).",
        ),
        "k_pct": st.column_config.NumberColumn("K%", format="%.1f"),
        "whiff_pct": st.column_config.NumberColumn("Whiff%", format="%.1f"),
        "xwoba_allowed": st.column_config.NumberColumn("xwOBA", format="%.3f"),
        "barrel_allowed": st.column_config.NumberColumn("Brl%", format="%.1f"),
        "opp_k_pct": st.column_config.NumberColumn(
            "Opp K%", format="%.1f%%",
            help=(
                "Opposing team's season K rate (% of PAs ending in strikeout).\n"
                "League avg ~22.5%. Higher = pitcher should get more Ks.\n"
                "Range: 18% (very contact-heavy) to 28% (whiff-prone team)."
            ),
        ),
        "opp_hr_per_pa": st.column_config.NumberColumn(
            "Opp HR%", format="%.2f%%",
            help=(
                "Opposing team's HR rate per plate appearance (%).\n"
                "League avg ~2.8%. Higher = pitcher faces more power.\n"
                "Range: 1.5% (low-power lineup) to 4.5% (Dodgers/Yankees-type)."
            ),
        ),
        "recent_era": st.column_config.NumberColumn("L5 ERA", format="%.2f"),
        "recent_k9": st.column_config.NumberColumn("L5 K/9", format="%.2f"),
        "days_rest": st.column_config.NumberColumn("Rest"),
        "avg_recent_pitches": st.column_config.NumberColumn("Pitches"),
        "reliability": st.column_config.NumberColumn(
            "Conf", format="%.2f", width="small",
            help="Reliability factor (0.3 - 1.0). Multiplier on Test/kHR scores. Low = small sample / reliever / bulk relief role.",
        ),
    }
    # Color-code the pitcher table
    def _style_pitcher_df(df_in):
        if df_in is None or df_in.empty:
            return df_in
        specs = [
            # (col, poor, elite, higher_is_better)
            ("test_score",  30,    75,    True),
            ("kHR",         30,    80,    True),
            ("hr_suppress", 25,    75,    True),
            ("proj_k",      3,     8,     True),
            ("era",         5.5,   2.5,   False),  # Lower ERA = better (poor=5.5 → red, elite=2.5 → green)
            ("whip",        1.5,   1.0,   False),
            ("k9",          6.5,   11,    True),
            ("bb9",         4.5,   2.0,   False),
            ("hr9",         1.8,   0.7,   False),
            ("ip",          15,    50,    True),
            ("ip_per_outing", 3.5, 6.0,   True),
            ("k_pct",       18,    30,    True),
            ("whiff_pct",   18,    32,    True),
            ("xwoba_allowed", 0.350, 0.270, False),  # Lower = better
            ("barrel_allowed", 11,  4,    False),
            ("opp_k_pct",      18,  28,   True),    # For PITCHER perspective: higher opp K% = good
            ("opp_hr_per_pa",  4.0, 1.5,  False),   # For PITCHER perspective: lower opp HR rate = good
            ("recent_era",  5.5,   2.5,   False),
            ("recent_k9",   6.5,   11,    True),
            ("reliability", 0.4,   0.9,   True),
        ]
        def color_cell(val, poor, elite, hb):
            if val is None or pd.isna(val):
                return ""
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if hb:
                ratio = (v - poor) / (elite - poor) if elite != poor else 0.5
            else:
                ratio = (poor - v) / (poor - elite) if poor != elite else 0.5
            ratio = max(0, min(1, ratio))
            hue = ratio * 120
            return f"background-color: hsl({hue:.0f}, 60%, 40%); color: white;"
        styled = df_in.style
        for col, poor, elite, hb in specs:
            if col in df_in.columns:
                styled = styled.map(
                    lambda v, p=poor, e=elite, hh=hb: color_cell(v, p, e, hh),
                    subset=[col],
                )
        # Number formatting
        fmt = {
            "test_score": "{:.1f}", "kHR": "{:.1f}", "hr_suppress": "{:.1f}",
            "proj_k": "{:.1f}", "era": "{:.2f}", "whip": "{:.2f}",
            "k9": "{:.2f}", "bb9": "{:.2f}", "hr9": "{:.2f}", "ip": "{:.1f}",
            "ip_per_outing": "{:.1f}", "k_pct": "{:.1f}", "whiff_pct": "{:.1f}",
            "xwoba_allowed": "{:.3f}", "barrel_allowed": "{:.1f}",
            "opp_k_pct": "{:.1f}", "opp_hr_per_pa": "{:.2f}",
            "recent_era": "{:.2f}", "recent_k9": "{:.2f}", "reliability": "{:.2f}",
        }
        valid_fmt = {k: v for k, v in fmt.items() if k in df_in.columns}
        if valid_fmt:
            styled = styled.format(valid_fmt, na_rep="—")
        return styled

    st.dataframe(
        _style_pitcher_df(p_slate[show_cols]),
        hide_index=True, use_container_width=True,
        column_config=col_config,
    )

st.divider()
# ============================================================================
# GAME CONTEXT (park, weather, ump, vegas) - build per-game
# ============================================================================

game_context_map = {}
matchup_tables = {}

for _, game in slate.iterrows():
    gpk = int(game["gamePk"])

    # Park factor
    venue = game.get("venue") or ""
    park_mult = 1.0
    if use_weather:
        try:
            park_info = get_park(venue) or {}
            # Convert factor from 100=neutral scale to multiplier
            park_mult = (park_info.get("hr_factor", 100) / 100.0)
        except Exception:
            park_mult = 1.0

    # Weather
    weather = {}
    wx_mult = 1.0
    wx_summary = ""
    if use_weather and venue:
        try:
            park_info = get_park(venue)
            lat = park_info.get("lat")
            lon = park_info.get("lon")
            cf_bearing = park_info.get("cf_bearing", 0)
            if lat is None or lon is None:
                raise ValueError("missing coords")
            game_dt = game.get("gameTime")
            if isinstance(game_dt, pd.Timestamp):
                wx_iso = game_dt.isoformat()
            else:
                wx_iso = None
            weather = fetch_weather(lat, lon, wx_iso) or {}
            if weather.get("error"):
                wx_summary = f"Weather API error: {weather.get('error', '')[:50]}"
            else:
                wx_mult, _summary = hr_multiplier(weather, park_info)
                wx_summary = _summary or "Neutral"
                # Hard rain warning - >80% chance suggests likely delay/postponement
                pp = weather.get("precip_prob")
                if pp is not None and pp >= 80:
                    wx_summary = f"⚠️ HEAVY RAIN ({pp:.0f}%) — possible delay/PPD — " + wx_summary
        except Exception as e:
            weather = {"error": str(e)}
            wx_mult = 1.0
            wx_summary = f"Weather failed: {str(e)[:60]}"
    elif not use_weather:
        wx_summary = "Weather disabled"
    elif not venue:
        wx_summary = "No venue info"

    full_hr_mult = park_mult * wx_mult

    # Umpire + catcher framing
    ump = {}
    framing = {}
    if use_ump and HAVE_GAME_CONTEXT:
        try:
            ump = get_umpire_for_game(gpk) or {}
            framing = get_catcher_framing_for_game(gpk) or {}
        except Exception:
            ump = {}
            framing = {}

    # Vegas
    vegas_row = {}
    if use_vegas and HAVE_GAME_CONTEXT:
        try:
            vegas_row = get_vegas_for_game(
                game.get("away_team"), game.get("home_team")
            ) or {}
        except Exception:
            vegas_row = {}

    def _fill_to_nine(existing_lineup, team_id):
        """Pad lineup to 9 position players using active roster as fallback.

        Returns (lineup, is_confirmed) where is_confirmed = True only if
        MLB has actually posted the lineup (not roster-padded).

        IMPORTANT: When using roster fill, we sort by season PA (descending)
        so the most-used position players appear first. Alphabetical order
        (the API default) was producing nonsense lineups where e.g. Pete
        Alonso would show up "batting 9th" just because A comes after L.
        """
        is_confirmed = len(existing_lineup) >= 8  # At least 8 of 9 = real lineup
        if not team_id:
            return existing_lineup, is_confirmed
        existing_ids = {p.get("id") for p in existing_lineup if p.get("id")}
        try:
            roster = get_team_roster(int(team_id))
        except Exception:
            return existing_lineup, is_confirmed
        # Filter to position players only (skip P, SP, RP, TWP)
        position_players = [
            p for p in roster
            if p.get("position") and str(p.get("position")).upper() not in
                ("P", "SP", "RP", "TWP")
            and p.get("id") not in existing_ids
        ]
        # SORT by season PA so likely starters come first.
        # This makes the roster-fill "lineup" produce sensible expected_PA
        # estimates instead of alphabetical noise.
        pa_lookup = {}
        try:
            if not hitter_stats.empty and "player_id" in hitter_stats.columns:
                for _, hs_row in hitter_stats.iterrows():
                    pid = hs_row.get("player_id")
                    pa_val = hs_row.get("pa")
                    if pd.notna(pid) and pd.notna(pa_val):
                        try:
                            pa_lookup[int(pid)] = float(pa_val)
                        except (ValueError, TypeError):
                            continue
        except Exception:
            pass
        position_players.sort(
            key=lambda p: pa_lookup.get(int(p.get("id", 0)) if p.get("id") else 0, 0),
            reverse=True,
        )
        # Pad up to 9 total
        needed = max(0, 9 - len(existing_lineup))
        for p in position_players[:needed]:
            existing_lineup.append({
                "id": p.get("id"), "name": p.get("name"),
                "position": p.get("position"), "bats": p.get("bats"),
                "is_roster_fill": True,  # mark these as not real lineup positions
            })
        return existing_lineup, is_confirmed

    try:
        away_lineup = get_lineup(int(game["gamePk"]), "away")
    except Exception:
        away_lineup = []
    away_lineup, away_confirmed = _fill_to_nine(away_lineup, game.get("away_team_id"))

    try:
        home_lineup = get_lineup(int(game["gamePk"]), "home")
    except Exception:
        home_lineup = []
    home_lineup, home_confirmed = _fill_to_nine(home_lineup, game.get("home_team_id"))

    # Backfill batting handedness for any hitters missing it
    needs_bats_ids = set()
    for p in away_lineup + home_lineup:
        if p.get("id") and not p.get("bats"):
            try:
                needs_bats_ids.add(int(p["id"]))
            except (ValueError, TypeError):
                continue
    if needs_bats_ids:
        try:
            bats_map = fill_hitter_bats([], ids=needs_bats_ids)
            for p in away_lineup + home_lineup:
                pid = p.get("id")
                if pid in bats_map and not p.get("bats"):
                    p["bats"] = bats_map[pid]
        except Exception:
            pass

    away_p_id = safe_int(game.get("away_pitcher_id"))
    home_p_id = safe_int(game.get("home_pitcher_id"))
    away_p_row, home_p_row = {}, {}
    if away_p_id:
        rows = pitcher_stats[pitcher_stats["player_id"] == away_p_id]
        if len(rows):
            away_p_row = rows.iloc[0].to_dict()
    if home_p_id:
        rows = pitcher_stats[pitcher_stats["player_id"] == home_p_id]
        if len(rows):
            home_p_row = rows.iloc[0].to_dict()

    # TBD pitcher fallback - if a probable pitcher is missing or unknown,
    # use team-level pitching as a proxy so hitters facing TBD still get
    # a sensible projection (flagged with tbd_proxy=True).
    try:
        from data_fetcher import get_team_pitching_proxy as _tbd_proxy
        if not away_p_row and game.get("away_team_id"):
            proxy = _tbd_proxy(int(game["away_team_id"]))
            if proxy:
                away_p_row = proxy
        if not home_p_row and game.get("home_team_id"):
            proxy = _tbd_proxy(int(game["home_team_id"]))
            if proxy:
                home_p_row = proxy
    except Exception:
        pass

    if use_recent_form:
        try:
            if away_p_id:
                away_p_row.update(pitcher_recent_map.get(away_p_id, {}))
            if home_p_id:
                home_p_row.update(pitcher_recent_map.get(home_p_id, {}))
        except Exception:
            pass

    # Hitter recent form
    recent_hitter_map = {}
    if use_recent_form:
        try:
            for p in (away_lineup + home_lineup):
                pid = safe_int(p.get("id"))
                if pid is None:
                    continue
                recent_hitter_map[pid] = get_hitter_recent_form_trad(pid) or {}
        except Exception:
            pass

    # Build matchup tables for each side
    away_matchup = build_matchup_table(
        away_lineup,
        pd.Series(home_p_row) if home_p_row else None,
        hitter_stats, pitcher_stats,
        recent_form_dict=recent_hitter_map,
        pitcher_arsenal_df=pitcher_arsenal_all,
    )
    home_matchup = build_matchup_table(
        home_lineup,
        pd.Series(away_p_row) if away_p_row else None,
        hitter_stats, pitcher_stats,
        recent_form_dict=recent_hitter_map,
        pitcher_arsenal_df=pitcher_arsenal_all,
    )

    # Power Score - composite HR-likelihood incorporating park/weather/pitcher
    try:
        from models import add_power_score
        away_matchup = add_power_score(
            away_matchup,
            park_mult=park_mult, weather_mult=wx_mult,
            pitcher_hr9=home_p_row.get("hr9") if home_p_row else None,
            pitcher_barrel_allowed=home_p_row.get("barrel_batted_rate") if home_p_row else None,
        )
        home_matchup = add_power_score(
            home_matchup,
            park_mult=park_mult, weather_mult=wx_mult,
            pitcher_hr9=away_p_row.get("hr9") if away_p_row else None,
            pitcher_barrel_allowed=away_p_row.get("barrel_batted_rate") if away_p_row else None,
        )
    except Exception:
        pass

    # Pitch match score - single call per hitter, captures all outputs
    def _apply_pitch_match(matchup_df, opp_p_row):
        if not opp_p_row or matchup_df is None or matchup_df.empty:
            return
        scores, hr_scores, bests, bestxw, worsts = [], [], [], [], []
        for _, hitter_row in matchup_df.iterrows():
            pid = hitter_row.get("player_id")
            ps = None
            if pid is not None and not pd.isna(pid):
                try:
                    ps = pitch_match_score_for_hitter(
                        int(pid), opp_p_row, pitcher_arsenal_all
                    )
                except Exception:
                    ps = None
            if isinstance(ps, dict):
                scores.append(ps.get("pitch_match_score"))
                hr_scores.append(ps.get("pitch_hr_score"))
                bests.append(ps.get("best_pitch"))
                bestxw.append(ps.get("best_pitch_xwoba"))
                worsts.append(ps.get("worst_pitch"))
            else:
                scores.append(None)
                hr_scores.append(None)
                bests.append(None)
                bestxw.append(None)
                worsts.append(None)
        matchup_df["pitch_match_score"] = scores
        matchup_df["pitch_hr_score"] = hr_scores
        matchup_df["best_pitch"] = bests
        matchup_df["best_pitch_xwoba"] = bestxw
        matchup_df["worst_pitch"] = worsts

    if use_pitch_match and HAVE_PITCH_MATCH:
        try:
            _apply_pitch_match(away_matchup, home_p_row)
            _apply_pitch_match(home_matchup, away_p_row)
        except Exception:
            pass

    # HR probability per hitter - now with hand-aware park + pull-wind multipliers
    from park_factors import get_park_hand_factor, wind_pull_side_multiplier

    venue_name = game.get("venue", "")
    wind_mph = (weather or {}).get("wind_mph")
    wind_dir = (weather or {}).get("wind_dir_deg")
    pull_summaries = {}  # for displaying in game header

    for matchup_df, opp_p_row in [(away_matchup, home_p_row), (home_matchup, away_p_row)]:
        if matchup_df is None or matchup_df.empty:
            continue
        hr_pa, hr_game, verdicts, signals = [], [], [], []
        pull_mults_col = []
        for _, hr in matchup_df.iterrows():
            row_dict = hr.to_dict()
            pa = safe_float(row_dict.get("pa"))
            sample = int(pa) if pa is not None else None
            bats = row_dict.get("bats", "R") or "R"

            # Hand-aware park factor (replaces the generic park_mult for this hitter)
            try:
                hand_park = get_park_hand_factor(venue_name, bats)
            except Exception:
                hand_park = park_mult

            # Pull-side wind multiplier
            try:
                pull_mult, pull_summary = wind_pull_side_multiplier(
                    venue_name, bats, wind_mph, wind_dir
                )
                if pull_summary:
                    pull_summaries[pull_summary] = pull_summaries.get(pull_summary, 0) + 1
            except Exception:
                pull_mult = 1.0

            pull_mults_col.append(round(pull_mult, 3))

            # Combined park factor for this specific hitter: hand-aware × pull-wind
            hitter_park_mult = hand_park * pull_mult

            # Pitch-type HR match multiplier
            # 50 = neutral, 90 = great barrel rates vs this pitcher's arsenal,
            # 10 = poor barrel rates. Modest 0.85x to 1.15x multiplier.
            pitch_hr_score = row_dict.get("pitch_hr_score")
            if pitch_hr_score is not None and not pd.isna(pitch_hr_score):
                # Center on 50 -> 1.0, +/- 1 score = 0.003 multiplier shift
                pitch_hr_mult = 1.0 + (pitch_hr_score - 50) * 0.003
                pitch_hr_mult = max(0.85, min(1.15, pitch_hr_mult))
            else:
                pitch_hr_mult = 1.0

            try:
                p_pa = hr_prob_per_pa(
                    row_dict, opp_p_row,
                    park_factor=hitter_park_mult, weather_mult=wx_mult,
                    pitch_match_score=row_dict.get("pitch_match_score"),
                )
                # Apply pitch_hr_score as an additional fine adjustment
                # BUT re-apply the soft squash so we don't blow past the cap.
                if p_pa is not None:
                    raw = float(p_pa) * pitch_hr_mult
                    # Soft squash: same logic as in props.py (7.0% per-PA ceiling,
                    # wider differentiation band)
                    if raw <= 0.04:
                        p_pa = raw
                    else:
                        excess = raw - 0.04
                        p_pa = 0.04 + 0.030 * np.tanh(excess / 0.040)
                    p_pa = max(0.001, p_pa)
            except TypeError:
                try:
                    p_pa = hr_prob_per_pa(
                        row_dict, opp_p_row,
                        park_hr_factor=hitter_park_mult, weather_hr_factor=wx_mult,
                    )
                except TypeError:
                    try:
                        p_pa = hr_prob_per_pa(row_dict, opp_p_row)
                    except Exception:
                        p_pa = None
            # Lineup-spot-aware expected PA per game
            # ONLY use lineup-position scaling when:
            #   - This player is in a REAL posted lineup (not roster-fill)
            #   - The whole game's lineup is confirmed
            # Otherwise default to 4.2 (league average) for everyone, because
            # roster-fill positions are alphabetical and not predictive.
            lp = row_dict.get("lineup_pos")
            is_fill = row_dict.get("is_roster_fill", False)
            game_confirmed = (
                away_confirmed if matchup_df is away_matchup else home_confirmed
            )
            if (lp is not None and not pd.isna(lp)
                    and not is_fill and game_confirmed):
                try:
                    lp_int = int(lp)
                    expected_pa = max(3.6, 4.7 - (lp_int - 1) * 0.1)
                except (ValueError, TypeError):
                    expected_pa = 4.2
            else:
                # Default to league avg - don't pretend roster order = lineup order
                expected_pa = 4.2

            p_game = hr_prob_full_game(p_pa, expected_pa=expected_pa) if p_pa is not None else None
            hr_pa.append(round(p_pa * 100, 2) if p_pa is not None else None)
            hr_game.append(round(p_game * 100, 2) if p_game is not None else None)
            verdicts.append(hr_verdict(
                round(p_game * 100, 2) if p_game is not None else None,
                sample, INSUFFICIENT_PA_THRESHOLD,
            ))
            signals.append(hr_signal_emoji(
                round(p_game * 100, 2) if p_game is not None else None,
                sample, INSUFFICIENT_PA_THRESHOLD,
            ))
        matchup_df["hr_pa_pct"] = hr_pa
        matchup_df["hr_game_pct"] = hr_game
        matchup_df["verdict"] = verdicts
        matchup_df["alert"] = signals
        matchup_df["pull_wind_mult"] = pull_mults_col

    # Sleeper score - uses sleepers.py functions correctly
    try:
        if not away_matchup.empty:
            away_matchup = hr_probability(
                away_matchup,
                pd.Series(home_p_row) if home_p_row else None,
                hr_mult=full_hr_mult,
            )
            away_matchup = find_sleepers(away_matchup, season_hr_col="home_run")
        if not home_matchup.empty:
            home_matchup = hr_probability(
                home_matchup,
                pd.Series(away_p_row) if away_p_row else None,
                hr_mult=full_hr_mult,
            )
            home_matchup = find_sleepers(home_matchup, season_hr_col="home_run")
    except Exception:
        # Fallback: compute sleeper score directly if helpers fail
        for matchup_df in [away_matchup, home_matchup]:
            if matchup_df.empty:
                continue
            if "hr_game_pct" in matchup_df.columns and "home_run" in matchup_df.columns:
                hr_pct = matchup_df["hr_game_pct"].rank(pct=True) * 100
                season_pct = matchup_df["home_run"].rank(pct=True) * 100
                matchup_df["sleeper_score"] = (hr_pct - season_pct).round(1)

    # Grand slam compound probability - real signature: (df, pitcher_row, hr_mult)
    # Adds a gs_score column per hitter. Sum to get a team-level value.
    away_gs = 0.0
    home_gs = 0.0
    try:
        if not away_matchup.empty:
            away_matchup = grand_slam_probability(
                away_matchup,
                pd.Series(home_p_row) if home_p_row else None,
                hr_mult=full_hr_mult,
            )
            if "gs_score" in away_matchup.columns:
                away_gs = float(away_matchup["gs_score"].fillna(0).sum())
        if not home_matchup.empty:
            home_matchup = grand_slam_probability(
                home_matchup,
                pd.Series(away_p_row) if away_p_row else None,
                hr_mult=full_hr_mult,
            )
            if "gs_score" in home_matchup.columns:
                home_gs = float(home_matchup["gs_score"].fillna(0).sum())
    except Exception:
        pass

    # K projection
    away_k_col = away_matchup["k_pct"] if "k_pct" in away_matchup.columns else pd.Series(dtype=float)
    home_k_col = home_matchup["k_pct"] if "k_pct" in home_matchup.columns else pd.Series(dtype=float)
    away_lineup_k_pct = float(away_k_col.mean()) if not away_k_col.empty and not away_k_col.isna().all() else None
    home_lineup_k_pct = float(home_k_col.mean()) if not home_k_col.empty and not home_k_col.isna().all() else None

    away_k_proj, home_k_proj = {}, {}
    try:
        # Park K factor (small effect, but worth including)
        pkf = park_k_factor(venue) if venue else 1.0
        if away_p_row:
            away_k_proj = k_total_projection(
                away_p_row, home_lineup_k_pct,
                ump_k_factor=ump.get("k_factor", 1.0),
                park_k_factor=pkf,
            )
        if home_p_row:
            home_k_proj = k_total_projection(
                home_p_row, away_lineup_k_pct,
                ump_k_factor=ump.get("k_factor", 1.0),
                park_k_factor=pkf,
            )
    except Exception:
        pass

    game_context_map[game["gamePk"]] = {
        "park": park_mult, "weather": weather, "wx_mult": wx_mult,
        "park_mult": park_mult, "hr_mult": full_hr_mult,
        "summary": wx_summary, "vegas": vegas_row, "ump": ump,
        "framing": framing,
        "away_matchup": away_matchup, "home_matchup": home_matchup,
        "away_p_row": away_p_row, "home_p_row": home_p_row,
        "away_k_proj": away_k_proj, "home_k_proj": home_k_proj,
        "away_gs": away_gs, "home_gs": home_gs,
        "pull_wind_summary": list(pull_summaries.keys()) if 'pull_summaries' in dir() else [],
        "away_lineup_confirmed": away_confirmed,
        "home_lineup_confirmed": home_confirmed,
    }
    matchup_tables[game["gamePk"]] = (away_matchup, home_matchup)


st.divider()


# ============================================================================
# LINEUP CONFIRMATION BANNER - tells user upfront if many games are unconfirmed
# ============================================================================
unconfirmed_games = []
for gpk, ctx in game_context_map.items():
    g_rows = slate[slate["gamePk"] == gpk]
    if g_rows.empty:
        continue
    g = g_rows.iloc[0]
    away_conf = ctx.get("away_lineup_confirmed", True)
    home_conf = ctx.get("home_lineup_confirmed", True)
    if not away_conf or not home_conf:
        sides = []
        if not away_conf:
            sides.append(g.get("away_team_abbr", "AWAY"))
        if not home_conf:
            sides.append(g.get("home_team_abbr", "HOME"))
        unconfirmed_games.append(f"{'/'.join(sides)} ({g.get('away_team_abbr')}@{g.get('home_team_abbr')})")

if unconfirmed_games:
    n_total = len(game_context_map)
    n_unconf = len(unconfirmed_games)
    st.warning(
        f"⚠️ **{n_unconf}/{n_total} games have unconfirmed lineups** — MLB hasn't "
        f"posted the actual batting order yet for: {', '.join(unconfirmed_games[:5])}"
        f"{'...' if len(unconfirmed_games) > 5 else ''}. "
        f"For those teams, hitters are sorted by season PA (likely starters first), "
        f"and **lineup-position PA scaling is DISABLED** (all use 4.2 PA). "
        f"**For best accuracy, refresh after 4-5 PM ET for evening games.**"
    )


# ============================================================================
# TOP 5 PICKS OF THE DAY — combined HR signal across all factors
# ============================================================================
st.subheader("🏆 Top 5 Picks of the Day")
st.caption(
    "Highest-confidence HR plays combining: HR Game%, matchup score, weather/park boost, "
    "recent form, barrel rate, and opposing pitcher quality. Qualified samples only."
)

# Gather all qualified hitters with game context
all_hitters_for_picks = []
for gpk, ctx in game_context_map.items():
    game_rows = slate[slate["gamePk"] == gpk]
    if game_rows.empty:
        continue
    g_row = game_rows.iloc[0]
    park_mult = ctx.get("park_mult", 1.0) or 1.0
    wx_mult = ctx.get("wx_mult", 1.0) or 1.0
    hr_mult = ctx.get("hr_mult", 1.0) or 1.0
    for side in ("away", "home"):
        m = ctx.get(f"{side}_matchup")
        if m is None or m.empty:
            continue
        x = m.copy()
        x["game"] = f"{g_row['away_team_abbr']} @ {g_row['home_team_abbr']}"
        x["team"] = g_row[f"{side}_team_abbr"]
        opp_side = "home" if side == "away" else "away"
        x["opp_pitcher"] = g_row.get(f"{opp_side}_pitcher", "TBD") or "TBD"
        opp_p_row = ctx.get(f"{opp_side}_p_row") or {}
        x["opp_pitcher_xwoba"] = opp_p_row.get("xwoba")
        x["env_boost"] = hr_mult
        all_hitters_for_picks.append(x)

if all_hitters_for_picks:
    combined_picks = pd.concat(all_hitters_for_picks, ignore_index=True)
    if "pa" in combined_picks.columns and "hr_game_pct" in combined_picks.columns:
        q = combined_picks[
            combined_picks["pa"].notna()
            & (combined_picks["pa"] >= INSUFFICIENT_PA_THRESHOLD)
            & combined_picks["hr_game_pct"].notna()
        ].copy()
    else:
        q = combined_picks.copy()

    if not q.empty:
        def _pct(s):
            return s.rank(pct=True) * 100 if s.notna().any() else pd.Series([50]*len(s), index=s.index)

        score_parts = []
        weights = []

        if "hr_game_pct" in q.columns:
            score_parts.append(_pct(q["hr_game_pct"]))
            weights.append(0.35)
        if "matchup" in q.columns:
            score_parts.append(_pct(q["matchup"]))
            weights.append(0.15)
        if "barrel_pct" in q.columns:
            score_parts.append(_pct(q["barrel_pct"]))
            weights.append(0.15)
        if "hr_form" in q.columns:
            score_parts.append(_pct(q["hr_form"]))
            weights.append(0.10)
        if "env_boost" in q.columns:
            score_parts.append(_pct(q["env_boost"]))
            weights.append(0.15)
        if "opp_pitcher_xwoba" in q.columns and q["opp_pitcher_xwoba"].notna().any():
            score_parts.append(_pct(q["opp_pitcher_xwoba"]))
            weights.append(0.10)

        if score_parts:
            total_w = sum(weights)
            weighted = sum(p * (w / total_w) for p, w in zip(score_parts, weights))
            q["pick_score"] = weighted.round(1)
        else:
            q["pick_score"] = q.get("hr_game_pct", 0)

        # Diversity rule: max 2 picks per game so the top 5 doesn't pile up
        # on one matchup. Greedy selection: sort by pick_score, take in order,
        # skipping any that would exceed 2-per-game.
        q_sorted = q.sort_values("pick_score", ascending=False)
        picks = []
        game_count = {}
        for _, row in q_sorted.iterrows():
            g = row.get("game", "")
            if game_count.get(g, 0) >= 2:
                continue
            picks.append(row)
            game_count[g] = game_count.get(g, 0) + 1
            if len(picks) >= 5:
                break
        if picks:
            top5 = pd.DataFrame(picks).reset_index(drop=True)
        else:
            top5 = q_sorted.head(5).reset_index(drop=True)
        top5["rank"] = range(1, len(top5) + 1)

        cols_to_show = [c for c in [
            "rank", "player_name", "team", "game", "opp_pitcher",
            "pick_score", "hr_game_pct", "matchup", "barrel_pct",
            "hr_form", "env_boost",
        ] if c in top5.columns]
        disp = top5[cols_to_show].copy()

        st.dataframe(
            disp, hide_index=True, use_container_width=True,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "player_name": st.column_config.TextColumn("Hitter"),
                "team": st.column_config.TextColumn("Tm", width="small"),
                "game": st.column_config.TextColumn("Game"),
                "opp_pitcher": st.column_config.TextColumn("vs Pitcher"),
                "pick_score": st.column_config.NumberColumn(
                    "Pick Score",
                    format="%.1f",
                    help="0-100 composite of HR Game%, matchup, barrel, form, park/weather, pitcher quality.",
                ),
                "hr_game_pct": st.column_config.NumberColumn("HR Game%", format="%.1f%%"),
                "matchup": st.column_config.NumberColumn("Matchup", format="%.1f"),
                "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
                "hr_form": st.column_config.NumberColumn("Form", format="%.0f"),
                "env_boost": st.column_config.NumberColumn(
                    "Env",
                    format="%.2f×",
                    help="Park × weather HR multiplier. >1.0 = boost, <1.0 = suppress.",
                ),
            },
        )
        glance = " · ".join(
            f"**#{r['rank']} {r['player_name']}** ({r['team']}, {r.get('hr_game_pct', 0):.1f}%)"
            for _, r in top5.iterrows()
        )
        st.markdown(glance)
    else:
        st.caption("Not enough qualified hitters with HR projections yet.")
else:
    st.caption("Waiting for matchup data to populate.")

st.divider()
# ============================================================================
# TOP SLEEPERS + TOP HR PLAYS ACROSS THE WHOLE SLATE
# ============================================================================
st.subheader("💎 Top Sleepers & Best HR Plays")
st.caption(
    "Combined view across every game today. **Sleepers**: hitters whose HR probability "
    "today exceeds their season pace (under-the-radar plays). **Top HR**: highest "
    "calibrated HR Game% regardless of name recognition."
)

all_hitters = []
for gpk, ctx in game_context_map.items():
    game_rows = slate[slate["gamePk"] == gpk]
    if game_rows.empty:
        continue
    g_row = game_rows.iloc[0]
    for side in ("away", "home"):
        m = ctx.get(f"{side}_matchup")
        if m is None or m.empty:
            continue
        x = m.copy()
        x["game"] = f"{g_row['away_team_abbr']} @ {g_row['home_team_abbr']}"
        x["team"] = g_row[f"{side}_team_abbr"]
        opp_side = "home" if side == "away" else "away"
        x["opp_pitcher"] = g_row.get(f"{opp_side}_pitcher", "TBD") or "TBD"
        all_hitters.append(x)

if all_hitters:
    combined_all = pd.concat(all_hitters, ignore_index=True)
    if "pa" in combined_all.columns:
        qualified = combined_all[combined_all["pa"].notna() & (combined_all["pa"] >= INSUFFICIENT_PA_THRESHOLD)]
    else:
        qualified = combined_all

    # Save-snapshot + full export buttons - always render
    snap_col1, snap_col2, snap_col3 = st.columns([1.2, 1.5, 3])
    with snap_col1:
        if st.button("💾 Save snapshot", help="Save today's projections so they can be evaluated against actual outcomes tomorrow."):
            try:
                from backtest import save_snapshot
                ok = save_snapshot(selected_date, combined_all, p_slate)
                if ok:
                    st.success(f"Saved snapshot for {selected_date}")
                else:
                    st.error("Snapshot save failed")
            except Exception as e:
                st.error(f"Backtest module error: {e}")

    with snap_col2:
        # Build combined export - try Excel, fall back to CSV-bundle
        import io as _io
        from datetime import datetime as _dt
        try:
            import openpyxl  # noqa: F401
            HAS_OPENPYXL = True
        except Exception:
            HAS_OPENPYXL = False

        if HAS_OPENPYXL:
            try:
                buffer = _io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    sheets_written = 0
                    if combined_all is not None and not combined_all.empty:
                        combined_all.to_excel(writer, sheet_name="Hitters", index=False)
                        sheets_written += 1
                    if p_slate is not None and not p_slate.empty:
                        p_slate.to_excel(writer, sheet_name="Pitchers", index=False)
                        sheets_written += 1
                    if "hr_game_pct" in qualified.columns:
                        top_hr = qualified.dropna(subset=["hr_game_pct"]).sort_values(
                            "hr_game_pct", ascending=False).head(20)
                        if not top_hr.empty:
                            top_hr.to_excel(writer, sheet_name="Top 20 HR", index=False)
                    if "power_score" in qualified.columns:
                        top_pow = qualified.dropna(subset=["power_score"]).sort_values(
                            "power_score", ascending=False).head(20)
                        if not top_pow.empty:
                            top_pow.to_excel(writer, sheet_name="Top 20 Power", index=False)
                    if "sleeper_score" in qualified.columns:
                        top_sl = qualified.dropna(subset=["sleeper_score"]).sort_values(
                            "sleeper_score", ascending=False).head(20)
                        if not top_sl.empty:
                            top_sl.to_excel(writer, sheet_name="Top 20 Sleepers", index=False)
                    # Ensure at least one sheet exists (openpyxl errors on empty)
                    if sheets_written == 0:
                        pd.DataFrame({"empty": ["no data"]}).to_excel(
                            writer, sheet_name="Empty", index=False)
                buffer.seek(0)
                st.download_button(
                    "📥 Export ALL to Excel",
                    data=buffer.getvalue(),
                    file_name=f"posey_mlb_{_dt.now().strftime('%Y-%m-%d_%H-%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Hitters + Pitchers + Top lists in one Excel workbook.",
                )
            except Exception as e:
                st.error(f"Excel export failed: {str(e)[:100]}")
        else:
            # Fallback: combined CSV with sections
            try:
                csv_buf = _io.StringIO()
                csv_buf.write("=== HITTERS ===\n")
                if combined_all is not None and not combined_all.empty:
                    combined_all.to_csv(csv_buf, index=False)
                csv_buf.write("\n\n=== PITCHERS ===\n")
                if p_slate is not None and not p_slate.empty:
                    p_slate.to_csv(csv_buf, index=False)
                st.download_button(
                    "📥 Export ALL to CSV",
                    data=csv_buf.getvalue(),
                    file_name=f"posey_mlb_{_dt.now().strftime('%Y-%m-%d_%H-%M')}.csv",
                    mime="text/csv",
                    help="Combined CSV - openpyxl not installed for Excel export.",
                )
                st.caption("ℹ️ Add `openpyxl` to requirements.txt for Excel output")
            except Exception as e:
                st.error(f"CSV export failed: {str(e)[:100]}")

    with snap_col3:
        st.caption(
            "📥 **Export** = full slate (hitters + pitchers + top lists) in one file. "
            "💾 **Snapshot** = save for tomorrow's backtest comparison."
        )

    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        st.markdown("**🎯 Top 10 HR Plays**")
        if "hr_game_pct" in qualified.columns:
            top_hr = qualified.dropna(subset=["hr_game_pct"]).sort_values(
                "hr_game_pct", ascending=False
            ).head(10)
            if not top_hr.empty:
                cols = [c for c in [
                    "player_name", "team", "game",
                    "hr_game_pct", "barrel_pct", "iso",
                ] if c in top_hr.columns]
                disp = top_hr[cols].copy().reset_index(drop=True)
                st.dataframe(
                    disp, hide_index=True, use_container_width=True,
                    column_config={
                        "player_name": st.column_config.TextColumn("Hitter"),
                        "team": st.column_config.TextColumn("Tm"),
                        "game": st.column_config.TextColumn("Game"),
                        "hr_game_pct": st.column_config.NumberColumn("HR%", format="%.1f%%"),
                        "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
                        "iso": st.column_config.NumberColumn("ISO", format="%.3f"),
                    },
                )
            else:
                st.caption("No HR projections available.")

    with col_mid:
        st.markdown("**💪 Top 10 Power**")
        st.caption("Composite incl. park, weather, matchup")
        if "power_score" in qualified.columns:
            top_pow = qualified.dropna(subset=["power_score"]).sort_values(
                "power_score", ascending=False
            ).head(10)
            if not top_pow.empty:
                cols = [c for c in [
                    "player_name", "team", "game",
                    "power_score", "barrel_pct", "avg_ev",
                ] if c in top_pow.columns]
                disp = top_pow[cols].copy().reset_index(drop=True)
                st.dataframe(
                    disp, hide_index=True, use_container_width=True,
                    column_config={
                        "player_name": st.column_config.TextColumn("Hitter"),
                        "team": st.column_config.TextColumn("Tm"),
                        "game": st.column_config.TextColumn("Game"),
                        "power_score": st.column_config.NumberColumn(
                            "Power", format="%.1f",
                            help="HR-likelihood composite blending raw power, "
                                 "form, park, weather, and pitcher.",
                        ),
                        "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
                        "avg_ev": st.column_config.NumberColumn("EV", format="%.1f"),
                    },
                )
            else:
                st.caption("Power scores not yet computed.")
        else:
            st.caption("Power Score requires per-game context to load.")

    with col_right:
        st.markdown("**💎 Top 10 Sleepers**")
        st.caption("Under-the-radar HR upside")
        if "sleeper_score" in qualified.columns:
            top_sleep = qualified.dropna(subset=["sleeper_score"]).sort_values(
                "sleeper_score", ascending=False
            ).head(10)
            if not top_sleep.empty:
                cols = [c for c in [
                    "player_name", "team", "game",
                    "sleeper_score", "hr_game_pct", "barrel_pct",
                ] if c in top_sleep.columns]
                disp = top_sleep[cols].copy().reset_index(drop=True)
                st.dataframe(
                    disp, hide_index=True, use_container_width=True,
                    column_config={
                        "player_name": st.column_config.TextColumn("Hitter"),
                        "team": st.column_config.TextColumn("Tm"),
                        "game": st.column_config.TextColumn("Game"),
                        "sleeper_score": st.column_config.NumberColumn(
                            "Sleeper", format="%.1f",
                            help="HR prob percentile MINUS season HR percentile.",
                        ),
                        "hr_game_pct": st.column_config.NumberColumn("HR%", format="%.1f%%"),
                        "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
                    },
                )
            else:
                st.caption("No sleeper scores computed yet.")
else:
    st.caption("Waiting for matchup data to populate.")

st.divider()


# ============================================================================
# GAME-BY-GAME MATCHUPS
# ============================================================================
st.subheader("🎮 Isolated Game-by-Game Matchups")
st.caption("Real data only. Empty cells = data not available for that player.")


def build_col_config():
    return {
        "alert": st.column_config.TextColumn(
            "Signal", width="small",
            help=(
                "🟢 = ELITE (HR Game% ≥ 22%) — top-tier HR play\n"
                "🟡 = STRONG (14-22%) — solid HR consideration\n"
                "🟠 = DECENT (7-14%) — middle of pack\n"
                "🔴 = WEAK (< 7%) — avoid for HR props\n"
                "⚪ = INSUFFICIENT SAMPLE — player has too few PA (< threshold) "
                "for a reliable projection. NOT a sleeper indicator."
            ),
        ),
        "player_name": st.column_config.TextColumn("Hitter"),
        "lineup_pos": st.column_config.NumberColumn(
            "#", width="small",
            help=(
                "Batting order position from MLB's posted lineup. "
                "If lineup hasn't posted yet, this is alphabetical roster order "
                "(see warning above each game)."
            ),
        ),
        "bats": st.column_config.TextColumn("B", width="small"),
        "position": st.column_config.TextColumn("Pos", width="small"),
        "power_score": st.column_config.NumberColumn(
            "Power", format="%.1f",
            help="Composite HR-likelihood score (0-99). Blends raw power "
                 "(barrel%, ISO, hard-hit%, exit velocity, LA, FB%, pulled "
                 "barrel%, SLG, recent ISO) percentile-ranked across this "
                 "lineup, then multiplied by park × weather × pitcher HR/9 "
                 "× pitcher barrel%-allowed. Higher = better HR play today."
        ),
        "matchup_opp": st.column_config.NumberColumn(
            "Opp", format="%.1f",
            help="Matchup Opportunity score (0-99). Like Power Score but "
                 "weighted more heavily toward environment (park, weather, "
                 "pitcher quality) and less on the hitter's raw power. "
                 "Catches contact hitters who happen to face a HR-friendly "
                 "matchup today."
        ),
        "hr_game_pct": st.column_config.NumberColumn(
            "HR Game%", format="%.1f%%",
            help=(
                "TRUE PROBABILITY of hitting ≥1 HR this game. "
                "Calculated from per-PA HR rate × expected PA (lineup-spot aware). "
                "Max realistic value ~26% (leadoff elite) to 22% (#9 spot). "
                "This is the betting probability."
            ),
        ),
        "hr_pa_pct": st.column_config.NumberColumn(
            "HR PA%", format="%.2f%%",
            help=(
                "TRUE PROBABILITY of hitting a HR on any single plate appearance. "
                "Blends observed HR/PA with barrel-derived xHR/PA to reduce noise. "
                "League avg ~2.8%. Elite (Judge-tier) caps near 6.5%."
            ),
        ),
        "likely_hr_pct": st.column_config.NumberColumn(
            "Likely HR%", format="%.2f%%",
            help=(
                "CONTEXT-FREE hitter trait (barrel × FB% × 0.75). "
                "Represents the hitter's underlying HR-producing skill regardless "
                "of who they face. Use HR PA% / HR Game% for today's actual matchup. "
                "Range: 0-5%."
            ),
        ),
        "hr_score": st.column_config.NumberColumn(
            "HR Score", format="%.1f",
            help=(
                "0-100 COMPOSITE SCORE (not a probability). "
                "Combines barrel%, ISO, xwOBA, recent form × today's park/weather mult. "
                "Used as input to sleeper_score percentile rank."
            ),
        ),
        "hr_prob": st.column_config.NumberColumn(
            "HR Score (alias)", format="%.1f",
            help="Same value as HR Score - kept for backward compatibility.",
        ),
        "matchup": st.column_config.NumberColumn(
            "Matchup", format="%.1f",
            help=(
                "0-100 composite of SEASON stats vs opposing pitcher: "
                "xwOBA, barrel%, ISO, recent HR + pitcher xwOBA-allowed, "
                "barrel-allowed, HR/9. Pure quality measure."
            ),
        ),
        "test_score": st.column_config.NumberColumn(
            "Test", format="%.1f",
            help=(
                "0-100 score blending: 70% Matchup (season vs today's pitcher) + "
                "30% Recent Form (last 15 games), scaled by PA reliability. "
                "Different from Matchup because it incorporates hot/cold streaks. "
                "Hot hitter facing same pitcher = higher Test than Matchup."
            ),
        ),
        "streak_label": st.column_config.TextColumn(
            "Streak", width="small",
            help="🔥 HR streak / hot · 🌶️ 2+ HR last 5 · ⚡ multi-HR game · ❄️ cold (no HR L10)"
        ),
        "pa": st.column_config.NumberColumn("PA", format="%d"),
        "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
        "iso": st.column_config.NumberColumn("ISO", format="%.3f"),
        "xwoba": st.column_config.NumberColumn("xwOBA", format="%.3f"),
        "xwobacon": st.column_config.NumberColumn("xwOBAcon", format="%.3f"),
        "pitch_match_score": st.column_config.NumberColumn(
            "Pitch Match", format="%.1f",
            help="xwOBA-weighted matchup score (0-100). Higher = hitter's per-pitch xwOBA "
                 "matches well against pitcher's arsenal usage.",
        ),
        "pitch_hr_score": st.column_config.NumberColumn(
            "Pitch HR", format="%.1f",
            help="HR-specific matchup score (0-100). Weighs pitcher's pitch usage "
                 "against hitter's barrel%/SLG vs each pitch type. >70 = hitter has "
                 "elite power vs this pitcher's arsenal.",
        ),
        "best_pitch": st.column_config.TextColumn("Best Pitch"),
        "best_pitch_xwoba": st.column_config.NumberColumn("Best xwOBA", format="%.3f"),
        "worst_pitch": st.column_config.TextColumn("Worst Pitch"),
        "fb_pct": st.column_config.NumberColumn("FB%", format="%.1f%%"),
        "la": st.column_config.NumberColumn("LA", format="%.1f"),
        "avg_ev": st.column_config.NumberColumn("EV", format="%.1f"),
        "hard_hit": st.column_config.NumberColumn("HH%", format="%.1f%%"),
        "sprint_speed": st.column_config.NumberColumn("Sprint", format="%.1f"),
        "k_pct": st.column_config.NumberColumn("K%", format="%.1f%%"),
        "bb_pct": st.column_config.NumberColumn("BB%", format="%.1f%%"),
        "whiff_pct": st.column_config.NumberColumn("Whiff%", format="%.1f%%"),
        "home_run": st.column_config.NumberColumn("HR", format="%d"),
        "recent_hr": st.column_config.NumberColumn("L15 HR", format="%d"),
        "sleeper_score": st.column_config.NumberColumn("Sleeper", format="%.1f"),
        "verdict": st.column_config.TextColumn("Verdict"),
    }


def _style_matchup_df(df: pd.DataFrame):
    """
    Returns a Styled DataFrame with color-coded columns - Kasper-style.
    Green = good for HR play, Red = bad. Color thresholds use absolute MLB
    benchmarks (not per-lineup percentiles) so colors mean the same thing
    every game.
    """
    if df is None or df.empty:
        return df

    # color spec: (col_name, poor_val, elite_val, higher_is_better)
    color_specs = [
        ("power_score",     20,    70,   True),   # Power Score
        ("matchup_opp",     30,    75,   True),   # Matchup Opportunity
        ("hr_game_pct",     5,     22,   True),
        ("hr_pa_pct",       1.5,   6.0,  True),
        ("matchup",         30,    75,   True),
        ("test_score",      30,    75,   True),
        ("barrel_pct",      4,     15,   True),
        ("iso",             0.110, 0.260, True),
        ("xwoba",           0.280, 0.380, True),
        ("xwobacon",        0.330, 0.460, True),
        ("pitch_match_score", 35,  70,   True),
        ("pitch_hr_score",    30,  75,   True),
        ("fb_pct",          22,    40,   True),
        ("la",              5,     22,   True),   # sweet spot is ~28, but capped here
        ("avg_ev",          87,    94,   True),
        ("hard_hit",        30,    52,   True),
        ("sprint_speed",    25,    29,   True),
        ("k_pct",           18,    30,   False),  # Lower K% = better
        ("bb_pct",          5,     12,   True),
        ("whiff_pct",       18,    32,   False),
        ("home_run",        3,     20,   True),
        ("recent_hr",       0,     3,    True),
        ("sleeper_score",   0,     30,   True),
    ]

    def color_cell(val, poor, elite, higher_better):
        if val is None or pd.isna(val):
            return ""
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if higher_better:
            ratio = (v - poor) / (elite - poor) if elite != poor else 0.5
        else:
            ratio = (poor - v) / (poor - elite) if poor != elite else 0.5
        ratio = max(0, min(1, ratio))
        # Map 0→red, 0.5→yellow, 1→green - use HSL hue
        # 0 = red (0°), 0.4 = orange (30°), 0.6 = yellow (60°), 1 = green (120°)
        hue = ratio * 120
        return f"background-color: hsl({hue:.0f}, 60%, 40%); color: white;"

    styled = df.style
    for col, poor, elite, higher_better in color_specs:
        if col in df.columns:
            styled = styled.map(
                lambda v, p=poor, e=elite, hb=higher_better: color_cell(v, p, e, hb),
                subset=[col],
            )
    # Format key numeric columns to limit decimal noise
    fmt_map = {
        "power_score": "{:.1f}",
        "matchup_opp": "{:.1f}",
        "hr_game_pct": "{:.1f}",
        "hr_pa_pct": "{:.2f}",
        "matchup": "{:.1f}",
        "test_score": "{:.1f}",
        "barrel_pct": "{:.1f}",
        "iso": "{:.3f}",
        "xwoba": "{:.3f}",
        "xwobacon": "{:.3f}",
        "fb_pct": "{:.1f}",
        "la": "{:.1f}",
        "avg_ev": "{:.1f}",
        "hard_hit": "{:.1f}",
        "k_pct": "{:.1f}",
        "bb_pct": "{:.1f}",
        "whiff_pct": "{:.1f}",
        "pitch_match_score": "{:.1f}",
        "pitch_hr_score": "{:.1f}",
        "sleeper_score": "{:.1f}",
        "best_pitch_xwoba": "{:.3f}",
    }
    valid_fmt = {k: v for k, v in fmt_map.items() if k in df.columns}
    if valid_fmt:
        styled = styled.format(valid_fmt, na_rep="—")
    return styled


def render_matchup_section(matchup_df: pd.DataFrame, team_label: str):
    if matchup_df is None or matchup_df.empty:
        st.caption(f"{team_label}: no lineup data available yet.")
        return

    if "pa" in matchup_df.columns:
        qualified = matchup_df[matchup_df["pa"].notna() & (matchup_df["pa"] >= INSUFFICIENT_PA_THRESHOLD)]
        insufficient = matchup_df[~(matchup_df["pa"].notna() & (matchup_df["pa"] >= INSUFFICIENT_PA_THRESHOLD))]
    else:
        qualified = matchup_df
        insufficient = pd.DataFrame()

    cols_to_show = [c for c in [
        "alert", "player_name", "lineup_pos", "bats", "position",
        "power_score", "matchup_opp", "hr_game_pct", "hr_pa_pct", "matchup", "test_score",
        "streak_label",
        "pa", "barrel_pct", "iso", "xwoba", "xwobacon",
        "pitch_match_score", "pitch_hr_score", "best_pitch", "best_pitch_xwoba", "worst_pitch",
        "fb_pct", "la", "avg_ev", "hard_hit",
        "k_pct", "bb_pct", "whiff_pct",
        "home_run", "recent_hr", "sleeper_score",
    ] if c in matchup_df.columns]

    # Auto-hide columns that are completely empty (no point showing them)
    cols_to_show = [c for c in cols_to_show if matchup_df[c].notna().any()]

    if not qualified.empty:
        st.markdown(f"**{team_label}**")
        st.dataframe(
            _style_matchup_df(qualified[cols_to_show]),
            hide_index=True, use_container_width=True,
            column_config=build_col_config(),
        )

    if not insufficient.empty:
        with st.expander(f"⚠️ {team_label} — Insufficient Sample ({len(insufficient)} hitters below {INSUFFICIENT_PA_THRESHOLD} PA)"):
            st.dataframe(
                _style_matchup_df(insufficient[cols_to_show]),
                hide_index=True, use_container_width=True,
                column_config=build_col_config(),
            )


for _, game in slate.iterrows():
    ctx = game_context_map.get(game["gamePk"], {})
    if not ctx:
        continue

    # RAIN-RISK BANNER - shown ABOVE the game header so it can't be missed
    wx = ctx.get("weather") or {}
    pp = wx.get("precip_prob")
    if pp is not None and pp >= 80:
        st.error(
            f"🌧️ **HIGH RAIN RISK ({pp:.0f}%)** — this game may be delayed or "
            f"postponed. Treat projections as conditional on the game being played."
        )
    elif pp is not None and pp >= 50:
        st.warning(
            f"☔ **Rain risk ({pp:.0f}%)** — possible delay. Projections still apply if game is played."
        )

    away_tab = f"✈️ {game['away_team_abbr']} @ {game['home_team_abbr']}"
    home_tab = f"🏠 {game['home_team_abbr']} vs {game['away_team_abbr']}"

    # Game header with start time
    away_k_mean = ctx["away_k_proj"].get("mean") if ctx.get("away_k_proj") else None
    home_k_mean = ctx["home_k_proj"].get("mean") if ctx.get("home_k_proj") else None

    # Format game time to user's local timezone (assumes ET, which is most common)
    game_dt = game.get("gameTime")
    time_str = ""
    if isinstance(game_dt, pd.Timestamp):
        try:
            # Convert UTC to US Eastern (most common for MLB schedules)
            local_dt = game_dt.tz_convert("US/Eastern") if game_dt.tzinfo else game_dt
            # 12-hour format, strip leading zero on hour (Windows-safe)
            time_str = local_dt.strftime("%I:%M %p ET").lstrip("0")
        except Exception:
            try:
                time_str = game_dt.strftime("%I:%M %p").lstrip("0")
            except Exception:
                time_str = ""

    header_bits = [f"### {game['away_team_abbr']} @ {game['home_team_abbr']}"]
    if time_str:
        header_bits.append(f"🕐 {time_str}")
    if game.get("away_pitcher") and game.get("home_pitcher"):
        header_bits.append(
            f"**{game['away_pitcher']}** vs **{game['home_pitcher']}**"
        )
    st.markdown(" · ".join(header_bits))

    # Lineup confirmation status - warn if using roster-fill
    away_conf = ctx.get("away_lineup_confirmed", True)
    home_conf = ctx.get("home_lineup_confirmed", True)
    if not away_conf or not home_conf:
        unconfirmed = []
        if not away_conf:
            unconfirmed.append(game.get("away_team_abbr", "AWAY"))
        if not home_conf:
            unconfirmed.append(game.get("home_team_abbr", "HOME"))
        st.warning(
            f"⚠️ **Lineup not yet posted for {', '.join(unconfirmed)}** — "
            "showing alphabetical roster fill. Lineup-position-based PA scaling "
            "is DISABLED for this game; all hitters use league-avg 4.2 PA. "
            "Refresh after lineups post for accurate per-PA projections."
        )

    # Pull-wind interaction summary, if applicable
    pull_summaries = ctx.get("pull_wind_summary", [])
    if pull_summaries:
        pull_msg = " · ".join(pull_summaries)
        st.caption(f"🎯 **Pull-side wind:** {pull_msg}")

    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("Venue", game.get("venue", "—"))
    with info_cols[1]:
        wx = ctx.get("weather") or {}
        if wx and not wx.get("error") and wx.get("temp_f") is not None:
            temp = wx.get("temp_f")
            wind = wx.get("wind_mph", 0)
            wind_dir_deg = wx.get("wind_dir_deg")
            # Convert degrees to compass direction
            if wind_dir_deg is not None:
                dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                idx = int((wind_dir_deg + 22.5) / 45) % 8
                wind_dir = dirs[idx]
            else:
                wind_dir = ""
            pp = wx.get("precip_prob", 0) or 0
            rain_str = f" · ☔{pp:.0f}%" if pp >= 30 else ""
            st.metric(
                "Weather",
                f"{temp:.0f}°F · {wind:.0f}mph {wind_dir}{rain_str}",
                help=ctx.get("summary", ""),
            )
        else:
            wx_help = ctx.get("summary", "Weather not loaded")
            st.metric("Weather", "—", help=wx_help)
    with info_cols[2]:
        st.metric("Park × Wx", f"{ctx.get('hr_mult', 1.0):.2f}×")
    with info_cols[3]:
        vg = ctx.get("vegas") or {}
        if vg.get("total"):
            st.metric("Vegas total", f"{vg['total']:.1f}")
        else:
            st.metric("Vegas total", "—")

    # Per-game Top HR Hitter + Top Sleeper, combining both lineups
    am = ctx.get("away_matchup")
    hm = ctx.get("home_matchup")
    pickable = []
    if am is not None and not am.empty:
        a_copy = am.copy()
        a_copy["_team"] = game.get("away_team_abbr", "")
        pickable.append(a_copy)
    if hm is not None and not hm.empty:
        h_copy = hm.copy()
        h_copy["_team"] = game.get("home_team_abbr", "")
        pickable.append(h_copy)
    if pickable:
        combined = pd.concat(pickable, ignore_index=True)
        pick_cols = st.columns(2)
        with pick_cols[0]:
            if "hr_game_pct" in combined.columns:
                valid = combined.dropna(subset=["hr_game_pct"])
                if not valid.empty:
                    top = valid.sort_values("hr_game_pct", ascending=False).iloc[0]
                    pct = top.get("hr_game_pct", 0)
                    alert = top.get("alert", "")
                    st.markdown(
                        f"**🎯 Best HR Play**: {alert} **{top['player_name']}** "
                        f"({top['_team']}) — {pct:.1f}% HR Game"
                    )
        with pick_cols[1]:
            if "sleeper_score" in combined.columns:
                valid = combined.dropna(subset=["sleeper_score"])
                if not valid.empty:
                    # Filter: sleeper means meaningful HR upside despite low season pace
                    sleepers_only = valid[valid["sleeper_score"] > 0]
                    if not sleepers_only.empty:
                        sl = sleepers_only.sort_values("sleeper_score", ascending=False).iloc[0]
                        sc = sl.get("sleeper_score", 0)
                        hr_pct = sl.get("hr_game_pct", 0) or 0
                        st.markdown(
                            f"**💎 Best Sleeper**: **{sl['player_name']}** "
                            f"({sl['_team']}) — sleeper {sc:.1f}, HR {hr_pct:.1f}%"
                        )

    tabs = st.tabs([away_tab, home_tab, "🎯 K Projections"])
    with tabs[0]:
        render_matchup_section(ctx.get("away_matchup"), game['away_team_abbr'])
    with tabs[1]:
        render_matchup_section(ctx.get("home_matchup"), game['home_team_abbr'])
    with tabs[2]:
        kp1, kp2 = st.columns(2)
        for col, side, label in [
            (kp1, "away", game.get("away_pitcher") or "TBD"),
            (kp2, "home", game.get("home_pitcher") or "TBD"),
        ]:
            with col:
                pp = ctx.get(f"{side}_k_proj") or {}
                if not pp or pp.get("mean") is None:
                    st.write(f"**{label}** — no real K/9 data available")
                    p_row = ctx.get(f"{side}_p_row") or {}
                    if p_row:
                        avail = [k for k, v in p_row.items()
                                  if v is not None and not (isinstance(v, float) and pd.isna(v))
                                  and k in ("k9", "k_percent", "era", "whip", "ip")]
                        if avail:
                            st.caption(f"Has: {', '.join(avail)}")
                        else:
                            st.caption("Pitcher has no real stat data this season yet")
                    continue
                st.markdown(
                    f"**{label}** · Projected K: **{pp['mean']:.1f}** "
                    f"(range {pp.get('low', 0):.1f}–{pp.get('high', 0):.1f})"
                )
                st.caption(
                    f"Blended K/9: {pp.get('blended_k9', 0):.2f} · "
                    f"Lineup adj: {pp.get('lineup_adj', 1):.2f}×"
                )
                lines = pd.DataFrame([
                    {"Line": f"Over {x} K", "Prob": round(pp.get(f"p_over_{x}", 0) * 100, 1)}
                    for x in ["5.5", "6.5", "7.5", "8.5"]
                ])
                st.dataframe(
                    lines, hide_index=True, use_container_width=True,
                    column_config={
                        "Line": st.column_config.TextColumn("Line"),
                        "Prob": st.column_config.NumberColumn("P(Over)", format="%.1f%%"),
                    },
                )

    if use_bvp or not pitcher_arsenal_all.empty:
        with st.expander("🔬 Deep dive: Pitcher arsenal + historical BvP"):
            st.caption(
                "**What this shows:** the actual pitches each pitcher throws (usage %, "
                "velocity, xwOBA allowed) and any career-history batter-vs-pitcher "
                "stats. Use this to spot if a hitter has seen this pitcher 10+ times "
                "and crushed them, or to see what pitches the pitcher leans on."
            )
            sub1, sub2 = st.columns(2)
            with sub1:
                if use_bvp:
                    h_pid = safe_int(game.get("home_pitcher_id"))
                    if h_pid:
                        st.markdown(f"**{game['away_team_abbr']} BvP history**")
                        st.caption(f"vs {game.get('home_pitcher', 'TBD')}")
                if not pitcher_arsenal_all.empty:
                    h_pid = safe_int(game.get("home_pitcher_id"))
                    if h_pid:
                        ars = pitcher_arsenal_all[pitcher_arsenal_all["player_id"] == h_pid]
                        if not ars.empty:
                            st.markdown(f"**{game.get('home_pitcher', 'TBD')} arsenal**")
                            st.dataframe(ars, hide_index=True, use_container_width=True)
            with sub2:
                if use_bvp:
                    a_pid = safe_int(game.get("away_pitcher_id"))
                    if a_pid:
                        st.markdown(f"**{game['home_team_abbr']} BvP history**")
                        st.caption(f"vs {game.get('away_pitcher', 'TBD')}")
                if not pitcher_arsenal_all.empty:
                    a_pid = safe_int(game.get("away_pitcher_id"))
                    if a_pid:
                        ars = pitcher_arsenal_all[pitcher_arsenal_all["player_id"] == a_pid]
                        if not ars.empty:
                            st.markdown(f"**{game.get('away_pitcher', 'TBD')} arsenal**")
                            st.dataframe(ars, hide_index=True, use_container_width=True)

    st.divider()


st.caption(
    f"Built {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
    f"Sources: MLB Stats API, Baseball Savant, Open-Meteo · "
    f"Posey MLB HR & K Data"
)
