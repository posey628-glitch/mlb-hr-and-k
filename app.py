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

from data_fetcher import (
    get_slate, get_lineup, get_team_roster,
    get_hitter_stats, get_pitcher_stats, get_pitcher_arsenal,
    get_pitcher_recent_form, get_hitter_recent_form_trad,
    get_sprint_speed,
)

# Optional newer functions added in updated data_fetcher.py
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
    from splits import get_career_bvp_aggregate, get_similar_arsenal_aggregate
    HAVE_SPLITS = True
except Exception:
    HAVE_SPLITS = False

try:
    from pitch_match import pitch_match_score_for_hitter
    HAVE_PITCH_MATCH = True
except Exception:
    HAVE_PITCH_MATCH = False

try:
    from game_context import get_umpire_for_game, get_catcher_framing_for_game, get_vegas_for_game
    HAVE_GAME_CONTEXT = True
except Exception:
    HAVE_GAME_CONTEXT = False


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
    use_sprint_speed = st.checkbox("Sprint speed", value=True)
    use_pitch_match = st.checkbox("Pitch match score", value=HAVE_PITCH_MATCH)
    use_bvp = st.checkbox("Batter vs Pitcher history", value=HAVE_SPLITS)
    use_weather = st.checkbox("Weather + park factors", value=True)
    use_ump = st.checkbox("Umpire + catcher framing", value=HAVE_GAME_CONTEXT)
    use_vegas = st.checkbox("Vegas totals", value=HAVE_GAME_CONTEXT)

    st.subheader("Display")
    show_diagnostic = st.checkbox("Show data diagnostic", value=False)
    show_legend = st.checkbox("Show legend / glossary", value=True)

    st.divider()
    st.caption("Real data only - empty cells mean we couldn't fetch that stat.")

INSUFFICIENT_PA_THRESHOLD = pa_threshold_for_date(selected_date)


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
if not hitter_trad.empty and "player_id" in hitter_stats.columns:
    hitter_stats["player_id"] = pd.to_numeric(hitter_stats["player_id"], errors="coerce").astype("Int64")
    hitter_trad["player_id"] = pd.to_numeric(hitter_trad["player_id"], errors="coerce").astype("Int64")
    drop = [c for c in ["player_name"] if c in hitter_trad.columns]
    hitter_stats = hitter_stats.merge(
        hitter_trad.drop(columns=drop, errors="ignore"),
        on="player_id", how="left", suffixes=("", "_t"),
    )
if not pitcher_trad.empty and "player_id" in pitcher_stats.columns:
    pitcher_stats["player_id"] = pd.to_numeric(pitcher_stats["player_id"], errors="coerce").astype("Int64")
    pitcher_trad["player_id"] = pd.to_numeric(pitcher_trad["player_id"], errors="coerce").astype("Int64")
    drop = [c for c in ["player_name"] if c in pitcher_trad.columns]
    pitcher_stats = pitcher_stats.merge(
        pitcher_trad.drop(columns=drop, errors="ignore"),
        on="player_id", how="left", suffixes=("", "_t"),
    )

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

p_slate = build_pitcher_slate(slate, pitcher_stats, pitcher_recent=pitcher_recent_map,
                                slate_date=selected_date)

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
        "test_score", "kHR", "proj_k", "form_arrow",
        "era", "whip", "k9", "bb9", "hr9",
        "ip", "games_started", "games_played", "ip_per_outing",
        "k_pct", "whiff_pct",
        "xwoba_allowed", "barrel_allowed",
        "recent_era", "recent_k9", "days_rest", "avg_recent_pitches",
        "reliability",
    ] if c in p_slate.columns]

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
            help="0-100 composite. K-blended (30%) + Whiff (20%) + xwOBA suppression (25%) + ERA (15%) + base K% (10%), multiplied by reliability factor.",
        ),
        "kHR": st.column_config.NumberColumn(
            "kHR", format="%.1f",
            help="Strikeout-focused score, also reliability-adjusted.",
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
        "recent_era": st.column_config.NumberColumn("L5 ERA", format="%.2f"),
        "recent_k9": st.column_config.NumberColumn("L5 K/9", format="%.2f"),
        "days_rest": st.column_config.NumberColumn("Rest"),
        "avg_recent_pitches": st.column_config.NumberColumn("Pitches"),
        "reliability": st.column_config.NumberColumn(
            "Conf", format="%.2f", width="small",
            help="Reliability factor (0.3 - 1.0). Multiplier on Test/kHR scores. Low = small sample / reliever / bulk relief role.",
        ),
    }
    st.dataframe(
        p_slate[show_cols], hide_index=True, use_container_width=True,
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
    if use_weather and venue in PARKS:
        try:
            park_info = PARKS[venue]
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
            wx_mult, _summary = hr_multiplier(weather, park_info)
            wx_summary = weather.get("summary", "")
        except Exception:
            weather = {}
            wx_mult = 1.0

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

    try:
        away_lineup = get_lineup(int(game["gamePk"]), "away")
    except Exception:
        away_lineup = []
    if not away_lineup:
        try:
            away_lineup = [
                {"id": p["id"], "name": p["name"], "position": p["position"]}
                for p in get_team_roster(int(game["away_team_id"]))[:9]
            ]
        except Exception:
            away_lineup = []

    try:
        home_lineup = get_lineup(int(game["gamePk"]), "home")
    except Exception:
        home_lineup = []
    if not home_lineup:
        try:
            home_lineup = [
                {"id": p["id"], "name": p["name"], "position": p["position"]}
                for p in get_team_roster(int(game["home_team_id"]))[:9]
            ]
        except Exception:
            home_lineup = []

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

    # Pitch match score
    if use_pitch_match and HAVE_PITCH_MATCH:
        try:
            if home_p_row and not away_matchup.empty:
                pitch_scores = []
                for _, hitter_row in away_matchup.iterrows():
                    pid = hitter_row.get("player_id")
                    if pid is None or pd.isna(pid):
                        pitch_scores.append(None)
                        continue
                    try:
                        ps = pitch_match_score_for_hitter(
                            int(pid), home_p_row, pitcher_arsenal_all
                        )
                        pitch_scores.append(ps.get("score") if isinstance(ps, dict) else ps)
                    except Exception:
                        pitch_scores.append(None)
                away_matchup["pitch_match_score"] = pitch_scores
                # Best pitch + worst pitch labels
                best_pitch_data = []
                for _, hitter_row in away_matchup.iterrows():
                    pid = hitter_row.get("player_id")
                    if pid is None or pd.isna(pid):
                        best_pitch_data.append({"best_pitch": None, "best_pitch_xwoba": None,
                                                  "worst_pitch": None})
                        continue
                    try:
                        ps = pitch_match_score_for_hitter(
                            int(pid), home_p_row, pitcher_arsenal_all
                        )
                        if isinstance(ps, dict):
                            best_pitch_data.append({
                                "best_pitch": ps.get("best_pitch"),
                                "best_pitch_xwoba": ps.get("best_pitch_xwoba"),
                                "worst_pitch": ps.get("worst_pitch"),
                            })
                        else:
                            best_pitch_data.append({"best_pitch": None,
                                                      "best_pitch_xwoba": None,
                                                      "worst_pitch": None})
                    except Exception:
                        best_pitch_data.append({"best_pitch": None,
                                                  "best_pitch_xwoba": None,
                                                  "worst_pitch": None})
                bpd_df = pd.DataFrame(best_pitch_data)
                for c in ["best_pitch", "best_pitch_xwoba", "worst_pitch"]:
                    away_matchup[c] = bpd_df[c].values

            if away_p_row and not home_matchup.empty:
                pitch_scores = []
                for _, hitter_row in home_matchup.iterrows():
                    pid = hitter_row.get("player_id")
                    if pid is None or pd.isna(pid):
                        pitch_scores.append(None)
                        continue
                    try:
                        ps = pitch_match_score_for_hitter(
                            int(pid), away_p_row, pitcher_arsenal_all
                        )
                        pitch_scores.append(ps.get("score") if isinstance(ps, dict) else ps)
                    except Exception:
                        pitch_scores.append(None)
                home_matchup["pitch_match_score"] = pitch_scores
                best_pitch_data = []
                for _, hitter_row in home_matchup.iterrows():
                    pid = hitter_row.get("player_id")
                    if pid is None or pd.isna(pid):
                        best_pitch_data.append({"best_pitch": None, "best_pitch_xwoba": None,
                                                  "worst_pitch": None})
                        continue
                    try:
                        ps = pitch_match_score_for_hitter(
                            int(pid), away_p_row, pitcher_arsenal_all
                        )
                        if isinstance(ps, dict):
                            best_pitch_data.append({
                                "best_pitch": ps.get("best_pitch"),
                                "best_pitch_xwoba": ps.get("best_pitch_xwoba"),
                                "worst_pitch": ps.get("worst_pitch"),
                            })
                        else:
                            best_pitch_data.append({"best_pitch": None,
                                                      "best_pitch_xwoba": None,
                                                      "worst_pitch": None})
                    except Exception:
                        best_pitch_data.append({"best_pitch": None,
                                                  "best_pitch_xwoba": None,
                                                  "worst_pitch": None})
                bpd_df = pd.DataFrame(best_pitch_data)
                for c in ["best_pitch", "best_pitch_xwoba", "worst_pitch"]:
                    home_matchup[c] = bpd_df[c].values
        except Exception:
            pass

    # HR probability per hitter
    for matchup_df, opp_p_row in [(away_matchup, home_p_row), (home_matchup, away_p_row)]:
        if matchup_df.empty:
            continue
        hr_pa, hr_game, verdicts, signals = [], [], [], []
        for _, hr in matchup_df.iterrows():
            row_dict = hr.to_dict()
            pa = safe_float(row_dict.get("pa"))
            sample = int(pa) if pa is not None else None
            # hr_prob_per_pa signatures vary across deployed versions of props.py.
            # Try the modern kwarg form first, fall back to positional, fall back to no env factors.
            try:
                p_pa = hr_prob_per_pa(
                    row_dict, opp_p_row,
                    park_factor=park_mult, weather_mult=wx_mult,
                )
            except TypeError:
                try:
                    p_pa = hr_prob_per_pa(
                        row_dict, opp_p_row,
                        park_hr_factor=park_mult, weather_hr_factor=wx_mult,
                    )
                except TypeError:
                    try:
                        p_pa = hr_prob_per_pa(row_dict, opp_p_row)
                    except Exception:
                        p_pa = None
            p_game = hr_prob_full_game(p_pa) if p_pa is not None else None
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
    }
    matchup_tables[game["gamePk"]] = (away_matchup, home_matchup)


st.divider()


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

        top5 = q.sort_values("pick_score", ascending=False).head(5).reset_index(drop=True)
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

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**🎯 Top 10 HR Plays**")
        if "hr_game_pct" in qualified.columns:
            top_hr = qualified.dropna(subset=["hr_game_pct"]).sort_values(
                "hr_game_pct", ascending=False
            ).head(10)
            if not top_hr.empty:
                cols = [c for c in [
                    "player_name", "team", "game", "opp_pitcher",
                    "hr_game_pct", "hr_pa_pct", "barrel_pct", "iso", "matchup",
                ] if c in top_hr.columns]
                disp = top_hr[cols].copy().reset_index(drop=True)
                st.dataframe(
                    disp, hide_index=True, use_container_width=True,
                    column_config={
                        "player_name": st.column_config.TextColumn("Hitter"),
                        "team": st.column_config.TextColumn("Tm"),
                        "game": st.column_config.TextColumn("Game"),
                        "opp_pitcher": st.column_config.TextColumn("vs Pitcher"),
                        "hr_game_pct": st.column_config.NumberColumn("HR Game%", format="%.1f%%"),
                        "hr_pa_pct": st.column_config.NumberColumn("HR PA%", format="%.2f%%"),
                        "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
                        "iso": st.column_config.NumberColumn("ISO", format="%.3f"),
                        "matchup": st.column_config.NumberColumn("Matchup", format="%.1f"),
                    },
                )
            else:
                st.caption("No HR projections available.")

    with col_right:
        st.markdown("**💎 Top 10 Sleepers**")
        if "sleeper_score" in qualified.columns:
            top_sleep = qualified.dropna(subset=["sleeper_score"]).sort_values(
                "sleeper_score", ascending=False
            ).head(10)
            if not top_sleep.empty:
                cols = [c for c in [
                    "player_name", "team", "game", "opp_pitcher",
                    "sleeper_score", "hr_game_pct", "home_run", "recent_hr", "barrel_pct",
                ] if c in top_sleep.columns]
                disp = top_sleep[cols].copy().reset_index(drop=True)
                st.dataframe(
                    disp, hide_index=True, use_container_width=True,
                    column_config={
                        "player_name": st.column_config.TextColumn("Hitter"),
                        "team": st.column_config.TextColumn("Tm"),
                        "game": st.column_config.TextColumn("Game"),
                        "opp_pitcher": st.column_config.TextColumn("vs Pitcher"),
                        "sleeper_score": st.column_config.NumberColumn(
                            "Sleeper",
                            format="%.1f",
                            help="HR prob percentile MINUS season HR percentile. "
                                 "Higher = bigger surprise candidate.",
                        ),
                        "hr_game_pct": st.column_config.NumberColumn("HR Game%", format="%.1f%%"),
                        "home_run": st.column_config.NumberColumn("Season HR", format="%d"),
                        "recent_hr": st.column_config.NumberColumn("L15 HR", format="%d"),
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
        "alert": st.column_config.TextColumn("Signal", width="small"),
        "player_name": st.column_config.TextColumn("Hitter"),
        "lineup_pos": st.column_config.NumberColumn("#", width="small"),
        "bats": st.column_config.TextColumn("B", width="small"),
        "position": st.column_config.TextColumn("Pos", width="small"),
        "hr_game_pct": st.column_config.NumberColumn("HR Game%", format="%.1f%%"),
        "hr_pa_pct": st.column_config.NumberColumn("HR PA%", format="%.2f%%"),
        "matchup": st.column_config.NumberColumn("Matchup", format="%.1f"),
        "test_score": st.column_config.NumberColumn("Test", format="%.1f"),
        "pa": st.column_config.NumberColumn("PA"),
        "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
        "iso": st.column_config.NumberColumn("ISO", format="%.3f"),
        "xwoba": st.column_config.NumberColumn("xwOBA", format="%.3f"),
        "xwobacon": st.column_config.NumberColumn("xwOBAcon", format="%.3f"),
        "pitch_match_score": st.column_config.NumberColumn("Pitch Match", format="%.1f"),
        "best_pitch": st.column_config.TextColumn("Best Pitch"),
        "best_pitch_xwoba": st.column_config.NumberColumn("Best xwOBA", format="%.3f"),
        "worst_pitch": st.column_config.TextColumn("Worst Pitch"),
        "fb_pct": st.column_config.NumberColumn("FB%", format="%.1f%%"),
        "la": st.column_config.NumberColumn("LA"),
        "sprint_speed": st.column_config.NumberColumn("Sprint"),
        "k_pct": st.column_config.NumberColumn("K%", format="%.1f%%"),
        "bb_pct": st.column_config.NumberColumn("BB%", format="%.1f%%"),
        "whiff_pct": st.column_config.NumberColumn("Whiff%", format="%.1f%%"),
        "home_run": st.column_config.NumberColumn("HR", format="%d"),
        "recent_hr": st.column_config.NumberColumn("L15 HR"),
        "sleeper_score": st.column_config.NumberColumn("Sleeper", format="%.1f"),
        "verdict": st.column_config.TextColumn("Verdict"),
    }


def render_matchup_section(matchup_df: pd.DataFrame, team_label: str):
    if matchup_df is None or matchup_df.empty:
        st.caption(f"{team_label}: no lineup data available yet.")
        return

    # Split qualified vs insufficient sample
    if "pa" in matchup_df.columns:
        qualified = matchup_df[matchup_df["pa"].notna() & (matchup_df["pa"] >= INSUFFICIENT_PA_THRESHOLD)]
        insufficient = matchup_df[~(matchup_df["pa"].notna() & (matchup_df["pa"] >= INSUFFICIENT_PA_THRESHOLD))]
    else:
        qualified = matchup_df
        insufficient = pd.DataFrame()

    cols_to_show = [c for c in [
        "alert", "player_name", "lineup_pos", "bats", "position",
        "hr_game_pct", "hr_pa_pct", "matchup", "test_score",
        "pa", "barrel_pct", "iso", "xwoba", "xwobacon",
        "pitch_match_score", "best_pitch", "best_pitch_xwoba", "worst_pitch",
        "fb_pct", "la", "sprint_speed",
        "k_pct", "bb_pct", "whiff_pct",
        "home_run", "recent_hr", "sleeper_score",
    ] if c in matchup_df.columns]

    if not qualified.empty:
        st.markdown(f"**{team_label}**")
        st.dataframe(
            qualified[cols_to_show], hide_index=True, use_container_width=True,
            column_config=build_col_config(),
        )

    if not insufficient.empty:
        with st.expander(f"⚠️ {team_label} — Insufficient Sample ({len(insufficient)} hitters below {INSUFFICIENT_PA_THRESHOLD} PA)"):
            st.dataframe(
                insufficient[cols_to_show], hide_index=True, use_container_width=True,
                column_config=build_col_config(),
            )


for _, game in slate.iterrows():
    ctx = game_context_map.get(game["gamePk"], {})
    if not ctx:
        continue

    away_tab = f"✈️ {game['away_team_abbr']} @ {game['home_team_abbr']}"
    home_tab = f"🏠 {game['home_team_abbr']} vs {game['away_team_abbr']}"

    # Game header
    away_k_mean = ctx["away_k_proj"].get("mean") if ctx.get("away_k_proj") else None
    home_k_mean = ctx["home_k_proj"].get("mean") if ctx.get("home_k_proj") else None
    header_bits = [f"### {game['away_team_abbr']} @ {game['home_team_abbr']}"]
    if game.get("away_pitcher") and game.get("home_pitcher"):
        header_bits.append(
            f"**{game['away_pitcher']}** vs **{game['home_pitcher']}**"
        )
    st.markdown(" · ".join(header_bits))

    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("Venue", game.get("venue", "—"))
    with info_cols[1]:
        wx = ctx.get("weather") or {}
        if wx:
            temp = wx.get("temp_f", "—")
            wind = wx.get("wind_mph", "—")
            wind_dir = wx.get("wind_dir", "—")
            st.metric("Weather", f"{temp}°F · {wind} mph {wind_dir}")
        else:
            st.metric("Weather", "—")
    with info_cols[2]:
        st.metric("Park × Wx", f"{ctx.get('hr_mult', 1.0):.2f}×")
    with info_cols[3]:
        vg = ctx.get("vegas") or {}
        if vg.get("total"):
            st.metric("Vegas total", f"{vg['total']:.1f}")
        else:
            st.metric("Vegas total", "—")

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
        with st.expander("🔬 Supplemental (BvP + Arsenals)"):
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
