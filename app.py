"""
app.py — Posey MLB HR & K Data Dashboard
==========================================
Real data only. No fake defaults or imputed averages.

Data sources (all free, no API keys):
  - MLB Stats API   - slate, lineups, traditional stats, umpire
  - Baseball Savant - Statcast (xwOBA, ISO, Barrel%, pitch arsenals,
                       sprint speed, run values, catcher framing)
  - Open-Meteo      - game-time weather
  - ESPN            - Vegas implied totals

Hitters with insufficient PA appear in a separate section below the main
matchup table. The PA threshold scales with how deep we are in the season
(60 PA in April, 100 by midseason, 150+ in September).

⚠️ Bet responsibly. Lines are sharp. Use as one input only.
"""

from __future__ import annotations

from datetime import date, datetime
import pandas as pd
import streamlit as st

from data_fetcher import (
    get_slate, get_lineup, get_team_roster,
    get_hitter_stats, get_pitcher_stats, get_pitcher_arsenal,
    get_hitter_traditional, get_pitcher_traditional,
    get_pitcher_recent_form, get_hitter_recent_form_trad,
    get_sprint_speed,
)
from models import build_matchup_table, build_pitcher_slate
from park_factors import get_park
from weather import fetch_weather, hr_multiplier
from sleepers import hr_probability, find_sleepers, grand_slam_probability
from splits import bvp_for_lineup
from pitch_match import get_hitter_pitch_arsenal, lineup_pitch_match
from game_context import (
    get_umpire_for_game, get_vegas_totals, get_pitcher_workload,
    ttop_multiplier, park_hand_factor,
)
from props import hr_prob_per_pa, hr_prob_full_game, k_total_projection


st.set_page_config(page_title="Posey MLB HR & K Data", layout="wide", page_icon="⚾")
st.markdown(
    "<style>div[data-testid='stMetric']{min-height:85px;}</style>",
    unsafe_allow_html=True,
)


# ===========================================================================
# DYNAMIC SAMPLE THRESHOLD - scales with how deep in the season
# ===========================================================================

def get_pa_threshold(slate_date):
    """
    Sample-size threshold for trusting season stats.
    MLB season starts ~March 28. By month into season:
      Month 1 (Apr):   40 PA
      Month 2 (May):   80 PA
      Month 3 (Jun): 120 PA
      Month 4 (Jul): 160 PA
      Month 5+:      200 PA
    """
    try:
        if isinstance(slate_date, str):
            d = datetime.fromisoformat(slate_date).date()
        else:
            d = slate_date
        season_start = date(d.year, 3, 28)
        if d < season_start:
            return 100  # offseason / spring training - arbitrary
        days_in = (d - season_start).days
        if days_in <= 30:
            return 40
        elif days_in <= 60:
            return 80
        elif days_in <= 90:
            return 120
        elif days_in <= 120:
            return 160
        return 200
    except Exception:
        return 100


# ===========================================================================
# HELPERS
# ===========================================================================

def calc_4tier(score, scale=(45, 65)):
    """Map score → 4-tier emoji. Returns ⚪ if score is None/NaN."""
    if score is None or pd.isna(score):
        return "⚪"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "⚪"
    low, high = scale
    mid = (low + high) / 2
    if s >= high:
        return "🟢"
    if s >= mid:
        return "🟡"
    if s >= low:
        return "🟠"
    return "🔴"


def safe_int(val, default=None):
    try:
        if val is None or pd.isna(val):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def split_by_sample_size(df, threshold):
    """Split into (sufficient, insufficient) based on PA."""
    if df is None or df.empty or "pa" not in df.columns:
        return df, pd.DataFrame()
    insufficient_mask = df["pa"].notna() & (df["pa"] < threshold)
    return df[~insufficient_mask].copy(), df[insufficient_mask].copy()


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.title("⚾ Posey MLB Props")
    selected_date = st.date_input("Slate date", value=date.today())

    pa_threshold = get_pa_threshold(selected_date)
    custom_threshold = st.number_input(
        f"PA threshold (auto: {pa_threshold})",
        min_value=0, max_value=500, value=pa_threshold, step=10,
        help="Hitters below this PA go to Insufficient Sample section. "
             "Auto-scales to season progress.",
    )

    st.markdown("### Data Sources")
    use_pitch_match = st.checkbox("Pitch-match (Statcast)", value=True)
    use_vegas = st.checkbox("Vegas totals (ESPN)", value=True)
    use_umpire = st.checkbox("Umpire (Stats API)", value=False)
    use_recent_form = st.checkbox("Recent form (slower)", value=False)
    use_sprint_speed = st.checkbox("Sprint Speed (Statcast)", value=True)
    use_bvp = st.checkbox("BvP supplemental", value=False)

    st.markdown("---")
    st.caption(f"Refreshed: {datetime.now().strftime('%I:%M %p')}")
    if st.button("🔄 Force refresh"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.warning(
        "**Bet responsibly.** Lines are sharp. Bet flat units. "
        "Shop across books. Set a daily limit."
    )

INSUFFICIENT_PA_THRESHOLD = custom_threshold


# ===========================================================================
# LOAD DATA
# ===========================================================================
with st.spinner("Loading slate..."):
    slate = get_slate(selected_date.isoformat())

if slate is None or slate.empty:
    st.warning(f"No games on {selected_date}.")
    st.stop()

with st.spinner("Loading Statcast..."):
    hitter_stats = get_hitter_stats()
    pitcher_stats = get_pitcher_stats()
    try:
        hitter_trad = get_hitter_traditional()
    except Exception:
        hitter_trad = pd.DataFrame()
    try:
        pitcher_trad = get_pitcher_traditional()
    except Exception:
        pitcher_trad = pd.DataFrame()

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

# Sprint speed
if use_sprint_speed:
    try:
        sprint_df = get_sprint_speed()
        if not sprint_df.empty and "player_id" in sprint_df.columns and "player_id" in hitter_stats.columns:
            sprint_cols = [c for c in ["player_id", "sprint_speed", "hp_to_1b"]
                            if c in sprint_df.columns]
            hitter_stats = hitter_stats.merge(
                sprint_df[sprint_cols], on="player_id", how="left",
            )
    except Exception:
        pass

# Pitch arsenals
hitter_pitch_arsenal = pd.DataFrame()
pitcher_arsenal_all = pd.DataFrame()
if use_pitch_match:
    with st.spinner("Loading pitch-match data..."):
        try:
            hitter_pitch_arsenal = get_hitter_pitch_arsenal()
        except Exception:
            pass
        try:
            pitcher_arsenal_all = get_pitcher_arsenal()
        except Exception:
            pass

# Vegas
vegas_df = pd.DataFrame()
if use_vegas:
    try:
        vegas_df = get_vegas_totals(selected_date.isoformat())
    except Exception:
        pass


# ===========================================================================
# HEADER + DATA AVAILABILITY
# ===========================================================================
st.title(f"⚾ Posey MLB HR & K Data — {selected_date.strftime('%A, %B %d, %Y')}")

# Detailed data-source diagnostic
src_cols = st.columns(7)
src_cols[0].metric("Games", len(slate))
src_cols[1].metric("Statcast Hitters", f"{len(hitter_stats)}")
src_cols[2].metric("Statcast Pitchers", f"{len(pitcher_stats)}")

# Check what % of pitchers have real K/9 data (this is the biggest signal)
if not pitcher_trad.empty and "k9" in pitcher_trad.columns:
    pct_with_k9 = (pitcher_trad["k9"].notna().sum() / len(pitcher_trad) * 100) if len(pitcher_trad) else 0
    src_cols[3].metric("Pitchers w/ K9", f"{pct_with_k9:.0f}%")
else:
    src_cols[3].metric("Pitchers w/ K9", "0%")

# Check sprint speed
if use_sprint_speed:
    sprint_ok = "sprint_speed" in hitter_stats.columns and hitter_stats["sprint_speed"].notna().any()
    src_cols[4].metric("Sprint", "✓" if sprint_ok else "—")
else:
    src_cols[4].metric("Sprint", "off")

src_cols[5].metric("Vegas", "✓" if not vegas_df.empty else "—")
src_cols[6].metric("Pitch Arsenals", "✓" if not pitcher_arsenal_all.empty else "—")

# Sample size info
st.caption(
    f"📊 PA threshold for HR projections: **{INSUFFICIENT_PA_THRESHOLD}**. "
    f"Hitters below this go to Insufficient Sample. "
    f"({(date.today() - date(date.today().year, 3, 28)).days} days into season)"
)

with st.expander("📖 Dashboard Legend", expanded=False):
    leg1, leg2, leg3 = st.columns(3)
    with leg1:
        st.markdown("**📈 Core Metrics**")
        st.markdown(
            "- **Test:** Algorithmic baseline grade\n"
            "- **kHR:** Lineup K-resistance vs Pitcher\n"
            "- **Proj K:** K projection\n"
            "- **HR Game%:** Probability ≥1 HR today"
        )
        st.markdown("**🟢 Color Signals**")
        st.markdown(
            "- 🟢 **Strong play** (top tier)\n"
            "- 🟡 **Lean** (slight edge)\n"
            "- 🟠 **Caution** (slight fade)\n"
            "- 🔴 **Heavy fade**\n"
            "- ⚪ **Insufficient data**"
        )
    with leg2:
        st.markdown("**🔥 Hitter Metrics**")
        st.markdown(
            "- **ISO:** Power (SLG-BA)\n"
            "- **xwOBA / xwOBAcon:** Expected wOBA\n"
            "- **Brl%:** Barrel rate\n"
            "- **FB% / LA:** Fly ball + launch angle\n"
            "- **Sprint:** ft/sec (Statcast)"
        )
    with leg3:
        st.markdown("**💎 Pitcher Metrics**")
        st.markdown(
            "- **ERA / WHIP / K9 / BB9 / HR9:** Standard rates\n"
            "- **Whiff% / CSW%:** Swing-miss + called strikes\n"
            "- **xwOBA allowed / Brl%:** Statcast quality"
        )

st.divider()


# ===========================================================================
# BUILD PER-GAME CONTEXT
# ===========================================================================
game_context_map = {}
progress = st.progress(0.0, text="Assembling game environments...")

for idx, (_, game) in enumerate(slate.iterrows()):
    progress.progress((idx + 1) / len(slate), text=f"Game {idx+1}/{len(slate)}")

    venue = game.get("venue", "") or ""
    park = get_park(venue)

    try:
        gt = pd.to_datetime(game.get("gameTime"))
        if pd.isna(gt):
            gt = datetime.now()
    except Exception:
        gt = datetime.now()

    try:
        weather = fetch_weather(park.get("lat"), park.get("lon"), gt) if park.get("lat") else {}
    except Exception:
        weather = {}
    try:
        wx_mult, wx_summary = hr_multiplier(weather, park)
    except Exception:
        wx_mult, wx_summary = 1.0, "—"
    park_mult = park.get("hr_factor", 100) / 100.0
    full_hr_mult = wx_mult * park_mult

    vegas_row = None
    if not vegas_df.empty:
        try:
            match = vegas_df[
                (vegas_df.get("away_abbr") == game["away_team_abbr"]) &
                (vegas_df.get("home_abbr") == game["home_team_abbr"])
            ]
            if len(match):
                vegas_row = match.iloc[0].to_dict()
        except Exception:
            pass

    ump = {"name": "TBD", "k_factor": 1.0, "bb_factor": 1.0}
    if use_umpire:
        try:
            ump = get_umpire_for_game(int(game["gamePk"]))
        except Exception:
            pass

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
        for pid, row_dict in [(away_p_id, away_p_row), (home_p_id, home_p_row)]:
            if pid:
                try:
                    row_dict.update(get_pitcher_recent_form(int(pid)))
                    row_dict.update(get_pitcher_workload(int(pid)))
                except Exception:
                    pass

    away_recent, home_recent = {}, {}
    if use_recent_form:
        for p in away_lineup:
            if p.get("id"):
                try:
                    away_recent[p["id"]] = get_hitter_recent_form_trad(int(p["id"]))
                except Exception:
                    pass
        for p in home_lineup:
            if p.get("id"):
                try:
                    home_recent[p["id"]] = get_hitter_recent_form_trad(int(p["id"]))
                except Exception:
                    pass

    home_p_series = pd.Series(home_p_row) if home_p_row else None
    away_p_series = pd.Series(away_p_row) if away_p_row else None

    try:
        away_matchup = build_matchup_table(
            away_lineup, home_p_series,
            hitter_stats, pitcher_stats, recent_form_dict=away_recent,
        )
    except Exception:
        away_matchup = pd.DataFrame()

    try:
        home_matchup = build_matchup_table(
            home_lineup, away_p_series,
            hitter_stats, pitcher_stats, recent_form_dict=home_recent,
        )
    except Exception:
        home_matchup = pd.DataFrame()

    if use_pitch_match and not pitcher_arsenal_all.empty and not hitter_pitch_arsenal.empty:
        pm_cols = ["player_id", "pitch_match_score", "best_pitch",
                    "best_pitch_xwoba", "worst_pitch", "weighted_xwoba"]
        try:
            if home_p_id:
                away_pm = lineup_pitch_match(
                    away_lineup, home_p_id,
                    hitter_pitch_arsenal, pitcher_arsenal_all,
                )
                if not away_pm.empty and not away_matchup.empty:
                    keep = [c for c in pm_cols if c in away_pm.columns]
                    if "player_id" in keep:
                        away_matchup = away_matchup.merge(
                            away_pm[keep], on="player_id", how="left",
                        )
            if away_p_id:
                home_pm = lineup_pitch_match(
                    home_lineup, away_p_id,
                    hitter_pitch_arsenal, pitcher_arsenal_all,
                )
                if not home_pm.empty and not home_matchup.empty:
                    keep = [c for c in pm_cols if c in home_pm.columns]
                    if "player_id" in keep:
                        home_matchup = home_matchup.merge(
                            home_pm[keep], on="player_id", how="left",
                        )
        except Exception:
            pass

    try:
        away_matchup = hr_probability(away_matchup, home_p_series, full_hr_mult)
        home_matchup = hr_probability(home_matchup, away_p_series, full_hr_mult)
        away_matchup = find_sleepers(away_matchup, season_hr_col="home_run")
        home_matchup = find_sleepers(home_matchup, season_hr_col="home_run")
        away_matchup = grand_slam_probability(away_matchup, home_p_series, full_hr_mult)
        home_matchup = grand_slam_probability(home_matchup, away_p_series, full_hr_mult)
    except Exception:
        pass

    # Calibrated HR probability per hitter - ONLY when real data sufficient
    for matchup_df, opp_p_row in [(away_matchup, home_p_row),
                                     (home_matchup, away_p_row)]:
        if matchup_df is None or matchup_df.empty:
            continue
        hr_pa, hr_game, verdicts = [], [], []
        for _, hrow in matchup_df.iterrows():
            try:
                ph_factor = park_hand_factor(venue, hrow.get("bats", ""))
                lpos = hrow.get("lineup_pos", 5) or 5
                ttop = ttop_multiplier(int(lpos))
                pm_score = None
                if "pitch_match_score" in matchup_df.columns:
                    pmv = hrow.get("pitch_match_score")
                    if pmv is not None and not pd.isna(pmv):
                        pm_score = float(pmv)

                p_pa = hr_prob_per_pa(
                    hitter_row=hrow.to_dict(),
                    pitcher_row=opp_p_row,
                    park_factor=park_mult,
                    park_hand_factor=ph_factor,
                    weather_mult=wx_mult,
                    pitch_match_score=pm_score,
                    ttop_mult=ttop,
                    defense_factor=1.0,
                    min_pa=INSUFFICIENT_PA_THRESHOLD,
                )

                if p_pa is None:
                    hr_pa.append(None)
                    hr_game.append(None)
                    verdicts.append("⚪")
                    continue

                expected_pa = 4.3 if int(lpos) <= 5 else 3.8
                p_game = hr_prob_full_game(p_pa, expected_pa=expected_pa)

                hr_pa.append(round(p_pa * 100, 2))
                hr_game.append(round(p_game * 100, 1) if p_game is not None else None)

                m_val = hrow.get("matchup")
                if m_val is None or pd.isna(m_val):
                    verdicts.append("⚪")
                else:
                    avg = (float(m_val) + (p_game or 0) * 200) / 2
                    verdicts.append(calc_4tier(avg, scale=(45, 65)))
            except Exception:
                hr_pa.append(None)
                hr_game.append(None)
                verdicts.append("⚪")

        matchup_df["hr_pa_pct"] = hr_pa
        matchup_df["hr_game_pct"] = hr_game
        matchup_df["verdict"] = verdicts

    # K projection
    away_k_col = away_matchup["k_pct"] if "k_pct" in away_matchup.columns else pd.Series(dtype=float)
    home_k_col = home_matchup["k_pct"] if "k_pct" in home_matchup.columns else pd.Series(dtype=float)
    away_lineup_k_pct = float(away_k_col.mean()) if not away_k_col.empty and not away_k_col.isna().all() else None
    home_lineup_k_pct = float(home_k_col.mean()) if not home_k_col.empty and not home_k_col.isna().all() else None

    away_k_proj, home_k_proj = {}, {}
    try:
        if away_p_row:
            away_k_proj = k_total_projection(
                away_p_row, home_lineup_k_pct,
                ump_k_factor=ump.get("k_factor", 1.0),
            )
        if home_p_row:
            home_k_proj = k_total_projection(
                home_p_row, away_lineup_k_pct,
                ump_k_factor=ump.get("k_factor", 1.0),
            )
    except Exception:
        pass

    game_context_map[game["gamePk"]] = {
        "park": park, "weather": weather, "wx_mult": wx_mult,
        "park_mult": park_mult, "hr_mult": full_hr_mult,
        "summary": wx_summary, "vegas": vegas_row, "ump": ump,
        "away_lineup": away_lineup, "home_lineup": home_lineup,
        "away_p_row": away_p_row, "home_p_row": home_p_row,
        "away_matchup": away_matchup, "home_matchup": home_matchup,
        "away_k_proj": away_k_proj, "home_k_proj": home_k_proj,
    }

progress.empty()


# ===========================================================================
# STARTING PITCHER OVERVIEW
# ===========================================================================
st.subheader("📋 Starting Pitcher Overview")

try:
    pitcher_recent_data = {}
    for gpk, ctx in game_context_map.items():
        game_rows = slate[slate["gamePk"] == gpk]
        if game_rows.empty:
            continue
        g_row = game_rows.iloc[0]
        for side in ("away", "home"):
            pid = safe_int(g_row.get(f"{side}_pitcher_id"))
            if pid:
                pitcher_recent_data[pid] = {
                    "recent_era": ctx[f"{side}_p_row"].get("recent_era"),
                    "recent_k9": ctx[f"{side}_p_row"].get("recent_k9"),
                    "days_rest": ctx[f"{side}_p_row"].get("days_rest"),
                    "avg_recent_pitches": ctx[f"{side}_p_row"].get("avg_recent_pitches"),
                }
    pitcher_slate = build_pitcher_slate(slate, pitcher_stats, pitcher_recent_data)
except Exception as e:
    st.warning(f"Pitcher slate error: {e}")
    pitcher_slate = pd.DataFrame()

if not pitcher_slate.empty:
    if "test_score" in pitcher_slate.columns:
        pitcher_slate["alert"] = pitcher_slate["test_score"].apply(
            lambda x: calc_4tier(x, scale=(45, 65))
        )
    else:
        pitcher_slate["alert"] = "⚪"

    base_cols = ["alert", "pitcher_name", "team", "home_away", "opp", "throws"]
    metric_cols = ["test_score", "kHR", "proj_k", "form_arrow",
                    "era", "whip", "k9", "bb9", "hr9", "ip",
                    "k_pct", "whiff_pct", "csw_pct",
                    "xwoba_allowed", "barrel_allowed",
                    "recent_era", "recent_k9", "days_rest", "avg_recent_pitches"]
    keep = [c for c in base_cols + metric_cols if c in pitcher_slate.columns]
    display = pitcher_slate[keep].copy().reset_index(drop=True)

    col_config = {
        "alert": st.column_config.TextColumn("Signal",
            help="🟢=Elite / 🟡=Pace / 🟠=Caution / 🔴=Fade / ⚪=No data"),
        "pitcher_name": st.column_config.TextColumn("Pitcher"),
        "team": st.column_config.TextColumn("Tm"),
        "home_away": st.column_config.TextColumn("", width="small"),
        "opp": st.column_config.TextColumn("Opp"),
        "throws": st.column_config.TextColumn("T"),
        "test_score": st.column_config.NumberColumn("Test", format="%.1f"),
        "kHR": st.column_config.NumberColumn("kHR", format="%.1f"),
        "proj_k": st.column_config.NumberColumn("Proj K", format="%.1f",
            help="Real K/9 × 5.5 IP. Blank if no real K/9 data."),
        "form_arrow": st.column_config.TextColumn("Trend"),
        "era": st.column_config.NumberColumn("ERA", format="%.2f"),
        "whip": st.column_config.NumberColumn("WHIP", format="%.2f"),
        "k9": st.column_config.NumberColumn("K/9", format="%.2f"),
        "bb9": st.column_config.NumberColumn("BB/9", format="%.2f"),
        "hr9": st.column_config.NumberColumn("HR/9", format="%.2f"),
        "ip": st.column_config.NumberColumn("IP", format="%.1f"),
        "k_pct": st.column_config.NumberColumn("K%", format="%.1f"),
        "whiff_pct": st.column_config.NumberColumn("Whiff%", format="%.1f"),
        "csw_pct": st.column_config.NumberColumn("CSW%", format="%.1f"),
        "xwoba_allowed": st.column_config.NumberColumn("xwOBA", format="%.3f"),
        "barrel_allowed": st.column_config.NumberColumn("Brl%", format="%.1f"),
        "recent_era": st.column_config.NumberColumn("L5 ERA", format="%.2f"),
        "recent_k9": st.column_config.NumberColumn("L5 K/9", format="%.2f"),
        "days_rest": st.column_config.TextColumn("Rest"),
        "avg_recent_pitches": st.column_config.NumberColumn("Pitches", format="%d"),
    }

    st.dataframe(
        display, hide_index=True, use_container_width=True, height=400,
        column_config=col_config,
    )

st.divider()


# ===========================================================================
# GAME-BY-GAME MATCHUPS
# ===========================================================================
st.subheader("🎮 Isolated Game-by-Game Matchups")
st.caption("Real data only. Empty cells = data not available for that player.")


def build_col_config():
    return {
        "alert": st.column_config.TextColumn("Signal"),
        "player_name": st.column_config.TextColumn("Hitter"),
        "lineup_pos": st.column_config.NumberColumn("#", format="%d"),
        "position": st.column_config.TextColumn("Pos"),
        "bats": st.column_config.TextColumn("B"),
        "hr_game_pct": st.column_config.NumberColumn("HR Game%", format="%.1f%%"),
        "hr_pa_pct": st.column_config.NumberColumn("HR PA%", format="%.2f%%"),
        "matchup": st.column_config.NumberColumn("Matchup", format="%.1f"),
        "test_score": st.column_config.NumberColumn("Test", format="%.1f"),
        "pa": st.column_config.NumberColumn("PA", format="%d"),
        "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
        "iso": st.column_config.NumberColumn("ISO", format="%.3f"),
        "xwoba": st.column_config.NumberColumn("xwOBA", format="%.3f"),
        "xwobacon": st.column_config.NumberColumn("xwOBAcon", format="%.3f"),
        "pitch_match_score": st.column_config.NumberColumn("Pitch Match", format="%.1f"),
        "best_pitch": st.column_config.TextColumn("Best Pitch"),
        "best_pitch_xwoba": st.column_config.NumberColumn("Best xwOBA", format="%.3f"),
        "worst_pitch": st.column_config.TextColumn("Worst Pitch"),
        "fb_pct": st.column_config.NumberColumn("FB%", format="%.1f%%"),
        "la": st.column_config.NumberColumn("LA", format="%.1f°"),
        "sprint_speed": st.column_config.NumberColumn("Sprint", format="%.1f"),
        "k_pct": st.column_config.NumberColumn("K%", format="%.1f%%"),
        "bb_pct": st.column_config.NumberColumn("BB%", format="%.1f%%"),
        "whiff_pct": st.column_config.NumberColumn("Whiff%", format="%.1f%%"),
        "home_run": st.column_config.NumberColumn("HR", format="%d"),
        "recent_hr": st.column_config.NumberColumn("L15 HR", format="%d"),
        "sleeper_score": st.column_config.NumberColumn("Sleeper", format="%.1f"),
        "gs_score": st.column_config.NumberColumn("GS", format="%.2f"),
    }


def get_display_columns(df):
    show_cols = [
        "alert", "player_name", "lineup_pos", "bats", "position",
        "hr_game_pct", "hr_pa_pct", "matchup", "test_score",
        "pa", "barrel_pct", "iso", "xwoba", "xwobacon",
        "pitch_match_score", "best_pitch", "best_pitch_xwoba", "worst_pitch",
        "fb_pct", "la", "sprint_speed",
        "k_pct", "bb_pct", "whiff_pct",
        "home_run", "recent_hr", "sleeper_score", "gs_score",
    ]
    return [c for c in show_cols if c in df.columns]


def render_main_matchup(df):
    if df is None or df.empty:
        st.write("No qualified hitters available")
        return

    df = df.copy()
    if "test_score" in df.columns:
        df["alert"] = df["test_score"].apply(lambda x: calc_4tier(x, scale=(45, 65)))
    else:
        df["alert"] = "⚪"

    keep = get_display_columns(df)
    if not keep:
        st.write("No columns to display")
        return

    display = df[keep].copy().reset_index(drop=True)
    st.dataframe(
        display, hide_index=True, use_container_width=True, height=380,
        column_config=build_col_config(),
    )


def render_insufficient_sample(df):
    if df is None or df.empty:
        return

    df = df.copy()
    df["alert"] = "⚪"
    # Blank the HR projection columns - tiny sample = no reliable projection
    if "hr_game_pct" in df.columns:
        df["hr_game_pct"] = None
    if "hr_pa_pct" in df.columns:
        df["hr_pa_pct"] = None

    keep = get_display_columns(df)
    if not keep:
        return

    display = df[keep].copy().reset_index(drop=True)
    st.caption(
        f"⚠️ **Insufficient Sample** — hitters with <{INSUFFICIENT_PA_THRESHOLD} PAs. "
        f"Real stats shown but HR projections withheld."
    )
    st.dataframe(
        display, hide_index=True, use_container_width=True,
        height=min(250, 50 + len(display) * 35),
        column_config=build_col_config(),
    )


def render_matchup_section(df, side_label):
    if df is None or df.empty:
        st.write("No data")
        return

    main, small = split_by_sample_size(df, INSUFFICIENT_PA_THRESHOLD)
    render_main_matchup(main)

    if not small.empty:
        with st.expander(f"📋 {len(small)} insufficient-sample player(s) for {side_label}",
                          expanded=False):
            render_insufficient_sample(small)


def game_label(row):
    try:
        t = pd.to_datetime(row["gameTime"]).tz_convert("US/Eastern")
        return f"🏟️ {row['away_team_abbr']} @ {row['home_team_abbr']} · ⏰ {t.strftime('%-I:%M %p ET')}"
    except Exception:
        return f"🏟️ {row['away_team_abbr']} @ {row['home_team_abbr']}"


for _, game in slate.iterrows():
    ctx = game_context_map.get(game["gamePk"])
    if not ctx:
        continue

    park = ctx["park"]
    vegas = ctx.get("vegas") or {}
    ump = ctx.get("ump", {})

    with st.container(border=True):
        st.markdown(f"### {game_label(game)}")

        env1, env2, env3, env4, env5 = st.columns(5)
        away_k_mean = ctx["away_k_proj"].get("mean") if ctx.get("away_k_proj") else None
        home_k_mean = ctx["home_k_proj"].get("mean") if ctx.get("home_k_proj") else None

        env1.metric(
            f"Away ({game['away_team_abbr']})",
            game.get("away_pitcher") or "TBD",
            delta=f"Proj K: {away_k_mean:.1f}" if away_k_mean is not None else "K data missing",
        )
        env2.metric(
            f"Home ({game['home_team_abbr']})",
            game.get("home_pitcher") or "TBD",
            delta=f"Proj K: {home_k_mean:.1f}" if home_k_mean is not None else "K data missing",
        )
        env3.metric("Weather Mult", f"{ctx['hr_mult']:.2f}×",
                    delta=f"{(ctx['hr_mult'] - 1) * 100:+.0f}%")
        if vegas and vegas.get("total"):
            tv = vegas["total"]
            ai = vegas.get("away_implied")
            hi = vegas.get("home_implied")
            ai_s = f"{ai:.1f}" if ai else "—"
            hi_s = f"{hi:.1f}" if hi else "—"
            env4.metric("Vegas O/U", f"{tv:.1f}", delta=f"A:{ai_s} H:{hi_s}")
        else:
            env4.metric("Vegas O/U", "—")
        env5.metric("Plate Ump", (ump.get("name") or "TBD")[:16])

        st.caption(
            f"📍 **Venue:** {game.get('venue', 'TBD')} ({park.get('roof', 'open')}) · "
            f"**Park HR:** {park.get('hr_factor', 100)} · "
            f"**Weather:** {ctx.get('summary') or 'No data'}"
        )

        # Top HR contender card
        main_only_a, _ = split_by_sample_size(ctx.get("away_matchup", pd.DataFrame()),
                                                INSUFFICIENT_PA_THRESHOLD)
        main_only_h, _ = split_by_sample_size(ctx.get("home_matchup", pd.DataFrame()),
                                                INSUFFICIENT_PA_THRESHOLD)
        all_qualified = []
        if not main_only_a.empty:
            all_qualified.append(main_only_a)
        if not main_only_h.empty:
            all_qualified.append(main_only_h)
        if all_qualified:
            combined = pd.concat(all_qualified, ignore_index=True)
            if "hr_game_pct" in combined.columns:
                valid = combined.dropna(subset=["hr_game_pct"])
                if not valid.empty:
                    top = valid.sort_values("hr_game_pct", ascending=False).iloc[0]
                    with st.container(border=True):
                        name = top.get("player_name", "?")
                        hr_g = top.get("hr_game_pct")
                        hr_pa = top.get("hr_pa_pct")
                        match_v = top.get("matchup")
                        iso_v = top.get("iso")
                        st.markdown(f"👑 **Top HR Contender:** `{name}`")
                        bullets = []
                        if hr_g is not None and not pd.isna(hr_g):
                            bullets.append(f"**HR Game%:** {hr_g:.1f}%")
                        if hr_pa is not None and not pd.isna(hr_pa):
                            bullets.append(f"**HR PA%:** {hr_pa:.2f}%")
                        if match_v is not None and not pd.isna(match_v):
                            bullets.append(f"**Matchup:** {match_v:.1f}")
                        if iso_v is not None and not pd.isna(iso_v):
                            bullets.append(f"**ISO:** {iso_v:.3f}")
                        if bullets:
                            st.markdown(" · ".join(bullets))

        away_tab = f"🏏 {game['away_team_abbr']} vs {game.get('home_pitcher') or 'TBD'}"
        home_tab = f"🏏 {game['home_team_abbr']} vs {game.get('away_pitcher') or 'TBD'}"
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
                        # Diagnostic: show what fields ARE available for this pitcher
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
                        {"Line": f"Over {x} K", "Prob": pp.get(f"p_over_{x}", 0)}
                        for x in ["5.5", "6.5", "7.5", "8.5"]
                    ])
                    st.dataframe(
                        lines, hide_index=True, use_container_width=True,
                        column_config={
                            "Line": st.column_config.TextColumn("Line"),
                            "Prob": st.column_config.NumberColumn("P(Over)",
                                format="%.0f%%"),
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
                            try:
                                bvp = bvp_for_lineup(ctx["away_lineup"], h_pid)
                                if not bvp.empty:
                                    st.dataframe(bvp.reset_index(drop=True),
                                                 hide_index=True,
                                                 use_container_width=True)
                                else:
                                    st.caption("No BvP history")
                            except Exception:
                                st.caption("BvP unavailable")
                with sub2:
                    if not pitcher_arsenal_all.empty:
                        for p_name, p_id in [
                            (game.get("away_pitcher"), safe_int(game.get("away_pitcher_id"))),
                            (game.get("home_pitcher"), safe_int(game.get("home_pitcher_id"))),
                        ]:
                            if not p_id:
                                continue
                            st.markdown(f"**{p_name}**")
                            try:
                                if "player_id" in pitcher_arsenal_all.columns:
                                    a = pitcher_arsenal_all[pitcher_arsenal_all["player_id"] == p_id]
                                else:
                                    a = pd.DataFrame()
                                if not a.empty:
                                    cols_a = [c for c in [
                                        "pitch_name", "pitch_usage", "ba", "slg",
                                        "woba", "whiff_percent",
                                    ] if c in a.columns]
                                    if cols_a:
                                        ad = a[cols_a].rename(columns={
                                            "pitch_name": "Pitch",
                                            "pitch_usage": "Usage%",
                                            "whiff_percent": "Whiff%",
                                        })
                                        st.dataframe(ad.reset_index(drop=True),
                                                     hide_index=True,
                                                     use_container_width=True)
                            except Exception:
                                pass
