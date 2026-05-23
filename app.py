"""
app.py — Posey MLB HR & K Data Dashboard
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
st.markdown("<style>div[data-testid='stMetric']{min-height:85px;}</style>", unsafe_allow_html=True)


def get_pa_threshold(slate_date):
    try:
        d = datetime.fromisoformat(slate_date).date() if isinstance(slate_date, str) else slate_date
        season_start = date(d.year, 3, 28)
        if d < season_start:
            return 100
        days_in = (d - season_start).days
        if days_in <= 30: return 40
        if days_in <= 60: return 80
        if days_in <= 90: return 120
        if days_in <= 120: return 160
        return 200
    except Exception:
        return 100


def calc_4tier(score, scale=(45, 65)):
    if score is None or pd.isna(score):
        return "⚪"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "⚪"
    low, high = scale
    mid = (low + high) / 2
    if s >= high: return "🟢"
    if s >= mid:  return "🟡"
    if s >= low:  return "🟠"
    return "🔴"


def safe_int(val, default=None):
    try:
        if val is None or pd.isna(val):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def split_by_sample_size(df, threshold):
    if df is None or df.empty or "pa" not in df.columns:
        return df, pd.DataFrame()
    mask = df["pa"].notna() & (df["pa"] < threshold)
    return df[~mask].copy(), df[mask].copy()


with st.sidebar:
    st.title("⚾ Posey MLB Props")
    selected_date = st.date_input("Slate date", value=date.today())
    pa_threshold = get_pa_threshold(selected_date)
    custom_threshold = st.number_input(
        f"PA threshold (auto: {pa_threshold})",
        min_value=0, max_value=500, value=pa_threshold, step=10,
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
    st.warning("**Bet responsibly.** Lines are sharp. Bet flat units.")

INSUFFICIENT_PA_THRESHOLD = custom_threshold

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

if not hitter_trad.empty and "player_id" in hitter_stats.columns:
    drop = [c for c in ["player_name"] if c in hitter_trad.columns]
    hitter_stats = hitter_stats.merge(hitter_trad.drop(columns=drop, errors="ignore"),
                                        on="player_id", how="left", suffixes=("", "_t"))
if not pitcher_trad.empty and "player_id" in pitcher_stats.columns:
    drop = [c for c in ["player_name"] if c in pitcher_trad.columns]
    pitcher_stats = pitcher_stats.merge(pitcher_trad.drop(columns=drop, errors="ignore"),
                                          on="player_id", how="left", suffixes=("", "_t"))

if use_sprint_speed:
    try:
        sprint_df = get_sprint_speed()
        if not sprint_df.empty and "player_id" in sprint_df.columns and "player_id" in hitter_stats.columns:
            sprint_cols = [c for c in ["player_id", "sprint_speed", "hp_to_1b"] if c in sprint_df.columns]
            hitter_stats = hitter_stats.merge(sprint_df[sprint_cols], on="player_id", how="left")
    except Exception:
        pass

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

vegas_df = pd.DataFrame()
if use_vegas:
    try:
        vegas_df = get_vegas_totals(selected_date.isoformat())
    except Exception:
        pass

st.title(f"⚾ Posey MLB HR & K Data — {selected_date.strftime('%A, %B %d, %Y')}")

src_cols = st.columns(7)
src_cols[0].metric("Games", len(slate))
src_cols[1].metric("Statcast Hitters", f"{len(hitter_stats)}")
src_cols[2].metric("Statcast Pitchers", f"{len(pitcher_stats)}")
if not pitcher_trad.empty and "k9" in pitcher_trad.columns:
    pct = (pitcher_trad["k9"].notna().sum() / len(pitcher_trad) * 100) if len(pitcher_trad) else 0
    src_cols[3].metric("Pitchers w/ K9", f"{pct:.0f}%")
else:
    src_cols[3].metric("Pitchers w/ K9", "0%")
if use_sprint_speed:
    sprint_ok = "sprint_speed" in hitter_stats.columns and hitter_stats["sprint_speed"].notna().any()
    src_cols[4].metric("Sprint", "✓" if sprint_ok else "—")
else:
    src_cols[4].metric("Sprint", "off")
src_cols[5].metric("Vegas", "✓" if not vegas_df.empty else "—")
src_cols[6].metric("Pitch Arsenals", "✓" if not pitcher_arsenal_all.empty else "—")

st.caption(f"📊 PA threshold: **{INSUFFICIENT_PA_THRESHOLD}**. ({(date.today() - date(date.today().year, 3, 28)).days} days into season)")

st.divider()

game_context_map = {}
progress = st.progress(0.0, text="Building games...")

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
            match = vegas_df[(vegas_df.get("away_abbr") == game["away_team_abbr"]) &
                              (vegas_df.get("home_abbr") == game["home_team_abbr"])]
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
            away_lineup = [{"id": p["id"], "name": p["name"], "position": p["position"]}
                            for p in get_team_roster(int(game["away_team_id"]))[:9]]
        except Exception:
            away_lineup = []
    try:
        home_lineup = get_lineup(int(game["gamePk"]), "home")
    except Exception:
        home_lineup = []
    if not home_lineup:
        try:
            home_lineup = [{"id": p["id"], "name": p["name"], "position": p["position"]}
                            for p in get_team_roster(int(game["home_team_id"]))[:9]]
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
        away_matchup = build_matchup_table(away_lineup, home_p_series, hitter_stats, pitcher_stats, recent_form_dict=away_recent)
    except Exception:
        away_matchup = pd.DataFrame()
    try:
        home_matchup = build_matchup_table(home_lineup, away_p_series, hitter_stats, pitcher_stats, recent_form_dict=home_recent)
    except Exception:
        home_matchup = pd.DataFrame()

    if use_pitch_match and not pitcher_arsenal_all.empty and not hitter_pitch_arsenal.empty:
        pm_cols = ["player_id", "pitch_match_score", "best_pitch", "best_pitch_xwoba", "worst_pitch", "weighted_xwoba"]
        try:
            if home_p_id:
                away_pm = lineup_pitch_match(away_lineup, home_p_id, hitter_pitch_arsenal, pitcher_arsenal_all)
                if not away_pm.empty and not away_matchup.empty:
                    keep = [c for c in pm_cols if c in away_pm.columns]
                    if "player_id" in keep:
                        away_matchup = away_matchup.merge(away_pm[keep], on="player_id", how="left")
            if away_p_id:
                home_pm = lineup_pitch_match(home_lineup, away_p_id, hitter_pitch_arsenal, pitcher_arsenal_all)
                if not home_pm.empty and not home_matchup.empty:
                    keep = [c for c in pm_cols if c in home_pm.columns]
                    if "player_id" in keep:
                        home_matchup = home_matchup.merge(home_pm[keep], on="player_id", how="left")
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

    for matchup_df, opp_p_row in [(away_matchup, home_p_row), (home_matchup, away_p_row)]:
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
                    hitter_row=hrow.to_dict(), pitcher_row=opp_p_row,
                    park_factor=park_mult, park_hand_factor=ph_factor,
                    weather_mult=wx_mult, pitch_match_score=pm_score,
                    ttop_mult=ttop, defense_factor=1.0,
                    min_pa=INSUFFICIENT_PA_THRESHOLD,
                )
                if p_pa is None:
                    hr_pa.append(None); hr_game.append(None); verdicts.append("⚪")
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
                hr_pa.append(None); hr_game.append(None); verdicts.append("⚪")
        matchup_df["hr_pa_pct"] = hr_pa
        matchup_df["hr_game_pct"] = hr_game
        matchup_df["verdict"] = verdicts

    away_k_col = away_matchup["k_pct"] if "k_pct" in away_matchup.columns else pd.Series(dtype=float)
    home_k_col = home_matchup["k_pct"] if "k_pct" in home_matchup.columns else pd.Series(dtype=float)
    away_lineup_k_pct = float(away_k_col.mean()) if not away_k_col.empty and not away_k_col.isna().all() else None
    home_lineup_k_pct = float(home_k_col.mean()) if not home_k_col.empty and not home_k_col.isna().all() else None

    away_k_proj, home_k_proj = {}, {}
    try:
        if away_p_row:
            away_k_proj = k_total_projection(away_p_row, home_lineup_k_pct, ump_k_factor=ump.get("k_factor", 1.0))
        if home_p_row:
            home_k_proj = k_total_projection(home_p_row, away_lineup_k_pct, ump_k_factor=ump.get("k_factor", 1.0))
    except Exception:
        pass

    game_context_map[game["gamePk"]] = {
        "park": park, "weather": weather, "wx_mult": wx_mult, "park_mult": park_mult,
        "hr_mult": full_hr_mult, "summary": wx_summary, "vegas": vegas_row, "ump": ump,
        "away_lineup": away_lineup, "home_lineup": home_lineup,
        "away_p_row": away_p_row, "home_p_row": home_p_row,
        "away_matchup": away_matchup, "home_matchup": home_matchup,
        "away_k_proj": away_k_proj, "home_k_proj": home_k_proj,
    }

progress.empty()

st.subheader("📋 Starting Pitcher Overview")
try:
    pitcher_recent_data = {}
    for gpk, ctx in game_context_map.items():
        game_rows = slate[slate["gamePk"] == gpk]
        if game_rows.empty: continue
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
        pitcher_slate["alert"] = pitcher_slate["test_score"].apply(lambda x: calc_4tier(x, scale=(45, 65)))
    else:
        pitcher_slate["alert"] = "⚪"
    base_cols = ["alert", "pitcher_name", "team", "home_away", "opp", "throws"]
    metric_cols = ["test_score", "kHR", "proj_k", "form_arrow", "era", "whip", "k9", "bb9", "hr9", "ip",
                    "k_pct", "whiff_pct", "csw_pct", "xwoba_allowed", "barrel_allowed",
                    "recent_era", "recent_k9", "days_rest", "avg_recent_pitches"]
    keep = [c for c in base_cols + metric_cols if c in pitcher_slate.columns]
    display = pitcher_slate[keep].copy().reset_index(drop=True)
    st.dataframe(display, hide_index=True, use_container_width=True, height=400)

st.divider()

st.subheader("🎮 Game-by-Game Matchups")

def render_matchup(df, threshold):
    if df is None or df.empty:
        st.write("No data")
        return
    main, small = split_by_sample_size(df, threshold)
    if main is not None and not main.empty:
        main = main.copy()
        if "test_score" in main.columns:
            main["alert"] = main["test_score"].apply(lambda x: calc_4tier(x, scale=(45, 65)))
        cols = [c for c in ["alert", "player_name", "lineup_pos", "bats", "position",
                              "hr_game_pct", "hr_pa_pct", "matchup", "test_score",
                              "pa", "barrel_pct", "iso", "xwoba", "xwobacon",
                              "pitch_match_score", "best_pitch", "best_pitch_xwoba", "worst_pitch",
                              "fb_pct", "la", "sprint_speed",
                              "k_pct", "bb_pct", "whiff_pct",
                              "home_run", "recent_hr", "sleeper_score", "gs_score"]
                if c in main.columns]
        st.dataframe(main[cols].reset_index(drop=True), hide_index=True, use_container_width=True, height=380)
    if small is not None and not small.empty:
        with st.expander(f"📋 {len(small)} insufficient-sample player(s)"):
            small = small.copy()
            small["alert"] = "⚪"
            if "hr_game_pct" in small.columns: small["hr_game_pct"] = None
            if "hr_pa_pct" in small.columns: small["hr_pa_pct"] = None
            cols = [c for c in ["alert", "player_name", "lineup_pos", "bats", "position",
                                  "pa", "barrel_pct", "iso", "xwoba", "k_pct", "home_run"]
                    if c in small.columns]
            st.caption(f"⚠️ Hitters with <{threshold} PAs. Stats shown but projections withheld.")
            st.dataframe(small[cols].reset_index(drop=True), hide_index=True, use_container_width=True)


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
        env1.metric(f"Away ({game['away_team_abbr']})", game.get("away_pitcher") or "TBD",
                    delta=f"Proj K: {away_k_mean:.1f}" if away_k_mean is not None else "K data missing")
        env2.metric(f"Home ({game['home_team_abbr']})", game.get("home_pitcher") or "TBD",
                    delta=f"Proj K: {home_k_mean:.1f}" if home_k_mean is not None else "K data missing")
        env3.metric("Weather Mult", f"{ctx['hr_mult']:.2f}×", delta=f"{(ctx['hr_mult'] - 1) * 100:+.0f}%")
        if vegas and vegas.get("total"):
            tv = vegas["total"]
            env4.metric("Vegas O/U", f"{tv:.1f}")
        else:
            env4.metric("Vegas O/U", "—")
        env5.metric("Plate Ump", (ump.get("name") or "TBD")[:16])
        st.caption(f"📍 {game.get('venue', 'TBD')} · Park HR: {park.get('hr_factor', 100)} · {ctx.get('summary') or 'No data'}")

        away_tab = f"🏏 {game['away_team_abbr']} vs {game.get('home_pitcher') or 'TBD'}"
        home_tab = f"🏏 {game['home_team_abbr']} vs {game.get('away_pitcher') or 'TBD'}"
        tabs = st.tabs([away_tab, home_tab, "🎯 K Projections"])

        with tabs[0]:
            render_matchup(ctx.get("away_matchup"), INSUFFICIENT_PA_THRESHOLD)
        with tabs[1]:
            render_matchup(ctx.get("home_matchup"), INSUFFICIENT_PA_THRESHOLD)
        with tabs[2]:
            kp1, kp2 = st.columns(2)
            for col, side, label in [(kp1, "away", game.get("away_pitcher") or "TBD"),
                                      (kp2, "home", game.get("home_pitcher") or "TBD")]:
                with col:
                    pp = ctx.get(f"{side}_k_proj") or {}
                    if not pp or pp.get("mean") is None:
                        st.write(f"**{label}** — no real K/9 data")
                        continue
                    st.markdown(f"**{label}** · Proj K: **{pp['mean']:.1f}** (range {pp.get('low', 0):.1f}–{pp.get('high', 0):.1f})")
                    lines = pd.DataFrame([{"Line": f"O {x}", "Prob": pp.get(f"p_over_{x}", 0)} for x in ["5.5", "6.5", "7.5", "8.5"]])
                    st.dataframe(lines, hide_index=True, use_container_width=True)
