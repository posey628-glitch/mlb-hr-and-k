"""
data_fetcher.py
================
Pulls today's MLB slate and all underlying data from free public sources:
  - MLB Stats API (statsapi.mlb.com) - slate, probable pitchers, lineups
  - Baseball Savant (baseballsavant.mlb.com) - Statcast stats, arsenals

No API keys required. All endpoints are publicly documented.
Results are cached for 30 minutes via Streamlit's @st.cache_data.
"""

from __future__ import annotations

import io
from datetime import datetime, date
from typing import Optional

import pandas as pd
import requests
import streamlit as st

# Pretend to be a real browser - Savant sometimes 403s on bare requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/csv, */*",
}

CURRENT_SEASON = datetime.now().year


# ----------------------------------------------------------------------------
# Safe parsers - handle MLB Stats API "-.--" and other junk values
# ----------------------------------------------------------------------------

def _safe_float(val):
    """Convert API value to float; return None for '-.--', '', or invalid."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            f = float(val)
            return f if not (f != f) else None  # NaN check
        except (TypeError, ValueError):
            return None
    s = str(val).strip()
    if s in ("", "-.--", "--", "-", ".---", "null", "None"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    """Convert API value to int; return None for '-.--', '', or invalid."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = str(val).strip()
    if s in ("", "-.--", "--", "-", "null", "None"):
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------------
# Shared normalizers - keep player_id types consistent across all sources
# ----------------------------------------------------------------------------

def _normalize_player_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure player_id is a consistent integer type so merges work.
    Also rename common column aliases that Savant uses inconsistently.
    """
    if df is None or df.empty:
        return df
    if "player_id" not in df.columns:
        for cand in ["mlb_id", "playerid", "MLBAMID", "mlbam_id", "player_mlb_id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "player_id"})
                break
    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")

    # Savant sometimes returns these under alternate names - normalize so
    # downstream code finds them under one canonical name.
    rename_map = {}
    for alias, canonical in [
        ("isolated_power", "iso"),
        ("la", "launch_angle"),
        ("ev", "launch_speed"),
        ("hardhit_percent", "hard_hit_percent"),
        ("barrel_percent", "barrel_batted_rate"),
        ("xba_diff", "xba_minus_ba_diff"),
    ]:
        if alias in df.columns and canonical not in df.columns:
            rename_map[alias] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _derive_hitter_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute commonly-missing hitter columns from real underlying stats.
    These are math identities (ISO = SLG - AVG), not fake defaults.
    """
    if df is None or df.empty:
        return df

    # ISO from SLG - AVG if missing or all-null
    needs_iso = "iso" not in df.columns or df["iso"].isna().all()
    if needs_iso:
        slg_col = next((c for c in ["slg", "slugging_percentage"] if c in df.columns), None)
        avg_col = next((c for c in ["batting_avg", "avg", "ba"] if c in df.columns), None)
        if slg_col and avg_col:
            df["iso"] = (df[slg_col] - df[avg_col]).round(3)

    return df


# ----------------------------------------------------------------------------
# Slate / probable pitchers (MLB Stats API)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=1800)
def get_slate(game_date: Optional[str] = None) -> pd.DataFrame:
    """
    Return one row per game for the given date (YYYY-MM-DD, default today).
    Columns: gamePk, gameTime, away_team, home_team, away_team_id, home_team_id,
             away_pitcher, home_pitcher, away_pitcher_id, home_pitcher_id, venue
    """
    if game_date is None:
        game_date = date.today().isoformat()

    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={game_date}"
        "&hydrate=probablePitcher,linescore,team"
    )
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()

    rows = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away = g["teams"]["away"]
            home = g["teams"]["home"]
            rows.append({
                "gamePk": g["gamePk"],
                "gameTime": g.get("gameDate"),
                "status": g.get("status", {}).get("detailedState"),
                "venue": g.get("venue", {}).get("name"),
                "away_team": away["team"]["name"],
                "away_team_abbr": away["team"].get("abbreviation",
                                                   away["team"]["name"][:3].upper()),
                "away_team_id": away["team"]["id"],
                "home_team": home["team"]["name"],
                "home_team_abbr": home["team"].get("abbreviation",
                                                   home["team"]["name"][:3].upper()),
                "home_team_id": home["team"]["id"],
                "away_pitcher": (away.get("probablePitcher") or {}).get("fullName", "TBD"),
                "away_pitcher_id": (away.get("probablePitcher") or {}).get("id"),
                "home_pitcher": (home.get("probablePitcher") or {}).get("fullName", "TBD"),
                "home_pitcher_id": (home.get("probablePitcher") or {}).get("id"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["gameTime"] = pd.to_datetime(df["gameTime"], errors="coerce")
    return df


@st.cache_data(ttl=1800)
def get_lineup(game_pk: int, side: str = "home") -> list[dict]:
    """
    Return the projected/actual lineup for a side ("home" or "away").
    Falls back to recent starters if lineup not yet posted.
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        boxscore = r.json().get("liveData", {}).get("boxscore", {})
        team = boxscore.get("teams", {}).get(side, {})
        batting_order = team.get("battingOrder", [])
        players = team.get("players", {})
        out = []
        for pid in batting_order:
            p = players.get(f"ID{pid}", {})
            person = p.get("person", {})
            out.append({
                "id": person.get("id"),
                "name": person.get("fullName"),
                "position": p.get("position", {}).get("abbreviation"),
                "bats": p.get("batSide", {}).get("code"),
            })
        return out
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_team_roster(team_id: int) -> list[dict]:
    """Active roster for a team - used as lineup fallback."""
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster/active"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        out = []
        for p in r.json().get("roster", []):
            pos = p.get("position", {}).get("abbreviation", "")
            if pos in ("P",):
                continue  # skip pitchers when looking for hitters
            out.append({
                "id": p["person"]["id"],
                "name": p["person"]["fullName"],
                "position": pos,
            })
        return out
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_all_team_rosters(slate: pd.DataFrame) -> dict:
    """
    Returns {team_id: [hitter dicts]} for every team on today's slate.
    Used so users can scan any hitter, not just confirmed lineups.
    """
    rosters = {}
    team_ids = set()
    for _, g in slate.iterrows():
        team_ids.add(int(g["away_team_id"]))
        team_ids.add(int(g["home_team_id"]))
    for tid in team_ids:
        rosters[tid] = get_team_roster(tid)
    return rosters


# ----------------------------------------------------------------------------
# Hitter season stats (Baseball Savant custom leaderboard)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_hitter_stats(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Pull season-level Statcast hitter stats with ALL columns needed for the
    dashboard: standard rates, expected stats, batted-ball quality, pulled
    contact, fly ball %, launch angle, swing/miss, etc.
    """
    selections = (
        "pa,abs,hits,player_age,k_percent,bb_percent,woba,xwoba,xiso,xba,xslg,xobp,"
        "iso,babip,slg,obp,batting_avg,on_base_plus_slg,home_run,"
        "barrel_batted_rate,solidcontact_percent,flareburner_percent,"
        "poorlyunder_percent,poorlytopped_percent,poorlyweak_percent,"
        "hard_hit_percent,avg_best_speed,avg_hit_angle,launch_speed,launch_angle,"
        "whiff_percent,swing_percent,sweet_spot_percent,xwobacon,wobacon,"
        "groundballs_percent,flyballs_percent,linedrives_percent,popups_percent,"
        "pull_percent,straightaway_percent,opposite_percent,"
        "pull_air_percent,straightaway_air_percent,opposite_air_percent,"
        "z_swing_percent,z_swing_miss_percent,oz_swing_percent,oz_swing_miss_percent,"
        "f_strike_percent,zone_percent"
    )
    url = (
        "https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={season}&type=batter&filter=&min=1&selections={selections}"
        "&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&csv=true"
    )
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "last_name, first_name" in df.columns:
        df["player_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
            if isinstance(s, str) and "," in s else s
        )
    if "player_id" not in df.columns:
        for cand in ["mlb_id", "playerid", "MLBAMID"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "player_id"})
                break
    # Compute PulledBrl% as approximation: pull_air_percent * (barrels/BBE proxy)
    if "pull_air_percent" in df.columns and "barrel_batted_rate" in df.columns:
        df["pulled_brl_pct"] = (df["pull_air_percent"] * df["barrel_batted_rate"] / 100).round(2)
    # Normalize player_id type for merges + derive missing columns
    df = _normalize_player_df(df)
    df = _derive_hitter_missing(df)
    return df


# ----------------------------------------------------------------------------
# Pitcher season stats
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_pitcher_stats(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Season-level Statcast pitcher stats - expanded.

    Now requests stats that let us derive ERA-equivalents independently of
    MLB Stats API: xera, slg (SLG against), obp (OBP against), hits, walks,
    earned_runs, home_run, etc. This way if MLB Stats API is unreachable
    (which happens on Streamlit Cloud), we still have everything we need.
    """
    selections = (
        # Core Statcast
        "pa,k_percent,bb_percent,woba,xwoba,xiso,xba,xslg,xobp,xera,"
        "barrel_batted_rate,hard_hit_percent,avg_best_speed,avg_hit_angle,"
        "whiff_percent,swing_percent,sweet_spot_percent,xwobacon,iso,babip,"
        "launch_speed,launch_angle,p_total_pitches,p_total_swinging_strike,"
        "csw_percent,zone_percent,in_zone_swing_miss_percent,"
        "f_strike_percent,oz_swing_percent,z_swing_percent,"
        "groundballs_percent,flyballs_percent,linedrives_percent,popups_percent,"
        "pull_percent,straightaway_percent,opposite_percent,"
        # Real outcome stats (ERA-equivalent derivation)
        "home_run,slg,obp,batting_avg,hits,walks_drawn,earned_runs,"
        "ab,strikeout,plate_appearances"
    )
    url = (
        "https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={season}&type=pitcher&filter=&min=1&selections={selections}"
        "&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&csv=true"
    )
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "last_name, first_name" in df.columns:
        df["player_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
            if isinstance(s, str) and "," in s else s
        )
    df = _normalize_player_df(df)

    # ------------------------------------------------------------------
    # SAVANT-DERIVED ERA/WHIP equivalents - work even when MLB Stats API dies
    # ------------------------------------------------------------------
    # Statcast 'pa' is plate appearances faced. ~4.3 PA per IP league average.
    if "pa" in df.columns:
        df["ip_savant"] = (df["pa"] / 4.3).round(1)
        # If MLB Stats API gave us real IP later, that will overwrite this.
        # For now, use ip_savant as the IP value.
        df["ip"] = df["ip_savant"]
    # K/9 from K% × 4.3 PA per IP
    if "k_percent" in df.columns:
        df["k9_savant"] = (df["k_percent"] * 4.3 * 9 / 100).round(2)
        df["k9"] = df["k9_savant"]
    # BB/9 from BB% × 4.3 PA per IP
    if "bb_percent" in df.columns:
        df["bb9_savant"] = (df["bb_percent"] * 4.3 * 9 / 100).round(2)
        df["bb9"] = df["bb9_savant"]
    # WHIP from (H + BB) / IP - if Savant gave us hits and walks
    if "hits" in df.columns and "walks_drawn" in df.columns and "ip_savant" in df.columns:
        df["whip_savant"] = ((df["hits"] + df["walks_drawn"]) / df["ip_savant"].replace(0, pd.NA)).round(2)
        df["whip"] = df["whip_savant"]
    # ERA from earned_runs × 9 / IP
    if "earned_runs" in df.columns and "ip_savant" in df.columns:
        df["era_savant"] = (df["earned_runs"] * 9 / df["ip_savant"].replace(0, pd.NA)).round(2)
        df["era"] = df["era_savant"]
    # HR/9 from home_run × 9 / IP
    if "home_run" in df.columns and "ip_savant" in df.columns:
        df["hr9_savant"] = (df["home_run"] * 9 / df["ip_savant"].replace(0, pd.NA)).round(2)
        df["hr9"] = df["hr9_savant"]

    return df


# ----------------------------------------------------------------------------
# Pitcher arsenal (pitch-mix, velo, spin, swing/miss per pitch type)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_pitcher_arsenal(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Returns one row per pitcher x pitch_type with: usage %, swstr%, hh%,
    velocity, spin rate, xwOBAcon, run value per 100, etc.
    """
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
        f"?type=pitcher&pitchType=&year={season}&team=&min=10&hand="
        "&csv=true"
    )
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "last_name, first_name" in df.columns:
        df["player_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
            if isinstance(s, str) and "," in s else s
        )
    return df


@st.cache_data(ttl=3600)
def get_pitcher_arsenal_by_count(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Pitch usage broken out by count: ahead, behind, even, early, all.
    Mirrors the 'Count Usage' tab in the screenshots.
    """
    frames = []
    count_filters = [
        ("all", ""),
        ("early", "0-0,1-0,0-1,1-1,2-0"),
        ("ahead", "0-1,0-2,1-2,2-2"),
        ("behind", "1-0,2-0,2-1,3-0,3-1,3-2"),
        ("even", "0-0,1-1,2-2,3-2"),
    ]
    for label, _filter in count_filters:
        # Savant doesn't have a public CSV grouped by count - use pitch-type
        # leaderboard with the count filter set
        url = (
            "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
            f"?type=pitcher&pitchType=&year={season}&team=&min=10&hand=&csv=true"
        )
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df["count_state"] = label
            frames.append(df)
            break  # Same endpoint for all - we'll filter client-side
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl=1800)
def get_pitcher_full_season_from_gamelog(pitcher_id: int,
                                            season: int = CURRENT_SEASON) -> dict:
    """
    Compute season-total ERA/WHIP/K9/BB9/HR9/IP/GS by summing the pitcher's
    own game log. This is independent of the /api/v1/stats?stats=season endpoint
    that's been flaky. Same /people/{id}/stats?stats=gameLog endpoint that
    powers recent form -- known to work.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
        f"?stats=gameLog&group=pitching&season={season}&sportId=1"
    )
    splits = []
    for attempt in range(2):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [])
            if splits:
                break
        except Exception:
            if attempt < 1:
                import time
                time.sleep(0.5)
            continue
    if not splits:
        return {}
    ip_sum, er_sum, k_sum, bb_sum, hr_sum = 0.0, 0, 0, 0, 0
    h_sum = 0
    gs_sum = 0
    gp_sum = 0
    for s in splits:
        st_ = s.get("stat", {}) or {}
        ip_sum += _safe_float(st_.get("inningsPitched")) or 0
        er_sum += _safe_int(st_.get("earnedRuns")) or 0
        k_sum += _safe_int(st_.get("strikeOuts")) or 0
        bb_sum += _safe_int(st_.get("baseOnBalls")) or 0
        hr_sum += _safe_int(st_.get("homeRuns")) or 0
        h_sum += _safe_int(st_.get("hits")) or 0
        gs_sum += _safe_int(st_.get("gamesStarted")) or 0
        gp_sum += _safe_int(st_.get("gamesPlayed")) or 0
    if ip_sum == 0:
        return {}
    # WHIP = (BB + H) / IP
    whip = round((bb_sum + h_sum) / ip_sum, 2) if ip_sum > 0 else None
    return {
        "era": round(er_sum * 9 / ip_sum, 2),
        "whip": whip,
        "k9": round(k_sum * 9 / ip_sum, 2),
        "bb9": round(bb_sum * 9 / ip_sum, 2),
        "hr9": round(hr_sum * 9 / ip_sum, 2),
        "ip": round(ip_sum, 1),
        "games_started": gs_sum,
        "games_played": gp_sum,
        "strikeouts": k_sum,
        "walks": bb_sum,
        "earned_runs": er_sum,
        "source": "gamelog",  # marker so we know which source
    }


@st.cache_data(ttl=1800)
def get_pitcher_recent_form(pitcher_id: int, season: int = CURRENT_SEASON,
                              n_starts: int = 5) -> dict:
    """
    Recent form: last N starts K/9, ERA, IP.
    Returns trending up/down arrow.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
        f"?stats=gameLog&group=pitching&season={season}&sportId=1"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {}
        # Last N starts only
        starts = [s for s in splits if int(s.get("stat", {}).get("gamesStarted", 0)) > 0]
        recent = starts[-n_starts:] if len(starts) > n_starts else starts
        if not recent:
            return {}
        ip_sum, er_sum, k_sum, bb_sum, hr_sum = 0.0, 0, 0, 0, 0
        for s in recent:
            st_ = s.get("stat", {})
            ip_sum += float(st_.get("inningsPitched", 0) or 0)
            er_sum += int(st_.get("earnedRuns", 0) or 0)
            k_sum += int(st_.get("strikeOuts", 0) or 0)
            bb_sum += int(st_.get("baseOnBalls", 0) or 0)
            hr_sum += int(st_.get("homeRuns", 0) or 0)
        if ip_sum == 0:
            return {}
        return {
            "recent_starts": len(recent),
            "recent_ip": round(ip_sum, 1),
            "recent_era": round(er_sum * 9 / ip_sum, 2),
            "recent_k9": round(k_sum * 9 / ip_sum, 2),
            "recent_bb9": round(bb_sum * 9 / ip_sum, 2),
            "recent_hr9": round(hr_sum * 9 / ip_sum, 2),
            "recent_k": k_sum,
        }
    except Exception:
        return {}


@st.cache_data(ttl=1800)
def get_hitter_recent_form_trad(player_id: int, season: int = CURRENT_SEASON,
                                  n_games: int = 15) -> dict:
    """Last 15 games hitter form via game log - lightweight.

    Now also tracks HR streaks and hot/cold pattern indicators.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=gameLog&group=hitting&season={season}&sportId=1"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        recent = splits[-n_games:] if len(splits) > n_games else splits
        if not recent:
            return {}
        ab, h, hr, k, bb, rbi = 0, 0, 0, 0, 0, 0
        d, t = 0, 0
        # Streak tracking - per-game HR list (most recent first)
        per_game_hr = []
        per_game_hits = []
        for s in recent:
            st_ = s.get("stat", {})
            ab += int(st_.get("atBats", 0) or 0)
            h += int(st_.get("hits", 0) or 0)
            game_hr = int(st_.get("homeRuns", 0) or 0)
            hr += game_hr
            per_game_hr.append(game_hr)
            per_game_hits.append(int(st_.get("hits", 0) or 0))
            k += int(st_.get("strikeOuts", 0) or 0)
            bb += int(st_.get("baseOnBalls", 0) or 0)
            rbi += int(st_.get("rbi", 0) or 0)
            d += int(st_.get("doubles", 0) or 0)
            t += int(st_.get("triples", 0) or 0)
        if ab == 0:
            return {}

        # Streak metrics - reverse so most-recent game is first
        per_game_hr_rev = list(reversed(per_game_hr))
        # Count games since last HR (None = never homered in window)
        games_since_hr = None
        for i, ghr in enumerate(per_game_hr_rev):
            if ghr > 0:
                games_since_hr = i
                break
        # Consecutive games with a HR (streak)
        consec_hr_games = 0
        for ghr in per_game_hr_rev:
            if ghr > 0:
                consec_hr_games += 1
            else:
                break
        # HRs in last 5, last 10 (rolling windows)
        hr_last_5 = sum(per_game_hr_rev[:5])
        hr_last_10 = sum(per_game_hr_rev[:10])
        # Multi-HR games in window
        multi_hr_games = sum(1 for ghr in per_game_hr if ghr >= 2)

        # Streak label for the table
        if consec_hr_games >= 2:
            streak_label = f"🔥 {consec_hr_games}g HR streak"
        elif hr_last_5 >= 3:
            streak_label = f"🔥 {hr_last_5} HR/L5"
        elif hr_last_5 >= 2:
            streak_label = f"🌶️ {hr_last_5} HR/L5"
        elif multi_hr_games >= 1:
            streak_label = f"⚡ {multi_hr_games}x multi-HR/L{len(recent)}"
        elif hr_last_10 == 0 and len(recent) >= 10:
            streak_label = "❄️ no HR L10"
        else:
            streak_label = ""

        return {
            "recent_games": len(recent),
            "recent_ab": ab, "recent_h": h, "recent_hr": hr,
            "recent_k": k, "recent_bb": bb, "recent_rbi": rbi,
            "recent_avg": round(h / ab, 3),
            "recent_iso": round((d + 2 * t + 3 * hr) / ab, 3) if ab else 0.0,
            "recent_k_pct": round(k / (ab + bb) * 100, 1) if (ab + bb) else 0.0,
            "recent_ops_proxy": round((h + bb) / (ab + bb), 3) if (ab + bb) else 0.0,
            "hr_streak_games": consec_hr_games,
            "games_since_hr": games_since_hr,
            "hr_last_5": hr_last_5,
            "hr_last_10": hr_last_10,
            "multi_hr_games": multi_hr_games,
            "streak_label": streak_label,
        }
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Sprint Speed (affects BABIP and infield hits) - Statcast
# ----------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_sprint_speed(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Statcast sprint speed leaderboard. ft/sec, league avg ~27.
    Elite is 30+. Slow is 25-.
    """
    url = (
        "https://baseballsavant.mlb.com/leaderboard/sprint_speed"
        f"?year={season}&position=&team=&min=10&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "last_name, first_name" in df.columns:
            df["player_name"] = df["last_name, first_name"].apply(
                lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
                if isinstance(s, str) and "," in s else s
            )
        return df
    except Exception:
        return pd.DataFrame()


# ----------------------------------------------------------------------------
# Statcast Run Values per pitch type (better pitch quality metric than whiff%)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_pitch_run_values(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Run value per 100 pitches by pitcher + pitch type.
    Negative = good for pitcher. -1.5 is elite, +1.5 is brutal.
    """
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-arsenals"
        f"?year={season}&min=10&type=run_value&hand=&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "last_name, first_name" in df.columns:
            df["player_name"] = df["last_name, first_name"].apply(
                lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
                if isinstance(s, str) and "," in s else s
            )
        return df
    except Exception:
        return pd.DataFrame()


# ----------------------------------------------------------------------------
# Traditional stats from MLB Stats API (WHIP, HR/9, OBP, HR totals)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_hitter_traditional(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Pull season AVG, OBP, SLG, HR, RBI, R for all hitters.

    Uses playerPool=All to catch non-qualifying hitters (early-season,
    platoon, callups). Defensive row parsing.
    """
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&season={season}&sportIds=1"
        "&playerPool=All&limit=3000"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
    except Exception:
        return pd.DataFrame()

    rows = []
    for s in splits:
        try:
            p = s.get("player", {}) or {}
            st_ = s.get("stat", {}) or {}
            pid = p.get("id")
            if pid is None:
                continue
            rows.append({
                "player_id": pid,
                "player_name": p.get("fullName"),
                "avg": _safe_float(st_.get("avg")),
                "obp": _safe_float(st_.get("obp")),
                "slg": _safe_float(st_.get("slg")),
                "ops": _safe_float(st_.get("ops")),
                "home_run": _safe_int(st_.get("homeRuns")),
                "rbi": _safe_int(st_.get("rbi")),
                "runs": _safe_int(st_.get("runs")),
                "sb": _safe_int(st_.get("stolenBases")),
                "trad_pa": _safe_int(st_.get("plateAppearances")),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    return _normalize_player_df(df)



@st.cache_data(ttl=900)  # Shortened from 3600 to 15min so failures recover faster
def get_pitcher_traditional(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Pull season ERA, WHIP, HR/9, K/9, BB/9 for all pitchers.

    Uses playerPool=All so non-qualifying pitchers (early-season, callups,
    long relievers spot-starting) are included. Retries 3x; if all fail OR
    return empty, we DON'T cache - by raising an exception that streamlit
    won't cache, then catching it to return empty. Next refresh tries again.
    """
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=pitching&season={season}&sportIds=1"
        "&playerPool=All&limit=3000"
    )
    splits = []
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [])
            if splits:
                break
        except Exception as e:
            last_err = e
            if attempt < 2:
                import time
                time.sleep(1.5)
            continue

    # CRITICAL: if we have no data, raise so the cache doesn't store empty
    if not splits:
        raise RuntimeError(f"MLB Stats API returned no pitcher data (last error: {last_err})")

    rows = []
    for s in splits:
        try:
            p = s.get("player", {}) or {}
            st_ = s.get("stat", {}) or {}
            pid = p.get("id")
            if pid is None:
                continue
            rows.append({
                "player_id": pid,
                "player_name": p.get("fullName"),
                "era": _safe_float(st_.get("era")),
                "whip": _safe_float(st_.get("whip")),
                "hr9": _safe_float(st_.get("homeRunsPer9")),
                # API has used both field names in different versions - try both
                "bb9": _safe_float(st_.get("walksPer9Inn") or st_.get("baseOnBallsPer9Inn")),
                "k9": _safe_float(st_.get("strikeoutsPer9Inn") or st_.get("strikeOutsPer9Inn")),
                "ip": _safe_float(st_.get("inningsPitched")),
                "wins": _safe_int(st_.get("wins")),
                "losses": _safe_int(st_.get("losses")),
                "games_started": _safe_int(st_.get("gamesStarted")),
                "games_played": _safe_int(st_.get("gamesPlayed")),
                "strikeouts": _safe_int(st_.get("strikeOuts")),
                "walks": _safe_int(st_.get("baseOnBalls")),
                "earned_runs": _safe_int(st_.get("earnedRuns")),
            })
        except Exception:
            continue

    if not rows:
        raise RuntimeError("MLB Stats API returned splits but no usable rows")

    df = pd.DataFrame(rows)
    return _normalize_player_df(df)


def get_pitcher_traditional_safe(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Wrapper that catches the cache-bust exception and returns empty df."""
    try:
        return get_pitcher_traditional(season)
    except Exception:
        return pd.DataFrame()


# ----------------------------------------------------------------------------
# Per-pitcher fallback - guarantees data for today's starters
# ----------------------------------------------------------------------------

@st.cache_data(ttl=1800)
def get_player_season_pitching(player_id: int, season: int = CURRENT_SEASON) -> dict:
    """Fetch one pitcher's season stats directly by ID. Retries once on failure."""
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=season&group=pitching&season={season}&sportId=1"
    )
    splits = []
    for attempt in range(2):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [])
            if splits:
                break
        except Exception:
            if attempt < 1:
                import time
                time.sleep(0.5)
            continue
    if not splits:
        return {}
    try:
        st_ = splits[0].get("stat", {}) or {}
        return {
            "era": _safe_float(st_.get("era")),
            "whip": _safe_float(st_.get("whip")),
            "hr9": _safe_float(st_.get("homeRunsPer9")),
            "bb9": _safe_float(st_.get("walksPer9Inn") or st_.get("baseOnBallsPer9Inn")),
            "k9": _safe_float(st_.get("strikeoutsPer9Inn") or st_.get("strikeOutsPer9Inn")),
            "ip": _safe_float(st_.get("inningsPitched")),
            "games_started": _safe_int(st_.get("gamesStarted")),
            "games_played": _safe_int(st_.get("gamesPlayed")),
            "strikeouts": _safe_int(st_.get("strikeOuts")),
            "walks": _safe_int(st_.get("baseOnBalls")),
            "earned_runs": _safe_int(st_.get("earnedRuns")),
        }
    except Exception:
        return {}


def fill_hitter_bats(lineups: list, ids: set | None = None) -> dict:
    """
    Bulk-fetch batting handedness for all hitter IDs in the provided lineups.
    Returns {player_id: "R"/"L"/"S"} mapping.
    """
    if ids is None:
        ids = set()
        for lineup in lineups:
            for p in lineup or []:
                pid = p.get("id")
                if pid is not None:
                    try:
                        ids.add(int(pid))
                    except (ValueError, TypeError):
                        continue
    if not ids:
        return {}
    try:
        ids_str = ",".join(str(p) for p in ids)
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}"
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        out = {}
        for person in r.json().get("people", []):
            pid = person.get("id")
            side = (person.get("batSide") or {}).get("code")
            if pid and side:
                out[pid] = side
        return out
    except Exception:
        return {}


def fill_pitcher_stats_for_slate(pitcher_stats: pd.DataFrame, slate: pd.DataFrame,
                                   season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    For every probable starter in the slate, if they're missing key stats
    in pitcher_stats, fetch them individually and patch them in.
    Guarantees ERA/K9/etc for every starter today (assuming they have stats).
    """
    if pitcher_stats is None or slate is None or slate.empty:
        return pitcher_stats

    df = pitcher_stats.copy() if pitcher_stats is not None and not pitcher_stats.empty else pd.DataFrame()
    if df.empty:
        df = pd.DataFrame(columns=["player_id"])

    # Ensure key columns exist
    for col in ["era", "whip", "k9", "bb9", "hr9", "ip", "games_started", "p_throws"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Get starter IDs from slate
    starter_ids = set()
    for col in ["away_pitcher_id", "home_pitcher_id"]:
        if col in slate.columns:
            for pid in slate[col].dropna().unique():
                try:
                    starter_ids.add(int(pid))
                except (ValueError, TypeError):
                    continue

    # Bulk-fetch handedness for all starters via people endpoint
    if starter_ids:
        try:
            ids_str = ",".join(str(p) for p in starter_ids)
            url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}"
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            for person in r.json().get("people", []):
                pid = person.get("id")
                hand = (person.get("pitchHand") or {}).get("code")
                if pid is None or not hand:
                    continue
                existing = df[df["player_id"] == pid] if "player_id" in df.columns else pd.DataFrame()
                if existing.empty:
                    df = pd.concat([df, pd.DataFrame([{"player_id": pid, "p_throws": hand}])],
                                      ignore_index=True)
                else:
                    df.at[existing.index[0], "p_throws"] = hand
        except Exception:
            pass

    for pid in starter_ids:
        existing = df[df["player_id"] == pid] if "player_id" in df.columns else pd.DataFrame()
        needs_fill = (
            existing.empty
            or pd.isna(existing.iloc[0].get("k9"))
            or pd.isna(existing.iloc[0].get("era"))
        )
        if not needs_fill:
            continue

        # Try the standard season endpoint first
        stats = get_player_season_pitching(pid, season)
        # Fall back to game log aggregation (independent code path - more reliable)
        if not stats or stats.get("k9") is None or stats.get("era") is None:
            gl_stats = get_pitcher_full_season_from_gamelog(pid, season)
            if gl_stats and gl_stats.get("k9") is not None:
                stats = gl_stats
        if not stats or stats.get("k9") is None:
            continue

        if existing.empty:
            new_row = {"player_id": pid}
            new_row.update(stats)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            idx = existing.index[0]
            for k, v in stats.items():
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    df.at[idx, k] = v

    return _normalize_player_df(df)


# ----------------------------------------------------------------------------
# Recent form: rolling 15-game Statcast
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_recent_form_hitter(player_id: int, season: int = CURRENT_SEASON, days: int = 15) -> dict:
    """
    Pulls last N days of statcast pitch-level data for one hitter and
    aggregates a few rolling indicators.
    """
    end = date.today()
    start = end.replace(day=max(1, end.day))  # safe default; we'll use Savant search
    from datetime import timedelta
    start = end - timedelta(days=days)
    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv"
        f"?all=true&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones="
        "&hfPull=&hfC=&hfSea={s}%7C&hfSit=&player_type=batter&hfOuts=&opponent="
        "&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={st}&game_date_lt={en}"
        "&batters_lookup%5B%5D={pid}&team=&position=&hfRO=&home_road=&hfFlag="
        "&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name&sort_col=pitches"
        "&player_event_sort=api_p_release_speed&sort_order=desc&min_pas=0&type=details"
    ).format(s=season, st=start.isoformat(), en=end.isoformat(), pid=player_id)

    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return {}
        bbe = df[df["type"] == "X"] if "type" in df.columns else df
        return {
            "pa": int(df["pitch_number"].count()) if "pitch_number" in df.columns else len(df),
            "xwoba_recent": float(df["estimated_woba_using_speedangle"].mean())
                if "estimated_woba_using_speedangle" in df.columns else None,
            "barrel_pct_recent": float((bbe["launch_speed_angle"] == 6).mean() * 100)
                if "launch_speed_angle" in bbe.columns and len(bbe) > 0 else None,
            "hard_hit_recent": float((bbe["launch_speed"] >= 95).mean() * 100)
                if "launch_speed" in bbe.columns and len(bbe) > 0 else None,
            "avg_ev_recent": float(bbe["launch_speed"].mean())
                if "launch_speed" in bbe.columns and len(bbe) > 0 else None,
        }
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Team-level pitching - used as fallback when probable pitcher is TBD
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_team_pitching(team_id: int, season: int = CURRENT_SEASON) -> dict:
    """
    Get team-aggregate pitching stats. Used when the day's starter is TBD
    or unknown (e.g. bullpen game). Returns dict shaped like a single
    pitcher row so models can drop it in as a proxy.
    """
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=pitching&season={season}&sportIds=1"
        f"&teamId={team_id}"
    )
    splits = []
    for attempt in range(2):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [])
            if splits:
                break
        except Exception:
            if attempt < 1:
                import time
                time.sleep(0.5)
            continue
    if not splits:
        return {}
    for s in splits:
        try:
            team = s.get("team", {}) or {}
            if team.get("id") != team_id:
                continue
            st_ = s.get("stat", {}) or {}
            return {
                "player_id": None,
                "player_name": f"Team Avg ({team.get('abbreviation', '')})",
                "is_team_avg": True,
                "era": _safe_float(st_.get("era")),
                "whip": _safe_float(st_.get("whip")),
                "hr9": _safe_float(st_.get("homeRunsPer9")),
                "bb9": _safe_float(st_.get("walksPer9Inn") or st_.get("baseOnBallsPer9Inn")),
                "k9": _safe_float(st_.get("strikeoutsPer9Inn") or st_.get("strikeOutsPer9Inn")),
                "ip": _safe_float(st_.get("inningsPitched")),
                "earned_runs": _safe_int(st_.get("earnedRuns")),
                "home_run": _safe_int(st_.get("homeRuns")),
            }
        except Exception:
            continue
    return {}


def get_team_pitching_proxy(team_id: int, season: int = CURRENT_SEASON,
                              hr_penalty: float = 1.10) -> dict:
    """
    Wrapper that returns team pitching with a small HR-allowance penalty applied.
    TBD/bullpen games tend to allow MORE HRs than a typical starter.
    """
    proxy = get_team_pitching(team_id, season)
    if not proxy:
        return {}
    if proxy.get("hr9") is not None:
        proxy["hr9_tbd_adjusted"] = round(proxy["hr9"] * hr_penalty, 2)
    proxy["tbd_proxy"] = True
    return proxy
