"""
data_fetcher.py
================
Pulls all data from public sources:
  - MLB Stats API (statsapi.mlb.com) - slate, lineups, traditional stats, handedness
  - Baseball Savant (baseballsavant.mlb.com) - Statcast stats, arsenals, sprint speed
"""

from __future__ import annotations

import io
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
import requests
import streamlit as st

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
# Shared normalizers
# ----------------------------------------------------------------------------

def _normalize_player_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "player_id" not in df.columns:
        for cand in ["mlb_id", "playerid", "MLBAMID", "mlbam_id", "player_mlb_id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "player_id"})
                break
    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    return df


def _derive_hitter_missing(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    needs_iso = "iso" not in df.columns or df["iso"].isna().all()
    if needs_iso:
        slg_col = next((c for c in ["slg", "slugging_percentage"] if c in df.columns), None)
        avg_col = next((c for c in ["batting_avg", "avg", "ba"] if c in df.columns), None)
        if slg_col and avg_col:
            df["iso"] = (df[slg_col] - df[avg_col]).round(3)
    return df


# ----------------------------------------------------------------------------
# Safe parsers
# ----------------------------------------------------------------------------

def _safe_float(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            f = float(val)
            return f if not (f != f) else None
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
# Slate / probable pitchers
# ----------------------------------------------------------------------------

@st.cache_data(ttl=1800)
def get_slate(game_date: Optional[str] = None) -> pd.DataFrame:
    if game_date is None:
        game_date = date.today().isoformat()
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={game_date}"
        "&hydrate=probablePitcher,linescore,team"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    rows = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            try:
                away = g["teams"]["away"]
                home = g["teams"]["home"]
                rows.append({
                    "gamePk": g["gamePk"],
                    "gameTime": g.get("gameDate"),
                    "status": g.get("status", {}).get("detailedState"),
                    "venue": g.get("venue", {}).get("name"),
                    "away_team": away["team"]["name"],
                    "away_team_abbr": away["team"].get("abbreviation", away["team"]["name"][:3].upper()),
                    "away_team_id": away["team"]["id"],
                    "home_team": home["team"]["name"],
                    "home_team_abbr": home["team"].get("abbreviation", home["team"]["name"][:3].upper()),
                    "home_team_id": home["team"]["id"],
                    "away_pitcher": (away.get("probablePitcher") or {}).get("fullName", "TBD"),
                    "away_pitcher_id": (away.get("probablePitcher") or {}).get("id"),
                    "home_pitcher": (home.get("probablePitcher") or {}).get("fullName", "TBD"),
                    "home_pitcher_id": (home.get("probablePitcher") or {}).get("id"),
                })
            except Exception:
                continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df["gameTime"] = pd.to_datetime(df["gameTime"], errors="coerce")
    return df


@st.cache_data(ttl=1800)
def get_lineup(game_pk: int, side: str = "home") -> list:
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
            try:
                p = players.get(f"ID{pid}", {})
                person = p.get("person", {})
                out.append({
                    "id": person.get("id"),
                    "name": person.get("fullName"),
                    "position": p.get("position", {}).get("abbreviation"),
                    "bats": p.get("batSide", {}).get("code"),
                })
            except Exception:
                continue
        return out
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_team_roster(team_id: int) -> list:
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster/active"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        out = []
        for p in r.json().get("roster", []):
            try:
                pos = p.get("position", {}).get("abbreviation", "")
                if pos == "P":
                    continue
                out.append({
                    "id": p["person"]["id"],
                    "name": p["person"]["fullName"],
                    "position": pos,
                })
            except Exception:
                continue
        return out
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_all_team_rosters(slate: pd.DataFrame) -> dict:
    rosters = {}
    if slate is None or slate.empty:
        return rosters
    team_ids = set()
    for _, g in slate.iterrows():
        try:
            team_ids.add(int(g["away_team_id"]))
            team_ids.add(int(g["home_team_id"]))
        except Exception:
            continue
    for tid in team_ids:
        rosters[tid] = get_team_roster(tid)
    return rosters


# ----------------------------------------------------------------------------
# Statcast hitter stats - min=1 to include all hitters
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_hitter_stats(season: int = CURRENT_SEASON) -> pd.DataFrame:
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
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    if "last_name, first_name" in df.columns:
        df["player_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
            if isinstance(s, str) and "," in s else s
        )
    if "pull_air_percent" in df.columns and "barrel_batted_rate" in df.columns:
        df["pulled_brl_pct"] = (df["pull_air_percent"] * df["barrel_batted_rate"] / 100).round(2)
    df = _normalize_player_df(df)
    df = _derive_hitter_missing(df)
    return df


# ----------------------------------------------------------------------------
# Statcast pitcher stats - min=1 to include all pitchers
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_pitcher_stats(season: int = CURRENT_SEASON) -> pd.DataFrame:
    selections = (
        "pa,k_percent,bb_percent,woba,xwoba,xiso,xba,xslg,xobp,"
        "barrel_batted_rate,hard_hit_percent,avg_best_speed,avg_hit_angle,"
        "whiff_percent,swing_percent,sweet_spot_percent,xwobacon,iso,babip,"
        "launch_speed,launch_angle,p_total_pitches,p_total_swinging_strike,"
        "csw_percent,zone_percent,in_zone_swing_miss_percent,"
        "f_strike_percent,oz_swing_percent,z_swing_percent,"
        "groundballs_percent,flyballs_percent,linedrives_percent,popups_percent,"
        "pull_percent,straightaway_percent,opposite_percent,home_run"
    )
    url = (
        "https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={season}&type=pitcher&filter=&min=1&selections={selections}"
        "&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    if "last_name, first_name" in df.columns:
        df["player_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
            if isinstance(s, str) and "," in s else s
        )
    df = _normalize_player_df(df)
    return df


# ----------------------------------------------------------------------------
# Pitcher arsenal
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_pitcher_arsenal(season: int = CURRENT_SEASON) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
        f"?type=pitcher&pitchType=&year={season}&team=&min=10&hand="
        "&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()
    if "last_name, first_name" in df.columns:
        df["player_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
            if isinstance(s, str) and "," in s else s
        )
    return df


@st.cache_data(ttl=3600)
def get_pitcher_arsenal_by_count(season: int = CURRENT_SEASON) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
        f"?type=pitcher&pitchType=&year={season}&team=&min=10&hand=&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df["count_state"] = "all"
        return df
    except Exception:
        return pd.DataFrame()


# ----------------------------------------------------------------------------
# Sprint speed + run values
# ----------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_sprint_speed(season: int = CURRENT_SEASON) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/sprint_speed"
        f"?year={season}&position=&team=&min=10&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()
    if "last_name, first_name" in df.columns:
        df["player_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
            if isinstance(s, str) and "," in s else s
        )
    return df


@st.cache_data(ttl=86400)
def get_pitch_run_values(season: int = CURRENT_SEASON) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-arsenals"
        f"?year={season}&min=10&type=run_value&hand=&csv=true"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()
    if "last_name, first_name" in df.columns:
        df["player_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
            if isinstance(s, str) and "," in s else s
        )
    return df


# ----------------------------------------------------------------------------
# Recent form (per player)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=1800)
def get_pitcher_recent_form(pitcher_id: int, season: int = CURRENT_SEASON, n_starts: int = 5) -> dict:
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
        starts = []
        for s in splits:
            gs = _safe_int(s.get("stat", {}).get("gamesStarted"))
            if gs and gs > 0:
                starts.append(s)
        recent = starts[-n_starts:] if len(starts) > n_starts else starts
        if not recent:
            return {}
        ip_sum, er_sum, k_sum, bb_sum, hr_sum = 0.0, 0, 0, 0, 0
        for s in recent:
            st_ = s.get("stat", {})
            ip = _safe_float(st_.get("inningsPitched")) or 0
            ip_sum += ip
            er_sum += _safe_int(st_.get("earnedRuns")) or 0
            k_sum += _safe_int(st_.get("strikeOuts")) or 0
            bb_sum += _safe_int(st_.get("baseOnBalls")) or 0
            hr_sum += _safe_int(st_.get("homeRuns")) or 0
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
def get_hitter_recent_form_trad(player_id: int, season: int = CURRENT_SEASON, n_games: int = 15) -> dict:
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
        ab, h, hr, k, bb, rbi, d, t = 0, 0, 0, 0, 0, 0, 0, 0
        for s in recent:
            st_ = s.get("stat", {})
            ab += _safe_int(st_.get("atBats")) or 0
            h += _safe_int(st_.get("hits")) or 0
            hr += _safe_int(st_.get("homeRuns")) or 0
            k += _safe_int(st_.get("strikeOuts")) or 0
            bb += _safe_int(st_.get("baseOnBalls")) or 0
            rbi += _safe_int(st_.get("rbi")) or 0
            d += _safe_int(st_.get("doubles")) or 0
            t += _safe_int(st_.get("triples")) or 0
        if ab == 0:
            return {}
        return {
            "recent_games": len(recent),
            "recent_ab": ab, "recent_h": h, "recent_hr": hr,
            "recent_k": k, "recent_bb": bb, "recent_rbi": rbi,
            "recent_avg": round(h / ab, 3),
            "recent_iso": round((d + 2 * t + 3 * hr) / ab, 3) if ab else 0.0,
            "recent_k_pct": round(k / (ab + bb) * 100, 1) if (ab + bb) else 0.0,
            "recent_ops_proxy": round((h + bb) / (ab + bb), 3) if (ab + bb) else 0.0,
        }
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Traditional stats - playerPool=All catches non-qualifying players
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_hitter_traditional(season: int = CURRENT_SEASON) -> pd.DataFrame:
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


@st.cache_data(ttl=3600)
def get_pitcher_traditional(season: int = CURRENT_SEASON) -> pd.DataFrame:
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=pitching&season={season}&sportIds=1"
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
                "era": _safe_float(st_.get("era")),
                "whip": _safe_float(st_.get("whip")),
                "hr9": _safe_float(st_.get("homeRunsPer9")),
                "bb9": _safe_float(st_.get("walksPer9Inn") or st_.get("baseOnBallsPer9Inn")),
                "k9": _safe_float(st_.get("strikeoutsPer9Inn") or st_.get("strikeOutsPer9Inn")),
                "ip": _safe_float(st_.get("inningsPitched")),
                "wins": _safe_int(st_.get("wins")),
                "losses": _safe_int(st_.get("losses")),
                "games_started": _safe_int(st_.get("gamesStarted")),
                "strikeouts": _safe_int(st_.get("strikeOuts")),
                "walks": _safe_int(st_.get("baseOnBalls")),
                "earned_runs": _safe_int(st_.get("earnedRuns")),
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    return _normalize_player_df(df)


# ----------------------------------------------------------------------------
# Per-pitcher fallback + handedness bulk fetch
# ----------------------------------------------------------------------------

@st.cache_data(ttl=1800)
def get_player_season_pitching(player_id: int, season: int = CURRENT_SEASON) -> dict:
    """Fetch one pitcher's season stats directly by ID."""
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=season&group=pitching&season={season}&sportId=1"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {}
        st_ = splits[0].get("stat", {}) or {}
        return {
            "era": _safe_float(st_.get("era")),
            "whip": _safe_float(st_.get("whip")),
            "hr9": _safe_float(st_.get("homeRunsPer9")),
            "bb9": _safe_float(st_.get("walksPer9Inn") or st_.get("baseOnBallsPer9Inn")),
            "k9": _safe_float(st_.get("strikeoutsPer9Inn") or st_.get("strikeOutsPer9Inn")),
            "ip": _safe_float(st_.get("inningsPitched")),
            "games_started": _safe_int(st_.get("gamesStarted")),
            "strikeouts": _safe_int(st_.get("strikeOuts")),
            "walks": _safe_int(st_.get("baseOnBalls")),
            "earned_runs": _safe_int(st_.get("earnedRuns")),
        }
    except Exception:
        return {}


def fill_pitcher_stats_for_slate(pitcher_stats: pd.DataFrame, slate: pd.DataFrame,
                                   season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    For every probable starter in the slate, fetch handedness and any missing
    season stats individually so the table is complete.
    """
    if pitcher_stats is None or slate is None or slate.empty:
        return pitcher_stats

    df = pitcher_stats.copy() if pitcher_stats is not None and not pitcher_stats.empty else pd.DataFrame()
    if df.empty:
        df = pd.DataFrame(columns=["player_id"])

    for col in ["era", "whip", "k9", "bb9", "hr9", "ip", "games_started", "p_throws"]:
        if col not in df.columns:
            df[col] = pd.NA

    starter_ids = set()
    for col in ["away_pitcher_id", "home_pitcher_id"]:
        if col in slate.columns:
            for pid in slate[col].dropna().unique():
                try:
                    starter_ids.add(int(pid))
                except (ValueError, TypeError):
                    continue

    # Bulk-fetch handedness for all starters at once
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

    # Per-pitcher season stats fill
    for pid in starter_ids:
        existing = df[df["player_id"] == pid] if "player_id" in df.columns else pd.DataFrame()
        needs_fill = (
            existing.empty
            or pd.isna(existing.iloc[0].get("k9"))
            or pd.isna(existing.iloc[0].get("era"))
        )
        if not needs_fill:
            continue

        stats = get_player_season_pitching(pid, season)
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
