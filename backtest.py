"""
Backtest framework - tracks model projections vs actual outcomes.

v2 (June 2026 fixes):
  - Snapshot persistence: uses Streamlit Community Cloud's st.session_state
    as an in-memory store with GitHub Gist as a durable remote backup.
    Falls back gracefully to /tmp if Gist token not set.
  - Outcome fetching: uses /api/v1/schedule?hydrate=linescore,boxscore to
    pull ALL game boxscores in ONE API call instead of N sequential calls.
    Adds retry logic with backoff. Caches outcomes per date.
  - Error surfacing: every failure now returns an error key so the UI can
    show the actual reason instead of "no outcomes matched".
"""

from datetime import datetime, timedelta, date
from pathlib import Path
import json
import time
import pandas as pd
import requests

try:
    import streamlit as st
    _HAVE_ST = True
except ImportError:
    _HAVE_ST = False

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, */*",
}


# ---------------------------------------------------------------------------
# Snapshot persistence — three-tier strategy
# ---------------------------------------------------------------------------
# Tier 1: st.session_state (in-memory, survives reruns in the same session)
# Tier 2: GitHub Gist (durable across deploys — requires GIST_TOKEN secret)
# Tier 3: /tmp local file (survives within the same container lifetime only)
#
# This means:
#   - Snapshots work immediately within a session (no setup needed)
#   - If GIST_TOKEN is set in Streamlit Secrets, they survive restarts/redeploys
#   - Even without Gist, the same-day backtest works fine in one session

def _session_key(d): return f"_snapshot_{d}"


def _gist_token():
    """Return GitHub token for Gist API if configured in Streamlit Secrets."""
    if not _HAVE_ST:
        return None
    try:
        return st.secrets.get("gist_token") or st.secrets.get("GIST_TOKEN")
    except Exception:
        return None


def _gist_id():
    """Return existing Gist ID for snapshot storage if configured."""
    if not _HAVE_ST:
        return None
    try:
        return st.secrets.get("gist_id") or st.secrets.get("GIST_ID")
    except Exception:
        return None


def _tmp_path(d):
    p = Path("/tmp/mlb_backtest")
    p.mkdir(exist_ok=True, parents=True)
    return p / f"snapshot_{d}.json"


def save_snapshot(snapshot_date, matchup_df: pd.DataFrame,
                  pitcher_slate_df: pd.DataFrame) -> bool:
    """
    Persist a slim version of today's projections.
    Saves to session_state immediately, then attempts Gist and /tmp.
    Returns True if at least session_state save succeeded.
    """
    try:
        def _records(df, keep_cols):
            sub = df[[c for c in keep_cols if c in df.columns]].copy()
            return sub.where(sub.notna(), None).to_dict("records")

        hitter_records = []
        if matchup_df is not None and not matchup_df.empty:
            hitter_records = _records(matchup_df, [
                "player_id", "player_name", "team", "opp", "lineup_pos",
                "power_score", "hr_game_pct", "hr_pa_pct",
                "matchup", "sleeper_score", "barrel_pct", "iso",
            ])

        pitcher_records = []
        if pitcher_slate_df is not None and not pitcher_slate_df.empty:
            pitcher_records = _records(pitcher_slate_df, [
                "player_id", "pitcher_name", "team", "opp",
                "test_score", "kHR", "hr_suppress", "proj_k", "role", "reliability",
            ])

        payload = {
            "date": str(snapshot_date),
            "saved_at": datetime.utcnow().isoformat(),
            "hitters": hitter_records,
            "pitchers": pitcher_records,
        }
        payload_str = json.dumps(payload, default=str)

        # Tier 1: session_state (always works)
        if _HAVE_ST:
            st.session_state[_session_key(str(snapshot_date))] = payload_str

        # Tier 2: GitHub Gist (durable — optional)
        token = _gist_token()
        gid = _gist_id()
        if token and gid:
            try:
                filename = f"snapshot_{snapshot_date}.json"
                r = requests.patch(
                    f"https://api.github.com/gists/{gid}",
                    headers={"Authorization": f"token {token}",
                             "Accept": "application/vnd.github.v3+json"},
                    json={"files": {filename: {"content": payload_str}}},
                    timeout=10,
                )
                # If patch fails (Gist not initialized), create it
                if r.status_code == 404:
                    requests.post(
                        "https://api.github.com/gists",
                        headers={"Authorization": f"token {token}",
                                 "Accept": "application/vnd.github.v3+json"},
                        json={"public": False, "description": "DingerMaven snapshots",
                              "files": {filename: {"content": payload_str}}},
                        timeout=10,
                    )
            except Exception:
                pass  # Gist failure is non-fatal

        # Tier 3: /tmp (same-container lifetime)
        try:
            _tmp_path(str(snapshot_date)).write_text(payload_str)
        except Exception:
            pass

        return True
    except Exception:
        return False


def load_snapshot(snapshot_date) -> dict | None:
    """Load snapshot - checks session_state → /tmp → Gist in that order."""
    key = str(snapshot_date)

    # Tier 1: session_state
    if _HAVE_ST:
        raw = st.session_state.get(_session_key(key))
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass

    # Tier 2: /tmp
    p = _tmp_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass

    # Tier 3: GitHub Gist
    token = _gist_token()
    gid = _gist_id()
    if token and gid:
        try:
            r = requests.get(
                f"https://api.github.com/gists/{gid}",
                headers={"Authorization": f"token {token}",
                         "Accept": "application/vnd.github.v3+json"},
                timeout=10,
            )
            if r.status_code == 200:
                files = r.json().get("files", {})
                fname = f"snapshot_{key}.json"
                if fname in files:
                    content = files[fname].get("content")
                    if content:
                        data = json.loads(content)
                        # Warm lower tiers
                        if _HAVE_ST:
                            st.session_state[_session_key(key)] = content
                        try:
                            _tmp_path(key).write_text(content)
                        except Exception:
                            pass
                        return data
        except Exception:
            pass

    return None


def list_snapshots() -> list[str]:
    """Return sorted list of dates with snapshots across all tiers."""
    dates = set()

    # Tier 1: session_state
    if _HAVE_ST:
        for k in st.session_state:
            if k.startswith("_snapshot_"):
                dates.add(k.replace("_snapshot_", ""))

    # Tier 2: /tmp
    try:
        p = Path("/tmp/mlb_backtest")
        if p.exists():
            for f in p.glob("snapshot_*.json"):
                dates.add(f.stem.replace("snapshot_", ""))
    except Exception:
        pass

    # Tier 3: Gist
    token = _gist_token()
    gid = _gist_id()
    if token and gid:
        try:
            r = requests.get(
                f"https://api.github.com/gists/{gid}",
                headers={"Authorization": f"token {token}",
                         "Accept": "application/vnd.github.v3+json"},
                timeout=8,
            )
            if r.status_code == 200:
                for fname in r.json().get("files", {}):
                    if fname.startswith("snapshot_") and fname.endswith(".json"):
                        dates.add(fname.replace("snapshot_", "").replace(".json", ""))
        except Exception:
            pass

    return sorted(dates)


# ---------------------------------------------------------------------------
# Outcome fetcher — ONE call per date using hydrated schedule
# ---------------------------------------------------------------------------

def _fetch_with_retry(url, headers=HEADERS, timeout=25, max_retries=3) -> requests.Response | None:
    """GET with exponential backoff. Returns Response or None on total failure."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                # Rate limited — wait longer
                time.sleep(3 * (2 ** attempt))
                continue
            if r.status_code == 403:
                # Blocked — no point retrying immediately, but try once more
                if attempt == 0:
                    time.sleep(2)
                    continue
                return None
            # Other error — log and return None
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < max_retries - 1:
                time.sleep(1 * (2 ** attempt))
            continue
        except Exception:
            return None
    return None


# Cache outcomes per date so re-evaluation doesn't re-fetch
_OUTCOME_CACHE: dict[str, dict] = {}


def fetch_hitter_outcomes(target_date) -> dict:
    """
    Fetch which hitters homered on target_date.

    Uses hydrated schedule (?hydrate=linescore,boxscore) to get all boxscores
    in ONE API call instead of N sequential calls. Falls back to per-game if
    hydrated schedule returns empty.

    Returns {player_id_int: {"hr": int, "ab": int, ...}} or {"_error": str}.
    """
    key = str(target_date)
    if key in _OUTCOME_CACHE:
        return _OUTCOME_CACHE[key]

    # Strategy 1: hydrated schedule (1 call for all games)
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={target_date}"
        f"&hydrate=linescore,boxscore"
    )
    r = _fetch_with_retry(url)
    if r is None:
        err = {"_error": f"MLB API unreachable for {target_date} (403/timeout after retries). "
                         f"This is a Streamlit Cloud IP-blocking issue with statsapi.mlb.com. "
                         f"The backtest will work when run locally or when MLB unblocks this IP."}
        _OUTCOME_CACHE[key] = err
        return err

    try:
        data = r.json()
    except Exception:
        err = {"_error": f"MLB API returned non-JSON response for {target_date}"}
        _OUTCOME_CACHE[key] = err
        return err

    out = {}
    games_processed = 0
    games_with_box = 0

    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            game_pk = g.get("gamePk")
            games_processed += 1

            # Try to get boxscore from hydrated response first
            box = g.get("liveData", {}) or {}
            boxscore = box.get("boxscore") or {}

            # If hydration didn't include boxscore data (game not yet final),
            # try a direct boxscore fetch as fallback
            if not boxscore:
                box_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
                br = _fetch_with_retry(box_url, timeout=12, max_retries=2)
                if br:
                    try:
                        bdata = br.json()
                        boxscore = bdata.get("liveData", {}).get("boxscore", {}) or {}
                    except Exception:
                        pass

            if not boxscore:
                continue

            games_with_box += 1
            for side in ("away", "home"):
                team_block = (boxscore.get("teams") or {}).get(side) or {}
                players = team_block.get("players") or {}
                for pid_key, player in players.items():
                    pid = (player.get("person") or {}).get("id")
                    if not pid:
                        continue
                    bat = (player.get("stats") or {}).get("batting") or {}
                    if not bat:
                        continue
                    # Must have appeared at bat
                    ab = int(bat.get("atBats") or 0)
                    if ab == 0:
                        continue
                    out[int(pid)] = {
                        "hr": int(bat.get("homeRuns") or 0),
                        "ab": ab,
                        "h": int(bat.get("hits") or 0),
                        "k": int(bat.get("strikeOuts") or 0),
                        "bb": int(bat.get("baseOnBalls") or 0),
                        "rbi": int(bat.get("rbi") or 0),
                    }

    if not out and games_processed > 0:
        # Games found but no player data — likely games not yet played
        out["_error"] = (
            f"Found {games_processed} games for {target_date} but "
            f"only {games_with_box} had boxscore data. "
            f"Games may not have been played yet, or boxscores not yet available."
        )
    elif not out and games_processed == 0:
        out["_error"] = f"No games found for {target_date} — may be an off-day or future date."

    out["_games_processed"] = games_processed
    out["_games_with_box"] = games_with_box
    _OUTCOME_CACHE[key] = out
    return out


def fetch_pitcher_outcomes(target_date) -> dict:
    """
    Fetch actual pitcher stats for target_date.
    Returns {player_id_int: {"ip", "k", "er", "bb", "hr", "h"}}
    """
    key = f"p_{target_date}"
    if key in _OUTCOME_CACHE:
        return _OUTCOME_CACHE[key]

    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={target_date}"
        f"&hydrate=linescore,boxscore"
    )
    r = _fetch_with_retry(url)
    if r is None:
        err = {"_error": f"MLB API unreachable for pitcher outcomes on {target_date}"}
        _OUTCOME_CACHE[key] = err
        return err

    try:
        data = r.json()
    except Exception:
        return {}

    out = {}
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            game_pk = g.get("gamePk")
            box = g.get("liveData", {}) or {}
            boxscore = box.get("boxscore") or {}
            if not boxscore:
                br = _fetch_with_retry(
                    f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
                    timeout=12, max_retries=2
                )
                if br:
                    try:
                        boxscore = br.json().get("liveData", {}).get("boxscore", {}) or {}
                    except Exception:
                        pass
            if not boxscore:
                continue
            for side in ("away", "home"):
                team_block = (boxscore.get("teams") or {}).get(side) or {}
                players = team_block.get("players") or {}
                for _, player in players.items():
                    pid = (player.get("person") or {}).get("id")
                    if not pid:
                        continue
                    pit = (player.get("stats") or {}).get("pitching") or {}
                    if not pit or not pit.get("gamesStarted"):
                        continue
                    try:
                        ip_str = str(pit.get("inningsPitched") or "0.0")
                        whole, _, frac = ip_str.partition(".")
                        ip = float(whole) + (float(frac or 0) / 3.0)
                    except Exception:
                        ip = 0.0
                    if ip == 0:
                        continue
                    out[int(pid)] = {
                        "ip": ip,
                        "k": int(pit.get("strikeOuts") or 0),
                        "er": int(pit.get("earnedRuns") or 0),
                        "bb": int(pit.get("baseOnBalls") or 0),
                        "hr": int(pit.get("homeRuns") or 0),
                        "h": int(pit.get("hits") or 0),
                    }
    _OUTCOME_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate_hitter_projections(snapshot: dict, actuals: dict) -> dict:
    """Compare projections to actuals. Returns metric dict with error key if failed."""
    if not snapshot:
        return {"error": "No snapshot data"}

    # Surface API errors from fetch step
    if actuals.get("_error"):
        return {
            "error": actuals["_error"],
            "_games_processed": actuals.get("_games_processed", 0),
            "_games_with_box": actuals.get("_games_with_box", 0),
        }

    if not actuals:
        return {"error": "Actuals dict is empty — games may not have been played yet"}

    hitters = snapshot.get("hitters", [])
    rows = []
    unmatched = 0
    for h in hitters:
        pid = h.get("player_id")
        if pid is None:
            continue
        try:
            pid = int(pid)
        except (ValueError, TypeError):
            continue
        # Try int key first, then str fallback
        actual = actuals.get(pid) or actuals.get(str(pid))
        if not actual or actual.get("ab", 0) == 0:
            unmatched += 1
            continue
        rows.append({
            "player_name": h.get("player_name"),
            "team": h.get("team"),
            "power_score": h.get("power_score"),
            "hr_game_pct": h.get("hr_game_pct"),
            "sleeper_score": h.get("sleeper_score"),
            "actual_hr": actual.get("hr", 0),
            "actual_ab": actual.get("ab", 0),
            "homered": actual.get("hr", 0) > 0,
        })

    if not rows:
        return {
            "error": (
                f"No hitters matched between snapshot ({len(hitters)} saved) "
                f"and actuals ({len([k for k in actuals if not str(k).startswith('_')])} returned). "
                f"Unmatched: {unmatched}. "
                f"Games fetched: {actuals.get('_games_processed', '?')}, "
                f"with boxscores: {actuals.get('_games_with_box', '?')}. "
                f"This usually means player_id types don't match or games haven't been played."
            )
        }

    df = pd.DataFrame(rows)
    df["homered"] = df["homered"].astype(int)
    n_played = len(df[df["actual_ab"] > 0])

    metrics = {
        "total_hitters_matched": len(df),
        "hitters_who_played": n_played,
        "total_actual_hrs": int(df["actual_hr"].sum()),
        "actual_hr_rate_pct": round(df["homered"].sum() / max(1, n_played) * 100, 1),
        "_unmatched_in_snapshot": unmatched,
        "_games_processed": actuals.get("_games_processed", "?"),
        "_games_with_box": actuals.get("_games_with_box", "?"),
    }

    df_played = df[df["actual_ab"] > 0].copy()
    if not df_played.empty and df_played["hr_game_pct"].notna().any():
        valid = df_played[df_played["hr_game_pct"].notna()].copy()
        if len(valid) > 0:
            valid["predicted_p"] = valid["hr_game_pct"] / 100
            brier = float(((valid["predicted_p"] - valid["homered"].astype(float)) ** 2).mean())
            metrics["brier_score"] = round(brier, 4)
            metrics["brier_n"] = int(len(valid))

        bands = [(0, 5), (5, 12), (12, 18), (18, 25), (25, 100)]
        band_summary = []
        for low, high in bands:
            sub = df_played[(df_played["hr_game_pct"] >= low) & (df_played["hr_game_pct"] < high)]
            if len(sub) == 0:
                continue
            band_summary.append({
                "band": f"{low}-{high}%",
                "n": len(sub),
                "predicted_rate": round(sub["hr_game_pct"].mean(), 1),
                "actual_rate": round(sub["homered"].mean() * 100, 1),
                "calibration_error": round(sub["homered"].mean() * 100 - sub["hr_game_pct"].mean(), 1),
            })
        metrics["hr_pct_bands"] = band_summary

    if not df_played.empty and df_played["power_score"].notna().any():
        ps_summary = []
        for low, high in [(0, 25), (25, 50), (50, 70), (70, 100)]:
            sub = df_played[(df_played["power_score"] >= low) & (df_played["power_score"] < high)]
            if len(sub) == 0:
                continue
            ps_summary.append({
                "band": f"PS {low}-{high}",
                "n": len(sub),
                "actual_hr_rate": round(sub["homered"].mean() * 100, 1),
            })
        metrics["power_score_bands"] = ps_summary

    if "hr_game_pct" in df_played.columns and df_played["hr_game_pct"].notna().any():
        top10 = df_played.nlargest(10, "hr_game_pct")
        metrics["top10_hr_predictions"] = [
            {"name": r["player_name"], "predicted_hr_pct": round(r["hr_game_pct"], 1),
             "homered": bool(r["homered"])}
            for _, r in top10.iterrows()
        ]
        metrics["top10_hr_hit_rate"] = round(
            top10["homered"].sum() / len(top10) * 100, 1) if len(top10) else 0.0

    if "sleeper_score" in df_played.columns and df_played["sleeper_score"].notna().any():
        top10_sl = df_played[df_played["sleeper_score"] > 0].nlargest(10, "sleeper_score")
        if not top10_sl.empty:
            metrics["top10_sleeper_predictions"] = [
                {"name": r["player_name"], "sleeper_score": round(r["sleeper_score"], 1),
                 "homered": bool(r["homered"])}
                for _, r in top10_sl.iterrows()
            ]
            metrics["top10_sleeper_hit_rate"] = round(
                top10_sl["homered"].sum() / len(top10_sl) * 100, 1)

    return metrics


def evaluate_pitcher_projections(snapshot: dict, actuals: dict) -> dict:
    """Compare K projections to actuals."""
    if not snapshot:
        return {"error": "No snapshot"}
    if actuals.get("_error"):
        return {"error": actuals["_error"]}
    if not actuals:
        return {"error": "No pitcher actuals"}

    pitchers = snapshot.get("pitchers", [])
    rows = []
    for p in pitchers:
        pid = p.get("player_id")
        if pid is None:
            continue
        try:
            pid = int(pid)
        except (ValueError, TypeError):
            continue
        actual = actuals.get(pid) or actuals.get(str(pid))
        if not actual or actual.get("ip", 0) == 0:
            continue
        rows.append({
            "pitcher_name": p.get("pitcher_name"),
            "team": p.get("team"),
            "proj_k": p.get("proj_k"),
            "actual_ip": actual.get("ip", 0),
            "actual_k": actual.get("k", 0),
        })
    if not rows:
        return {"error": f"No pitchers matched (snapshot had {len(pitchers)}, actuals had {len(actuals)})"}

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["proj_k"])
    if df.empty:
        return {"error": "No pitchers with both projected and actual K data"}

    df["k_error"] = df["actual_k"] - df["proj_k"]
    metrics = {
        "total_pitchers_matched": len(df),
        "k_projection_rmse": round(((df["k_error"] ** 2).mean()) ** 0.5, 2),
        "k_projection_bias": round(df["k_error"].mean(), 2),
        "k_projections_detail": [
            {"name": r["pitcher_name"], "projected_k": round(r["proj_k"], 1),
             "actual_k": int(r["actual_k"]), "actual_ip": round(r["actual_ip"], 1),
             "diff": round(r["k_error"], 1)}
            for _, r in df.sort_values("proj_k", ascending=False).iterrows()
        ],
    }
    return metrics


# ---------------------------------------------------------------------------
# Rolling aggregate
# ---------------------------------------------------------------------------

def _rolling_aggregate_uncached(snapshot_dates_key: str, max_days: int) -> dict:
    available = list_snapshots()
    if not available:
        return {"error": "No snapshots available. The app auto-saves a snapshot each day it loads. Check the sidebar for snapshot count."}

    dates_to_use = sorted(available)[-max_days:]

    n_snapshots = 0
    all_top10_rows = []
    all_brier = []
    per_day = []
    fetch_errors = []

    for sd in dates_to_use:
        snapshot = load_snapshot(sd)
        if not snapshot:
            continue
        hitter_actuals = fetch_hitter_outcomes(sd)
        if hitter_actuals.get("_error"):
            fetch_errors.append(f"{sd}: {hitter_actuals['_error'][:80]}")
            continue
        h_metrics = evaluate_hitter_projections(snapshot, hitter_actuals)
        if not h_metrics or h_metrics.get("error"):
            fetch_errors.append(f"{sd}: {h_metrics.get('error', 'evaluate failed')[:80]}")
            continue

        n_snapshots += 1
        per_day.append({
            "date": sd,
            "hitters_played": h_metrics.get("hitters_who_played", 0),
            "actual_hrs": h_metrics.get("total_actual_hrs", 0),
            "slate_hr_rate_pct": h_metrics.get("actual_hr_rate_pct", 0),
            "top10_hit_rate_pct": h_metrics.get("top10_hr_hit_rate", 0),
            "brier": h_metrics.get("brier_score"),
        })
        if h_metrics.get("brier_score") is not None:
            all_brier.append(h_metrics["brier_score"])
        for entry in h_metrics.get("top10_hr_predictions", []):
            all_top10_rows.append({
                "date": sd, "name": entry.get("name"),
                "predicted_pct": entry.get("predicted_hr_pct"),
                "homered": int(entry.get("homered", False)),
            })

    if n_snapshots == 0:
        error_detail = "\n".join(fetch_errors[:5]) if fetch_errors else "No snapshots had actuals"
        return {
            "error": (
                f"No snapshots had matchable outcomes. "
                f"Fetch errors ({len(fetch_errors)}): {error_detail}"
            )
        }

    top10_total = len(all_top10_rows)
    top10_hr_count = sum(r["homered"] for r in all_top10_rows)
    top10_hr_rate = (top10_hr_count / top10_total * 100) if top10_total else 0
    slate_rates = [d["slate_hr_rate_pct"] for d in per_day if d["slate_hr_rate_pct"]]
    avg_slate = sum(slate_rates) / len(slate_rates) if slate_rates else 0
    by_date = {}
    for r in all_top10_rows:
        by_date.setdefault(r["date"], []).append(r["homered"])
    days_with_hit = sum(1 for picks in by_date.values() if sum(picks) > 0)

    return {
        "n_snapshots": n_snapshots,
        "top10_picks_total": top10_total,
        "top10_hrs_hit": top10_hr_count,
        "top10_hr_rate_pct": round(top10_hr_rate, 1),
        "slate_baseline_hr_rate_pct": round(avg_slate, 1),
        "edge_vs_slate_pp": round(top10_hr_rate - avg_slate, 1),
        "days_with_any_top10_hit": days_with_hit,
        "days_total": len(by_date),
        "any_hit_rate_pct": round(days_with_hit / len(by_date) * 100, 1) if by_date else 0,
        "brier_mean": round(sum(all_brier) / len(all_brier), 4) if all_brier else None,
        "per_day_summary": per_day,
        "_fetch_errors": fetch_errors,
    }


if _HAVE_ST:
    rolling_aggregate_hitters = st.cache_data(ttl=1800)(
        lambda snapshot_dates=None, max_days=30: _rolling_aggregate_uncached(
            ",".join(sorted(snapshot_dates)) if snapshot_dates else "", max_days
        )
    )
else:
    def rolling_aggregate_hitters(snapshot_dates=None, max_days=30):
        key = ",".join(sorted(snapshot_dates)) if snapshot_dates else ""
        return _rolling_aggregate_uncached(key, max_days)
