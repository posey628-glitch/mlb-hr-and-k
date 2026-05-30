"""
Backtest framework - tracks model projections vs actual outcomes.

V1: in-memory snapshot via st.session_state and snapshot-fetch on demand.
The user clicks a button to evaluate yesterday's predictions and we
fetch actual outcomes from MLB Stats API to compare.

For long-term tracking across container restarts, this same data can be
mirrored to a Gist or GitHub repo (future enhancement).
"""

from datetime import datetime, timedelta, date
from pathlib import Path
import json
import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}


# ----------------------------------------------------------------------------
# Local persistence (best effort - works on local dev, may not on Streamlit Cloud)
# ----------------------------------------------------------------------------

# Storage location - try a persistent path first, fall back to /tmp.
# Streamlit Cloud doesn't persist /tmp across container restarts, but it does
# persist files written to relative paths within the app directory (sometimes).
# For long-term persistence across days/deploys, snapshots should ultimately be
# pushed to a remote store (Gist, S3, Supabase). V1 uses local disk best-effort.
def _get_snapshot_dir():
    candidates = [
        Path("/mount/src/mlb-hr-and-k/backtest_data"),  # Streamlit Cloud relative
        Path.cwd() / "backtest_data",                    # local dev
        Path("/tmp/mlb_backtest"),                       # always-writable fallback
    ]
    for c in candidates:
        try:
            c.mkdir(exist_ok=True, parents=True)
            # Test writability
            test = c / ".write_test"
            test.write_text("ok")
            test.unlink()
            return c
        except Exception:
            continue
    # Last resort
    return Path("/tmp")

SNAPSHOT_DIR = _get_snapshot_dir()


def save_snapshot(snapshot_date, matchup_df: pd.DataFrame,
                    pitcher_slate_df: pd.DataFrame) -> bool:
    """
    Persist a slim version of today's projections to a JSON file.
    Returns True if saved.
    """
    try:
        path = SNAPSHOT_DIR / f"snapshot_{snapshot_date}.json"

        # Slim hitter projections.
        # CRITICAL: do NOT fillna(0) here. NaN HR Game% means "insufficient
        # sample, no projection" — converting to 0 would land those hitters
        # in the 0-5% calibration band when evaluated, inflating that band's
        # apparent accuracy (since "0% predicted, didn't homer" looks correct
        # but was never actually a prediction). Use None (JSON null) instead
        # so evaluate_hitter_projections can filter them out.
        def _na_safe_records(df, keep_cols):
            sub = df[keep_cols].copy()
            # Convert NaN to None so JSON serializes as null, not 0
            return sub.where(sub.notna(), None).to_dict("records")

        hitter_records = []
        if matchup_df is not None and not matchup_df.empty:
            keep_cols = [c for c in [
                "player_id", "player_name", "team", "opp", "lineup_pos",
                "power_score", "hr_game_pct", "hr_pa_pct",
                "matchup", "sleeper_score", "barrel_pct", "iso",
            ] if c in matchup_df.columns]
            hitter_records = _na_safe_records(matchup_df, keep_cols)

        pitcher_records = []
        if pitcher_slate_df is not None and not pitcher_slate_df.empty:
            keep_cols = [c for c in [
                "player_id", "pitcher_name", "team", "opp",
                "test_score", "kHR", "hr_suppress",
                "proj_k", "role", "reliability",
            ] if c in pitcher_slate_df.columns]
            pitcher_records = _na_safe_records(pitcher_slate_df, keep_cols)

        payload = {
            "date": str(snapshot_date),
            "saved_at": datetime.utcnow().isoformat(),
            "hitters": hitter_records,
            "pitchers": pitcher_records,
        }
        with open(path, "w") as f:
            json.dump(payload, f, default=str)
        return True
    except Exception:
        return False


def load_snapshot(snapshot_date) -> dict | None:
    """Load a previously saved snapshot for the given date."""
    try:
        path = SNAPSHOT_DIR / f"snapshot_{snapshot_date}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def list_snapshots() -> list[str]:
    """Return list of dates we have snapshots for."""
    try:
        return sorted([
            p.stem.replace("snapshot_", "")
            for p in SNAPSHOT_DIR.glob("snapshot_*.json")
        ])
    except Exception:
        return []


# ----------------------------------------------------------------------------
# Actual outcome fetcher - what actually happened on a given date
# ----------------------------------------------------------------------------

def fetch_hitter_outcomes(target_date) -> dict:
    """
    For a given date, fetch which hitters homered and their game line.
    Returns {player_id: {"hr": int, "ab": int, "h": int, "k": int, "bb": int, "rbi": int}}
    """
    try:
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={target_date}&hydrate=team,probablePitcher"
        )
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    out = {}
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            game_pk = g.get("gamePk")
            if not game_pk:
                continue
            # Fetch detailed boxscore for this game
            try:
                box_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
                br = requests.get(box_url, headers=HEADERS, timeout=15)
                br.raise_for_status()
                box = br.json()
            except Exception:
                continue
            try:
                liveData = box.get("liveData", {})
                boxscore = liveData.get("boxscore", {}) or {}
                for side in ("away", "home"):
                    team_block = boxscore.get("teams", {}).get(side, {}) or {}
                    players = team_block.get("players", {}) or {}
                    for pid_key, player in players.items():
                        pid = (player.get("person") or {}).get("id")
                        if not pid:
                            continue
                        bat = (player.get("stats") or {}).get("batting") or {}
                        if not bat:
                            continue
                        out[pid] = {
                            "hr": int(bat.get("homeRuns") or 0),
                            "ab": int(bat.get("atBats") or 0),
                            "h": int(bat.get("hits") or 0),
                            "k": int(bat.get("strikeOuts") or 0),
                            "bb": int(bat.get("baseOnBalls") or 0),
                            "rbi": int(bat.get("rbi") or 0),
                        }
            except Exception:
                continue
    return out


def fetch_pitcher_outcomes(target_date) -> dict:
    """
    For a given date, fetch what each starting pitcher actually did.
    Returns {player_id: {"ip": float, "k": int, "er": int, "bb": int, "hr": int, "h": int}}
    """
    try:
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={target_date}&hydrate=team"
        )
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    out = {}
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            game_pk = g.get("gamePk")
            if not game_pk:
                continue
            try:
                box_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
                br = requests.get(box_url, headers=HEADERS, timeout=15)
                br.raise_for_status()
                box = br.json()
            except Exception:
                continue
            try:
                liveData = box.get("liveData", {})
                boxscore = liveData.get("boxscore", {}) or {}
                for side in ("away", "home"):
                    team_block = boxscore.get("teams", {}).get(side, {}) or {}
                    players = team_block.get("players", {}) or {}
                    for pid_key, player in players.items():
                        pid = (player.get("person") or {}).get("id")
                        if not pid:
                            continue
                        pit = (player.get("stats") or {}).get("pitching") or {}
                        if not pit:
                            continue
                        # Only count starters - inningsPitched > 0 and gameStarted
                        gs_flag = pit.get("gamesStarted", 0)
                        if not gs_flag:
                            continue
                        try:
                            ip_str = str(pit.get("inningsPitched") or "0.0")
                            # MLB IP format: "5.2" means 5 and 2/3 innings
                            whole, _, frac = ip_str.partition(".")
                            ip = float(whole) + (float(frac or 0) / 3.0)
                        except Exception:
                            ip = 0.0
                        out[pid] = {
                            "ip": ip,
                            "k": int(pit.get("strikeOuts") or 0),
                            "er": int(pit.get("earnedRuns") or 0),
                            "bb": int(pit.get("baseOnBalls") or 0),
                            "hr": int(pit.get("homeRuns") or 0),
                            "h": int(pit.get("hits") or 0),
                        }
            except Exception:
                continue
    return out


# ----------------------------------------------------------------------------
# Accuracy evaluation
# ----------------------------------------------------------------------------

def evaluate_hitter_projections(snapshot: dict, actuals: dict) -> dict:
    """
    Given a snapshot of hitter projections and the actual outcomes,
    compute accuracy metrics:
      - HR hit rate by Power Score band
      - HR hit rate by HR Game% prediction band
      - Sleeper accuracy
      - Brier score (probabilistic forecast quality)
    """
    if not snapshot or not actuals:
        return {}

    hitters = snapshot.get("hitters", [])
    rows = []
    for h in hitters:
        pid = h.get("player_id")
        if pid is None:
            continue
        try:
            pid = int(pid)
        except (ValueError, TypeError):
            continue
        actual = actuals.get(pid) or actuals.get(str(pid))
        if not actual:
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
        return {"error": "No matched hitters between snapshot and actuals"}

    df = pd.DataFrame(rows)
    df["homered"] = df["homered"].astype(int)
    n_played = len(df[df["actual_ab"] > 0])

    metrics = {
        "total_hitters_matched": len(df),
        "hitters_who_played": n_played,
        "total_actual_hrs": int(df["actual_hr"].sum()),
        "actual_hr_rate_pct": round(df["homered"].sum() / max(1, n_played) * 100, 1),
    }

    # Accuracy by HR Game% band
    df_played = df[df["actual_ab"] > 0].copy()
    if not df_played.empty and df_played["hr_game_pct"].notna().any():
        # BRIER SCORE — the single most informative calibration metric for
        # probabilistic forecasts. Lower = better. For rare events like HRs,
        # well-calibrated models score around 0.04-0.06; values <0.03 are
        # excellent, >0.08 mean systematic over-prediction.
        # Formula: mean((predicted_probability - actual_outcome)^2)
        valid = df_played[df_played["hr_game_pct"].notna()].copy()
        if len(valid) > 0:
            valid["predicted_p"] = valid["hr_game_pct"] / 100
            valid["homered_int"] = valid["homered"].astype(int)
            brier = float(((valid["predicted_p"] - valid["homered_int"]) ** 2).mean())
            metrics["brier_score"] = round(brier, 4)
            metrics["brier_n"] = int(len(valid))

        bands = [(0, 5), (5, 12), (12, 18), (18, 25), (25, 100)]
        band_summary = []
        for low, high in bands:
            sub = df_played[(df_played["hr_game_pct"] >= low) & (df_played["hr_game_pct"] < high)]
            if len(sub) == 0:
                continue
            actual_rate = sub["homered"].mean() * 100
            predicted_rate = sub["hr_game_pct"].mean()
            band_summary.append({
                "band": f"{low}-{high}%",
                "n": len(sub),
                "predicted_rate": round(predicted_rate, 1),
                "actual_rate": round(actual_rate, 1),
                "calibration_error": round(actual_rate - predicted_rate, 1),
            })
        metrics["hr_pct_bands"] = band_summary

    # Power Score accuracy
    if not df_played.empty and df_played["power_score"].notna().any():
        ps_bands = [(0, 25), (25, 50), (50, 70), (70, 100)]
        ps_summary = []
        for low, high in ps_bands:
            sub = df_played[(df_played["power_score"] >= low) & (df_played["power_score"] < high)]
            if len(sub) == 0:
                continue
            actual_rate = sub["homered"].mean() * 100
            ps_summary.append({
                "band": f"PS {low}-{high}",
                "n": len(sub),
                "actual_hr_rate": round(actual_rate, 1),
            })
        metrics["power_score_bands"] = ps_summary

    # Top 10 predicted vs actual
    if "hr_game_pct" in df_played.columns and df_played["hr_game_pct"].notna().any():
        top10 = df_played.nlargest(10, "hr_game_pct")
        metrics["top10_hr_predictions"] = [
            {
                "name": r["player_name"],
                "predicted_hr_pct": round(r["hr_game_pct"], 1),
                "homered": bool(r["homered"]),
            }
            for _, r in top10.iterrows()
        ]
        metrics["top10_hr_hit_rate"] = round(
            top10["homered"].sum() / len(top10) * 100, 1
        ) if len(top10) > 0 else 0.0

    # Sleeper accuracy - top 10 by sleeper_score
    if "sleeper_score" in df_played.columns and df_played["sleeper_score"].notna().any():
        top10_sl = df_played[df_played["sleeper_score"] > 0].nlargest(10, "sleeper_score")
        if not top10_sl.empty:
            metrics["top10_sleeper_predictions"] = [
                {
                    "name": r["player_name"],
                    "sleeper_score": round(r["sleeper_score"], 1),
                    "homered": bool(r["homered"]),
                }
                for _, r in top10_sl.iterrows()
            ]
            metrics["top10_sleeper_hit_rate"] = round(
                top10_sl["homered"].sum() / len(top10_sl) * 100, 1
            )

    return metrics


def evaluate_pitcher_projections(snapshot: dict, actuals: dict) -> dict:
    """
    Compare projected Ks to actual Ks for each starter.
    """
    if not snapshot or not actuals:
        return {}
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
            "test_score": p.get("test_score"),
            "kHR": p.get("kHR"),
            "proj_k": p.get("proj_k"),
            "actual_ip": actual.get("ip", 0),
            "actual_k": actual.get("k", 0),
            "actual_er": actual.get("er", 0),
            "actual_hr_allowed": actual.get("hr", 0),
        })
    if not rows:
        return {"error": "No matched pitchers between snapshot and actuals"}

    df = pd.DataFrame(rows)
    df["k_error"] = df["actual_k"] - df["proj_k"]
    metrics = {
        "total_pitchers_matched": len(df),
        "k_projection_rmse": round(((df["k_error"] ** 2).mean()) ** 0.5, 2),
        "k_projection_bias": round(df["k_error"].mean(), 2),
    }

    # K accuracy detail
    df_sorted = df.sort_values("proj_k", ascending=False)
    metrics["k_projections_detail"] = [
        {
            "name": r["pitcher_name"],
            "projected_k": round(r["proj_k"], 1),
            "actual_k": int(r["actual_k"]),
            "actual_ip": round(r["actual_ip"], 1),
            "diff": round(r["k_error"], 1),
        }
        for _, r in df_sorted.iterrows()
    ]

    return metrics
