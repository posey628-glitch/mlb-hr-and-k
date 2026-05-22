"""
splits.py
==========
Pulls:
  - Hitter vs LHP / RHP season splits (xwOBA, ISO, K%, etc.)
  - Pitcher vs LHH / RHH season splits
  - Batter-vs-Pitcher career history (PA, H, HR, AVG, K%)
  - Performance vs similar-arsenal pitchers (proxy for "this kind of stuff")

All from MLB Stats API + Baseball Savant. No keys needed.
"""

from __future__ import annotations
import io
from typing import Optional

import pandas as pd
import requests
import streamlit as st

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/csv, */*",
}


# ---------------------------------------------------------------------------
# Hitter splits vs LHP / RHP
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_hitter_splits(season: int) -> pd.DataFrame:
    """
    Pulls Statcast hitter splits vs RHP and vs LHP.
    Returns long-format df with one row per (player, opponent_hand).
    """
    frames = []
    for hand, code in [("R", "vsR"), ("L", "vsL")]:
        url = (
            "https://baseballsavant.mlb.com/leaderboard/custom"
            f"?year={season}&type=batter&filter=&min=q"
            f"&selections=pa,k_percent,bb_percent,woba,xwoba,xiso,iso,"
            f"barrel_batted_rate,hard_hit_percent,sweet_spot_percent,"
            f"xwobacon,whiff_percent"
            f"&chart=false&x=pa&y=pa&r=no&chartType=beeswarm"
            f"&hand={hand}&csv=true"
        )
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df["opp_pitcher_hand"] = hand
            if "last_name, first_name" in df.columns:
                df["player_name"] = df["last_name, first_name"].apply(
                    lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
                    if isinstance(s, str) and "," in s else s
                )
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl=3600)
def get_pitcher_splits(season: int) -> pd.DataFrame:
    """Pitcher vs LHH / RHH splits."""
    frames = []
    for hand, code in [("R", "vsR"), ("L", "vsL")]:
        url = (
            "https://baseballsavant.mlb.com/leaderboard/custom"
            f"?year={season}&type=pitcher&filter=&min=q"
            f"&selections=pa,k_percent,bb_percent,woba,xwoba,iso,"
            f"barrel_batted_rate,hard_hit_percent,whiff_percent,csw_percent"
            f"&chart=false&x=pa&y=pa&r=no&chartType=beeswarm"
            f"&hand={hand}&csv=true"
        )
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df["batter_hand"] = hand
            if "last_name, first_name" in df.columns:
                df["player_name"] = df["last_name, first_name"].apply(
                    lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")]))
                    if isinstance(s, str) and "," in s else s
                )
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Batter vs Pitcher (career history)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400)  # 24h - this rarely changes
def get_bvp(batter_id: int, pitcher_id: int) -> dict:
    """
    Career stats for a specific batter vs specific pitcher via MLB Stats API.
    Returns dict with pa, ab, h, hr, bb, k, avg, ops, etc.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{batter_id}/stats"
        f"?stats=vsPlayer&group=hitting&opposingPlayerId={pitcher_id}&sportId=1"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {"pa": 0}
        # Aggregate across seasons
        agg = {"pa": 0, "ab": 0, "h": 0, "hr": 0, "bb": 0, "k": 0, "double": 0,
               "triple": 0, "rbi": 0}
        for s in splits:
            st_ = s.get("stat", {})
            agg["pa"] += int(st_.get("plateAppearances", 0) or 0)
            agg["ab"] += int(st_.get("atBats", 0) or 0)
            agg["h"] += int(st_.get("hits", 0) or 0)
            agg["hr"] += int(st_.get("homeRuns", 0) or 0)
            agg["bb"] += int(st_.get("baseOnBalls", 0) or 0)
            agg["k"] += int(st_.get("strikeOuts", 0) or 0)
            agg["double"] += int(st_.get("doubles", 0) or 0)
            agg["triple"] += int(st_.get("triples", 0) or 0)
            agg["rbi"] += int(st_.get("rbi", 0) or 0)
        # Derived
        agg["avg"] = round(agg["h"] / agg["ab"], 3) if agg["ab"] else 0.0
        agg["k_pct"] = round(agg["k"] / agg["pa"] * 100, 1) if agg["pa"] else 0.0
        agg["bb_pct"] = round(agg["bb"] / agg["pa"] * 100, 1) if agg["pa"] else 0.0
        agg["iso"] = round(
            (agg["double"] + agg["triple"] * 2 + agg["hr"] * 3) / agg["ab"], 3
        ) if agg["ab"] else 0.0
        return agg
    except Exception:
        return {"pa": 0}


def bvp_for_lineup(lineup: list[dict], pitcher_id: int) -> pd.DataFrame:
    """Build BvP rows for an entire lineup against one pitcher."""
    if not pitcher_id or pd.isna(pitcher_id):
        return pd.DataFrame()
    rows = []
    for p in lineup:
        if not p.get("id"):
            continue
        h = get_bvp(int(p["id"]), int(pitcher_id))
        rows.append({"player_id": p["id"], "player_name": p["name"], **h})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Similar-arsenal pitchers (using pitch-mix similarity)
# ---------------------------------------------------------------------------

def find_similar_pitchers(
    target_pitcher_id: int,
    target_throws: str,
    arsenal_df: pd.DataFrame,
    n: int = 10,
) -> list[int]:
    """
    Given a pitcher, find the N most similar pitchers by:
      - Same throwing hand
      - Closest pitch-mix usage % distribution (cosine similarity)

    Returns list of pitcher_ids.
    """
    if arsenal_df.empty or "player_id" not in arsenal_df.columns:
        return []

    # Build pitcher x pitch_type usage matrix
    pivot = arsenal_df.pivot_table(
        index="player_id", columns="pitch_name",
        values="pitch_usage", aggfunc="sum", fill_value=0,
    )
    if target_pitcher_id not in pivot.index:
        return []

    target_vec = pivot.loc[target_pitcher_id].values
    target_norm = (target_vec ** 2).sum() ** 0.5
    if target_norm == 0:
        return []

    # Cosine similarity
    sims = {}
    for pid in pivot.index:
        if pid == target_pitcher_id:
            continue
        vec = pivot.loc[pid].values
        n_ = (vec ** 2).sum() ** 0.5
        if n_ == 0:
            continue
        cos = float((target_vec * vec).sum() / (target_norm * n_))
        sims[pid] = cos

    # Optionally filter by same handedness if we have that mapped
    ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in ranked[:n]]


def hitter_vs_similar(
    batter_id: int,
    similar_pitcher_ids: list[int],
) -> dict:
    """
    Aggregate batter's career stats vs a list of similar-arsenal pitchers.
    """
    if not similar_pitcher_ids:
        return {"pa": 0}
    agg = {"pa": 0, "ab": 0, "h": 0, "hr": 0, "k": 0, "bb": 0,
           "double": 0, "triple": 0, "rbi": 0}
    for pid in similar_pitcher_ids:
        h = get_bvp(int(batter_id), int(pid))
        for k in ("pa", "ab", "h", "hr", "k", "bb", "double", "triple", "rbi"):
            agg[k] += h.get(k, 0)
    agg["avg"] = round(agg["h"] / agg["ab"], 3) if agg["ab"] else 0.0
    agg["k_pct"] = round(agg["k"] / agg["pa"] * 100, 1) if agg["pa"] else 0.0
    agg["iso"] = round(
        (agg["double"] + agg["triple"] * 2 + agg["hr"] * 3) / agg["ab"], 3
    ) if agg["ab"] else 0.0
    return agg
