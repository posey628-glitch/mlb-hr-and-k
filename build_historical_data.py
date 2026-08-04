#!/usr/bin/env python3
"""
build_historical_data.py — PRECOMPUTE historical data ONCE per season.

WHY THIS EXISTS (the user's insight, v46.38):
  Prior-completed-season stats are STATIC — they never change. Fetching them
  live on every page load was making the app crawl (~5,400 API calls before
  render). Instead we build them ONCE into historical_data.json, which the app
  loads INSTANTLY from disk. Only the CURRENT season's daily game results change
  day-to-day, and those flow through the normal live fetch — not this file.

WHEN TO RUN:
  - Once at the START of each season (the 3-year window rolls automatically).
  - Optionally re-run mid-season to add newly-called-up players who weren't in
    the initial universe (their history still doesn't change — you're just
    adding rows).
  - You do NOT run this daily. Daily game results are handled by the app's
    normal current-season fetch + the snapshot/pattern loop.

HOW IT WORKS:
  1. Pull the universe of relevant hitters (all players with recent MLB time).
  2. For each, fetch 3-yr priors (get_hitter_historical_priors) + 3-yr splits
     (get_hitter_historical_splits) — the SAME functions the app's fallback
     uses, so the data is identical.
  3. Write everything to historical_data.json keyed by player_id.

USAGE:
  python build_historical_data.py                 # current-season universe
  python build_historical_data.py --season 2027   # force a season
  python build_historical_data.py --ids 592450,660271   # specific players

  The output file (historical_data.json) is committed to the repo so Streamlit
  Cloud serves it — no API calls at render.
"""
import argparse
import json
import sys
import time
from datetime import datetime

# The app's own fetchers — reuse them so the data matches exactly.
import data_fetcher as d


def get_hitter_universe(season: int) -> list:
    """All hitters with MLB playing time in the current or prior season — the
    set worth precomputing history for. Uses the stats leaderboard so we don't
    miss anyone who might appear in a lineup."""
    ids = set()
    for yr in (season, season - 1):
        try:
            # season hitting leaderboard — everyone with PAs that year
            url = (f"https://statsapi.mlb.com/api/v1/stats"
                   f"?stats=season&group=hitting&season={yr}&sportId=1"
                   f"&limit=2000&playerPool=All")
            r = d.requests.get(url, headers=d.HEADERS, timeout=30)
            if r.status_code == 200:
                for sg in r.json().get("stats", []):
                    for sp in sg.get("splits", []):
                        pid = (sp.get("player") or {}).get("id")
                        if pid:
                            ids.add(int(pid))
        except Exception as e:
            print(f"  ! universe fetch {yr} failed: {e}")
    return sorted(ids)


def build(season: int, ids: list, out_path: str):
    print(f"Building historical data for season {season} "
          f"(prior 3 yrs: {season-1}, {season-2}, {season-3})")
    print(f"Universe: {len(ids)} hitters\n")

    priors, splits = {}, {}
    t0 = time.time()
    # process in chunks so partial progress is visible + resumable
    CHUNK = 25
    for i in range(0, len(ids), CHUNK):
        chunk = tuple(ids[i:i + CHUNK])
        # call the UNWRAPPED functions (skip st.cache_data in a plain script)
        pf = getattr(d.get_hitter_historical_priors, "__wrapped__",
                     d.get_hitter_historical_priors)
        sf = getattr(d.get_hitter_historical_splits, "__wrapped__",
                     d.get_hitter_historical_splits)
        try:
            pr = pf(chunk, current_year=season)
            for _, row in pr.iterrows():
                pid = str(int(row["player_id"]))
                priors[pid] = {k: (None if v != v else v)
                               for k, v in row.items() if k != "player_id"}
        except Exception as e:
            print(f"  ! priors chunk {i} failed: {e}")
        try:
            sr = sf(chunk, current_year=season)
            for _, row in sr.iterrows():
                pid = str(int(row["player_id"]))
                splits[pid] = {k: (None if v != v else v)
                               for k, v in row.items() if k != "player_id"}
        except Exception as e:
            print(f"  ! splits chunk {i} failed: {e}")
        done = min(i + CHUNK, len(ids))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed else 0
        eta = (len(ids) - done) / rate if rate else 0
        print(f"  {done}/{len(ids)} hitters  "
              f"({len(priors)} priors, {len(splits)} splits)  "
              f"~{eta:.0f}s left")

    payload = {
        "meta": {
            "built_at": datetime.utcnow().isoformat() + "Z",
            "season": season,
            "prior_seasons": [season - 1, season - 2, season - 3],
            "n_priors": len(priors),
            "n_splits": len(splits),
        },
        "priors": priors,
        "splits": splits,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"\n✅ wrote {out_path}")
    print(f"   {len(priors)} priors, {len(splits)} splits, "
          f"built in {time.time()-t0:.0f}s")
    print(f"   Commit this file so Streamlit Cloud serves it (instant load).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=datetime.now().year)
    ap.add_argument("--ids", type=str, default="",
                    help="comma-separated player_ids (else full universe)")
    ap.add_argument("--out", type=str, default="historical_data.json")
    args = ap.parse_args()

    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        print("Fetching hitter universe...")
        ids = get_hitter_universe(args.season)
        if not ids:
            print("! could not build universe (network?). Aborting.")
            sys.exit(1)

    build(args.season, ids, args.out)


if __name__ == "__main__":
    main()
