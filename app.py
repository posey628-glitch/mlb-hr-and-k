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
import requests
import streamlit as st

# APP VERSION - bump this any time we ship a bugfix that needs to invalidate
# the data cache (e.g. IL detection logic changed, park database updated, etc.)
# On startup we compare this against the cached version and clear @st.cache_data
# if they differ. This avoids the "user uploads new code but Streamlit serves
# the old cached function output until 1-hour TTL expires" problem.
APP_VERSION = "2026.05.30-leaders-v17b"

# Core imports - make each one defensive so a single missing function
# doesn't kill the whole app
try:
    from data_fetcher import get_slate
except ImportError:
    def get_slate(*a, **k): return pd.DataFrame()

try:
    from data_fetcher import _stats_day_key
except ImportError:
    # Fallback so callers using this don't break if module is older
    def _stats_day_key() -> str:
        from datetime import date
        return date.today().isoformat()

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

# splits.py was deleted — never used in main data flow.
# Per-pitch matchup data is captured by pitch_match_score (pitch_match.py),
# and pitcher handedness splits come via data_fetcher.get_pitcher_handedness_splits.
HAVE_SPLITS = False

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

# AUTO CACHE INVALIDATION — proper version
# Use @st.cache_data to store the deployed version PROCESS-WIDE (survives
# across user sessions, unlike st.session_state which is per-browser-tab).
# When the deployed version differs from what's cached, we clear.
# This means: clear ONCE per actual code deploy, not once per browser tab.
@st.cache_resource
def _deployed_version_tracker() -> dict:
    """Process-wide singleton storing the version that's currently 'live'.
    Returns a mutable dict so we can update it in-place when we detect a
    new deploy without invalidating the cache_resource itself."""
    return {"version": None}

_version_state = _deployed_version_tracker()
if _version_state["version"] != APP_VERSION:
    # First load after a deploy — clear data caches once for everyone.
    try:
        st.cache_data.clear()
    except Exception:
        pass
    _version_state["version"] = APP_VERSION
# Also remember in session_state so other code paths can read it (e.g. the
# diagnostic at the top of the page, sidebar metadata).
st.session_state["_app_version"] = APP_VERSION


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


def hr_grade(hr_game_pct, sample_size=None, pa_threshold=80):
    """Letter grade (A+/A/B+/B/C/D/F) for HR Game% - more intuitive than %.

    Calibrated to real-world MLB rates:
      A+ : ≥22%  (elite matchup - top 5% of plays)
      A  : 19-22% (very strong - top 15%)
      B+ : 16-19% (strong matchup - top 25%)
      B  : 12-16% (solid)
      C+ : 9-12%  (modest)
      C  : 6-9%   (below average)
      D  : 3-6%   (poor)
      F  : <3%    (avoid)
    """
    if hr_game_pct is None or pd.isna(hr_game_pct):
        return "—"
    # NaN PA = insufficient sample
    if sample_size is None or pd.isna(sample_size) or sample_size < pa_threshold:
        return "—"
    if hr_game_pct >= 22:
        return "A+"
    if hr_game_pct >= 19:
        return "A"
    if hr_game_pct >= 16:
        return "B+"
    if hr_game_pct >= 12:
        return "B"
    if hr_game_pct >= 9:
        return "C+"
    if hr_game_pct >= 6:
        return "C"
    if hr_game_pct >= 3:
        return "D"
    return "F"


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


def pitcher_grade(test_score, hr_suppress=None, sample_size=None, pa_threshold=80):
    """Letter grade + label for pitchers — matches BetGravy style.

    Combines test_score (K-focused) and hr_suppress (HR-friendly avoidance)
    to produce a pitcher matchup grade from the batter's perspective. This
    grade reflects the PITCHER'S SKILL ONLY — park and weather are NOT
    factored in (use env_adj_grade for that, which adjusts for tonight's
    environment).

      EXPLOITABLE+ : test ≤ 30 OR hr_suppress ≤ 30  (target this pitcher)
      EXPLOITABLE  : test ≤ 45 OR hr_suppress ≤ 45
      MIXED        : test 45-65, hr_suppress 45-65 (neutral)
      TOUGH        : test ≥ 65 AND hr_suppress ≥ 65
      ELITE        : test ≥ 80 AND hr_suppress ≥ 75 (avoid HR plays here)
    """
    if test_score is None or pd.isna(test_score):
        return "—"
    # NO sample (NaN/None) = insufficient. A pitcher with no PA faced shouldn't
    # get a grade. Was previously skipping the check for NaN which let pitchers
    # like Jordan Wicks (no data) get EXPLOIT+ grades.
    if sample_size is None or pd.isna(sample_size) or sample_size < pa_threshold:
        return "—"
    hr_s = hr_suppress if (hr_suppress is not None and not pd.isna(hr_suppress)) else test_score
    # Combined indicator - higher = harder to score against
    combined_min = min(test_score, hr_s)
    combined_max = max(test_score, hr_s)
    avg = (test_score + hr_s) / 2

    if avg >= 80 and combined_min >= 70:
        return "ELITE"
    if avg >= 65 and combined_min >= 55:
        return "TOUGH"
    if combined_max <= 30 or combined_min <= 25:
        return "EXPLOIT+"
    if combined_max <= 45 or combined_min <= 35:
        return "EXPLOIT"
    return "MIXED"


def pitcher_grade_env_adj(base_grade, env_mult):
    """Adjust the base pitcher grade for tonight's park × weather.

    The intuition: an ELITE pitcher in Coors with wind out (env_mult=1.20)
    is closer to TOUGH in HR-suppression terms tonight, and an EXPLOIT
    pitcher in Petco with cold wind in (env_mult=0.80) is closer to MIXED.

    Rules (env_mult is the game's park × weather HR multiplier):
      env >= 1.18 (very hitter-friendly):  shift one tier toward EXPLOIT
      env >= 1.10 (hitter-friendly):       shift half a tier (only soft tiers)
      env <= 0.85 (pitcher-friendly):      shift one tier toward TOUGH
      env <= 0.92 (mildly pitcher-friendly): shift half a tier (only soft tiers)
      else: no change

    Half-tier means: a borderline grade gets nudged; established tiers
    (ELITE / EXPLOIT+) hold unless the env shift is strong.
    """
    if base_grade in (None, "—"):
        return "—"
    if env_mult is None or pd.isna(env_mult):
        return base_grade

    em = float(env_mult)
    order = ["EXPLOIT+", "EXPLOIT", "MIXED", "TOUGH", "ELITE"]
    if base_grade not in order:
        return base_grade
    idx = order.index(base_grade)

    if em >= 1.18:
        new_idx = max(0, idx - 1)
    elif em >= 1.10:
        # Soft shift: only nudge MIXED→EXPLOIT and TOUGH→MIXED. Don't drag
        # ELITE down for a mild park boost.
        if base_grade in ("MIXED", "TOUGH"):
            new_idx = idx - 1
        else:
            new_idx = idx
    elif em <= 0.85:
        new_idx = min(len(order) - 1, idx + 1)
    elif em <= 0.92:
        if base_grade in ("EXPLOIT", "MIXED"):
            new_idx = idx + 1
        else:
            new_idx = idx
    else:
        new_idx = idx
    return order[new_idx]


def pitcher_grade_sort_key(grade):
    """Return numeric sort key for pitcher grade. Lower = more exploitable.
    Used so users can sort the grade column and get a sensible order.

    From batter perspective: EXPLOIT+ is BEST (target), ELITE is WORST (avoid).
    Sort ascending → most exploitable first (best for HR betting).
    """
    order = {
        "EXPLOIT+": 1,
        "EXPLOIT": 2,
        "MIXED": 3,
        "TOUGH": 4,
        "ELITE": 5,
        "—": 99,  # insufficient data goes last
    }
    return order.get(grade, 99)



# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚾ Posey MLB Props")
    selected_date = st.date_input("Slate date", value=date.today())

    # ========================================================================
    # OWNER MODE — toggle to expose owner-only features
    #
    # 🔑 EASIEST SETUP (no Streamlit secrets needed):
    #   1. Change OWNER_KEY below from "posey-mlb-owner-2026" to your own
    #      secret passphrase (any string — make it long and random)
    #   2. Save and deploy
    #   3. Bookmark: https://your-app-url/?owner=YOUR_KEY_HERE
    #      (replace YOUR_KEY_HERE with what you set above)
    #   4. Clicking that bookmark gives you owner mode automatically
    #
    # 🔒 MORE SECURE (optional - hides key from your GitHub repo):
    #   1. In Streamlit Cloud → Settings → Secrets, add:
    #      owner_key = "your-secret-string"
    #   2. If set, st.secrets["owner_key"] overrides the hardcoded value
    #
    # Public users see no owner-only sections. Owner sees: Twitter post,
    # My Picks editor, and any future owner-exclusive tools.
    # ========================================================================

    # 👇 Key is now loaded from Streamlit Secrets (set in app Settings → Secrets):
    #    owner_key = "Posey628628!"
    # Leaving this empty keeps your secret out of the public GitHub repo.
    OWNER_KEY = ""

    # Try to override from Streamlit secrets (if you set it up); otherwise use hardcoded
    try:
        _secret_key = st.secrets.get("owner_key", "")
        if _secret_key:
            OWNER_KEY = _secret_key
    except Exception:
        pass

    owner_mode = False
    if OWNER_KEY:
        # Method 1: URL param ?owner=...  (silent, instant via bookmark)
        try:
            qp = st.query_params
            url_key = qp.get("owner", "")
            if isinstance(url_key, list):
                url_key = url_key[0] if url_key else ""
            if url_key and url_key == OWNER_KEY:
                owner_mode = True
        except Exception:
            pass
        # Method 2: Password entry (collapsed expander — public users won't notice)
        if not owner_mode:
            with st.expander("🔐 Owner login", expanded=False):
                pwd = st.text_input(
                    "Owner key", type="password", key="owner_pwd",
                    help="Hidden field — only the app owner needs this.",
                )
                if pwd and pwd == OWNER_KEY:
                    owner_mode = True
                    st.success("✓ Owner mode active")
                elif pwd:
                    st.error("Invalid key.")
    if owner_mode:
        st.caption("👑 **Owner mode active** — extra tools enabled below.")

    st.subheader("Data sources")
    use_recent_form = st.checkbox("Recent form (L15)", value=True)
    use_pitch_match = st.checkbox("Pitch match score", value=HAVE_PITCH_MATCH)
    # BvP checkbox removed (was dead UI — the get_career_bvp_aggregate
    # function was never actually called in the data flow, only the arsenal
    # block ran). Per-pitch matchup is captured by pitch_match_score.
    use_bvp = False
    use_weather = st.checkbox("Weather + park factors", value=True)
    # Catcher framing remains disabled — was fetched but never wired into
    # k_total_projection. Reintroducing it cleanly would require pulling the
    # game's starting catcher from the lineup card then joining their framing
    # stat — non-trivial. Skip until needed.
    #
    # Umpire factor RE-ENABLED with a real lookup table (UMPIRE_K_FACTORS in
    # game_context.py). Most umpires are neutral; this only meaningfully moves
    # K projections on the ~10% of slates with an extreme umpire assigned.
    use_ump = st.checkbox(
        "Umpire K-zone tendencies",
        value=HAVE_GAME_CONTEXT,
        help=(
            "Apply known umpire K-rate tendencies to K projections. "
            "Most umpires are neutral (1.0×). Top K-friendly umpires "
            "add ~7% to K projections; tight-zone umpires subtract ~5%."
        ),
    )
    # Vegas totals removed (May 2026). We pull them from ESPN but never
    # actually USED them in any HR or K calculation — only displayed as a
    # metric. For HR/K props, the underlying signals (park × weather,
    # opp team K%/HR%, pitcher quality) already cover what Vegas total
    # represents in market-derived form. Keeping a stale "Vegas total"
    # display added load time and an extra API dependency for no analytical
    # value. The variable is kept = False so downstream code paths still work.
    use_vegas = False
    # Sprint speed has no impact on HR or K projections - removed from UI.
    # If you want speed-based metrics (steals, infield hits) in the future, re-enable.
    use_sprint_speed = False

    st.subheader("Display")
    show_diagnostic = st.checkbox("Show data diagnostic", value=False)
    show_legend = st.checkbox("Show legend / glossary", value=True)
    show_transactions = st.checkbox(
        "Show recent transactions", value=True,
        help=(
            "Display trades, signings, DFAs, releases, call-ups, IL moves "
            "from the last 2 days. Critical for catching mid-day roster changes "
            "that might affect tonight's slate."
        ),
    )

    st.divider()
    if st.button("🔄 Force refresh all data", help="Clears the data cache and re-fetches from APIs. Use if you see stale or missing data."):
        st.cache_data.clear()
        st.rerun()
    if st.button("⚾ Refresh slate ONLY (probable pitcher changes)",
                 help="Use when a team announces a new starter mid-day. Re-pulls "
                      "the schedule + probables from MLB Stats API without "
                      "re-fetching all hitter/pitcher stats (much faster)."):
        # Clear only the slate-related cache entries
        try:
            from data_fetcher import get_slate as _g
            if hasattr(_g, "clear"):
                _g.clear()
        except Exception:
            pass
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
            st.caption(f"Snapshots saved: **{len(existing)}** (most recent: {existing[-1]})")
            # Check coverage in the last 14 days
            from datetime import datetime, timedelta
            today = datetime.now().date()
            expected_days = set()
            for i in range(14):
                expected_days.add(str(today - timedelta(days=i)))
            covered = set(existing).intersection(expected_days)
            coverage_pct = len(covered) / 14 * 100
            if coverage_pct >= 80:
                st.caption(f"✅ 14-day coverage: {len(covered)}/14 days ({coverage_pct:.0f}%)")
            elif coverage_pct >= 50:
                st.caption(f"🟡 14-day coverage: {len(covered)}/14 days ({coverage_pct:.0f}%)")
            else:
                st.caption(f"⚠️ 14-day coverage: {len(covered)}/14 days — load app daily to build calibration data.")
            # NEW: Show staleness of most recent snapshot
            try:
                latest_date = datetime.strptime(existing[-1], "%Y-%m-%d").date()
                days_old = (today - latest_date).days
                if days_old == 0:
                    st.caption("✅ Today's snapshot exists — backtest data is current.")
                elif days_old == 1:
                    st.caption("✅ Last snapshot: yesterday.")
                elif days_old <= 3:
                    st.caption(f"🟡 Last snapshot: {days_old} days ago. Load the app today to capture it.")
                else:
                    st.caption(f"⚠️ Last snapshot: {days_old} days ago. Backtest data is stale.")
            except Exception:
                pass
        else:
            st.caption("⚠️ No snapshots yet. Auto-save happens when the app loads.")
    except Exception:
        st.caption("Backtest module unavailable")
    show_backtest = st.checkbox("Show backtest panel", value=False,
                                  help="See accuracy of past projections vs actual outcomes.")


# ============================================================================
# DATA LOAD
# ============================================================================

with st.spinner("Loading slate and stats..."):
    try:
        slate = get_slate(selected_date.isoformat())
    except (ConnectionError, requests.exceptions.RequestException) as _e:
        # MLB Stats API is temporarily unavailable. Show a friendly message
        # and stop instead of letting the raw traceback through.
        st.error(
            f"⚠️ **MLB Stats API is currently unreachable.**\n\n"
            f"This is usually temporary (MLB's servers occasionally overload). "
            f"Try one of:\n"
            f"- Wait 30-60 seconds and refresh the page\n"
            f"- Click '🔄 Force refresh all data' in the sidebar\n"
            f"- Check https://statsapi.mlb.com/api/v1/schedule?sportId=1 in your browser to confirm MLB is back up\n\n"
            f"_Technical detail: {type(_e).__name__}_"
        )
        st.stop()
    except Exception as _e:
        st.error(
            f"⚠️ **Unexpected error loading slate.**\n\n"
            f"Error: {type(_e).__name__}: {_e}\n\n"
            f"Try clicking '🔄 Force refresh all data' in the sidebar."
        )
        st.stop()

if slate.empty:
    st.warning(f"No MLB games found on {selected_date}. Try another date.")
    st.stop()

# ============================================================================
# FILTER: hide games that are already in progress or final
# ============================================================================
# When user is checking the slate mid-day, the 1pm games are already started
# and the data is no longer actionable for HR props. Only show upcoming games.
hide_started = st.sidebar.checkbox(
    "Hide games already started/final",
    value=True,
    help=(
        "Removes games whose first pitch was before 'now'. "
        "Uncheck if you want to review every game on the schedule including in-progress ones."
    ),
)
if hide_started and selected_date == datetime.now().date():
    try:
        now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
        if "gameTime" in slate.columns:
            def _is_upcoming(t):
                if pd.isna(t):
                    return True  # if no time, default to showing
                try:
                    if hasattr(t, "tzinfo") and t.tzinfo is not None:
                        return t > now_utc
                    # Assume UTC if naive
                    return pd.Timestamp(t).tz_localize("UTC") > now_utc
                except Exception:
                    return True
            mask = slate["gameTime"].apply(_is_upcoming)
            n_filtered = (~mask).sum()
            slate = slate[mask].reset_index(drop=True)
            if n_filtered > 0:
                st.info(
                    f"⏱️ Hiding {n_filtered} game{'s' if n_filtered != 1 else ''} "
                    f"that already started. Uncheck 'Hide games already started/final' "
                    f"in sidebar to see them."
                )
    except Exception as _e:
        # If filtering fails for any reason, don't block the user — show everything
        pass

if slate.empty:
    st.warning(
        f"All games on {selected_date} have started or finished. "
        f"Uncheck 'Hide games already started/final' in sidebar to see them."
    )
    st.stop()

# Statcast pulls — wrap with try/except so transient timeouts don't crash app
try:
    hitter_stats = get_hitter_stats(_stats_day=_stats_day_key()) if not slate.empty else pd.DataFrame()
except (ConnectionError, requests.exceptions.RequestException) as _e:
    st.error(
        f"⚠️ **Baseball Savant (hitter data) is currently unreachable.**\n\n"
        f"This is usually temporary. Try:\n"
        f"- Wait 30-60 seconds and refresh\n"
        f"- Click '🔄 Force refresh all data' in the sidebar\n\n"
        f"_Error: {type(_e).__name__}_"
    )
    st.stop()

try:
    pitcher_stats = get_pitcher_stats(_stats_day=_stats_day_key()) if not slate.empty else pd.DataFrame()
except (ConnectionError, requests.exceptions.RequestException) as _e:
    st.error(
        f"⚠️ **Baseball Savant (pitcher data) is currently unreachable.**\n\n"
        f"_Error: {type(_e).__name__}_"
    )
    st.stop()

try:
    pitcher_arsenal_all = get_pitcher_arsenal() if not slate.empty else pd.DataFrame()
except Exception:
    # Arsenal is non-critical; fall back to empty
    pitcher_arsenal_all = pd.DataFrame()

# Pitcher handedness splits (vs LHB / vs RHB)
# Use MLB Stats API statSplits which IS documented and works on Streamlit Cloud.
# We only fetch splits for pitchers in TODAY'S slate (typically 26-30 pitchers)
# instead of every pitcher in the league.
try:
    from data_fetcher import get_pitcher_handedness_splits
    slate_pitcher_ids = []
    for col in ["away_pitcher_id", "home_pitcher_id"]:
        if col in slate.columns:
            for pid in slate[col].dropna().unique():
                try:
                    slate_pitcher_ids.append(int(pid))
                except (TypeError, ValueError):
                    continue
    slate_pitcher_ids = tuple(set(slate_pitcher_ids))
    pitcher_splits = get_pitcher_handedness_splits(pitcher_ids=slate_pitcher_ids) if slate_pitcher_ids else pd.DataFrame()
    if (pitcher_splits is not None and not pitcher_splits.empty
            and "player_id" in pitcher_stats.columns):
        pitcher_stats["player_id"] = pd.to_numeric(
            pitcher_stats["player_id"], errors="coerce").astype("Int64")
        pitcher_splits["player_id"] = pd.to_numeric(
            pitcher_splits["player_id"], errors="coerce").astype("Int64")
        # Drop any conflicting columns then merge
        merge_cols = [c for c in pitcher_splits.columns if c != "player_id"]
        existing = set(pitcher_stats.columns)
        merge_cols = [c for c in merge_cols if c not in existing]
        if merge_cols:
            pitcher_stats = pitcher_stats.merge(
                pitcher_splits[["player_id"] + merge_cols],
                on="player_id", how="left"
            )
except Exception as _e:
    # Don't block app if MLB Stats API splits fetch fails
    pass

# NEW: Pitcher day/night splits + IL status
# Only fetch for slate pitchers (~26-30 calls each)
_pitcher_dn_il_err = None
try:
    from data_fetcher import get_pitcher_day_night_splits, get_pitchers_il_status
    if slate_pitcher_ids:
        pitcher_dn = get_pitcher_day_night_splits(pitcher_ids=slate_pitcher_ids)
        if pitcher_dn is not None and not pitcher_dn.empty and "player_id" in pitcher_stats.columns:
            pitcher_dn["player_id"] = pd.to_numeric(
                pitcher_dn["player_id"], errors="coerce").astype("Int64")
            merge_cols = [c for c in pitcher_dn.columns if c != "player_id"]
            existing = set(pitcher_stats.columns)
            merge_cols = [c for c in merge_cols if c not in existing]
            if merge_cols:
                pitcher_stats = pitcher_stats.merge(
                    pitcher_dn[["player_id"] + merge_cols],
                    on="player_id", how="left"
                )
        # IL status
        pitcher_il = get_pitchers_il_status(slate_pitcher_ids)
        if pitcher_il is not None and not pitcher_il.empty and "player_id" in pitcher_stats.columns:
            pitcher_il["player_id"] = pd.to_numeric(
                pitcher_il["player_id"], errors="coerce").astype("Int64")
            merge_cols = [c for c in pitcher_il.columns if c != "player_id"]
            existing = set(pitcher_stats.columns)
            merge_cols = [c for c in merge_cols if c not in existing]
            if merge_cols:
                pitcher_stats = pitcher_stats.merge(
                    pitcher_il[["player_id"] + merge_cols],
                    on="player_id", how="left"
                )
except NameError as _e:
    # slate_pitcher_ids not defined - handedness fetch failed earlier
    _pitcher_dn_il_err = f"NameError: {_e}"
except Exception as _e:
    _pitcher_dn_il_err = f"{type(_e).__name__}: {_e}"

# Surface the error in a small caption so we can debug
if _pitcher_dn_il_err:
    with st.sidebar:
        st.caption(f"⚠️ Day/Night+IL fetch issue: {_pitcher_dn_il_err}")

# Traditional stats
hitter_trad = get_hitter_traditional(_stats_day=_stats_day_key()) if not slate.empty else pd.DataFrame()
pitcher_trad = get_pitcher_traditional(_stats_day=_stats_day_key()) if not slate.empty else pd.DataFrame()

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
    # Coalesce era, whip, ip, hr9, k9, bb9 — PREFER trad (real MLB Stats API)
    # over Savant-derived estimates. This was a real bug: Savant unconditionally
    # set k9 = k_percent * 4.3 * 9 / 100, so the trad fillna() never fired.
    # Now trad wins where it exists, Savant-derived fills NaN.
    for col in ["era", "whip", "ip", "hr9", "k9", "bb9"]:
        trad_col = f"{col}_trad"
        if trad_col in pitcher_stats.columns:
            if col in pitcher_stats.columns:
                # Trad first, Savant-derived as fallback for NaN positions
                pitcher_stats[col] = pitcher_stats[trad_col].fillna(pitcher_stats[col])
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
    # Trigger LA fill if >30% of hitters are missing LA. Previously we only
    # triggered on 100% missing, which meant a single hitter with LA from a
    # different source would suppress the fill for everyone else.
    if "launch_angle" not in hitter_stats.columns:
        needs_la = True
        la_missing_pct_before = 100.0
        la_n_total_before = len(hitter_stats)
    else:
        la_n_total_before = len(hitter_stats)
        la_n_missing_before = hitter_stats["launch_angle"].isna().sum()
        la_missing_pct_before = (la_n_missing_before / la_n_total_before * 100.0
                                    if la_n_total_before > 0 else 0.0)
        needs_la = (la_n_total_before > 0) and (la_missing_pct_before > 30.0)

    # Persist coverage state to session_state so a diagnostic expander
    # near the bottom of the page can show it. Even when the fill is skipped
    # because coverage is already good, we want to be able to see that.
    st.session_state["_la_missing_pct_before"] = la_missing_pct_before
    st.session_state["_la_n_total"] = la_n_total_before
    st.session_state["_la_fill_ran"] = bool(needs_la)
    st.session_state["_la_fill_error"] = None

    if needs_la:
        try:
            from data_fetcher import fill_hitter_la_for_slate
            with st.spinner("Fetching real launch angles from Statcast (bulk first, ~5s)..."):
                hitter_stats = fill_hitter_la_for_slate(hitter_stats, slate)
            # Re-measure coverage AFTER fill so we can report the delta
            if "launch_angle" in hitter_stats.columns:
                n_missing_after = hitter_stats["launch_angle"].isna().sum()
                n_total_after = len(hitter_stats)
                pct_after = (n_missing_after / n_total_after * 100.0
                                if n_total_after > 0 else 0.0)
                st.session_state["_la_missing_pct_after"] = pct_after
            else:
                st.session_state["_la_missing_pct_after"] = 100.0
        except Exception as e:
            st.session_state["_la_fill_error"] = str(e)[:200]
            st.warning(f"LA fill skipped: {e}")
    else:
        # Already healthy — record current coverage as the "after" too
        st.session_state["_la_missing_pct_after"] = la_missing_pct_before

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

# BIG visible deploy version. If this doesn't say v6-NUCLEAR, deploy hasn't taken effect.
st.markdown(
    f"### 🔧 DEPLOY VERSION: `{APP_VERSION}`"
)
st.caption(
    "👆 **This version string must match the latest fix from chat.** "
    "If it says an older version (v4, v5, etc.), the Streamlit deploy didn't "
    "complete — wait 60 seconds and refresh, or check Streamlit Cloud status."
)

# Display current ET time + when each data source was last refreshed
try:
    _now_et = pd.Timestamp.now(tz="US/Eastern")
    _time_caption = (
        f"🕐 **Current time: {_now_et.strftime('%-I:%M %p ET')}** · "
        f"Slate refreshes every 5 min · Lineups every 3 min · Rosters every 15 min · "
        f"All game times shown in ET"
    )
    st.caption(_time_caption)
except Exception:
    pass

n_games = len(slate)
n_pitchers = pitcher_stats["k9"].notna().sum() if "k9" in pitcher_stats.columns else 0
n_hitters = hitter_stats["xwoba"].notna().sum() if "xwoba" in hitter_stats.columns else 0
hdr_col1, hdr_col2, hdr_col3, hdr_col4 = st.columns(4)
hdr_col1.metric("Games", n_games)
hdr_col2.metric("Pitchers w/ data", int(n_pitchers))
hdr_col3.metric("Hitters w/ data", int(n_hitters))
hdr_col4.metric("PA threshold", INSUFFICIENT_PA_THRESHOLD)

# =============================================================================
# RECENT TRANSACTIONS — trades, signings, DFAs, IL, call-ups in last 2 days
# =============================================================================
if show_transactions:
    try:
        from data_fetcher import get_recent_transactions
        txn_df = get_recent_transactions(days_back=2, _stats_day=_stats_day_key())
    except Exception:
        txn_df = pd.DataFrame()
    if not txn_df.empty:
        # Categorize transactions into types user cares about
        IMPACT_CODES = {
            # Roster moves that change WHERE a player is
            "TR":  ("🔁", "TRADE"),
            "SC":  ("✍️", "SIGNING"),
            "DFA": ("⚠️", "DFA"),
            "REL": ("❌", "RELEASE"),
            "OUT": ("⬇️", "OUTRIGHTED"),
            # Roster moves that change WHO is available
            "CU":  ("⬆️", "CALL-UP"),
            "SD":  ("⬇️", "SENT DOWN"),
            "SCL": ("✨", "SELECTED"),
            "STA": ("📝", "STATUS"),
            # IL moves
            "IL":  ("🏥", "TO IL"),
            "RTN": ("🟢", "FROM IL"),
        }
        # Filter to "high-impact" moves only
        impact_df = txn_df[txn_df["type_code"].isin(IMPACT_CODES.keys())].copy()
        if not impact_df.empty:
            impact_df["icon"] = impact_df["type_code"].map(lambda c: IMPACT_CODES.get(c, ("", ""))[0])
            impact_df["category"] = impact_df["type_code"].map(lambda c: IMPACT_CODES.get(c, ("", ""))[1])
            # Group by category for a clean summary count
            cat_counts = impact_df["category"].value_counts()
            count_pills = " · ".join(
                f"{IMPACT_CODES[code][0]} {IMPACT_CODES[code][1]}: **{cat_counts.get(IMPACT_CODES[code][1], 0)}**"
                for code in ["TR", "SC", "DFA", "REL", "CU", "SD", "SCL", "IL", "RTN"]
                if cat_counts.get(IMPACT_CODES[code][1], 0) > 0
            )
            with st.expander(
                f"🔄 Recent roster moves (last 2 days): {len(impact_df)} transactions — {count_pills}",
                expanded=False,
            ):
                st.caption(
                    "Mid-day roster moves that could affect tonight's slate. "
                    "If a player you're targeting was DFA'd or sent down, you'll see it here. "
                    "If a hitter was traded, the data may still show their previous team "
                    "until our caches roll over. Click '🔄 Force refresh all data' if needed."
                )
                show_cols = ["date", "icon", "category", "player_name", "from_team", "to_team", "description"]
                show_cols = [c for c in show_cols if c in impact_df.columns]
                # Trim description for readability
                if "description" in impact_df.columns:
                    impact_df["description"] = impact_df["description"].astype(str).str[:140]
                st.dataframe(
                    impact_df[show_cols].head(50),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "date": st.column_config.TextColumn("Date", width="small"),
                        "icon": st.column_config.TextColumn("", width="small"),
                        "category": st.column_config.TextColumn("Type", width="small"),
                        "player_name": st.column_config.TextColumn("Player"),
                        "from_team": st.column_config.TextColumn("From"),
                        "to_team": st.column_config.TextColumn("To"),
                        "description": st.column_config.TextColumn("Details"),
                    },
                )
            # Build a set of player IDs that recently moved teams — used below
            # to flag affected hitters/pitchers in the slate.
            try:
                _recently_moved_ids = set(
                    int(pid) for pid in impact_df["player_id"].dropna().tolist()
                )
            except Exception:
                _recently_moved_ids = set()
        else:
            _recently_moved_ids = set()
    else:
        _recently_moved_ids = set()
else:
    _recently_moved_ids = set()

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

                                # Brier score — the single best calibration metric.
                                # Lower = better. For HRs, ~0.04-0.06 is good,
                                # <0.03 is excellent, >0.08 = systematic miscalibration.
                                brier = h_metrics.get("brier_score")
                                if brier is not None:
                                    b_col, _ = st.columns([1, 3])
                                    if brier < 0.04:
                                        b_label = "Excellent"
                                    elif brier < 0.06:
                                        b_label = "Good"
                                    elif brier < 0.08:
                                        b_label = "OK"
                                    else:
                                        b_label = "Needs tuning"
                                    b_col.metric(
                                        "Brier score",
                                        f"{brier:.4f}",
                                        delta=b_label,
                                        delta_color="off",
                                        help=(
                                            "Mean squared error between predicted "
                                            "HR probability and actual outcome (0/1). "
                                            "Lower = better. Rare-event reference: "
                                            "<0.03 excellent · 0.04-0.06 good · "
                                            ">0.08 systematic over-prediction."
                                        ),
                                    )

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
    with st.expander("📖 Legend & glossary — what every column means", expanded=False):
        # ROW 1: Letter grades (the NEW addition the user asked for)
        st.markdown("### 🎯 Letter Grades — quick at-a-glance interpretation")
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown(
                "**HITTER Grade (HR Game% based)** — applies to the **batter facing this pitcher**.\n\n"
                "| Grade | HR Game% | Meaning |\n"
                "|---|---|---|\n"
                "| **A+** | ≥22% | Elite — top-tier HR play |\n"
                "| **A** | 19-22% | Very strong matchup |\n"
                "| **B+** | 16-19% | Strong, above-average |\n"
                "| **B** | 12-16% | Solid, league-good |\n"
                "| **C+** | 9-12% | Modest, slight edge |\n"
                "| **C** | 6-9% | Below average |\n"
                "| **D** | 3-6% | Poor matchup |\n"
                "| **F** | <3% | Avoid |\n"
                "| **—** | n/a | Insufficient sample |\n"
            )
        with gcol2:
            st.markdown(
                "**PITCHER Grade — from the BATTER's perspective**. "
                "(Tells you: is this pitcher an HR target or someone to avoid?)\n\n"
                "| Grade | What it means |\n"
                "|---|---|\n"
                "| **EXPLOIT+** | 🔥 Target! Worst test/suppress scores (<30). HRs likely. |\n"
                "| **EXPLOIT** | Solid HR target. Below-average pitcher (test/suppress ≤45). |\n"
                "| **MIXED** | Neutral. No clear edge either way. |\n"
                "| **TOUGH** | Avoid HR plays. Pitcher has the edge (test+suppress avg ≥65). |\n"
                "| **ELITE** | 🚫 Strong avoid. Top-tier K stuff AND top suppress (avg ≥80). |\n"
                "| **—** | Insufficient sample (<80 PA faced) |\n\n"
                "**Why no 'EXPLOIT' on hitters?** Hitter grade rates THIS hitter's HR chance "
                "today; pitcher grade rates THIS pitcher as a target. Different perspectives."
            )

        st.markdown("---")
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
                "📉 SAMPLE NOISE — stat inconsistency (low IP)\n\n"
                "🌱 ROOKIE — debut in current season"
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
            st.markdown("**Verdict labels (hitters)**")
            st.markdown(
                "🟢 **STRONG** — A+/A grade play\n\n"
                "🟢 **SOLID** — B+/B grade play\n\n"
                "🟡 **WATCH** — C+ borderline play\n\n"
                "🟠 **AVOID** — C/D grade\n\n"
                "🔴 **WEAK** — F grade or terrible matchup"
            )
            st.markdown("**HR Form Arrow**")
            st.markdown(
                "↑ Recent ISO is 10%+ above season — hot\n\n"
                "→ Within ±10% of season pace — steady\n\n"
                "↓ Recent ISO is 10%+ below — cold"
            )
        with leg3:
            st.markdown("**Streak labels**")
            st.markdown(
                "🔥 **3+ HR L5** — 3+ HR in last 5 games (very hot)\n\n"
                "🔥 **HR L3** — homered in last 3 games\n\n"
                "⚡ **HR L5** — homered in last 5 games\n\n"
                "🌡️ **HR L10** — homered in last 10\n\n"
                "❄️ **no HR L10** — 0 HR in last 10 games (cold)"
            )
            st.markdown("**HR Environment flag (per game)**")
            st.markdown(
                "🔥 **HR-FRIENDLY** (env ≥1.15× boost): favor hitters\n\n"
                "⬆️ Slightly HR-friendly (1.08-1.15×)\n\n"
                "⬇️ Slightly HR-hostile (0.85-0.92×)\n\n"
                "❄️ **HR-HOSTILE** (env ≤0.85×): favor pitchers\n\n"
                "🌧️ Rain ≥50%: HR potential suppressed"
            )

        st.markdown("---")
        st.markdown("### 📊 Column Definitions")
        col_def1, col_def2 = st.columns(2)
        with col_def1:
            st.markdown(
                "**Hitter projection columns:**\n\n"
                "**HR Game%** — Probability of ≥1 HR this game. Top metric.\n\n"
                "**HR PA%** — Per-PA HR probability. Use for prop pricing.\n\n"
                "**Power Score** — 0-99 composite: barrel%, ISO, hard-hit, EV, FB%, pull-air%.\n\n"
                "**Matchup Opp** — Opportunity score: weights env factors more heavily (catches contact hitters in great spots).\n\n"
                "**Matchup** — Overall matchup quality (xwoba, barrel, opp pitcher).\n\n"
                "**Test Score** — Matchup × PA reliability + recent form blend.\n\n"
                "**Pitch Match** — Hitter's xwOBA vs THIS pitcher's specific arsenal (weighted by usage).\n\n"
                "**Pitch HR** — Same as above but barrel-based (HR-specific).\n\n"
                "**Pick Score** — Daily ranking composite (see formula below).\n\n"
                "**Sleeper Score** — HR%-percentile MINUS season-HR-percentile. Positive = today is better than season pace.\n\n"
                "**GS Score** — Grand Slam composite (HR% × lineup traffic)."
            )
        with col_def2:
            st.markdown(
                "**Pitcher projection columns:**\n\n"
                "**Test Score** — K-focused 0-95 composite (blended K/9 30%, Whiff% 20%, xwOBA suppress 25%, ERA 15%, K% 10%) × reliability.\n\n"
                "**kHR** — K-rating composite (50% blended K/9, 20% K%, 30% Whiff%).\n\n"
                "**HR Suppress** — HR-suppression score (32% barrel allowed, 28% xwOBA, 20% HR/9, 20% batted-ball mix) × opp HR% × park.\n\n"
                "**Proj K** — Blended K/9 × expected IP × opponent K% multiplier.\n\n"
                "**Reliability** — 0.3-1.0. Captures sample size reliability. Relievers ~0.4-0.65; full starters 1.0.\n\n"
                "**vs LHB/RHB splits** — pitcher's actual data vs each handedness. Used to OVERRIDE overall HR/9 in projections (when sample ≥40 PA).\n\n"
                "**Opp K%** — Opposing team's season K rate. Adjusts proj_k.\n\n"
                "**Opp HR%** — Opposing team's HR/PA. Adjusts hr_suppress."
            )

        st.markdown("---")
        st.markdown("### 🎯 Top Picks Formula (how the daily Top 10 is ranked)")
        st.markdown(
            "**Pick Score** = 0-100 weighted blend of:\n\n"
            "- **HR Game%** (25%) — today's HR probability (most important)\n"
            "- **Matchup vs Opponent** (15%) — pitcher quality × park\n"
            "- **Power Score** (15%) — underlying season power skills\n"
            "- **Pitch HR Match** (10%) — barrel rate vs this pitcher's specific arsenal\n"
            "- **HR Form** (12%) — recent ISO trend\n"
            "- **Sleeper Lift** (8%) — today vs season pace\n"
            "- **Env Boost** (15%) — park × weather × pull-wind\n\n"
            "**Lineup bonuses:** +3 if confirmed lineup, -2 if roster-fill (lineup unknown)\n\n"
            "Diversity rule: max 2 hitters per game (3 if slate is small)."
        )

        st.markdown("---")
        st.markdown("### ☀️🌙 Day vs Night Splits + IL Status")
        st.markdown(
            "**Day/Night splits** — some hitters/pitchers perform very differently in "
            "day vs night games (visibility, sleep cycles, lighting). The model now "
            "uses these as a modest adjustment:\n\n"
            "**Hitters:** ±15% bound on h_base adjustment (caps so it can't overpower other factors). "
            "Sample shrinkage with 100 PA prior. Only triggers with ≥40 PA in the split.\n\n"
            "**Pitchers:** ±10% on pitcher_mult. Sample shrinkage with 60 PA prior. "
            "Only triggers with ≥50 PA in the split.\n\n"
            "**Why bounded:** day/night effects are real but smaller than handedness or park. "
            "If a hitter has .250 AVG in day games and .280 at night, that's a 12% difference — "
            "applying 50% of it (the model's choice) gives us a 6% nudge in the right direction.\n\n"
            "**Game header now shows:** ☀️ DAY GAME or 🌙 NIGHT GAME next to the HR env flag.\n\n"
            "---\n\n"
            "**🚑 ON IL / 🏥 FRESH IL flags (pitchers)**\n\n"
            "Pitchers fresh off the IL throw fewer pitches and may be rusty. The model:\n"
            "- Caps expected_ip at 3.5 for first 3 days back (was 5.5 for full starters)\n"
            "- Reduces expected_ip by 15% for days 4-7 since return\n"
            "- This affects proj_k (fewer innings = fewer Ks projected)\n"
            "- Shows a 🏥 FRESH IL ({days}d) warning in the pitcher table\n\n"
            "Data source: MLB Stats API transactions endpoint."
        )

        st.markdown("---")
        st.markdown("### 🔥 Smash Spots — Triple-Threat Alignment Flag")
        st.markdown(
            "**The 'all stars align' flag** — appears in the hitter table when a batter has "
            "multiple advantages stacking together. Look for these FIRST when building your slate:\n\n"
            "| Tier | Conditions Required |\n"
            "|---|---|\n"
            "| 🔥🔥🔥 **ELITE SMASH** | EXPLOIT+ pitcher + favorable env (≥1.05) + favorable park (≥1.04) + HR Game% ≥19% |\n"
            "| 🔥🔥 **STRONG SMASH** | EXPLOIT/EXPLOIT+ pitcher + favorable env + favorable park + HR Game% ≥15% |\n"
            "| 🔥 **SMASH** | EXPLOIT/EXPLOIT+ pitcher + (favorable env OR park) + HR Game% ≥15% |\n\n"
            "Where:\n"
            "- **Favorable env** = hand-aware park × weather × pull-wind ≥ 1.05\n"
            "- **Favorable park** = hand-aware park × pull-wind ≥ 1.04\n"
            "- **Pitcher grade** must NOT be TBD (we need real pitcher stats)\n\n"
            "Why this matters: when a hitter faces a bad pitcher (EXPLOIT) in a hitter-friendly "
            "ballpark with wind blowing out and good handedness platoon — that's where the model "
            "becomes most confident. Each factor alone is moderate; together they compound.\n\n"
            "**Critical filters:**\n"
            "- LINEUP MUST BE CONFIRMED — we need real batting order to project HR Game% accurately\n"
            "- PITCHER MUST NOT BE TBD — no grade = no smash flag\n"
            "- **Max 2 per team in the leaderboard** — avoids stacking the same lineup\n\n"
            "**Why some games have 0 Smash Spots:** if the pitcher is TOUGH/ELITE/MIXED, or "
            "the weather/park is HR-hostile, no batter on that team can qualify regardless of "
            "their personal stats. This is correct behavior — the model is honest about when "
            "all the dominoes don't line up."
        )

        st.markdown("---")
        st.markdown("### 🔬 Why hitters with low barrel% can still rank high")
        st.markdown(
            "Sometimes you'll see a hitter with mediocre barrel% rank near elites. Possible reasons:\n\n"
            "1. **Pitch-specific edge** — they happen to crush THIS pitcher's arsenal (high Pitch HR score)\n"
            "2. **Great park/weather** — HR-friendly venue + wind blowing out (env_mult >1.15)\n"
            "3. **Platoon advantage** — opposite-handed matchup with reverse-split pitcher\n"
            "4. **Lineup spot** — leadoff/#2 hitters get 4.6 PAs vs #9's 3.6 (more chances)\n"
            "5. **Sample noise** — recent power surge can inflate rolling stats\n\n"
            "When this happens, check `Pitch HR`, `Env`, and `Lineup_pos` columns. "
            "If all three favor them, the model is correctly elevating them."
        )

        # Concrete scale guide - what's "good" vs "bad" for each metric
        st.markdown("---")
        st.markdown("### 📏 What's Good vs Bad — Real MLB Scales")
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
    "Look for **🚨 RELIEVER** and **⚠️ LOW IP** flags — those scores are intentionally scaled down. "
    "**New to the columns?** See the **📖 Legend & glossary** section above the pitcher table "
    "for grade definitions (A+, EXPLOIT, MIXED, TOUGH, ELITE), column meanings, and scoring formulas."
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

# Enrich p_slate with MLB's primaryPosition designation (SP/RP/P).
# This gives us MLB's actual role label rather than inferring from games-started.
# Coleman Crow example: rookie with 2 starts looks like a "RELIEVER" by our
# games-started inference, but MLB lists him as SP. Trust MLB's label.
if not p_slate.empty and "pitcher_id" in p_slate.columns:
    try:
        from data_fetcher import get_pitcher_primary_positions
        _pid_tuple = tuple(int(x) for x in p_slate["pitcher_id"].dropna().tolist())
        _pos_map = get_pitcher_primary_positions(_pid_tuple)
        p_slate["primary_position"] = p_slate["pitcher_id"].apply(
            lambda pid: _pos_map.get(int(pid), "") if pd.notna(pid) else ""
        )
        # Now re-run role classification using the new position signal.
        # The first pass inside build_pitcher_slate ran without primary_position.
        try:
            from models import recompute_pitcher_roles
            p_slate = recompute_pitcher_roles(p_slate, slate_date=selected_date)
        except (ImportError, AttributeError):
            pass  # older models.py without recompute function
    except Exception:
        p_slate["primary_position"] = ""

# HARD OVERRIDE: probable starters cannot be on IL.
# MLB doesn't list IL'd pitchers as probable starters. If our transactions
# parser thinks Webb is on IL but he's the probable starter today, the slate
# is the authority — override unconditionally.
# NUCLEAR v6 FIX: previous version only cleared on_il. But days_since_return
# and il_count_this_season are SEPARATE fields the warn flag + role check
# also read. Even with on_il=False, if days_since_return=3, you'd see
# "🏥 FRESH IL (3d)" — and if il_count>0, role check thinks pitcher is
# "returning from IL". Clear EVERYTHING for these pitchers.
if not p_slate.empty:
    for il_col in ("on_il", "days_since_return", "il_count_this_season"):
        if il_col in p_slate.columns:
            if il_col == "on_il":
                p_slate[il_col] = False
            else:
                p_slate[il_col] = pd.NA
    # Also re-run role classification with the cleaned data
    try:
        from models import recompute_pitcher_roles
        p_slate = recompute_pitcher_roles(p_slate, slate_date=selected_date)
    except (ImportError, AttributeError):
        pass

    # FINAL ROLE NORMALIZATION: catch any deprecated labels that leak through.
    # "🚨 RELIEVER" and "🔄 BULK" are dead labels — replace with "🚨 OPENER".
    # Also catches case mismatches and partial matches defensively.
    if "role" in p_slate.columns:
        def _normalize_role(r):
            s = str(r) if r is not None else ""
            # Case-insensitive substring matching to catch any variant
            s_lower = s.lower()
            if "reliever" in s_lower:
                # Preserve any emoji/prefix but replace the word
                s = s.replace("🚨 RELIEVER", "🚨 OPENER")
                s = s.replace("RELIEVER", "OPENER")
                s = s.replace("Reliever", "OPENER")
                s = s.replace("reliever", "OPENER")
            if "bulk" in s_lower:
                s = s.replace("🔄 BULK", "🚨 OPENER")
                s = s.replace("BULK", "OPENER")
            return s
        p_slate["role"] = p_slate["role"].apply(_normalize_role)

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
    # Add letter-style matchup grade (EXPLOITABLE/MIXED/TOUGH/ELITE)
    p_slate["grade"] = p_slate.apply(
        lambda r: pitcher_grade(
            r.get("test_score"), r.get("hr_suppress"), r.get("pa"),
            INSUFFICIENT_PA_THRESHOLD,
        ),
        axis=1,
    )
    # Add hidden numeric rank for sortable Grade column.
    # When users click the Grade header, they want EXPLOIT+ first (best HR targets),
    # not alphabetical order. We sort the dataframe by this BEFORE display.
    p_slate["_grade_rank"] = p_slate["grade"].apply(pitcher_grade_sort_key)

    # Add a visible warning flag column combining role + sample_noise + IL
    if not p_slate.empty:
        def _warn_flag(r):
            flags = []
            if r.get("sample_noise"):
                flags.append("📉")
            # NOTE: Removed 🚑 ON IL flag entirely. By definition, every pitcher
            # in p_slate is a probable starter for today's game — they CANNOT
            # be on IL. Logan Webb showed ON IL repeatedly because of transaction-
            # parsing edge cases (paternity, restricted list, etc.) that no
            # amount of pattern matching can fully prevent. The slate itself is
            # the authority: if MLB lists you as the probable starter, you're
            # not on IL. Period.
            # 🏥 FRESH IL (just back) is still meaningful for workload context.
            days_since = r.get("days_since_return")
            if (days_since is not None and not pd.isna(days_since)
                    and 0 <= days_since <= 7):
                flags.append(f"🏥 FRESH IL ({int(days_since)}d)")
            return " ".join(flags) if flags else ""
        p_slate["warn"] = p_slate.apply(_warn_flag, axis=1)

        # =====================================================================
        # PLATOON HR VULNERABILITY FLAG
        # =====================================================================
        # User example: "Fedde has allowed 10 of last 12 HRs to RHB" — even
        # if his overall HR rate looks fine, the side-split matters a lot.
        # Use season vs_lhb_hr_per_pa and vs_rhb_hr_per_pa (which we already
        # pull) to compute a platoon ratio. If a pitcher's HR rate to one
        # handedness is significantly higher than the other, flag the
        # vulnerable side so the user can target that side's hitters.
        def _platoon_hr_flag(r):
            l_hr_pa = r.get("vs_lhb_hr_per_pa")
            r_hr_pa = r.get("vs_rhb_hr_per_pa")
            # NA-safe PA extraction — `or 0` raises NAType.__bool__ on pd.NA
            _l_raw = r.get("vs_lhb_pa", 0)
            _r_raw = r.get("vs_rhb_pa", 0)
            try:
                l_pa = 0 if pd.isna(_l_raw) else float(_l_raw or 0)
            except (TypeError, ValueError):
                l_pa = 0
            try:
                r_pa = 0 if pd.isna(_r_raw) else float(_r_raw or 0)
            except (TypeError, ValueError):
                r_pa = 0
            # Need meaningful samples to make a call (min 40 PA each side)
            if (l_hr_pa is None or pd.isna(l_hr_pa)
                or r_hr_pa is None or pd.isna(r_hr_pa)
                or l_pa < 40 or r_pa < 40):
                return ""
            l_hr_pa = float(l_hr_pa)
            r_hr_pa = float(r_hr_pa)
            # Avoid div-by-zero — if a side is 0% HR allowed, use minimum 0.001
            l_safe = max(l_hr_pa, 0.001)
            r_safe = max(r_hr_pa, 0.001)
            ratio_r_to_l = r_safe / l_safe  # >1 = more HR to RHB
            # Require: large ratio AND vulnerable side has high absolute HR rate
            if ratio_r_to_l >= 2.0 and r_hr_pa >= 0.035:
                return "💥 RHB"  # Very vulnerable to RHB
            if ratio_r_to_l >= 1.5 and r_hr_pa >= 0.030:
                return "💢 RHB"  # Vulnerable to RHB
            if ratio_r_to_l <= 0.5 and l_hr_pa >= 0.035:
                return "💥 LHB"
            if ratio_r_to_l <= 0.67 and l_hr_pa >= 0.030:
                return "💢 LHB"
            return ""
        p_slate["platoon_hr_flag"] = p_slate.apply(_platoon_hr_flag, axis=1)

        # =====================================================================
        # RECENT HR ALLOWED FLAG
        # =====================================================================
        # User: "Severino 3 HRs over last 3 starts" — surface this directly.
        # We already have recent_hr9 (HR/9 in last 5 starts). Translate that
        # into an absolute count using recent_ip and recent_starts so user
        # sees "🔥 5 HR L5" rather than just "HR/9 = 1.94".
        def _recent_hr_flag(r):
            recent_ip = r.get("recent_ip")
            recent_hr9 = r.get("recent_hr9")
            recent_starts = r.get("recent_starts")
            if (recent_ip is None or pd.isna(recent_ip) or recent_ip <= 0
                or recent_hr9 is None or pd.isna(recent_hr9)
                or recent_starts is None or pd.isna(recent_starts)):
                return ""
            # Approximate total recent HRs allowed
            hr_count = round(float(recent_hr9) * float(recent_ip) / 9.0)
            n = int(recent_starts)
            if hr_count >= 6:
                return f"🔥 {hr_count}HR L{n}"
            if hr_count >= 4:
                return f"⚠️ {hr_count}HR L{n}"
            if hr_count >= 3 and n <= 3:
                return f"⚠️ {hr_count}HR L{n}"  # 3 HRs in 3 starts is meaningful
            return ""
        p_slate["recent_hr_flag"] = p_slate.apply(_recent_hr_flag, axis=1)

    show_cols = [c for c in [
        "alert", "grade", "role", "warn", "platoon_hr_flag", "recent_hr_flag",
        "pitcher_name", "team", "home_away", "opp", "throws",
        "test_score", "kHR", "hr_suppress", "proj_k", "form_arrow",
        "era", "whip", "k9", "bb9", "hr9",
        "ip", "games_started", "games_played", "ip_per_outing",
        "k_pct", "whiff_pct",
        "xwoba_allowed", "barrel_allowed",
        # NEW: handedness splits (MLB Stats API fields)
        "vs_lhb_pa", "vs_lhb_avg", "vs_lhb_obp", "vs_lhb_slg",
        "vs_lhb_hr_per_pa", "vs_lhb_k_percent",
        "vs_rhb_pa", "vs_rhb_avg", "vs_rhb_obp", "vs_rhb_slg",
        "vs_rhb_hr_per_pa", "vs_rhb_k_percent",
        # NEW: day/night splits
        "vs_day_pa", "vs_day_avg", "vs_day_slg", "vs_day_hr_per_pa",
        "vs_night_pa", "vs_night_avg", "vs_night_slg", "vs_night_hr_per_pa",
        "opp_k_pct", "opp_hr_per_pa",
        "recent_era", "recent_k9", "days_rest", "avg_recent_pitches",
        # NEW: IL info
        "days_since_return", "il_count_this_season",
        "reliability",
    ] if c in p_slate.columns]

    # Auto-hide empty columns
    show_cols = [c for c in show_cols if p_slate[c].notna().any()]

    col_config = {
        "alert": st.column_config.TextColumn("Signal", width="small"),
        "grade": st.column_config.TextColumn(
            "Matchup", width="small",
            help=(
                "Matchup grade from the BATTER's perspective:\n"
                "EXPLOIT+ = target these pitchers (worst test/suppress scores)\n"
                "EXPLOIT  = solid HR target, somewhat exploitable\n"
                "MIXED    = neutral, no edge either way\n"
                "TOUGH    = avoid HR plays, pitcher has the edge\n"
                "ELITE    = strong avoid (top suppress + top K stuff)\n"
                "—        = insufficient sample"
            ),
        ),
        "role": st.column_config.TextColumn(
            "Role", width="small",
            help=(
                "✓ Established starter (full IP expected) · "
                "🌱 NEW STARTER (rookie/recent recall, short leash) · "
                "🏥 RETURNING (just back from IL) · "
                "⚠️ LOW IP (below-expected workload) · "
                "🔄 SWING (used in starts + relief alternately) · "
                "🚨 OPENER (MLB lists as RP, ~1-2 IP expected)"
            ),
        ),
        "warn": st.column_config.TextColumn(
            "Flag", width="small",
            help="📉 = sample noise (ERA-WHIP mismatch or zero Statcast values at low IP)",
        ),
        "platoon_hr_flag": st.column_config.TextColumn(
            "Vuln", width="small",
            help=(
                "Platoon HR vulnerability based on season-long splits:\n"
                "💥 RHB / 💥 LHB = SEVERE — pitcher's HR/PA to that side is 2x+ "
                "the other AND ≥3.5%. Target that side's hitters hard.\n"
                "💢 RHB / 💢 LHB = NOTABLE — 1.5x+ ratio and ≥3.0%. Target that side.\n"
                "Blank = balanced or sample too small (need 40+ PA each side)."
            ),
        ),
        "recent_hr_flag": st.column_config.TextColumn(
            "L5 HR",
            help=(
                "Recent HRs allowed in last 5 starts:\n"
                "🔥 = 6+ HRs (extremely vulnerable lately)\n"
                "⚠️ 4-5 HRs (vulnerable) or 3 HRs in 3 starts (concerning trend)\n"
                "Blank = normal recent HR rate."
            ),
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
        # Handedness splits (from MLB Stats API statSplits endpoint)
        "vs_lhb_pa": st.column_config.NumberColumn(
            "vs L PA", format="%d",
            help="Plate appearances vs LHB this season. Larger = more reliable split.",
        ),
        "vs_lhb_avg": st.column_config.NumberColumn(
            "vs L AVG", format="%.3f",
            help="Batting avg allowed to LHB. <.220 = elite, .280+ = LHB hit this pitcher.",
        ),
        "vs_lhb_obp": st.column_config.NumberColumn("vs L OBP", format="%.3f"),
        "vs_lhb_slg": st.column_config.NumberColumn(
            "vs L SLG", format="%.3f",
            help="Slugging allowed to LHB. .500+ = LHB crush this pitcher.",
        ),
        "vs_lhb_hr_per_pa": st.column_config.NumberColumn(
            "vs L HR%", format="%.2f%%",
            help=(
                "HR rate per PA allowed to LHB. League avg ~2.8%.\n"
                "<2% = strong, >4% = LHB hammer this pitcher.\n"
                "Model uses THIS rate (not overall HR/9) when LHB at bat."
            ),
        ),
        "vs_lhb_k_percent": st.column_config.NumberColumn(
            "vs L K%", format="%.1f%%",
            help="K rate vs LHB. >25% strong; <18% pitcher struggles vs LHB.",
        ),
        "vs_rhb_pa": st.column_config.NumberColumn("vs R PA", format="%d"),
        "vs_rhb_avg": st.column_config.NumberColumn("vs R AVG", format="%.3f"),
        "vs_rhb_obp": st.column_config.NumberColumn("vs R OBP", format="%.3f"),
        "vs_rhb_slg": st.column_config.NumberColumn("vs R SLG", format="%.3f"),
        "vs_rhb_hr_per_pa": st.column_config.NumberColumn(
            "vs R HR%", format="%.2f%%",
            help="HR rate per PA allowed to RHB. League avg ~2.8%.",
        ),
        "vs_rhb_k_percent": st.column_config.NumberColumn(
            "vs R K%", format="%.1f%%",
            help="K rate vs RHB.",
        ),
        # Day/night pitcher splits
        "vs_day_pa": st.column_config.NumberColumn(
            "Day PA", format="%d",
            help="Plate appearances in day games. Larger = more reliable split.",
        ),
        "vs_day_avg": st.column_config.NumberColumn("Day AVG", format="%.3f"),
        "vs_day_slg": st.column_config.NumberColumn("Day SLG", format="%.3f"),
        "vs_day_hr_per_pa": st.column_config.NumberColumn(
            "Day HR%", format="%.2f%%",
            help=(
                "HR rate per PA in day games. League avg ~2.8%.\n"
                "Some pitchers are dramatically worse at day games (visibility, sleep cycles)."
            ),
        ),
        "vs_night_pa": st.column_config.NumberColumn("Night PA", format="%d"),
        "vs_night_avg": st.column_config.NumberColumn("Night AVG", format="%.3f"),
        "vs_night_slg": st.column_config.NumberColumn("Night SLG", format="%.3f"),
        "vs_night_hr_per_pa": st.column_config.NumberColumn(
            "Night HR%", format="%.2f%%",
            help="HR rate per PA in night games. League avg ~2.8%.",
        ),
        "days_since_return": st.column_config.NumberColumn(
            "Days Off IL", format="%d",
            help=(
                "Days since this pitcher was reinstated from the IL.\n"
                "≤3 days = first start back, expect fewer innings.\n"
                "4-7 days = still ramping back up.\n"
                "Blank = healthy / no recent IL stint."
            ),
        ),
        "il_count_this_season": st.column_config.NumberColumn(
            "IL Stints", format="%d",
            help="Number of times this pitcher has been placed on IL this season.",
        ),
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

    # Sort by grade rank (EXPLOIT+ first → ELITE last → — at the bottom),
    # then by test_score descending within same grade. Gives users a
    # meaningful default order when they don't manually sort.
    p_slate_display = p_slate.copy()
    if "_grade_rank" in p_slate_display.columns:
        sort_cols = ["_grade_rank"]
        sort_asc = [True]
        if "test_score" in p_slate_display.columns:
            sort_cols.append("test_score")
            sort_asc.append(True)  # within EXPLOIT+, lower test = MORE exploitable
        p_slate_display = p_slate_display.sort_values(sort_cols, ascending=sort_asc)

    st.dataframe(
        _style_pitcher_df(p_slate_display[show_cols]),
        hide_index=True, use_container_width=True,
        column_config=col_config,
    )

    # DIAGNOSTIC: show LITERAL column values so user can verify what's actually
    # in the data when they're confused about what they see. Especially useful
    # for emoji-based flags that might be misread (🏥 ≠ "on IL", 🚨 ≠ "reliever").
    with st.expander("🔍 Diagnostic: show literal flag/role values for any pitcher"):
        st.caption(
            "If you think a pitcher is flagged incorrectly, find them here. "
            "Shows the EXACT text in each flag column (with emojis spelled out). "
            "💡 Note: 🏥 RETURNING means 'starter just back from IL' — NOT 'on IL'. "
            "🚨 OPENER means 'MLB lists as RP, ~1-2 IP expected' — NOT 'reliever'."
        )
        if not p_slate.empty and "pitcher_name" in p_slate.columns:
            search = st.text_input(
                "Filter pitchers (partial name match):",
                key="diag_pitcher_search",
                placeholder="webb, giolito, crow...",
            )
            diag_df = p_slate.copy()
            if search:
                diag_df = diag_df[
                    diag_df["pitcher_name"].str.contains(search, case=False, na=False)
                ]
            diag_cols = [c for c in [
                "pitcher_name", "team", "role", "warn", "primary_position",
                "on_il", "days_since_return", "il_count_this_season",
                "ip", "games_started", "games_played", "is_rookie",
            ] if c in diag_df.columns]
            if not diag_df.empty:
                st.dataframe(
                    diag_df[diag_cols].head(30),
                    hide_index=True, use_container_width=True,
                )

    # DIAGNOSTIC: Launch angle coverage. If the LA column shows 0% across the
    # board in the export, this expander tells you whether the fill RAN, what
    # the before/after coverage was, and what error (if any) happened.
    with st.expander("🔍 Diagnostic: launch angle coverage"):
        st.caption(
            "Launch angle feeds into the Power Score (sweet-spot bonus at ~28°). "
            "If LA shows all NaN in the export, check the state below."
        )
        la_total = st.session_state.get("_la_n_total", 0)
        la_before = st.session_state.get("_la_missing_pct_before")
        la_after = st.session_state.get("_la_missing_pct_after")
        la_ran = st.session_state.get("_la_fill_ran", False)
        la_err = st.session_state.get("_la_fill_error")
        c1, c2, c3 = st.columns(3)
        c1.metric("Hitters in stats", la_total)
        if la_before is not None:
            c2.metric(
                "Missing LA before fill",
                f"{la_before:.1f}%",
                help="If <30%, fill is skipped (already healthy).",
            )
        if la_after is not None:
            delta = None
            if la_before is not None:
                delta = f"{la_after - la_before:+.1f}pp"
            c3.metric(
                "Missing LA after fill",
                f"{la_after:.1f}%",
                delta=delta,
                delta_color="inverse",
                help="After fill ran (if it ran). Target: <10% missing.",
            )
        st.write(f"**LA fill ran this session:** {'✓ yes' if la_ran else '— no (already ≤30% missing)'}")
        if la_err:
            st.error(f"LA fill error: {la_err}")
        if not la_ran and (la_before is not None and la_before > 30):
            st.warning(
                "LA fill was needed but didn't run. This usually means an "
                "exception fired before the fill block (check above)."
            )
        if la_after is not None and la_after > 50:
            st.warning(
                "More than half of hitters still missing LA after fill. "
                "Statcast bulk endpoint may be down or rate-limited. Power "
                "Score will be reduced for affected players."
            )

st.divider()
# ============================================================================
# GAME CONTEXT (park, weather, ump, vegas) - build per-game
# ============================================================================

game_context_map = {}
matchup_tables = {}

for _, game in slate.iterrows():
    gpk = int(game["gamePk"])

    # NEW: Classify game as day/night for split adjustments
    # CRITICAL: Use the VENUE's local timezone, not ET. A 3:45 PM Arizona
    # game is a day game even though it's 6:45 PM ET.
    try:
        from data_fetcher import classify_game_day_night, get_venue_timezone
        game_time_iso = game.get("gameTime") or ""
        venue_tz = get_venue_timezone(game.get("venue") or "")
        # Handle pd.Timestamp gametime
        if hasattr(game_time_iso, "isoformat"):
            game_time_iso = game_time_iso.isoformat()
            if not game_time_iso.endswith("Z") and "+" not in game_time_iso:
                game_time_iso = game_time_iso + "+00:00"
        game_type = classify_game_day_night(game_time_iso, venue_tz) if game_time_iso else "night"
    except Exception:
        game_type = "night"

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
    wx_mult_nowind = 1.0  # wind-stripped wx_mult for per-hitter HR calc
    wx_summary = ""
    # ---- REAL ROOF STATUS from MLB game feed (overrides our guessing) ----
    # Only matters for venues with a roof — we pull this regardless of venue
    # to keep code simple; it just returns roof_closed=None for outdoor parks.
    roof_status = {"roof_closed": None, "condition": "", "temp": None, "wind": ""}
    try:
        from data_fetcher import get_game_roof_status
        roof_status = get_game_roof_status(int(game["gamePk"])) or roof_status
    except Exception:
        pass
    real_roof_closed = roof_status.get("roof_closed")  # True / False / None

    if use_weather and venue:
        try:
            park_info = get_park(venue)
            lat = park_info.get("lat")
            lon = park_info.get("lon")
            cf_bearing = park_info.get("cf_bearing", 0)
            if park_info.get("unknown"):
                # Venue not in our park database — likely a renamed park or
                # spring-training site. Surface this so user knows what's going
                # on instead of silently producing neutral data.
                raise ValueError(f"venue '{venue}' not in park database — neutral park/weather used")
            if lat is None or lon is None:
                raise ValueError(f"venue '{venue}' missing coordinates")
            game_dt = game.get("gameTime")
            if isinstance(game_dt, pd.Timestamp):
                wx_iso = game_dt.isoformat()
            else:
                wx_iso = None
            weather = fetch_weather(lat, lon, wx_iso) or {}
            if weather.get("error"):
                wx_summary = f"Weather API error: {weather.get('error', '')[:50]}"
            else:
                # If MLB CONFIRMS the roof is closed, ignore outside weather
                # entirely — treat as indoor neutral.
                if real_roof_closed is True:
                    wx_mult = 1.0
                    wx_mult_nowind = 1.0
                    cond_label = roof_status.get("condition") or "Roof Closed"
                    wx_summary = f"🏟️ {cond_label} (MLB-confirmed) — indoor neutral"
                else:
                    wx_mult, _summary = hr_multiplier(weather, park_info)
                    # ALSO compute a wind-stripped multiplier for per-hitter
                    # HR calc, since wind is applied per-handedness via
                    # wind_pull_side_multiplier (avoids double-counting).
                    wx_mult_nowind, _ = hr_multiplier(weather, park_info, skip_wind=True)
                    wx_summary = _summary or "Neutral"
                    # Hard rain warning - >80% chance suggests likely delay/postponement
                    pp = weather.get("precip_prob")
                    if pp is not None and pp >= 80:
                        wx_summary = f"⚠️ HEAVY RAIN ({pp:.0f}%) — possible delay/PPD — " + wx_summary
                    # If MLB CONFIRMS roof is OPEN, note that
                    if real_roof_closed is False and park_info.get("roof") == "retractable":
                        wx_summary += " · 🏟️ Roof confirmed OPEN"
        except Exception as e:
            weather = {"error": str(e)}
            wx_mult = 1.0
            wx_mult_nowind = 1.0
            wx_summary = f"Weather failed: {str(e)[:60]}"
    elif not use_weather:
        wx_summary = "Weather disabled"
    elif not venue:
        wx_summary = "No venue info"

    full_hr_mult = park_mult * wx_mult

    # ===== HR Environment Flag =====
    # Categorize the COMBINED park × weather × wind impact for hitters
    # This gives users an immediate "is this a HR-friendly spot?" signal.
    hr_env_flag = ""
    hr_env_color = ""  # for st.success/info/warning/error
    if full_hr_mult >= 1.15:
        hr_env_flag = f"🔥 **HR-FRIENDLY ENVIRONMENT** (boost: {full_hr_mult:.2f}×) — favor hitters/sluggers, avoid pitchers"
        hr_env_color = "success"
    elif full_hr_mult >= 1.08:
        hr_env_flag = f"⬆️ **Slightly HR-friendly** ({full_hr_mult:.2f}×) — modest hitter boost"
        hr_env_color = "info"
    elif full_hr_mult <= 0.85:
        hr_env_flag = f"❄️ **HR-HOSTILE ENVIRONMENT** ({full_hr_mult:.2f}×) — favor pitchers, avoid hitter HR props"
        hr_env_color = "error"
    elif full_hr_mult <= 0.92:
        hr_env_flag = f"⬇️ **Slightly HR-hostile** ({full_hr_mult:.2f}×) — pitchers preferred"
        hr_env_color = "warning"
    # else: neutral, no flag

    # Umpire factor — uses UMPIRE_K_FACTORS lookup table in game_context.py.
    # Most umpires return neutral 1.0; only ~10% have meaningfully non-neutral
    # K-rate tendencies (±5% or more from neutral).
    ump = {}
    framing = {}
    if use_ump and HAVE_GAME_CONTEXT:
        try:
            ump = get_umpire_for_game(gpk) or {}
        except Exception:
            ump = {}

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

        Returns (lineup, is_confirmed, bench) where:
          lineup     = the 9 hitters shown as the lineup (real if posted, else padded)
          is_confirmed = True only if MLB has actually posted the lineup
          bench      = remaining active-roster position players (NOT in the 9)
                       Used to surface late-swap candidates like Grichuk who
                       just signed but might pinch-hit or fill in.

        IMPORTANT: When using roster fill, we sort by season PA (descending)
        so the most-used position players appear first. Alphabetical order
        (the API default) was producing nonsense lineups where e.g. Pete
        Alonso would show up "batting 9th" just because A comes after L.
        """
        is_confirmed = len(existing_lineup) >= 8  # At least 8 of 9 = real lineup
        if not team_id:
            return existing_lineup, is_confirmed, []
        existing_ids = {p.get("id") for p in existing_lineup if p.get("id")}
        try:
            roster = get_team_roster(int(team_id))
        except Exception:
            return existing_lineup, is_confirmed, []
        # Filter to position players only (skip P, SP, RP).
        # IMPORTANT: TWP (two-way players like Ohtani) ARE included — they
        # bat in the lineup even on days they pitch. Excluding them was a bug
        # that hid Ohtani when the Dodgers' lineup wasn't yet posted.
        # We also INCLUDE them when they pitch — because Ohtani still hits.
        position_players = [
            p for p in roster
            if p.get("position") and str(p.get("position")).upper() not in
                ("P", "SP", "RP")
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
        # Pad up to 9 total when lineup isn't fully posted
        needed = max(0, 9 - len(existing_lineup))
        for p in position_players[:needed]:
            existing_lineup.append({
                "id": p.get("id"), "name": p.get("name"),
                "position": p.get("position"), "bats": p.get("bats"),
                "is_roster_fill": True,  # mark these as not real lineup positions
            })
        # Bench: every other position player NOT in the 9 we display.
        # These are the late-swap / pinch-hit candidates.
        bench_pool = position_players[needed:] if needed > 0 else position_players
        bench = [{
            "id": p.get("id"), "name": p.get("name"),
            "position": p.get("position"), "bats": p.get("bats"),
        } for p in bench_pool]
        return existing_lineup, is_confirmed, bench

    try:
        away_lineup = get_lineup(int(game["gamePk"]), "away")
    except Exception:
        away_lineup = []
    away_lineup, away_confirmed, away_bench = _fill_to_nine(away_lineup, game.get("away_team_id"))

    try:
        home_lineup = get_lineup(int(game["gamePk"]), "home")
    except Exception:
        home_lineup = []
    home_lineup, home_confirmed, home_bench = _fill_to_nine(home_lineup, game.get("home_team_id"))

    # Backfill batting handedness for any hitters missing it
    # IMPORTANT: include bench players too — they show "bats: None" otherwise
    # because /roster/active doesn't always populate batSide.
    needs_bats_ids = set()
    for p in away_lineup + home_lineup + away_bench + home_bench:
        if p.get("id") and not p.get("bats"):
            try:
                needs_bats_ids.add(int(p["id"]))
            except (ValueError, TypeError):
                continue
    if needs_bats_ids:
        try:
            bats_map = fill_hitter_bats([], ids=needs_bats_ids)
            for p in away_lineup + home_lineup + away_bench + home_bench:
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

    # Hitter recent form + day/night splits
    # Day/night splits are fetched ONLY for confirmed-lineup hitters to limit
    # API calls (typically 18 hitters per game = ~270 calls if every team is
    # confirmed). When lineups aren't posted yet, we skip the day/night fetch.
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

    # Day/night splits — only if at least one side has confirmed lineup
    try:
        from data_fetcher import get_hitter_day_night_splits
        if away_confirmed or home_confirmed:
            hitter_ids_for_dn = []
            for p in (away_lineup + home_lineup):
                pid = safe_int(p.get("id"))
                if pid is not None:
                    hitter_ids_for_dn.append(pid)
            hitter_ids_for_dn = tuple(set(hitter_ids_for_dn))
            if hitter_ids_for_dn:
                hitter_dn_df = get_hitter_day_night_splits(hitter_ids=hitter_ids_for_dn)
                if hitter_dn_df is not None and not hitter_dn_df.empty:
                    for _, dn_row in hitter_dn_df.iterrows():
                        pid = safe_int(dn_row.get("player_id"))
                        if pid is None:
                            continue
                        # Add day/night fields to the recent_hitter_map entry so
                        # they flow through build_matchup_table -> row_dict -> props
                        if pid not in recent_hitter_map:
                            recent_hitter_map[pid] = {}
                        for k, v in dn_row.items():
                            if k != "player_id" and pd.notna(v):
                                recent_hitter_map[pid][k] = v
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

    # Bench matchups — same calculations but for roster-active players NOT
    # in tonight's 9-man lineup. Used to surface late-swap candidates
    # (pinch hitters, players who could replace someone in the lineup last minute).
    away_bench_matchup = pd.DataFrame()
    home_bench_matchup = pd.DataFrame()
    try:
        if away_bench:
            away_bench_matchup = build_matchup_table(
                away_bench,
                pd.Series(home_p_row) if home_p_row else None,
                hitter_stats, pitcher_stats,
                recent_form_dict=recent_hitter_map,
                pitcher_arsenal_df=pitcher_arsenal_all,
            )
        if home_bench:
            home_bench_matchup = build_matchup_table(
                home_bench,
                pd.Series(away_p_row) if away_p_row else None,
                hitter_stats, pitcher_stats,
                recent_form_dict=recent_hitter_map,
                pitcher_arsenal_df=pitcher_arsenal_all,
            )
    except Exception:
        pass

    # Power Score - composite HR-likelihood incorporating park/weather/pitcher.
    # NOTE: Power Score intentionally uses the GAME-LEVEL wx_mult (with wind),
    # while per-hitter HR Game% below uses wx_mult_nowind + per-hand pull-wind.
    # Reason: Power Score is a hitter-quality ranking that gets one game-wide
    # environment adjustment. The per-hitter HR Game% needs handedness-precise
    # wind (LHB pull side ≠ RHB pull side ≠ CF direction), which the game-level
    # wind can't represent. Different scopes, intentionally different wind models.
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

        # Look up opposing pitcher's grade for SMASH SPOT detection
        # Smash Spot = batter facing EXPLOIT/EXPLOIT+ pitcher AND favorable env+park.
        # This is the "all stars align" flag the user requested.
        opp_pitcher_grade = None
        opp_pitcher_id = opp_p_row.get("player_id") if opp_p_row else None
        if opp_pitcher_id is not None and not p_slate.empty and "grade" in p_slate.columns:
            try:
                # Coerce both sides to int for reliable matching (pitcher_id can
                # end up as Int64/float/object across different pandas operations)
                opp_pid_int = int(opp_pitcher_id)
                _ids = pd.to_numeric(p_slate["pitcher_id"], errors="coerce")
                _match = p_slate[_ids == opp_pid_int]
                if not _match.empty:
                    opp_pitcher_grade = _match.iloc[0].get("grade")
            except Exception:
                pass

        hr_pa, hr_game, verdicts, signals, grades = [], [], [], [], []
        smash_spots = []  # NEW: triple-threat HR opportunity flags
        pull_mults_col = []
        for _, hr in matchup_df.iterrows():
            row_dict = hr.to_dict()
            # Inject game_type for day/night split adjustment in props.py.
            # game_type is the local variable set at the top of this game loop.
            row_dict["game_type"] = game_type
            # opp_p_row gets game_type too (defensive)
            if isinstance(opp_p_row, dict) and "game_type" not in opp_p_row:
                opp_p_row["game_type"] = game_type
            pa = safe_float(row_dict.get("pa"))
            sample = int(pa) if pa is not None else None
            bats = row_dict.get("bats", "R") or "R"

            # Hand-aware park factor (replaces the generic park_mult for this hitter)
            try:
                hand_park = get_park_hand_factor(venue_name, bats)
            except Exception:
                hand_park = park_mult

            # Pull-side wind multiplier (passes temp for retractable-roof awareness)
            try:
                _temp_f_for_wind = (weather or {}).get("temp_f")
                pull_mult, pull_summary = wind_pull_side_multiplier(
                    venue_name, bats, wind_mph, wind_dir,
                    temp_f=_temp_f_for_wind,
                    real_roof_closed=real_roof_closed,  # ground truth if available
                )
                if pull_summary:
                    pull_summaries[pull_summary] = pull_summaries.get(pull_summary, 0) + 1
            except Exception:
                pull_mult = 1.0

            pull_mults_col.append(round(pull_mult, 3))

            # Combined park factor for this specific hitter: hand-aware × pull-wind
            hitter_park_mult = hand_park * pull_mult

            # Pitch-type HR match multiplier
            # NOTE: pitch_match_score is ALREADY applied inside hr_prob_per_pa
            # via pm_mult. This is an ADDITIONAL fine adjustment using the
            # barrel-based pitch_hr_score (vs the xwoba-based pitch_match_score).
            # These two signals are correlated, so we use a NARROWER range here
            # to avoid double-counting the "hitter mashes this pitcher" effect.
            # Was 0.85-1.15 (too generous). Now 0.92-1.08 = max ±8% adjustment.
            pitch_hr_score = row_dict.get("pitch_hr_score")
            if pitch_hr_score is not None and not pd.isna(pitch_hr_score):
                # Center on 50 -> 1.0, +/- 1 score = 0.0016 shift
                pitch_hr_mult = 1.0 + (pitch_hr_score - 50) * 0.0016
                pitch_hr_mult = max(0.92, min(1.08, pitch_hr_mult))
            else:
                pitch_hr_mult = 1.0

            try:
                p_pa = hr_prob_per_pa(
                    row_dict, opp_p_row,
                    park_factor=hitter_park_mult, weather_mult=wx_mult_nowind,
                    pitch_match_score=row_dict.get("pitch_match_score"),
                )
                # Apply pitch_hr_score as an additional fine adjustment
                # BUT re-apply the soft squash so we don't blow past the cap.
                if p_pa is not None:
                    raw = float(p_pa) * pitch_hr_mult
                    # Soft squash: same logic as in props.py.
                    # Theoretical asymptote 7% per PA, practical ceiling ~6.3%
                    # (tanh never reaches 1.0).
                    if raw <= 0.04:
                        p_pa = raw
                    else:
                        excess = raw - 0.04
                        p_pa = 0.04 + 0.032 * np.tanh(excess / 0.045)
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
            #
            # NEW (May 2026): Home/Away PA adjustment.
            # Away teams average ~4.31 PA per game; home teams ~4.21 PA per game.
            # The reason: home team doesn't bat in the bottom of the 9th when
            # HOME/AWAY PA OFFSET — refined from 0.03 to 0.055 (May 2026).
            # Real MLB data 2022-2024: away teams average ~4.31 PA/game vs
            # home ~4.20 (away advantage because home team doesn't bat in the
            # bottom of the 9th when winning, ~50% of games). The ±0.055 split
            # gives 4.255 vs 4.145 — within 0.005 PA of empirical average and
            # adds about 0.15-0.30 percentage points to away hitters' HR Game%.
            # Still small enough that it doesn't dominate top-10 selection, but
            # directionally correct and research-backed.
            is_away_team = (matchup_df is away_matchup)
            ha_offset = 0.055 if is_away_team else -0.055

            lp = row_dict.get("lineup_pos")
            is_fill = row_dict.get("is_roster_fill", False)
            game_confirmed = (
                away_confirmed if matchup_df is away_matchup else home_confirmed
            )
            if (lp is not None and not pd.isna(lp)
                    and not is_fill and game_confirmed):
                try:
                    lp_int = int(lp)
                    # Base PA by lineup position; add home/away offset
                    expected_pa = max(3.6, 4.7 - (lp_int - 1) * 0.1) + ha_offset
                except (ValueError, TypeError):
                    expected_pa = 4.2 + ha_offset
            else:
                # Default to league avg + home/away offset
                expected_pa = 4.2 + ha_offset

            p_game = hr_prob_full_game(p_pa, expected_pa=expected_pa) if p_pa is not None else None
            hr_pa.append(round(p_pa * 100, 2) if p_pa is not None else None)
            hr_game.append(round(p_game * 100, 2) if p_game is not None else None)
            game_pct_val = round(p_game * 100, 2) if p_game is not None else None
            verdicts.append(hr_verdict(
                game_pct_val, sample, INSUFFICIENT_PA_THRESHOLD,
            ))
            signals.append(hr_signal_emoji(
                game_pct_val, sample, INSUFFICIENT_PA_THRESHOLD,
            ))
            grades.append(hr_grade(
                game_pct_val, sample, INSUFFICIENT_PA_THRESHOLD,
            ))

            # SMASH SPOT calculation - the "all stars align" flag.
            # User's request: highlight when a batter has:
            #   1. EXPLOIT or EXPLOIT+ pitcher matchup (NOT TBD)
            #   2. Favorable park dimensions for their handedness
            #   3. Favorable combined environment (park × weather × wind)
            #   4. PLUS reasonable HR Game% so we don't flag bad hitters in great spots
            #
            # Tiers (stricter than before):
            #   🔥🔥🔥 ELITE SMASH = EXPLOIT+ pitcher + favorable park + favorable env + HR%≥19%
            #   🔥🔥 STRONG SMASH = EXPLOIT/EXPLOIT+ + favorable park + favorable env + HR%≥15%
            #   🔥 SMASH = EXPLOIT/EXPLOIT+ + (favorable park OR env) + HR%≥15%
            #
            # NEW REQUIREMENTS (fixing bugs from May 26 export):
            # - LINEUP MUST BE CONFIRMED (otherwise we can't trust HR Game% projection)
            # - PITCHER MUST NOT BE TBD (no grade = no confidence in matchup)
            # - park_favorable now uses hand_park × pull_wind combined (not just either)
            smash_label = ""
            try:
                # Combined env factor (park × weather × pull-wind for this hitter)
                # NOTE: uses wx_mult_nowind to avoid double-counting wind
                # (pull_mult already encodes the pull-side wind effect).
                full_env = hand_park * wx_mult_nowind * pull_mult
                # Combined park-side factor (hand-aware × pull-wind)
                full_park = hand_park * pull_mult

                pitcher_name = (row_dict.get("opp_pitcher") or "").upper()
                pitcher_is_tbd = pitcher_name in ("TBD", "TBA", "")

                pitcher_exploitable = (
                    opp_pitcher_grade in ("EXPLOIT", "EXPLOIT+")
                    and not pitcher_is_tbd
                )
                pitcher_super_exploitable = (
                    opp_pitcher_grade == "EXPLOIT+"
                    and not pitcher_is_tbd
                )
                game_pct_ok = game_pct_val is not None and game_pct_val >= 15
                # Slightly looser thresholds: env ≥1.03 (was 1.05), park ≥1.02 (was 1.04).
                # User was seeing 0 smash spots at 1.05/1.04 even with 100+ confirmed
                # lineups. The stricter cutoff was excluding legitimate plays where
                # park is roof-controlled (≈1.03) and weather is modest.
                env_favorable = full_env >= 1.03
                park_favorable = full_park >= 1.02

                # CRITICAL: Smash spots only apply to CONFIRMED LINEUPS.
                # is_roster_fill is the authoritative flag set in models.py when
                # the player came from roster-padding (not actual lineup).
                is_fill_player = bool(row_dict.get("is_roster_fill", False))
                lineup_truly_confirmed = (
                    game_confirmed and not is_fill_player
                    and lp is not None and not pd.isna(lp)
                )

                if (lineup_truly_confirmed and game_pct_ok and pitcher_super_exploitable
                        and env_favorable and park_favorable
                        and game_pct_val >= 19):
                    smash_label = "🔥🔥🔥 ELITE SMASH"
                elif (lineup_truly_confirmed and game_pct_ok and pitcher_exploitable
                        and env_favorable and park_favorable):
                    smash_label = "🔥🔥 STRONG SMASH"
                elif (lineup_truly_confirmed and game_pct_ok and pitcher_exploitable
                        and (env_favorable or park_favorable)):
                    smash_label = "🔥 SMASH"
            except Exception:
                smash_label = ""
            smash_spots.append(smash_label)
        matchup_df["hr_pa_pct"] = hr_pa
        matchup_df["hr_game_pct"] = hr_game
        matchup_df["verdict"] = verdicts
        matchup_df["alert"] = signals
        matchup_df["grade"] = grades
        matchup_df["smash_spot"] = smash_spots
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
        # Look up role-aware expected_ip from p_slate (models.py computed it)
        # Was a real bug: k_total_projection defaults to expected_ip=5.5 for
        # every pitcher, so openers and rookie starters had inflated K projections.
        def _exp_ip_for_pitcher(p_row):
            if not p_row:
                return 5.5
            pid = p_row.get("player_id")
            if pid is not None and not p_slate.empty and "pitcher_id" in p_slate.columns:
                try:
                    _ids = pd.to_numeric(p_slate["pitcher_id"], errors="coerce")
                    _match = p_slate[_ids == int(pid)]
                    if not _match.empty and "expected_ip" in _match.columns:
                        v = _match.iloc[0].get("expected_ip")
                        if v is not None and not pd.isna(v):
                            return float(v)
                except Exception:
                    pass
            return 5.5  # fallback to legacy default
        # Pull umpire K-factor (neutral 1.0 for ~90% of games)
        ump_k = float(ump.get("k_factor", 1.0)) if ump else 1.0
        if away_p_row:
            away_exp_ip = _exp_ip_for_pitcher(away_p_row)
            away_k_proj = k_total_projection(
                away_p_row, home_lineup_k_pct,
                ump_k_factor=ump_k,
                park_k_factor=pkf,
                expected_ip=away_exp_ip,
            )
        if home_p_row:
            home_exp_ip = _exp_ip_for_pitcher(home_p_row)
            home_k_proj = k_total_projection(
                home_p_row, away_lineup_k_pct,
                ump_k_factor=ump_k,
                park_k_factor=pkf,
                expected_ip=home_exp_ip,
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
        "pull_wind_summary": list(pull_summaries.keys()),
        "away_lineup_confirmed": away_confirmed,
        "home_lineup_confirmed": home_confirmed,
        "hr_env_flag": hr_env_flag,
        "hr_env_color": hr_env_color,
        "game_type": game_type,  # "day" or "night" for splits
        "real_roof_closed": real_roof_closed,  # True/False/None (from MLB feed)
        "roof_condition": roof_status.get("condition", ""),
        "away_bench_matchup": away_bench_matchup,
        "home_bench_matchup": home_bench_matchup,
    }
    matchup_tables[game["gamePk"]] = (away_matchup, home_matchup)


# After all games processed, add env-adjusted grade to p_slate.
# Each pitcher gets a grade that reflects today's PARK × WEATHER context,
# in addition to the base "grade" which is pitcher-skill-only.
# Convention: a pitcher's "opposing environment" is the environment HE faces,
# i.e. the game's own park × weather. (Hr_mult is the same for both pitchers
# in a game.)
if not p_slate.empty and "grade" in p_slate.columns:
    # Build pitcher_id → game env_mult map from game_context_map
    env_by_pid = {}
    for gpk, ctx in game_context_map.items():
        env = ctx.get("hr_mult", 1.0)
        a_row = ctx.get("away_p_row") or {}
        h_row = ctx.get("home_p_row") or {}
        a_pid = a_row.get("player_id") if a_row else None
        h_pid = h_row.get("player_id") if h_row else None
        if a_pid is not None and not pd.isna(a_pid):
            env_by_pid[int(a_pid)] = float(env)
        if h_pid is not None and not pd.isna(h_pid):
            env_by_pid[int(h_pid)] = float(env)

    def _env_adj(row):
        try:
            pid = row.get("pitcher_id")
            if pid is None or pd.isna(pid):
                return row.get("grade", "—")
            env = env_by_pid.get(int(pid))
            if env is None:
                return row.get("grade", "—")
            return pitcher_grade_env_adj(row.get("grade"), env)
        except Exception:
            return row.get("grade", "—")

    p_slate["env_adj_grade"] = p_slate.apply(_env_adj, axis=1)
    # Store env_mult on each pitcher row too for the UI/diagnostic
    p_slate["game_env_mult"] = p_slate["pitcher_id"].apply(
        lambda pid: env_by_pid.get(int(pid)) if pid is not None and not pd.isna(pid) else None
    )

st.divider()


# ============================================================================
# LINEUP CONFIRMATION BANNER - tells user upfront if many games are unconfirmed
# ============================================================================
unconfirmed_games = []
unconfirmed_with_time = []  # tuples of (label, hours_until_first_pitch)
# Try US/Eastern, fall back to UTC if zoneinfo can't find it (Python 3.14 + missing tzdata)
try:
    now_et = pd.Timestamp.now(tz="US/Eastern")
except Exception:
    try:
        now_et = pd.Timestamp.now(tz="America/New_York")
    except Exception:
        # Last resort: UTC. Time-to-game calc will be off by ET offset but app won't crash.
        now_et = pd.Timestamp.now(tz="UTC")
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
        label = f"{'/'.join(sides)} ({g.get('away_team_abbr')}@{g.get('home_team_abbr')})"
        unconfirmed_games.append(label)
        # Compute time-until-first-pitch
        game_time = g.get("gameTime")
        time_label = ""
        hours_until = None
        if isinstance(game_time, pd.Timestamp):
            try:
                if game_time.tzinfo:
                    local_dt = game_time.tz_convert("US/Eastern")
                else:
                    local_dt = game_time.tz_localize("UTC").tz_convert("US/Eastern")
                time_label = local_dt.strftime("%I:%M %p ET").lstrip("0")
                hours_until = (local_dt - now_et).total_seconds() / 3600
            except Exception:
                pass
        unconfirmed_with_time.append((label, time_label, hours_until))

if unconfirmed_games:
    n_total = len(game_context_map)
    n_unconf = len(unconfirmed_games)
    n_conf = n_total - n_unconf
    # Categorize: imminent (< 2hr), upcoming (2-6hr), later (6hr+)
    imminent = [u for u in unconfirmed_with_time if u[2] is not None and u[2] < 2]
    soon = [u for u in unconfirmed_with_time if u[2] is not None and 2 <= u[2] < 6]
    later = [u for u in unconfirmed_with_time if u[2] is not None and u[2] >= 6]
    no_time = [u for u in unconfirmed_with_time if u[2] is None]

    confirmed_note = f" ✅ {n_conf} game{'s' if n_conf != 1 else ''} fully confirmed." if n_conf else ""
    msg_parts = [
        f"⚠️ **{n_unconf}/{n_total} games have unconfirmed lineups.**{confirmed_note} "
        f"For unconfirmed teams: hitters sorted by season PA (likely starters first), "
        f"the # column shows '—', and lineup-position PA scaling is DISABLED "
        f"(everyone uses 4.2 PA = league avg)."
    ]
    if imminent:
        labels = ", ".join([f"{u[0]} @ {u[1]}" for u in imminent[:5]])
        msg_parts.append(f"\n\n🚨 **Imminent (<2hr to first pitch):** {labels} — refresh NOW if available")
    if soon:
        labels = ", ".join([f"{u[0]} @ {u[1]}" for u in soon[:5]])
        msg_parts.append(f"\n\n⏰ **Upcoming (2-6hr away):** {labels} — refresh ~1hr before each game")
    if later:
        labels = ", ".join([f"{u[0]} @ {u[1]}" for u in later[:5]])
        msg_parts.append(f"\n\n🕐 **Later tonight:** {labels} — refresh after 4 PM ET")
    if no_time:
        labels = ", ".join([u[0] for u in no_time[:5]])
        msg_parts.append(f"\n\n❓ **Unknown time:** {labels}")
    st.warning("".join(msg_parts))
else:
    n_total = len(game_context_map)
    if n_total > 0:
        st.success(f"✅ **All {n_total} games have confirmed lineups** — projections use real batting positions.")


# ============================================================================
# SLATE LEADERS — who's #1 in each meaningful category across the whole slate
# ============================================================================
# Build a single dataframe combining all hitters across all games, then find
# the leader in each category. Mark them with 🏆 in the main displays via a
# slate_leader_flags column on combined_all (built below this section).
slate_leader_cats = []  # list of (category_label, player_id, player_name, value, fmt)
slate_leader_pid_map = {}  # pid → list of category labels (for icon display)

def _track_leader(label, pid, name, value, fmt="{:.2f}"):
    """Record a slate leader for display + emoji marking."""
    if pid is None or pd.isna(pid) or name is None:
        return
    slate_leader_cats.append((label, int(pid), name, value, fmt))
    slate_leader_pid_map.setdefault(int(pid), []).append(label)

# Hitter leaders (require sufficient sample for stats-based categories)
if all_hitters:
    _combined_lead = pd.concat(all_hitters, ignore_index=True)
    # Dedupe by player_id — same hitter may appear in multiple rosters
    if "player_id" in _combined_lead.columns:
        _combined_lead["_pid"] = pd.to_numeric(_combined_lead["player_id"], errors="coerce")
        _combined_lead = _combined_lead.drop_duplicates(subset="_pid", keep="first")

    def _leader_by(df, col, label, min_pa=100, ascending=False, fmt="{:.2f}"):
        if col not in df.columns:
            return
        sub = df[df[col].notna()].copy()
        if "pa" in sub.columns:
            sub = sub[sub["pa"].fillna(0) >= min_pa]
        if sub.empty:
            return
        sub = sub.sort_values(col, ascending=ascending)
        top = sub.iloc[0]
        _track_leader(label, top.get("player_id"), top.get("player_name"),
                       top[col], fmt)

    # Quality leaders (stats)
    _leader_by(_combined_lead, "barrel_pct",  "💥 Slate-best Barrel%",       fmt="{:.1f}%")
    _leader_by(_combined_lead, "iso",         "⚡ Slate-best ISO",            fmt="{:.3f}")
    _leader_by(_combined_lead, "xwoba",       "🎯 Slate-best xwOBA",          fmt="{:.3f}")
    _leader_by(_combined_lead, "hard_hit",    "💪 Slate-best Hard-Hit%",      fmt="{:.1f}%")
    _leader_by(_combined_lead, "avg_ev",      "🚀 Slate-best Exit Velo",      fmt="{:.1f} mph")
    _leader_by(_combined_lead, "home_run",    "🏟️ Slate-best Season HRs",     min_pa=50, fmt="{:.0f}")
    _leader_by(_combined_lead, "recent_hr",   "🔥 Slate-best Recent HRs (L15)", min_pa=50, fmt="{:.0f}")

    # Today's matchup leaders (need projection columns to be populated)
    _leader_by(_combined_lead, "hr_game_pct", "🎲 Slate-best HR Game%",       min_pa=50, fmt="{:.1f}%")
    _leader_by(_combined_lead, "power_score", "💎 Slate-best Power Score",    min_pa=50, fmt="{:.1f}")
    _leader_by(_combined_lead, "sleeper_score", "💤 Slate-best Sleeper",      min_pa=100, fmt="{:.1f}")

    # Environment-leader: best HR-friendly env on the slate
    if "env_boost" in _combined_lead.columns and _combined_lead["env_boost"].notna().any():
        env_sub = _combined_lead.dropna(subset=["env_boost"])
        max_env = env_sub["env_boost"].max()
        top_env = env_sub[env_sub["env_boost"] >= max_env - 0.001]
        # Among hitters in the best env, pick the one with the highest barrel%
        # so the leader is also a quality bat, not someone in Coors with no skill
        if "barrel_pct" in top_env.columns and top_env["barrel_pct"].notna().any():
            top_env = top_env.sort_values("barrel_pct", ascending=False)
        top = top_env.iloc[0]
        _track_leader("🌞 Slate-best Environment + bat",
                        top.get("player_id"), top.get("player_name"),
                        top["env_boost"], "{:.2f}×")

# Pitcher leaders (from p_slate)
if not p_slate.empty:
    def _p_leader(df, col, label, ascending=False, fmt="{:.1f}", min_ip=20):
        if col not in df.columns:
            return
        sub = df[df[col].notna()].copy()
        if "ip" in sub.columns:
            sub = sub[sub["ip"].fillna(0) >= min_ip]
        if sub.empty:
            return
        sub = sub.sort_values(col, ascending=ascending)
        top = sub.iloc[0]
        _track_leader(label, top.get("pitcher_id"), top.get("pitcher_name"),
                       top[col], fmt)

    _p_leader(p_slate, "test_score",   "🛡️ Slate-best Test Score (pitcher)", fmt="{:.1f}")
    _p_leader(p_slate, "hr_suppress",  "🚫 Slate-best HR Suppress (pitcher)", fmt="{:.1f}")
    _p_leader(p_slate, "k9",           "⚡ Slate-best K/9 (pitcher)",         fmt="{:.2f}")
    _p_leader(p_slate, "whiff_pct",    "💨 Slate-best Whiff% (pitcher)",      fmt="{:.1f}%")
    _p_leader(p_slate, "proj_k",       "📊 Slate-best Projected K (pitcher)", fmt="{:.1f}")

    # Worst pitchers — these are smash targets for HITTERS
    _p_leader(p_slate, "test_score",   "🎯 Slate-easiest Test Score (smash)",  ascending=True, fmt="{:.1f}")
    _p_leader(p_slate, "hr9",          "🔥 Slate-worst HR/9 (smash)",         fmt="{:.2f}", min_ip=20)

if slate_leader_cats:
    st.subheader("🏆 Slate Leaders — who tops the slate in each category")
    st.caption(
        "These players have the slate-best value in each named category. "
        "Players appearing here will be highlighted with 🏆 in the top picks "
        "and matchup tables. A player can lead multiple categories."
    )
    lc1, lc2 = st.columns(2)
    half = (len(slate_leader_cats) + 1) // 2
    for col, items in [(lc1, slate_leader_cats[:half]), (lc2, slate_leader_cats[half:])]:
        with col:
            for label, pid, name, val, fmt in items:
                try:
                    val_str = fmt.format(val)
                except Exception:
                    val_str = str(val)
                st.markdown(f"- {label}: **{name}** ({val_str})")

st.divider()


# ============================================================================
# TOP 5 PICKS OF THE DAY — combined HR signal across all factors
# ============================================================================
st.subheader("🏆 Top 10 Picks of the Day")
st.caption(
    "Best HR plays combining: HR Game%, matchup, recent form, power, "
    "park/weather, pitch-specific match, and lineup confirmation. "
    "**Note:** A hitter facing an ace (Misiorowski-tier) can still appear "
    "if they have an elite per-pitch match against that pitcher's arsenal. "
    "Check the Matchup column — if it's <50, the model is acknowledging the "
    "tough matchup but other factors are compensating. Max 2 per game; "
    "max 3 if slate is small."
)

# Gather all qualified hitters with game context
# Pre-initialize variables to None so any unexpected reference order can't
# NameError. They get their real values inside the if-block below.
all_hitters_for_picks = []
combined_picks = None
combined_all = None
top10 = None
top_picks_export = None
two_leg_df = None
three_leg_df = None
two_leg_parlay_export = None
three_leg_parlay_export = None
sleeper_parlay_export = None
rr_export = None
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
    # Slate-leader flag — re-uses the map built in the Slate Leaders section.
    if slate_leader_pid_map and "player_id" in combined_picks.columns:
        def _leader_flag_pick(pid):
            try:
                cats = slate_leader_pid_map.get(int(pid), [])
            except (TypeError, ValueError):
                return ""
            if not cats:
                return ""
            return f"🏆×{len(cats)}" if len(cats) >= 3 else "🏆"
        combined_picks["slate_leader_flag"] = combined_picks["player_id"].apply(_leader_flag_pick)
    # Enrich with opp pitcher grade for downstream sleeper-parlay filtering.
    # (combined_all gets the same enrichment later; we duplicate it here so
    # the parlay code that uses `q` has access too.)
    if (not p_slate.empty and "pitcher_name" in p_slate.columns
            and "grade" in p_slate.columns
            and "opp_pitcher" in combined_picks.columns):
        try:
            _grade_map = p_slate.set_index("pitcher_name")["grade"].to_dict()
            _hr9_map = p_slate.set_index("pitcher_name")["hr9"].to_dict() if "hr9" in p_slate.columns else {}
            _barrel_a_map = p_slate.set_index("pitcher_name")["barrel_allowed"].to_dict() if "barrel_allowed" in p_slate.columns else {}
            combined_picks["opp_pitcher_grade"] = combined_picks["opp_pitcher"].map(_grade_map)
            combined_picks["opp_pitcher_hr9"] = combined_picks["opp_pitcher"].map(_hr9_map)
            combined_picks["opp_pitcher_barrel_allowed"] = combined_picks["opp_pitcher"].map(_barrel_a_map)
            if "platoon_hr_flag" in p_slate.columns:
                _platoon_map = p_slate.set_index("pitcher_name")["platoon_hr_flag"].to_dict()
                combined_picks["opp_platoon_hr"] = combined_picks["opp_pitcher"].map(_platoon_map)
            if "recent_hr_flag" in p_slate.columns:
                _recent_hr_map = p_slate.set_index("pitcher_name")["recent_hr_flag"].to_dict()
                combined_picks["opp_recent_hr"] = combined_picks["opp_pitcher"].map(_recent_hr_map)
        except Exception:
            pass
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

        # PICK SCORE FORMULA — comprehensive blend of everything we've built
        # The goal: identify hitters with the best COMBINATION of:
        #   • Today's matchup quality (40% — present matters most)
        #   • Real underlying power (25% — skills don't fluctuate game-to-game)
        #   • Recent form + sleeper lift (20% — hot streaks + upside today)
        #   • Today's environment (15% — park × weather × pull-wind)
        #
        # Bonuses:
        #   • Confirmed lineup +3 points (more certainty about PA)
        #   • Roster-fill -2 points (PA is league-avg estimate, more uncertain)
        #
        # Then filter: must have sample size, pitch matchup data when available.
        # Diversity rule remains: max 2 per game.
        score_parts = []
        weights = []

        # === Today's matchup (40%) ===
        # HR Game% is the highest-signal predictor we have
        if "hr_game_pct" in q.columns:
            score_parts.append(_pct(q["hr_game_pct"]))
            weights.append(0.25)
        # matchup_opp captures pitcher quality + park factor against this hitter
        if "matchup_opp" in q.columns:
            score_parts.append(_pct(q["matchup_opp"]))
            weights.append(0.15)

        # === Underlying power (25%) ===
        # power_score is our composite — barrel + ISO + EV + hard-hit + FB%
        if "power_score" in q.columns:
            score_parts.append(_pct(q["power_score"]))
            weights.append(0.15)
        # pitch_hr_score: hitter's barrel% vs THIS pitcher's specific pitches
        if "pitch_hr_score" in q.columns and q["pitch_hr_score"].notna().any():
            score_parts.append(_pct(q["pitch_hr_score"]))
            weights.append(0.10)

        # === Recent form + sleeper lift (20%) ===
        # hr_form: hot/cold streak indicator
        if "hr_form" in q.columns:
            score_parts.append(_pct(q["hr_form"]))
            weights.append(0.12)
        # sleeper_score: today's HR percentile MINUS season pace
        # High positive sleeper = today's matchup is much better than season avg.
        # Use percentile rank (same as other components) so we never lose
        # differentiation at the high end. The previous fixed-scale formula
        # `((score + 70) / 1.5).clip(0, 100)` mapped all scores above ~77 to
        # ≈98-100, blurring rankings among the strongest sleepers (Julien 56,
        # Acuña 44 OK, but Schwarber-tier 80+ all collapsed).
        if "sleeper_score" in q.columns and q["sleeper_score"].notna().any():
            score_parts.append(_pct(q["sleeper_score"]))
            weights.append(0.08)

        # === Today's environment (15%) ===
        # env_boost includes park × weather × pull-side wind
        if "env_boost" in q.columns:
            score_parts.append(_pct(q["env_boost"]))
            weights.append(0.15)

        if score_parts:
            total_w = sum(weights)
            weighted = sum(p * (w / total_w) for p, w in zip(score_parts, weights))
            q["pick_score"] = weighted

            # Lineup confirmation bonus/penalty
            if "is_roster_fill" in q.columns:
                q["pick_score"] = q["pick_score"] - q["is_roster_fill"].fillna(False).astype(float) * 2.0
                # Confirmed (non-fill) gets +3
                q.loc[~q["is_roster_fill"].fillna(False).astype(bool), "pick_score"] += 3.0

            # PLATOON HR BONUS — if hitter's handedness matches the side the
            # opposing pitcher gives up the most HRs to, bump their score.
            # 💥 = +4 (severe), 💢 = +2 (notable). Switch hitters (S) get the
            # bonus regardless since they can pick the side.
            if "opp_platoon_hr" in q.columns and "bats" in q.columns:
                def _platoon_bonus(row):
                    # NA-safe: pd.NA in `or` expression crashes. Extract first,
                    # then check NA, then coerce.
                    _flag_raw = row.get("opp_platoon_hr")
                    _bats_raw = row.get("bats")
                    flag = "" if (_flag_raw is None or pd.isna(_flag_raw)) else str(_flag_raw)
                    bats = "" if (_bats_raw is None or pd.isna(_bats_raw)) else str(_bats_raw).upper()
                    if not flag or not bats:
                        return 0
                    is_severe = "💥" in flag
                    is_notable = "💢" in flag
                    target_side = "RHB" if "RHB" in flag else ("LHB" if "LHB" in flag else "")
                    if not target_side:
                        return 0
                    hitter_matches = (
                        bats == "S"
                        or (bats == "R" and target_side == "RHB")
                        or (bats == "L" and target_side == "LHB")
                    )
                    if hitter_matches:
                        return 4.0 if is_severe else (2.0 if is_notable else 0)
                    return 0
                q["pick_score"] = q["pick_score"] + q.apply(_platoon_bonus, axis=1)

            # RECENT HR ALLOWED BONUS — pitcher giving up HRs lately = ride the wave
            if "opp_recent_hr" in q.columns:
                def _recent_hr_bonus(row):
                    _raw = row.get("opp_recent_hr")
                    flag = "" if (_raw is None or pd.isna(_raw)) else str(_raw)
                    if "🔥" in flag:
                        return 3.0
                    if "⚠️" in flag:
                        return 1.5
                    return 0
                q["pick_score"] = q["pick_score"] + q.apply(_recent_hr_bonus, axis=1)

            q["pick_score"] = q["pick_score"].round(1)
        else:
            q["pick_score"] = q.get("hr_game_pct", 0)

        # Diversity rule: max 2 picks per game so the top 5 doesn't pile up
        # on one matchup. Greedy selection: sort by pick_score, take in order,
        # skipping any that would exceed 2-per-game.
        q_sorted = q.sort_values("pick_score", ascending=False)
        picks = []
        game_count = {}
        # First pass: max 2 per game (diversity rule)
        for _, row in q_sorted.iterrows():
            g = row.get("game", "")
            if game_count.get(g, 0) >= 2:
                continue
            picks.append(row)
            game_count[g] = game_count.get(g, 0) + 1
            if len(picks) >= 10:
                break
        # Second pass: if we have <10 picks (small slate), allow up to 3 per game
        if len(picks) < 10:
            picks_set = {(r.get("player_name"), r.get("team")) for r in picks}
            for _, row in q_sorted.iterrows():
                key = (row.get("player_name"), row.get("team"))
                if key in picks_set:
                    continue
                g = row.get("game", "")
                if game_count.get(g, 0) >= 3:
                    continue
                picks.append(row)
                picks_set.add(key)
                game_count[g] = game_count.get(g, 0) + 1
                if len(picks) >= 10:
                    break
        if picks:
            top10 = pd.DataFrame(picks).reset_index(drop=True)
        else:
            top10 = q_sorted.head(10).reset_index(drop=True)
        top10["rank"] = range(1, len(top10) + 1)

        cols_to_show = [c for c in [
            "rank", "slate_leader_flag", "player_name", "team", "game", "opp_pitcher",
            "pick_score", "hr_game_pct", "matchup", "barrel_pct",
            "hr_form", "env_boost",
        ] if c in top10.columns]
        disp = top10[cols_to_show].copy()

        st.dataframe(
            disp, hide_index=True, use_container_width=True,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "slate_leader_flag": st.column_config.TextColumn(
                    "🏆", width="small",
                    help="🏆 = slate leader in at least one category. ×N = leader in N categories.",
                ),
                "player_name": st.column_config.TextColumn("Hitter"),
                "team": st.column_config.TextColumn("Tm", width="small"),
                "game": st.column_config.TextColumn("Game"),
                "opp_pitcher": st.column_config.TextColumn("vs Pitcher"),
                "pick_score": st.column_config.NumberColumn(
                    "Pick Score",
                    format="%.1f",
                    help=(
                        "0-100 composite. Blends: HR Game% (25%), matchup quality (15%), "
                        "power score (15%), pitch-HR match (10%), recent form (12%), "
                        "sleeper lift (8%), env boost (15%). Plus +3 for confirmed "
                        "lineup, -2 for roster-fill."
                    ),
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
        # Store for export
        top_picks_export = top10.copy()
        glance = " · ".join(
            f"**#{r['rank']} {r['player_name']}** ({r['team']}, {r.get('hr_game_pct', 0):.1f}%)"
            for _, r in top10.head(5).iterrows()
        )
        st.markdown(f"**Top 5 at a glance:** {glance}")

        # ====================================================================
        # PARLAY SUGGESTIONS — combine top picks into 2-leg and 3-leg HR parlays
        # ====================================================================
        st.markdown("---")
        st.markdown("### 🎯 Parlay Suggestions — HR-to-hit combinations")
        st.caption(
            "Suggested HR parlays combining today's top plays. **Parlay HR%** = "
            "joint probability that ALL legs homer (computed by multiplying "
            "individual game HR%). Higher = more likely to cash but lower payout. "
            "We diversify across games when possible to avoid correlated outcomes "
            "(same lineup → same opposing pitcher = correlated)."
        )

        # Build parlay pool — but FIRST collapse to ONE hitter per team.
        # User request: parlays should never include two hitters from the
        # same team OR same game. Take the best (highest HR%) from each team.
        parlay_pool_full = top10[top10["hr_game_pct"].notna()].head(15).copy()
        if "team" in parlay_pool_full.columns:
            # For each team, keep only the row with highest hr_game_pct
            parlay_pool = (
                parlay_pool_full.sort_values("hr_game_pct", ascending=False)
                .drop_duplicates(subset=["team"], keep="first")
                .reset_index(drop=True)
            )
        else:
            parlay_pool = parlay_pool_full.reset_index(drop=True)
        if len(parlay_pool) >= 2:
            from itertools import combinations
            game_col = "game" if "game" in parlay_pool.columns else None

            # 2-LEG parlays — one per game AND one per team (latter is automatic
            # since pool was de-duped by team above)
            two_leg_parlays = []
            for combo in combinations(parlay_pool.index, 2):
                rows = [parlay_pool.loc[i] for i in combo]
                # Skip if both legs from same game (correlated)
                if game_col and rows[0].get(game_col) == rows[1].get(game_col):
                    continue
                # Joint probability (independent assumption since cross-game)
                p1 = rows[0].get("hr_game_pct", 0) / 100
                p2 = rows[1].get("hr_game_pct", 0) / 100
                joint = p1 * p2
                two_leg_parlays.append({
                    "Leg 1": f"{rows[0]['player_name']} ({rows[0].get('team','')})",
                    "Leg 2": f"{rows[1]['player_name']} ({rows[1].get('team','')})",
                    "L1 HR%": p1 * 100,
                    "L2 HR%": p2 * 100,
                    "Parlay HR%": joint * 100,
                    "Approx Odds": f"+{int(round(1 / joint * 100 - 100))}" if joint > 0 else "N/A",
                    "_p1_name": rows[0]['player_name'],
                    "_p2_name": rows[1]['player_name'],
                })
            two_leg_all = pd.DataFrame(two_leg_parlays).sort_values(
                "Parlay HR%", ascending=False
            ).reset_index(drop=True)

            # GREEDY DIVERSITY: cap each player at appearing in at most MAX_USES
            # parlay rows. User feedback was Wood appearing in 5+ rows was too
            # repetitive, but capping at 1 use dropped the parlay count to ~3.
            # Cap of 2 strikes the balance: shows variety but uses real top plays.
            MAX_USES = 2
            two_leg_selected = []
            player_use_count = {}
            for _, prow in two_leg_all.iterrows():
                p1 = prow["_p1_name"]
                p2 = prow["_p2_name"]
                if (player_use_count.get(p1, 0) >= MAX_USES
                        or player_use_count.get(p2, 0) >= MAX_USES):
                    continue
                two_leg_selected.append(prow)
                player_use_count[p1] = player_use_count.get(p1, 0) + 1
                player_use_count[p2] = player_use_count.get(p2, 0) + 1
                if len(two_leg_selected) >= 10:
                    break
            two_leg_df = (pd.DataFrame(two_leg_selected)
                          .drop(columns=["_p1_name", "_p2_name"])
                          if two_leg_selected else
                          two_leg_all.drop(columns=["_p1_name", "_p2_name"]).head(10))
            two_leg_df = two_leg_df.reset_index(drop=True)

            # 3-LEG parlays - one per game AND one per team
            three_leg_parlays = []
            for combo in combinations(parlay_pool.index[:12], 3):
                rows = [parlay_pool.loc[i] for i in combo]
                # Skip if any two legs share a game (correlated)
                if game_col:
                    games = {r.get(game_col) for r in rows}
                    if len(games) < 3:
                        continue
                p1 = rows[0].get("hr_game_pct", 0) / 100
                p2 = rows[1].get("hr_game_pct", 0) / 100
                p3 = rows[2].get("hr_game_pct", 0) / 100
                joint = p1 * p2 * p3
                three_leg_parlays.append({
                    "Leg 1": f"{rows[0]['player_name']}",
                    "Leg 2": f"{rows[1]['player_name']}",
                    "Leg 3": f"{rows[2]['player_name']}",
                    "Avg HR%": (p1 + p2 + p3) / 3 * 100,
                    "Parlay HR%": joint * 100,
                    "Approx Odds": f"+{int(round(1 / joint * 100 - 100))}" if joint > 0 else "N/A",
                    "_p1_name": rows[0]['player_name'],
                    "_p2_name": rows[1]['player_name'],
                    "_p3_name": rows[2]['player_name'],
                })
            three_leg_all = pd.DataFrame(three_leg_parlays).sort_values(
                "Parlay HR%", ascending=False
            ).reset_index(drop=True)
            # Same MAX_USES rule for 3-leg
            three_leg_selected = []
            player_use_count_3 = {}
            for _, prow in three_leg_all.iterrows():
                names = [prow["_p1_name"], prow["_p2_name"], prow["_p3_name"]]
                if any(player_use_count_3.get(n, 0) >= MAX_USES for n in names):
                    continue
                three_leg_selected.append(prow)
                for n in names:
                    player_use_count_3[n] = player_use_count_3.get(n, 0) + 1
                if len(three_leg_selected) >= 8:
                    break
            three_leg_df = (pd.DataFrame(three_leg_selected)
                            .drop(columns=["_p1_name", "_p2_name", "_p3_name"])
                            if three_leg_selected else
                            three_leg_all.drop(
                                columns=["_p1_name", "_p2_name", "_p3_name"]
                            ).head(8))
            three_leg_df = three_leg_df.reset_index(drop=True)

            tab1, tab2 = st.tabs(["🎯 2-Leg Parlays (safer)", "💎 3-Leg Parlays (better odds)"])
            with tab1:
                st.caption(
                    "Top 10 two-leg HR parlays from different games. "
                    "Real-world fair odds shown next to projected hit rate."
                )
                st.dataframe(
                    two_leg_df, hide_index=True, use_container_width=True,
                    column_config={
                        "L1 HR%": st.column_config.NumberColumn(format="%.1f%%"),
                        "L2 HR%": st.column_config.NumberColumn(format="%.1f%%"),
                        "Parlay HR%": st.column_config.NumberColumn(
                            format="%.2f%%",
                            help="Probability BOTH legs hit. ~3-5% = strong play.",
                        ),
                        "Approx Odds": st.column_config.TextColumn(
                            help="Fair-odds American format if the parlay HR% is correct.",
                        ),
                    },
                )
                # Store for export
                two_leg_parlay_export = two_leg_df.copy()
            with tab2:
                st.caption(
                    "Top 8 three-leg HR parlays from different games. "
                    "Lower hit rate but bigger payout. Use sparingly."
                )
                st.dataframe(
                    three_leg_df, hide_index=True, use_container_width=True,
                    column_config={
                        "Avg HR%": st.column_config.NumberColumn(format="%.1f%%"),
                        "Parlay HR%": st.column_config.NumberColumn(
                            format="%.3f%%",
                            help="Probability ALL THREE legs hit. ~0.5-1% = strong 3-leg.",
                        ),
                    },
                )
                three_leg_parlay_export = three_leg_df.copy()

            st.caption(
                "⚠️ **Honest disclaimer:** parlay odds compound risk. Each leg's "
                "miss-rate is real. Even a 4% 2-leg parlay misses 96% of the time. "
                "Use these as one signal among many, not a betting guarantee."
            )

            # ====================================================================
            # 💎 SLEEPER PARLAYS — separate section with explicit warning
            # ====================================================================
            # User asked for sleeper parlays in their own section with a warning.
            # Sleepers = high upside today vs season pace BUT lower absolute HR%,
            # so parlays are higher variance. Use only sleepers that aren't facing
            # TOUGH/ELITE pitchers (trap protection).
            sleeper_parlay_export = None
            if ("sleeper_score" in q.columns and "opp_pitcher_grade" in q.columns
                    and "team" in q.columns):
                sleeper_pool = q[
                    (q["sleeper_score"] > 15)  # meaningful sleeper lift
                    & (~q["opp_pitcher_grade"].isin(["TOUGH", "ELITE"]))
                    & (q["hr_game_pct"].fillna(0) >= 8)  # still need a real HR chance
                ].sort_values("sleeper_score", ascending=False)
                # One per team rule
                sleeper_pool = sleeper_pool.drop_duplicates(
                    subset=["team"], keep="first"
                ).head(10).reset_index(drop=True)

                if len(sleeper_pool) >= 2:
                    with st.expander("💎 Sleeper Parlays (higher variance — different game/team only)"):
                        st.warning(
                            "⚠️ **Higher variance than top-pick parlays.** "
                            "Sleepers are hitters with HR upside today vs season pace, "
                            "but their absolute HR Game% is lower than the top picks. "
                            "Hit rates here are lower; payouts are bigger. "
                            "TOUGH/ELITE pitcher matchups already filtered out. "
                            "Use SPARINGLY — these are dart throws compared to top-pick parlays."
                        )
                        sleeper_2leg = []
                        for combo in combinations(sleeper_pool.index, 2):
                            rows = [sleeper_pool.loc[i] for i in combo]
                            # Skip same game
                            if game_col and rows[0].get(game_col) == rows[1].get(game_col):
                                continue
                            p1 = rows[0].get("hr_game_pct", 0) / 100
                            p2 = rows[1].get("hr_game_pct", 0) / 100
                            joint = p1 * p2
                            sleeper_2leg.append({
                                "Leg 1": f"{rows[0]['player_name']} ({rows[0].get('team','')}) [Sleep {rows[0].get('sleeper_score',0):.0f}]",
                                "Leg 2": f"{rows[1]['player_name']} ({rows[1].get('team','')}) [Sleep {rows[1].get('sleeper_score',0):.0f}]",
                                "L1 HR%": p1 * 100,
                                "L2 HR%": p2 * 100,
                                "Parlay HR%": joint * 100,
                                "Approx Odds": f"+{int(round(1 / joint * 100 - 100))}" if joint > 0 else "N/A",
                            })
                        sleeper_2leg_df = pd.DataFrame(sleeper_2leg).sort_values(
                            "Parlay HR%", ascending=False
                        ).head(8).reset_index(drop=True)
                        st.markdown(f"**💎 Top sleeper 2-leg parlays ({len(sleeper_2leg_df)})**")
                        st.dataframe(
                            sleeper_2leg_df, hide_index=True, use_container_width=True,
                            column_config={
                                "L1 HR%": st.column_config.NumberColumn(format="%.1f%%"),
                                "L2 HR%": st.column_config.NumberColumn(format="%.1f%%"),
                                "Parlay HR%": st.column_config.NumberColumn(format="%.2f%%"),
                            },
                        )
                        sleeper_parlay_export = sleeper_2leg_df.copy()

                        # Sleeper round robin (3-pick)
                        if len(sleeper_pool) >= 3:
                            st.markdown("**💎 Sleeper Round Robin (top 3)**")
                            rr3 = sleeper_pool.head(3)
                            sleeper_rr = []
                            for n_legs in range(2, 4):
                                for combo in combinations(rr3.index, n_legs):
                                    rows = [rr3.loc[i] for i in combo]
                                    # Same-game check
                                    if game_col and len({r.get(game_col) for r in rows}) < n_legs:
                                        continue
                                    joint = 1.0
                                    names = []
                                    for r in rows:
                                        joint *= r.get("hr_game_pct", 0) / 100
                                        names.append(r["player_name"])
                                    if joint > 0:
                                        sleeper_rr.append({
                                            "Legs": n_legs,
                                            "Combination": " + ".join(names),
                                            "Hit %": joint * 100,
                                            "Fair Odds": f"+{int(round(1/joint*100 - 100))}",
                                        })
                            sleeper_rr_df = pd.DataFrame(sleeper_rr)
                            if not sleeper_rr_df.empty:
                                st.dataframe(
                                    sleeper_rr_df, hide_index=True, use_container_width=True,
                                    column_config={
                                        "Hit %": st.column_config.NumberColumn(format="%.3f%%"),
                                    },
                                )

            # ====================================================================
            # ROUND ROBIN — pick N hitters, generate every parlay combination
            # ====================================================================
            with st.expander("🎰 Round Robin — pick 3-5 hitters, see every combination"):
                st.caption(
                    "**What's a round robin?** Pick N hitters. The model generates "
                    "EVERY possible parlay combination at every leg count. Trade-off: "
                    "you wager on more parlays (more cost) but only need SOME to cash. "
                    "If 1 of your 5 misses, you still hit ~6 of the smaller parlays.\n\n"
                    "Default uses the top picks already shown above. Toggle Owner-mode "
                    "and use the My Picks editor below to round-robin YOUR personal list."
                )

                # Pool: top N from the parlay candidates
                rr_size = st.slider(
                    "Round Robin size (number of hitters):",
                    min_value=3, max_value=5, value=4,
                    help="More hitters = more parlay combinations but more cost.",
                    key="rr_size_slider",
                )

                # Take top N picks already built (parlay_pool from above)
                rr_picks = parlay_pool.head(rr_size).copy()
                if len(rr_picks) < rr_size:
                    st.warning(f"Only {len(rr_picks)} qualified picks available — using all.")

                if len(rr_picks) >= 3:
                    # Show the pool first
                    pool_display = rr_picks[["player_name", "team", "game", "hr_game_pct"]].copy()
                    pool_display["#"] = range(1, len(pool_display) + 1)
                    st.markdown("**Your Round Robin pool:**")
                    st.dataframe(
                        pool_display[["#", "player_name", "team", "game", "hr_game_pct"]],
                        hide_index=True, use_container_width=True,
                        column_config={
                            "hr_game_pct": st.column_config.NumberColumn("HR%", format="%.1f%%"),
                        },
                    )

                    # Generate every combination at each leg count
                    rr_results = []
                    for leg_count in range(2, len(rr_picks) + 1):
                        combos = list(combinations(rr_picks.index, leg_count))
                        for combo in combos:
                            rows = [rr_picks.loc[i] for i in combo]
                            # Independence assumption — typical for unrelated hitter HR props
                            joint = 1.0
                            names = []
                            for r in rows:
                                pct = r.get("hr_game_pct", 0) / 100
                                joint *= pct
                                names.append(r["player_name"])
                            if joint > 0:
                                odds = int(round(1 / joint * 100 - 100))
                                rr_results.append({
                                    "Legs": leg_count,
                                    "Combination": " + ".join(names),
                                    "Hit %": joint * 100,
                                    "Fair Odds": f"+{odds}",
                                })

                    rr_df = pd.DataFrame(rr_results)
                    if not rr_df.empty:
                        # Summary stats by leg count
                        n_legs = sorted(rr_df["Legs"].unique())
                        summary_lines = []
                        for lc in n_legs:
                            sub = rr_df[rr_df["Legs"] == lc]
                            avg_hit = sub["Hit %"].mean()
                            summary_lines.append(
                                f"**{lc}-leg:** {len(sub)} parlays · avg {avg_hit:.2f}% hit rate"
                            )
                        st.markdown(" · ".join(summary_lines))

                        # Display by leg count tabs
                        leg_tabs = st.tabs([f"{lc}-leg ({len(rr_df[rr_df['Legs']==lc])})" for lc in n_legs])
                        for tab_idx, lc in enumerate(n_legs):
                            with leg_tabs[tab_idx]:
                                sub = rr_df[rr_df["Legs"] == lc].sort_values(
                                    "Hit %", ascending=False
                                ).reset_index(drop=True)
                                st.dataframe(
                                    sub[["Combination", "Hit %", "Fair Odds"]],
                                    hide_index=True, use_container_width=True,
                                    column_config={
                                        "Hit %": st.column_config.NumberColumn(
                                            format="%.3f%%",
                                            help="Joint probability all legs hit.",
                                        ),
                                        "Fair Odds": st.column_config.TextColumn(
                                            help="What odds would be fair if our hit% is right.",
                                        ),
                                    },
                                )

                        # Round robin export
                        rr_export = rr_df.copy()

                        # Sportsbook-style summary
                        total_parlays = len(rr_df)
                        # If user wagers $1 on each combo, expected return:
                        # E[$] = sum( hit_prob * (odds_decimal - 1) ) - (1 - hit_prob) ...
                        # but odds vary so we'll just show the cost summary
                        st.caption(
                            f"💰 **Cost summary:** A full round robin = {total_parlays} parlays. "
                            f"At $1 each = ${total_parlays} total wager. "
                            f"On most sportsbooks you can select 'Round Robin' as a single ticket "
                            f"with a fixed cost = total parlays × wager-per-parlay."
                        )
                    else:
                        st.caption("No valid combinations generated.")
                else:
                    st.caption("Need at least 3 picks with HR projections.")

            # ====================================================================
            # MY PICKS — OWNER-ONLY editable personal pick list
            # ====================================================================
            if owner_mode:
                with st.expander("📝 My Picks — your personal HR picks for today (owner only)"):
                    st.caption(
                        "Build your own daily pick list by selecting hitters from "
                        "the slate. Picks save during your session and can be exported. "
                        "Use this to track YOUR plays separately from the model's suggestions."
                    )

                    # Initialize session state for picks
                    picks_key = f"my_picks_{selected_date.isoformat()}"
                    if picks_key not in st.session_state:
                        st.session_state[picks_key] = []

                    # Build hitter selection from current slate.
                    # combined_picks is the all-hitters pool built earlier in this
                    # section (around line 2547). combined_all isn't built until much
                    # later, so we can't use it here.
                    if combined_picks is not None and not combined_picks.empty:
                        # Sort by HR Game% so best plays appear first in selector
                        selector_df = combined_picks[
                            combined_picks["hr_game_pct"].notna()
                        ].sort_values("hr_game_pct", ascending=False).copy()
                        # Build label "Name (TEAM) — HR% / grade"
                        selector_df["_label"] = selector_df.apply(
                            lambda r: f"{r['player_name']} ({r.get('team','')}) — {r.get('hr_game_pct', 0):.1f}% / {r.get('grade','—')}",
                            axis=1,
                        )
                        all_labels = selector_df["_label"].tolist()

                        chosen = st.multiselect(
                            "Add hitters to your pick list:",
                            options=all_labels,
                            default=st.session_state[picks_key],
                            help="Type to filter. Select as many as you want.",
                            key=f"picks_selector_{selected_date.isoformat()}",
                        )
                        st.session_state[picks_key] = chosen

                        if chosen:
                            # Filter selector_df to chosen picks
                            my_picks_df = selector_df[selector_df["_label"].isin(chosen)].copy()
                            display_cols = [c for c in [
                                "player_name", "team", "game", "opp_pitcher",
                                "bats", "hr_game_pct", "grade", "smash_spot",
                                "barrel_pct", "iso", "matchup_opp", "env_mult",
                            ] if c in my_picks_df.columns]
                            st.dataframe(
                                my_picks_df[display_cols],
                                hide_index=True, use_container_width=True,
                                column_config={
                                    "hr_game_pct": st.column_config.NumberColumn(
                                        "HR Game%", format="%.2f%%"),
                                    "barrel_pct": st.column_config.NumberColumn(
                                        "Brl%", format="%.1f%%"),
                                    "iso": st.column_config.NumberColumn("ISO", format="%.3f"),
                                    "env_mult": st.column_config.NumberColumn(
                                        "Env×", format="%.3f"),
                                },
                            )

                            # Stats summary
                            avg_pct = my_picks_df["hr_game_pct"].mean()
                            n_picks = len(my_picks_df)
                            # If treating as a parlay
                            if n_picks >= 2 and n_picks <= 10:
                                # Joint probability if all are independent
                                joint_prob = 1.0
                                for p in my_picks_df["hr_game_pct"].dropna():
                                    joint_prob *= (p / 100)
                                summary_cols = st.columns(3)
                                with summary_cols[0]:
                                    st.metric("My picks", f"{n_picks}")
                                with summary_cols[1]:
                                    st.metric("Avg HR%", f"{avg_pct:.1f}%")
                                with summary_cols[2]:
                                    if joint_prob > 0:
                                        odds = int(round(1 / joint_prob * 100 - 100))
                                        st.metric(
                                            "As a parlay",
                                            f"{joint_prob*100:.3f}%",
                                            help=f"Fair odds: +{odds}",
                                        )

                            # Tweet-ready My Picks post
                            st.markdown("**My Picks tweet:**")
                            picks_lines = "\n".join(
                                f"{i+1}. {r['player_name']} ({r['team']})"
                                for i, (_, r) in enumerate(my_picks_df.iterrows())
                            )
                            mypicks_tweet = (
                                f"⚾ My HR Picks — {selected_date.strftime('%b %-d')}\n\n"
                                f"{picks_lines}\n\n"
                                f"#MLB #DFS #HRprops"
                            )
                            st.text_area(
                                "Copy My Picks for Twitter:",
                                value=mypicks_tweet, height=160,
                                key=f"mypicks_tweet_{selected_date.isoformat()}",
                            )
                            chars = len(mypicks_tweet)
                            if chars > 280:
                                st.warning(f"⚠️ {chars}/280 — too long, trim picks")
                            else:
                                st.caption(f"✅ {chars}/280 characters")

                            # Clear button
                            if st.button("🗑️ Clear my picks", key=f"clear_picks_{selected_date.isoformat()}"):
                                st.session_state[picks_key] = []
                                st.rerun()
                        else:
                            st.caption("No picks selected yet. Use the dropdown above.")
                    else:
                        st.caption("Slate data not loaded yet.")
            if owner_mode:
                with st.expander("🐦 Twitter-ready daily post — copy + paste (owner only)"):
                    date_str = selected_date.strftime("%b %-d")
                    top3 = top10.head(3)
                    top3_lines = [
                        f"{i+1}. {r['player_name']} ({r['team']}) {r.get('hr_game_pct', 0):.1f}%"
                        for i, (_, r) in enumerate(top3.iterrows())
                    ]
                    best_parlay = ""
                    if not two_leg_df.empty:
                        bp = two_leg_df.iloc[0]
                        best_parlay = (
                            f"\n\nTop Parlay: {bp['Leg 1']} + {bp['Leg 2']} "
                            f"({bp['Parlay HR%']:.1f}% / {bp['Approx Odds']})"
                        )

                    twitter_text = (
                        f"⚾ MLB HR Picks — {date_str}\n\n"
                        f"Top 3 HR Plays:\n"
                        + "\n".join(top3_lines)
                        + best_parlay
                        + "\n\n#MLB #DFS #HRprops"
                    )
                    st.text_area(
                        "Copy this for Twitter (280 chars limit):",
                        value=twitter_text, height=200,
                        help=(
                            "Paste this directly into Twitter. "
                            "Tips for growing followers: post daily at consistent times "
                            "(4-5 PM ET after lineups confirm), tag #MLB and #HRprops, "
                            "follow back accounts that engage. Show your work — when "
                            "picks hit, post the result the next morning to build credibility."
                        ),
                    )
                    chars = len(twitter_text)
                    if chars > 280:
                        st.warning(
                            f"⚠️ Post is {chars} characters — Twitter limit is 280. "
                            f"Trim the parlay or hashtags before posting."
                        )
                    else:
                        st.caption(f"✅ {chars}/280 characters — fits in one tweet.")
        else:
            st.caption("Need at least 2 picks with HR projections to suggest parlays.")
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

    # Tag slate leaders with 🏆 in a new column. The slate_leader_pid_map was
    # built earlier (Slate Leaders section). Each leader gets a flag that
    # shows in the matchup tables and export.
    if slate_leader_pid_map:
        def _leader_flag(pid):
            try:
                cats = slate_leader_pid_map.get(int(pid), [])
            except (TypeError, ValueError):
                return ""
            if not cats:
                return ""
            # Count of categories led; 🏆 prefix with category count
            if len(cats) >= 3:
                return f"🏆×{len(cats)}"
            return "🏆"
        combined_all["slate_leader_flag"] = combined_all["player_id"].apply(_leader_flag)
        # Also annotate p_slate the same way
        if not p_slate.empty:
            p_slate["slate_leader_flag"] = p_slate["pitcher_id"].apply(_leader_flag)

    # SLATE-WIDE SLEEPER RECOMPUTE
    # find_sleepers() is called per-lineup (9 hitters at a time), so
    # sleeper_score percentiles are computed within 9-row frames. A hitter
    # ranking #1 in a weak lineup would get ~95 sleeper score even if he'd
    # rank #50 slate-wide. Recompute on the full combined frame so percentiles
    # reflect the whole slate (200-300 hitters).
    try:
        score_col = "hr_score" if "hr_score" in combined_all.columns else (
            "hr_prob" if "hr_prob" in combined_all.columns else None
        )
        if (score_col and "home_run" in combined_all.columns
                and combined_all[score_col].notna().any()):
            hr_pct_slate = combined_all[score_col].rank(pct=True) * 100
            season_pct_slate = combined_all["home_run"].rank(pct=True) * 100
            combined_all["sleeper_score"] = (hr_pct_slate - season_pct_slate).round(1)
            # Suppress when PA below 100 (same threshold as find_sleepers)
            if "pa" in combined_all.columns:
                combined_all.loc[
                    combined_all["pa"].isna() | (combined_all["pa"] < 100),
                    "sleeper_score"
                ] = np.nan
    except Exception:
        pass

    # Enrich with the OPPOSING pitcher's grade so we can filter trap sleepers
    # (e.g. don't recommend a sleeper facing a TOUGH/ELITE pitcher).
    if (not p_slate.empty and "pitcher_name" in p_slate.columns
            and "grade" in p_slate.columns
            and "opp_pitcher" in combined_all.columns):
        try:
            grade_map = p_slate.set_index("pitcher_name")["grade"].to_dict()
            hr9_map = p_slate.set_index("pitcher_name")["hr9"].to_dict() if "hr9" in p_slate.columns else {}
            barrel_a_map = p_slate.set_index("pitcher_name")["barrel_allowed"].to_dict() if "barrel_allowed" in p_slate.columns else {}
            combined_all["opp_pitcher_grade"] = combined_all["opp_pitcher"].map(grade_map)
            combined_all["opp_pitcher_hr9"] = combined_all["opp_pitcher"].map(hr9_map)
            combined_all["opp_pitcher_barrel_allowed"] = combined_all["opp_pitcher"].map(barrel_a_map)
            # NEW: surface pitcher platoon HR vulnerability and recent HR streak
            # on each hitter row so it's visible in the matchup table.
            if "platoon_hr_flag" in p_slate.columns:
                platoon_map = p_slate.set_index("pitcher_name")["platoon_hr_flag"].to_dict()
                combined_all["opp_platoon_hr"] = combined_all["opp_pitcher"].map(platoon_map)
            if "recent_hr_flag" in p_slate.columns:
                recent_hr_map = p_slate.set_index("pitcher_name")["recent_hr_flag"].to_dict()
                combined_all["opp_recent_hr"] = combined_all["opp_pitcher"].map(recent_hr_map)
        except Exception:
            pass
    # Flag hitters whose player_id appears in recent transactions — their team
    # data or stats may not yet reflect the move. User can investigate manually.
    if _recently_moved_ids and "player_id" in combined_all.columns:
        try:
            combined_all["recently_moved"] = combined_all["player_id"].apply(
                lambda pid: "🔄" if (pd.notna(pid) and int(pid) in _recently_moved_ids) else ""
            )
        except Exception:
            pass
    # Drop hr_prob (it's a duplicate of hr_score, just confusingly named).
    # Both are 0-100 composite scores — keeping both confused users who
    # thought hr_prob was a 0-1 probability.
    if "hr_prob" in combined_all.columns:
        combined_all = combined_all.drop(columns=["hr_prob"])
    if "pa" in combined_all.columns:
        qualified = combined_all[combined_all["pa"].notna() & (combined_all["pa"] >= INSUFFICIENT_PA_THRESHOLD)]
    else:
        qualified = combined_all

    # ==== AUTO-SNAPSHOT: critical for the model to learn ====
    # The model is meaningless without outcome data to calibrate against.
    # Every time the app loads, IF:
    #   - selected_date is today
    #   - we have decent data (at least 1 confirmed lineup or 100+ qualified hitters)
    #   - we don't already have a snapshot for today
    # ...then automatically save a snapshot. This ensures we capture EVERY day,
    # not just days when the user remembers to click the button.
    auto_snap_status = ""
    try:
        from backtest import save_snapshot, list_snapshots
        existing_snaps = set(list_snapshots())
        snap_key = str(selected_date)
        if (snap_key not in existing_snaps
                and selected_date == datetime.now().date()
                and combined_all is not None and len(combined_all) >= 100):
            ok = save_snapshot(selected_date, combined_all, p_slate)
            if ok:
                auto_snap_status = f"✅ Auto-snapshot saved for {selected_date}"
            else:
                auto_snap_status = f"⚠️ Auto-snapshot failed - click button to save manually"
        elif snap_key in existing_snaps:
            auto_snap_status = f"✅ Snapshot exists for {selected_date}"
    except Exception:
        pass

    # Save-snapshot + full export buttons - always render
    snap_col1, snap_col2, snap_col3 = st.columns([1.2, 1.5, 3])
    with snap_col1:
        if st.button("💾 Save snapshot", help="Manually save today's projections for backtest comparison. Note: auto-snapshot already happens on every load."):
            try:
                from backtest import save_snapshot
                ok = save_snapshot(selected_date, combined_all, p_slate)
                if ok:
                    st.success(f"Saved snapshot for {selected_date}")
                else:
                    st.error("Snapshot save failed")
            except Exception as e:
                st.error(f"Backtest module error: {e}")
        if auto_snap_status:
            st.caption(auto_snap_status)

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
                        # Apply same TOUGH/ELITE filter to export as to the display widget
                        sleep_export = qualified.dropna(subset=["sleeper_score"]).copy()
                        if "opp_pitcher_grade" in sleep_export.columns:
                            sleep_export = sleep_export[
                                ~sleep_export["opp_pitcher_grade"].isin(["TOUGH", "ELITE"])
                            ]
                        top_sl = sleep_export.sort_values(
                            "sleeper_score", ascending=False).head(20)
                        if not top_sl.empty:
                            top_sl.to_excel(writer, sheet_name="Top 20 Sleepers", index=False)
                    # NEW: Top 10 Grand Slam scores
                    if "gs_score" in qualified.columns and qualified["gs_score"].notna().any():
                        top_gs_export = qualified.dropna(subset=["gs_score", "hr_game_pct"]).sort_values(
                            "gs_score", ascending=False).head(10)
                        if not top_gs_export.empty:
                            top_gs_export.to_excel(writer, sheet_name="Top 10 Grand Slam", index=False)
                    # NEW: Top 10 Picks (the curated daily picks from Top Picks section)
                    try:
                        if 'top_picks_export' in dir() and top_picks_export is not None and not top_picks_export.empty:
                            top_picks_export.to_excel(writer, sheet_name="Top 10 Picks", index=False)
                        # NEW: Parlay suggestion sheets
                        if 'two_leg_parlay_export' in dir() and two_leg_parlay_export is not None and not two_leg_parlay_export.empty:
                            two_leg_parlay_export.to_excel(writer, sheet_name="2-Leg Parlays", index=False)
                        if 'three_leg_parlay_export' in dir() and three_leg_parlay_export is not None and not three_leg_parlay_export.empty:
                            three_leg_parlay_export.to_excel(writer, sheet_name="3-Leg Parlays", index=False)
                        if 'sleeper_parlay_export' in dir() and sleeper_parlay_export is not None and not sleeper_parlay_export.empty:
                            sleeper_parlay_export.to_excel(writer, sheet_name="Sleeper Parlays", index=False)
                        if 'rr_export' in dir() and rr_export is not None and not rr_export.empty:
                            rr_export.to_excel(writer, sheet_name="Round Robin", index=False)
                    except Exception:
                        pass
                    # NEW: Smash Spots — triple-threat HR opportunities (max 2 per team)
                    if combined_all is not None and "smash_spot" in combined_all.columns:
                        smash_export = combined_all[combined_all["smash_spot"] != ""].copy()
                        if not smash_export.empty:
                            tier_order = {"🔥🔥🔥 ELITE SMASH": 3, "🔥🔥 STRONG SMASH": 2, "🔥 SMASH": 1}
                            smash_export["_tier"] = smash_export["smash_spot"].map(tier_order).fillna(0)
                            # Sort: tier desc, then HR Game% desc
                            smash_export = smash_export.sort_values(
                                ["_tier", "hr_game_pct"], ascending=[False, False]
                            )
                            # Limit to top 2 per team
                            smash_export = smash_export.groupby(
                                "team", group_keys=False
                            ).head(2)
                            # Re-sort final result
                            smash_export = smash_export.sort_values(
                                ["_tier", "hr_game_pct"], ascending=[False, False]
                            ).drop(columns=["_tier"])
                            smash_export.to_excel(writer, sheet_name="Smash Spots", index=False)
                    # NEW: Green Signal sheets - hitters & pitchers with 🟢 alert
                    if combined_all is not None and "alert" in combined_all.columns:
                        green_hitters = combined_all[combined_all["alert"] == "🟢"].sort_values(
                            "hr_game_pct", ascending=False, na_position="last")
                        if not green_hitters.empty:
                            green_hitters.to_excel(writer, sheet_name="Green Hitters", index=False)
                    if p_slate is not None and "alert" in p_slate.columns:
                        green_pitchers = p_slate[p_slate["alert"] == "🟢"].sort_values(
                            "test_score", ascending=False, na_position="last")
                        if not green_pitchers.empty:
                            green_pitchers.to_excel(writer, sheet_name="Green Pitchers", index=False)
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
        st.caption("Under-the-radar HR upside — TOUGH/ELITE pitcher matchups filtered out")
        if "sleeper_score" in qualified.columns:
            top_sleep = qualified.dropna(subset=["sleeper_score"]).copy()
            # TRAP PROTECTION: sleepers shouldn't be facing TOUGH/ELITE pitchers.
            # User's rationale: even if the model identifies HR upside vs season pace,
            # facing an ace is too risky for a sleeper play. Better to avoid entirely.
            # Keep MIXED/EXPLOIT/EXPLOIT+/— (no grade ≈ no data on pitcher).
            if "opp_pitcher_grade" in top_sleep.columns:
                avoid_grades = {"TOUGH", "ELITE"}
                pre_n = len(top_sleep)
                top_sleep = top_sleep[~top_sleep["opp_pitcher_grade"].isin(avoid_grades)]
                filtered_n = pre_n - len(top_sleep)
                if filtered_n > 0:
                    st.caption(f"_({filtered_n} candidates filtered out — facing TOUGH/ELITE pitcher)_")
            top_sleep = top_sleep.sort_values("sleeper_score", ascending=False).head(10)
            if not top_sleep.empty:
                cols = [c for c in [
                    "player_name", "team", "game", "opp_pitcher", "opp_pitcher_grade",
                    "sleeper_score", "hr_game_pct", "barrel_pct",
                ] if c in top_sleep.columns]
                disp = top_sleep[cols].copy().reset_index(drop=True)
                st.dataframe(
                    disp, hide_index=True, use_container_width=True,
                    column_config={
                        "player_name": st.column_config.TextColumn("Hitter"),
                        "team": st.column_config.TextColumn("Tm"),
                        "game": st.column_config.TextColumn("Game"),
                        "opp_pitcher": st.column_config.TextColumn("vs P", width="small"),
                        "opp_pitcher_grade": st.column_config.TextColumn(
                            "Grd", width="small",
                            help="Opp pitcher grade. EXPLOIT+/EXPLOIT = target. MIXED = neutral.",
                        ),
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

    # ====================================================================
    # A+/A GRADE LEADERBOARD — quick access to all elite grade plays
    # ====================================================================
    if "grade" in qualified.columns:
        elite_grades = qualified[qualified["grade"].isin(["A+", "A"])].copy()
        if not elite_grades.empty:
            st.markdown("---")
            st.markdown(f"**🏆 A+/A Grade Hitters Today ({len(elite_grades)})**")
            st.caption(
                "Every hitter rated A+ (HR Game% ≥22%) or A (19-22%). These are the "
                "top-tier HR plays on the slate. Sort by HR% to see best plays first."
            )
            elite_sorted = elite_grades.sort_values("hr_game_pct", ascending=False)
            elite_cols = [c for c in [
                "grade", "player_name", "team", "game", "opp_pitcher",
                "opp_pitcher_grade", "hr_game_pct", "smash_spot",
                "barrel_pct", "iso", "env_mult",
            ] if c in elite_sorted.columns]
            st.dataframe(
                elite_sorted[elite_cols], hide_index=True, use_container_width=True,
                column_config={
                    "grade": st.column_config.TextColumn("Grd", width="small"),
                    "player_name": st.column_config.TextColumn("Hitter"),
                    "team": st.column_config.TextColumn("Tm", width="small"),
                    "game": st.column_config.TextColumn("Game"),
                    "opp_pitcher": st.column_config.TextColumn("vs P"),
                    "opp_pitcher_grade": st.column_config.TextColumn("PGrd", width="small"),
                    "hr_game_pct": st.column_config.NumberColumn("HR%", format="%.1f%%"),
                    "smash_spot": st.column_config.TextColumn("Smash", width="small"),
                    "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
                    "iso": st.column_config.NumberColumn("ISO", format="%.3f"),
                    "env_mult": st.column_config.NumberColumn("Env×", format="%.2f"),
                },
            )

    # ====================================================================
    # 🎯 ELITE HITTERS vs VULNERABLE PITCHERS — the perfect-storm leaderboard
    # ====================================================================
    # User requested: "good matchup and graded hitters vs exploit/exploit+ pitchers
    # who let up above average home runs or hard hits or something like that
    # especially if park factors and weather are good for the hitter too"
    #
    # Qualifying criteria:
    #   - Hitter grade A/A+ OR HR Game% ≥ 16% (top-tier matchup plays)
    #   - Opposing pitcher is EXPLOIT or EXPLOIT+ grade
    #   - Pitcher HR/9 ≥ 1.2 (above-average HR rate allowed)
    #   - OR pitcher barrel% allowed ≥ 7.5 (above-average hard contact allowed)
    #   - Env mult ≥ 1.00 (at minimum neutral park/weather)
    if ("opp_pitcher_grade" in qualified.columns
            and "opp_pitcher_hr9" in qualified.columns):
        perfect_storm = qualified.copy()
        # Filter to hitters facing exploitable pitchers with real vulnerability
        ps_mask = (
            perfect_storm["opp_pitcher_grade"].isin(["EXPLOIT", "EXPLOIT+"])
            & (perfect_storm["hr_game_pct"].fillna(0) >= 14)
            & (
                (perfect_storm["opp_pitcher_hr9"].fillna(0) >= 1.2)
                | (perfect_storm.get("opp_pitcher_barrel_allowed", pd.Series([0]*len(perfect_storm))).fillna(0) >= 7.5)
            )
        )
        if "env_mult" in perfect_storm.columns:
            ps_mask = ps_mask & (perfect_storm["env_mult"].fillna(1.0) >= 1.00)
        perfect_storm = perfect_storm[ps_mask].copy()
        # Cap to 1 per team to spread plays
        if not perfect_storm.empty and "team" in perfect_storm.columns:
            perfect_storm = (
                perfect_storm.sort_values("hr_game_pct", ascending=False)
                .drop_duplicates(subset=["team"], keep="first")
                .head(15)
            )
        if not perfect_storm.empty:
            st.markdown("---")
            st.markdown(f"**🎯 Elite Hitters vs Vulnerable Pitchers ({len(perfect_storm)})**")
            st.caption(
                "Hitters with HR Game% ≥14% facing EXPLOIT/EXPLOIT+ pitchers who allow "
                "above-average HR/9 (≥1.2) or barrel% (≥7.5%) — plus favorable env "
                "(neutral or better). **Max 1 per team.** These are the strongest "
                "combinations of hitter quality, pitcher vulnerability, and environment."
            )
            ps_cols = [c for c in [
                "player_name", "team", "game", "grade", "hr_game_pct",
                "opp_pitcher", "opp_pitcher_grade", "opp_pitcher_hr9",
                "opp_pitcher_barrel_allowed", "env_mult", "barrel_pct",
            ] if c in perfect_storm.columns]
            st.dataframe(
                perfect_storm[ps_cols], hide_index=True, use_container_width=True,
                column_config={
                    "player_name": st.column_config.TextColumn("Hitter"),
                    "team": st.column_config.TextColumn("Tm", width="small"),
                    "game": st.column_config.TextColumn("Game"),
                    "grade": st.column_config.TextColumn("HGrd", width="small"),
                    "hr_game_pct": st.column_config.NumberColumn("HR%", format="%.1f%%"),
                    "opp_pitcher": st.column_config.TextColumn("vs P"),
                    "opp_pitcher_grade": st.column_config.TextColumn(
                        "PGrd", width="small",
                        help="Pitcher grade from batter's perspective.",
                    ),
                    "opp_pitcher_hr9": st.column_config.NumberColumn(
                        "P HR/9", format="%.2f",
                        help="Pitcher's HR/9 allowed. ≥1.2 = above-average vulnerability.",
                    ),
                    "opp_pitcher_barrel_allowed": st.column_config.NumberColumn(
                        "P Brl%", format="%.1f%%",
                        help="Pitcher's barrel% allowed. ≥7.5% = above-average hard contact.",
                    ),
                    "env_mult": st.column_config.NumberColumn(
                        "Env×", format="%.2f",
                        help="Park × weather × wind multiplier.",
                    ),
                    "barrel_pct": st.column_config.NumberColumn("H Brl%", format="%.1f%%"),
                },
            )

    # ====================================================================
    # ⚾ Recently homered warning — flag top picks who hit yesterday
    # ====================================================================
    # User concern: don't repeat "this guy homered yesterday → he'll homer today".
    # The model doesn't do this automatically, but we surface it visually to remind
    # the user to check WHY a hitter is on the board, not just trust momentum.
    if ("recent_hr" in qualified.columns and top_picks_export is not None
            and not top_picks_export.empty):
        # Check which top picks have homered in recent games (any in last 3)
        recent_homered = top_picks_export.copy()
        # We don't have "homered yesterday" directly, but recent_hr (L15) > 0
        # combined with a high recent_hr_weighted_rate signals a current hot streak.
        if "recent_hr" in recent_homered.columns:
            hot_picks = recent_homered[recent_homered["recent_hr"].fillna(0) >= 2].copy()
            if not hot_picks.empty:
                st.caption(
                    f"⚠️ **Recency check:** {len(hot_picks)} of today's top picks have "
                    f"≥2 HRs in their last 15 games ({', '.join(hot_picks['player_name'].tolist()[:5])}). "
                    f"Model rates these based on full season profile + today's matchup, "
                    f"NOT just recent results — but verify their season stats look strong, "
                    f"not just their streak."
                )
    # ====================================================================
    # ⏳ DUE TO HOMER — hitters with strong stats but no recent HR (regression)
    # ====================================================================
    # User asked: "if a player hasnt homered in a while but has all the stats
    # indicating that they have just had bad luck or will go soon..."
    # Modernized criteria (May 28, 2026): uses richer Statcast signals to
    # match competitor "He's Due List" methodology:
    #   - barrel% (quality of contact)
    #   - hard-hit% (consistent loud contact)
    #   - xwOBA / xSLG (expected production from underlying contact)
    #   - ISO (real power output)
    #   - pull% (HRs come from pulled contact for most hitters)
    #   - avg exit velocity
    # A hitter who's elite in MULTIPLE of these but has zero recent HRs is
    # the strongest regression candidate.
    if all(c in qualified.columns for c in ["recent_hr", "barrel_pct", "xwoba", "iso"]):
        due_pool = qualified.copy()
        # Looser threshold than before — let more candidates in, score harder
        due_pool = due_pool[
            (due_pool["recent_hr"].fillna(99) == 0)  # 0 HR in last 15
            & (due_pool["barrel_pct"].fillna(0) >= 7)
            & (due_pool["xwoba"].fillna(0) >= 0.320)
            & (due_pool["iso"].fillna(0) >= 0.140)
        ]
        # Avoid the trap of recommending guys vs aces
        if "opp_pitcher_grade" in due_pool.columns:
            due_pool = due_pool[~due_pool["opp_pitcher_grade"].isin(["TOUGH", "ELITE"])]
        if not due_pool.empty:
            due_pool = due_pool.copy()
            # Composite "due score" — higher weight on quality-of-contact metrics
            # because that's what predicts future HRs better than past HR count.
            score_components = []
            # Barrel% is THE strongest single HR predictor
            if "barrel_pct" in due_pool.columns:
                score_components.append(due_pool["barrel_pct"].fillna(0) * 3.0)
            # Hard-hit% (95+ mph) reinforces barrel signal
            if "hard_hit" in due_pool.columns:
                score_components.append(due_pool["hard_hit"].fillna(0) * 0.8)
            # xwOBA — expected production from full plate appearance profile
            if "xwoba" in due_pool.columns:
                score_components.append(due_pool["xwoba"].fillna(0) * 100)
            # xSLG — expected slugging based on quality of contact
            if "xslg" in due_pool.columns:
                score_components.append(pd.to_numeric(due_pool["xslg"], errors="coerce").fillna(0) * 50)
            # ISO — real isolated power
            if "iso" in due_pool.columns:
                score_components.append(due_pool["iso"].fillna(0) * 80)
            # Pull rate boost — pulled contact produces ~70% of MLB HRs
            if "pull_pct" in due_pool.columns:
                score_components.append(due_pool["pull_pct"].fillna(0) * 0.3)
            # Exit velocity — raw power
            if "avg_ev" in due_pool.columns:
                score_components.append(due_pool["avg_ev"].fillna(0) * 0.5)
            if score_components:
                due_pool["due_score"] = sum(score_components)
            else:
                due_pool["due_score"] = due_pool["barrel_pct"].fillna(0) * 2
            due_pool = due_pool.sort_values("due_score", ascending=False).head(10)
            st.markdown("---")
            st.markdown(f"**⏳ Due to Homer ({len(due_pool)}) — bad luck candidates**")
            st.caption(
                "Hitters with elite power metrics (barrel% ≥7, xwOBA ≥.320, ISO ≥.140) "
                "but ZERO HRs in their last 15 games. Ranked by composite score combining "
                "barrel%, hard-hit%, xwOBA, xSLG, ISO, pull%, and exit velo. "
                "These underlying signals predict future HRs better than recent HR count. "
                "**TOUGH/ELITE pitcher matchups filtered out.**"
            )
            due_cols = [c for c in [
                "player_name", "team", "game", "opp_pitcher", "opp_pitcher_grade",
                "recent_hr", "barrel_pct", "hard_hit", "xwoba", "xslg", "iso",
                "pull_pct", "avg_ev", "hr_game_pct",
            ] if c in due_pool.columns]
            st.dataframe(
                due_pool[due_cols], hide_index=True, use_container_width=True,
                column_config={
                    "player_name": st.column_config.TextColumn("Hitter"),
                    "team": st.column_config.TextColumn("Tm", width="small"),
                    "game": st.column_config.TextColumn("Game"),
                    "opp_pitcher": st.column_config.TextColumn("vs P"),
                    "opp_pitcher_grade": st.column_config.TextColumn("PGrd", width="small"),
                    "recent_hr": st.column_config.NumberColumn(
                        "L15 HR", format="%d", width="small",
                        help="HRs in last 15 games. 0 = cold streak."),
                    "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
                    "hard_hit": st.column_config.NumberColumn("HH%", format="%.1f%%"),
                    "xwoba": st.column_config.NumberColumn("xwOBA", format="%.3f"),
                    "xslg": st.column_config.NumberColumn("xSLG", format="%.3f"),
                    "iso": st.column_config.NumberColumn("ISO", format="%.3f"),
                    "pull_pct": st.column_config.NumberColumn("Pull%", format="%.1f%%"),
                    "avg_ev": st.column_config.NumberColumn("EV", format="%.1f"),
                    "hr_game_pct": st.column_config.NumberColumn("HR%", format="%.1f%%"),
                },
            )

    if "smash_spot" in qualified.columns and (qualified["smash_spot"] != "").any():
        st.markdown("---")
        st.markdown("**🔥 Today's Smash Spots — Triple-Threat HR Opportunities**")
        st.caption(
            "Hitters where multiple factors align: facing an EXPLOIT/EXPLOIT+ pitcher, "
            "favorable env (park × weather × pull-wind), AND strong HR Game%. "
            "**ELITE SMASH** = EXPLOIT+ + favorable env + favorable park + HR%≥19%. "
            "**STRONG SMASH** = EXPLOIT/+ + favorable env + favorable park + HR%≥15%. "
            "**SMASH** = EXPLOIT/+ + (favorable env OR park) + HR%≥15%. "
            "**Max 2 hitters per team** (the top 2 by HR Game% on any team facing an "
            "exploit pitcher — avoids stacking the same lineup)."
        )
        smash_df = qualified[qualified["smash_spot"] != ""].copy()
        # Sort by: ELITE first, then STRONG, then SMASH; within each tier by HR Game%
        tier_order = {"🔥🔥🔥 ELITE SMASH": 3, "🔥🔥 STRONG SMASH": 2, "🔥 SMASH": 1}
        smash_df["_tier"] = smash_df["smash_spot"].map(tier_order).fillna(0)
        smash_df = smash_df.sort_values(
            ["_tier", "hr_game_pct"], ascending=[False, False]
        )
        # CRITICAL: Limit to top 2 per team to avoid stacking same lineup.
        # User asked: "limit the smashes to 1 or 2 per team just the top 2"
        # We use groupby + head(2) which keeps the top 2 hitters per team
        # (already pre-sorted by tier then HR%).
        smash_df = smash_df.groupby("team", group_keys=False).head(2)
        # Re-sort the final result so tiers display in order
        smash_df = smash_df.sort_values(
            ["_tier", "hr_game_pct"], ascending=[False, False]
        ).head(15)
        smash_cols = [c for c in [
            "smash_spot", "player_name", "team", "game", "opp_pitcher",
            "hr_game_pct", "grade", "barrel_pct", "env_mult", "matchup_opp",
        ] if c in smash_df.columns]
        st.dataframe(
            smash_df[smash_cols], hide_index=True, use_container_width=True,
            column_config={
                "smash_spot": st.column_config.TextColumn("Flag", width="medium"),
                "player_name": st.column_config.TextColumn("Hitter"),
                "team": st.column_config.TextColumn("Tm", width="small"),
                "game": st.column_config.TextColumn("Game"),
                "opp_pitcher": st.column_config.TextColumn("vs Pitcher"),
                "hr_game_pct": st.column_config.NumberColumn("HR%", format="%.1f%%"),
                "grade": st.column_config.TextColumn("Grd", width="small"),
                "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
                "env_mult": st.column_config.NumberColumn("Env×", format="%.2f"),
                "matchup_opp": st.column_config.NumberColumn("Opp", format="%.1f"),
            },
        )

    # ====================================================================
    # Top 10 Grand Slam — new leaderboard
    # ====================================================================
    if "gs_score" in qualified.columns and qualified["gs_score"].notna().any():
        st.markdown("---")
        st.markdown("**⚡ Top 10 Grand Slam Opportunities**")
        st.caption(
            "Per-hitter grand slam score: combines HR probability × lineup traffic "
            "(runners on base from teammates' OBP and pitcher's WHIP). High score = "
            "elite power hitter ALSO getting bases-loaded chances tonight. Note: "
            "lineup traffic estimates are more accurate when batting order is "
            "confirmed."
        )
        # CRITICAL FIX: Filter out players whose hr_game_pct is NaN (small sample).
        # Without this filter, low-PA players like Conforto (80 PA) and Goldschmidt
        # (98 PA) ranked #1-#3 with NaN HR%, because their hr_score (composite) was
        # high but hr_game_pct correctly returned NaN for insufficient sample.
        # We can't make grand slam projections without a valid HR probability.
        top_gs = qualified.dropna(subset=["gs_score", "hr_game_pct"]).sort_values(
            "gs_score", ascending=False
        ).head(10)
        if not top_gs.empty:
            gs_cols = [c for c in [
                "player_name", "team", "game", "lineup_pos",
                "gs_score", "hr_game_pct", "barrel_pct", "iso",
            ] if c in top_gs.columns]
            gs_disp = top_gs[gs_cols].copy().reset_index(drop=True)
            st.dataframe(
                gs_disp, hide_index=True, use_container_width=True,
                column_config={
                    "player_name": st.column_config.TextColumn("Hitter"),
                    "team": st.column_config.TextColumn("Tm"),
                    "game": st.column_config.TextColumn("Game"),
                    "lineup_pos": st.column_config.NumberColumn(
                        "#", help="Batting order if confirmed; '—' otherwise.",
                    ),
                    "gs_score": st.column_config.NumberColumn(
                        "GS Score", format="%.1f",
                        help="Grand Slam composite. 0-100 scale.",
                    ),
                    "hr_game_pct": st.column_config.NumberColumn("HR%", format="%.1f%%"),
                    "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
                    "iso": st.column_config.NumberColumn("ISO", format="%.3f"),
                },
            )
        else:
            st.caption("No grand slam scores computed yet.")
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
                "TRUE PROBABILITY of hitting ≥1 HR this game. Calibrated to "
                "match real MLB rates — NOT inflated.\n\n"
                "REAL-WORLD REFERENCE POINTS:\n"
                "• Aaron Judge (2024 MVP year): 32% of games had ≥1 HR\n"
                "• Schwarber 38-HR season: ~33% of games had ≥1 HR\n"
                "• Soto 41-HR season: ~29% of games\n"
                "• Average MLB power hitter (20-25 HR): 18-20% of games\n"
                "• League-average hitter: 5-8% of games\n\n"
                "DraftKings/FanDuel typically priced Judge at +280 to +320 "
                "(implied 24-26%). Our 23% projection is in line with the market.\n\n"
                "Formula: 1 - (1 - HR per PA)^expected_PA. Our cap is ~26% game-max."
            ),
        ),
        "grade": st.column_config.TextColumn(
            "Grade", width="small",
            help=(
                "Letter grade equivalent of HR Game%:\n"
                "A+ : ≥22% (elite - top tier)\n"
                "A  : 19-22% (very strong)\n"
                "B+ : 16-19% (strong)\n"
                "B  : 12-16% (solid)\n"
                "C+ : 9-12% (modest)\n"
                "C  : 6-9% (below avg)\n"
                "D  : 3-6% (poor)\n"
                "F  : <3% (avoid)\n"
                "—  : insufficient sample"
            ),
        ),
        "smash_spot": st.column_config.TextColumn(
            "Smash", width="small",
            help=(
                "THE 'all stars align' flag — triple-threat HR opportunity.\n\n"
                "🔥🔥🔥 ELITE SMASH = facing EXPLOIT+ pitcher AND favorable env "
                "(park × wx × wind ≥1.05) AND favorable park (≥1.04) AND HR Game% ≥19%.\n\n"
                "🔥🔥 STRONG SMASH = EXPLOIT/+ pitcher + favorable env + favorable park + "
                "HR Game% ≥15%.\n\n"
                "🔥 SMASH = EXPLOIT/+ pitcher + (favorable env OR park) + HR Game% ≥15%.\n\n"
                "Requires: confirmed lineup, real pitcher (not TBD).\n"
                "Limited to 2 per team in the leaderboard."
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
        "obp": st.column_config.NumberColumn(
            "OBP", format="%.3f",
            help="On-base percentage. Reaches base safely / PA. League avg ~.320.",
        ),
        "slg": st.column_config.NumberColumn(
            "SLG", format="%.3f",
            help="Slugging percentage. Total bases / AB. League avg ~.398.",
        ),
        "ops": st.column_config.NumberColumn(
            "OPS", format="%.3f",
            help="On-base + Slugging. League avg ~.720. Elite hitters > .900.",
        ),
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
        "recent_hr_weighted_rate": st.column_config.NumberColumn(
            "L15 wHR%", format="%.2f%%",
            help=(
                "Recency-WEIGHTED HR/PA rate over last 15 games. "
                "Last 3 games weighted 3×, games 4-7 weighted 2×, games 8-15 weighted 1×. "
                "Captures hot streaks better than simple last-N average.\n\n"
                "Used as a modest adjuster (capped at ±25%) on the model's HR base rate "
                "so true hot streaks slightly boost projections, slumps slightly suppress."
            ),
        ),
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

    # Detect if this is a roster-fill (unconfirmed) lineup
    is_unconfirmed = (
        "is_roster_fill" in matchup_df.columns
        and matchup_df["is_roster_fill"].any()
    )
    if is_unconfirmed:
        st.caption(
            f"📋 **{team_label}** — lineup not posted. "
            f"Hitters below are sorted by season PA (likely starters first). "
            f"The **#** column shows '—' because real lineup positions are unknown. "
            f"DON'T read row order as batting order — Murakami showing first just "
            f"means he's the highest-PA hitter, not that he's leading off."
        )

    if "pa" in matchup_df.columns:
        qualified = matchup_df[matchup_df["pa"].notna() & (matchup_df["pa"] >= INSUFFICIENT_PA_THRESHOLD)]
        insufficient = matchup_df[~(matchup_df["pa"].notna() & (matchup_df["pa"] >= INSUFFICIENT_PA_THRESHOLD))]
    else:
        qualified = matchup_df
        insufficient = pd.DataFrame()

    cols_to_show = [c for c in [
        "alert", "grade", "smash_spot", "player_name", "lineup_pos", "bats", "position",
        "power_score", "matchup_opp", "hr_game_pct", "hr_pa_pct", "matchup", "test_score",
        "streak_label",
        "pa", "barrel_pct", "iso", "xwoba", "xwobacon",
        "obp", "slg", "ops",
        "pitch_match_score", "pitch_hr_score", "best_pitch", "best_pitch_xwoba", "worst_pitch",
        "fb_pct", "la", "avg_ev", "hard_hit",
        "k_pct", "bb_pct", "whiff_pct",
        "home_run", "recent_hr", "recent_hr_weighted_rate", "sleeper_score",
    ] if c in matchup_df.columns]

    # Auto-hide columns that are completely empty (no point showing them)
    cols_to_show = [c for c in cols_to_show if matchup_df[c].notna().any()]

    if not qualified.empty:
        st.markdown(f"**{team_label}**")

        # ====================================================================
        # CATEGORY LEADERS — who leads what on this team?
        # Shows the top hitter in each key category so you can see at a glance
        # who's the best play in different dimensions.
        # ====================================================================
        leader_categories = [
            ("HR Game%", "hr_game_pct", True, "%.1f%%"),
            ("Power Score", "power_score", True, "%.0f"),
            ("Matchup", "matchup", True, "%.0f"),
            ("HR Form", "hr_form", True, "%.0f"),
            ("Barrel%", "barrel_pct", True, "%.1f%%"),
            ("ISO", "iso", True, "%.3f"),
            ("xwOBA", "xwoba", True, "%.3f"),
            ("Recent HRs", "recent_hr", True, "%.0f"),
            ("L15 wHR%", "recent_hr_weighted_rate", True, "%.2f%%"),
            ("Pitch Match", "pitch_match_score", True, "%.0f"),
            ("Pitch HR Match", "pitch_hr_score", True, "%.0f"),
        ]
        # Count how many categories each player leads in
        leader_counts = {}
        leader_details = []
        for cat_label, col, descending, fmt in leader_categories:
            if col in qualified.columns and qualified[col].notna().any():
                # Best (max) row in this category
                best_idx = qualified[col].idxmax() if descending else qualified[col].idxmin()
                best_row = qualified.loc[best_idx]
                name = best_row.get("player_name", "?")
                val = best_row[col]
                try:
                    val_str = fmt % val
                except Exception:
                    val_str = str(val)
                leader_counts[name] = leader_counts.get(name, 0) + 1
                leader_details.append({"Category": cat_label, "Leader": name, "Value": val_str})

        if leader_details:
            with st.expander(f"🏅 {team_label} Category Leaders ({len(leader_details)} categories)"):
                # Display leaders table
                ldr_df = pd.DataFrame(leader_details)
                st.dataframe(ldr_df, hide_index=True, use_container_width=True)
                # Summary: hitters leading multiple categories
                multi_leaders = {n: c for n, c in leader_counts.items() if c >= 2}
                if multi_leaders:
                    sorted_leaders = sorted(multi_leaders.items(), key=lambda x: -x[1])
                    summary = " · ".join(
                        f"**{name}** ({count} categories)"
                        for name, count in sorted_leaders[:5]
                    )
                    st.markdown(f"**Top multi-category leaders:** {summary}")
                    st.caption(
                        "Hitters who lead in multiple categories are your strongest "
                        "all-around plays on this team. If someone leads 3+ categories, "
                        "they're likely the best HR pick from this lineup."
                    )

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
    # Status flag — surface in progress / final / postponed games prominently
    game_status_str = (game.get("status") or "").strip()
    if game_status_str:
        in_progress = any(kw in game_status_str.lower() for kw in
                          ("progress", "delayed", "warmup", "pre-game", "manager"))
        is_final = any(kw in game_status_str.lower() for kw in
                       ("final", "completed", "game over", "ended"))
        is_postponed = any(kw in game_status_str.lower() for kw in
                           ("postponed", "suspended", "cancelled", "canceled"))
        if in_progress:
            header_bits.append(f"🟠 **{game_status_str.upper()}** — bets locked")
        elif is_final:
            header_bits.append(f"🔴 **{game_status_str.upper()}** — game over")
        elif is_postponed:
            header_bits.append(f"⚫ **{game_status_str.upper()}**")
    st.markdown(" · ".join(header_bits))

    # Loud full-width banner for games that have started or finished — user
    # asked for an unmistakable signal so they don't pick from these.
    if game_status_str:
        gs_lower = game_status_str.lower()
        if any(kw in gs_lower for kw in ("progress", "delayed", "warmup")):
            st.error(
                f"🟠 **GAME IN PROGRESS — picks no longer available** "
                f"(except live betting). Status: {game_status_str}"
            )
        elif any(kw in gs_lower for kw in ("final", "completed", "game over", "ended")):
            st.error(
                f"🔴 **GAME FINAL — outcomes are set.** Status: {game_status_str}"
            )
        elif any(kw in gs_lower for kw in ("postponed", "suspended", "cancelled", "canceled")):
            st.error(
                f"⚫ **GAME {game_status_str.upper()}** — props void / refunded."
            )

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
            "hitters sorted by season PA (likely starters first). The # column "
            "shows '—' and lineup-position PA scaling is DISABLED for this game "
            "(everyone uses 4.2 PA = league avg). Refresh after lineups post."
        )

    # Rain delay warning (>50% precipitation chance)
    wx_info = ctx.get("weather") or {}
    precip = wx_info.get("precip_prob")
    if precip is not None and not pd.isna(precip):
        if precip >= 80:
            st.error(
                f"🌧️ **HEAVY RAIN — {precip:.0f}% precipitation probability.** "
                f"Game may be delayed or postponed. Reduce position sizes / "
                f"consider waiting until lineups + game status confirmed."
            )
        elif precip >= 50:
            st.warning(
                f"🌦️ **Rain possible — {precip:.0f}% precipitation.** "
                f"Conditions may slow ball flight, suppress HR potential."
            )

    # HR environment flag - prominent color-coded summary
    hr_env_flag = ctx.get("hr_env_flag", "")
    hr_env_color = ctx.get("hr_env_color", "")
    # NEW: append day/night indicator to env flag
    game_type_str = ctx.get("game_type", "")
    if game_type_str == "day":
        day_night_label = "☀️ DAY GAME"
    elif game_type_str == "night":
        day_night_label = "🌙 NIGHT GAME"
    else:
        day_night_label = ""

    # ROOF STATUS — show what MLB reports, with manual override option
    # for retractable parks where MLB hasn't reported status yet.
    real_roof_closed = ctx.get("real_roof_closed")  # True / False / None
    roof_condition = ctx.get("roof_condition", "")
    venue_park = get_park(game.get("venue", "")) or {}
    venue_roof_type = venue_park.get("roof", "open")

    roof_label = ""
    if real_roof_closed is True:
        roof_label = f"🏟️ ROOF CLOSED ({roof_condition})" if roof_condition else "🏟️ ROOF CLOSED (MLB-confirmed)"
    elif real_roof_closed is False and venue_roof_type == "retractable":
        roof_label = "🏟️ ROOF OPEN (MLB-confirmed)"
    elif venue_roof_type == "dome":
        roof_label = "🏟️ DOME (always closed)"
    elif venue_roof_type == "retractable":
        roof_label = "🏟️ Retractable roof — status unknown, using temp guess"

    if hr_env_flag:
        combined_msg = hr_env_flag
        if day_night_label:
            combined_msg += f" · {day_night_label}"
        if roof_label:
            combined_msg += f" · {roof_label}"
        if hr_env_color == "success":
            st.success(combined_msg)
        elif hr_env_color == "info":
            st.info(combined_msg)
        elif hr_env_color == "warning":
            st.warning(combined_msg)
        elif hr_env_color == "error":
            st.error(combined_msg)
        else:
            st.caption(combined_msg)
    elif day_night_label or roof_label:
        st.caption(" · ".join(s for s in [day_night_label, roof_label] if s))

    # Pull-wind interaction summary — show prominently when wind significantly
    # helps or hurts a handedness side. User asked for easy-to-see flag.
    pull_summaries = ctx.get("pull_wind_summary", [])
    if pull_summaries:
        # Determine which sides are affected
        all_msgs = " · ".join(pull_summaries)
        lhb_helped = "RF" in all_msgs and "boost" in all_msgs.lower()
        rhb_helped = "LF" in all_msgs and "boost" in all_msgs.lower()
        lhb_hurt = "RF" in all_msgs and "suppress" in all_msgs.lower()
        rhb_hurt = "LF" in all_msgs and "suppress" in all_msgs.lower()
        if lhb_helped and rhb_helped:
            st.info(f"🌬️ **STRONG WIND — boosts BOTH LHB & RHB:** {all_msgs}")
        elif lhb_helped:
            st.info(f"🌬️ **Wind FAVORS LHB:** {all_msgs}")
        elif rhb_helped:
            st.info(f"🌬️ **Wind FAVORS RHB:** {all_msgs}")
        elif lhb_hurt and rhb_hurt:
            st.warning(f"🌬️ **WIND SUPPRESSES HRs both sides:** {all_msgs}")
        elif lhb_hurt:
            st.warning(f"🌬️ **Wind hurts LHB:** {all_msgs}")
        elif rhb_hurt:
            st.warning(f"🌬️ **Wind hurts RHB:** {all_msgs}")
        else:
            st.caption(f"🎯 **Pull-side wind:** {all_msgs}")

    # PITCHER GRADE BANNER — show each starter's base grade alongside the
    # env-adjusted grade (post park × weather). Helps user see when an ELITE
    # pitcher in Coors becomes TOUGH, or an EXPLOIT in Petco becomes MIXED.
    if not p_slate.empty and "grade" in p_slate.columns:
        a_pid = (ctx.get("away_p_row") or {}).get("player_id")
        h_pid = (ctx.get("home_p_row") or {}).get("player_id")
        env_mult_show = ctx.get("hr_mult", 1.0)

        def _pitcher_grade_str(pid, label, side_team):
            if pid is None or pd.isna(pid):
                return None
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                return None
            match = p_slate[p_slate["pitcher_id"] == pid_int]
            if match.empty:
                return None
            base = match.iloc[0].get("grade") or "—"
            env_adj = match.iloc[0].get("env_adj_grade") or base
            name = match.iloc[0].get("pitcher_name") or label
            if base == env_adj or env_adj == "—":
                return f"**{name}** ({side_team}): {base}"
            # Different — show transition with arrow
            return f"**{name}** ({side_team}): {base} → **{env_adj}** (env-adj)"

        away_str = _pitcher_grade_str(a_pid, game.get("away_pitcher", "TBD"),
                                        game.get("away_team_abbr", ""))
        home_str = _pitcher_grade_str(h_pid, game.get("home_pitcher", "TBD"),
                                        game.get("home_team_abbr", ""))
        pitcher_strs = [s for s in (away_str, home_str) if s]
        if pitcher_strs:
            st.caption(" · ".join(pitcher_strs) + f" · env_mult: {env_mult_show:.2f}×")

    info_cols = st.columns(3)
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
        st.metric(
            "Park × Wx", f"{ctx.get('hr_mult', 1.0):.2f}×",
            help="Park HR factor × weather multiplier. >1.0 = HR-friendly.",
        )
        # Show umpire flag if K-factor is meaningfully non-neutral (±3%+)
        ump_info = ctx.get("ump") or {}
        ump_k = ump_info.get("k_factor", 1.0)
        ump_name = ump_info.get("name")
        if ump_name and (ump_k >= 1.03 or ump_k <= 0.97):
            if ump_k >= 1.05:
                st.caption(f"👨‍⚖️ **HP: {ump_name}** · K-friendly ({ump_k:.2f}×)")
            elif ump_k >= 1.03:
                st.caption(f"👨‍⚖️ HP: {ump_name} · slight K boost ({ump_k:.2f}×)")
            elif ump_k <= 0.95:
                st.caption(f"👨‍⚖️ **HP: {ump_name}** · K-suppressing ({ump_k:.2f}×)")
            else:
                st.caption(f"👨‍⚖️ HP: {ump_name} · slight K dampener ({ump_k:.2f}×)")

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
                    # Show slate-leader flag if this player tops any category
                    leader_badge = ""
                    try:
                        _pid = top.get("player_id")
                        if _pid is not None and not pd.isna(_pid):
                            cats = slate_leader_pid_map.get(int(_pid), [])
                            if cats:
                                leader_badge = " 🏆"
                    except Exception:
                        pass
                    st.markdown(
                        f"**🎯 Best HR Play**: {alert} **{top['player_name']}**{leader_badge} "
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
                        # NA-safe: pandas Series .get(...) can return pd.NA
                        _hr_raw = sl.get("hr_game_pct", 0)
                        try:
                            hr_pct = 0 if pd.isna(_hr_raw) else float(_hr_raw)
                        except (TypeError, ValueError):
                            hr_pct = 0
                        try:
                            sc = 0 if pd.isna(sc) else float(sc)
                        except (TypeError, ValueError):
                            sc = 0
                        st.markdown(
                            f"**💎 Best Sleeper**: **{sl['player_name']}** "
                            f"({sl['_team']}) — sleeper {sc:.1f}, HR {hr_pct:.1f}%"
                        )

    tabs = st.tabs([away_tab, home_tab, "🎯 K Projections"])
    with tabs[0]:
        render_matchup_section(ctx.get("away_matchup"), game['away_team_abbr'])
        # Late-swap candidates from active roster but not in tonight's 9
        bench = ctx.get("away_bench_matchup")
        if bench is not None and not bench.empty:
            with st.expander(
                f"🔄 {game['away_team_abbr']} bench / possible late-swap candidates "
                f"({len(bench)} active-roster hitters not in tonight's 9)"
            ):
                st.caption(
                    "These hitters are on the active roster but NOT in the posted "
                    "starting lineup. If a player gets scratched late, one of these "
                    "could be the replacement. Stats shown so you're prepared. "
                    "**HR Game% NOT computed** — these aren't expected starters."
                )
                bench_cols = [c for c in [
                    "player_name", "position", "bats", "pa",
                    "barrel_pct", "iso", "xwoba", "home_run", "recent_hr",
                ] if c in bench.columns]
                st.dataframe(
                    bench[bench_cols].sort_values("pa", ascending=False, na_position="last"),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "player_name": st.column_config.TextColumn("Hitter"),
                        "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
                        "iso": st.column_config.NumberColumn("ISO", format="%.3f"),
                        "xwoba": st.column_config.NumberColumn("xwOBA", format="%.3f"),
                    },
                )
    with tabs[1]:
        render_matchup_section(ctx.get("home_matchup"), game['home_team_abbr'])
        bench = ctx.get("home_bench_matchup")
        if bench is not None and not bench.empty:
            with st.expander(
                f"🔄 {game['home_team_abbr']} bench / possible late-swap candidates "
                f"({len(bench)} active-roster hitters not in tonight's 9)"
            ):
                st.caption(
                    "Active-roster hitters NOT in the posted starting lineup. "
                    "If someone gets scratched late, one of these is the likely replacement."
                )
                bench_cols = [c for c in [
                    "player_name", "position", "bats", "pa",
                    "barrel_pct", "iso", "xwoba", "home_run", "recent_hr",
                ] if c in bench.columns]
                st.dataframe(
                    bench[bench_cols].sort_values("pa", ascending=False, na_position="last"),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "player_name": st.column_config.TextColumn("Hitter"),
                        "barrel_pct": st.column_config.NumberColumn("Brl%", format="%.1f%%"),
                        "iso": st.column_config.NumberColumn("ISO", format="%.3f"),
                        "xwoba": st.column_config.NumberColumn("xwOBA", format="%.3f"),
                    },
                )
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

    if not pitcher_arsenal_all.empty:
        with st.expander("🔬 Deep dive: Pitcher arsenal"):
            st.caption(
                "**What this shows:** the actual pitches each pitcher throws — "
                "usage %, velocity, xwOBA allowed per pitch type. "
                "Use this to see what pitches the pitcher leans on and which "
                "ones give up the most damage."
            )
            sub1, sub2 = st.columns(2)
            with sub1:
                h_pid = safe_int(game.get("home_pitcher_id"))
                if h_pid:
                    ars = pitcher_arsenal_all[pitcher_arsenal_all["player_id"] == h_pid]
                    if not ars.empty:
                        st.markdown(f"**{game.get('home_pitcher', 'TBD')} arsenal**")
                        st.dataframe(ars, hide_index=True, use_container_width=True)
            with sub2:
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
