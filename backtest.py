"""
Backtest framework - tracks model projections vs actual outcomes.

V1: in-memory snapshot via st.session_state and snapshot-fetch on demand.
The user clicks a button to evaluate yesterday's predictions and we
fetch actual outcomes from MLB Stats API to compare.

For long-term tracking across container restarts, this same data can be
mirrored to a Gist or GitHub repo (future enhancement).
"""

from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import requests

try:
    import streamlit as st
    _HAVE_ST = True
except ImportError:
    _HAVE_ST = False

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


# ----------------------------------------------------------------------------
# GITHUB GIST PERSISTENCE (v41c)
# ----------------------------------------------------------------------------
# Streamlit Cloud's local disk gets wiped on every container restart (every
# redeploy, every wake-from-sleep). Without a durable backend, snapshots
# evaporate. GitHub Gist solves this:
#   - Free, no S3/Supabase signup
#   - Survives every restart
#   - Simple JSON read/write
#   - User-owned (private gist; only their token can read/write)
#
# Setup (one-time, user-side):
#   1. Create GitHub PAT with `gist` scope
#   2. Create a secret gist with filename `dingermaven_snapshots.json`, content `{}`
#   3. Add to Streamlit Secrets:
#        gist_token = "ghp_..."
#        gist_id = "abc123..."
#
# Storage shape: ONE gist file holds ALL snapshots as a dict keyed by date.
# {"2026-06-08": {hitters: [...], pitchers: [...], saved_at: "..."}, "2026-06-09": ...}
# Single file means one HTTP write per save, easy to reason about, no
# pagination concerns.
# ----------------------------------------------------------------------------
GIST_FILENAME = "dingermaven_snapshots.json"
GIST_API = "https://api.github.com/gists"


def _gist_token() -> str | None:
    """Return GitHub PAT from Streamlit secrets, or None if not configured."""
    if not _HAVE_ST:
        return None
    try:
        return st.secrets.get("gist_token") or None
    except Exception:
        return None


def _gist_id() -> str | None:
    """Return target gist ID from Streamlit secrets, or None if not configured."""
    if not _HAVE_ST:
        return None
    try:
        return st.secrets.get("gist_id") or None
    except Exception:
        return None


def durable_storage_configured() -> bool:
    """True if Gist tier (survives redeploys) is set up.

    Local disk and /tmp are both wiped on container restart. Without Gist,
    "Saved snapshot" is technically true for the session but doesn't survive
    redeploys. This function lets the UI honestly tell users whether their
    saves are durable.
    """
    return bool(_gist_token()) and bool(_gist_id())


# v43.40: module-level diagnostic for the most recent gist read attempt.
# Lets the UI surface the EXACT error code (401=token, 404=gist gone,
# 429=rate limit) instead of the generic "network/auth/rate limit" message.
_LAST_GIST_READ_ERROR: str | None = None

def last_gist_read_error() -> str | None:
    """Returns the most recent gist read failure reason, or None if last read succeeded."""
    return _LAST_GIST_READ_ERROR


def _gist_read_all() -> dict | None:
    """Fetch the gist contents and return the parsed snapshot dict.

    Returns:
      dict (possibly empty) on success
      None on failure (network error, auth error, malformed JSON)

    v42i CRITICAL FIX: previously returned {} on ANY exception, which
    caused catastrophic data loss — _save_snapshot_to_gist would do:
        all_snaps = _gist_read_all()  # {} on transient failure
        all_snaps[key] = payload      # only the new snapshot remains
        _gist_write_all(all_snaps)    # writes {key: payload} → WIPES ALL PRIOR
    A single transient GitHub API hiccup could destroy weeks of accumulated
    backtest data. Now we return None on failure so callers can abort the
    write instead of silently wiping the gist.

    v43.40: Capture the specific HTTP status code into _LAST_GIST_READ_ERROR
    so the UI can show "401 Unauthorized — token expired" instead of the
    vague "network/auth/rate limit" diagnostic that left users guessing.
    """
    global _LAST_GIST_READ_ERROR
    _LAST_GIST_READ_ERROR = None

    token = _gist_token()
    gist_id = _gist_id()
    if not token:
        _LAST_GIST_READ_ERROR = "gist_token missing from Streamlit Secrets"
        return None
    if not gist_id:
        _LAST_GIST_READ_ERROR = "gist_id missing from Streamlit Secrets"
        return None
    try:
        r = requests.get(
            f"{GIST_API}/{gist_id}",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            timeout=10,
        )
        if r.status_code == 401:
            _LAST_GIST_READ_ERROR = (
                "HTTP 401 Unauthorized — GitHub token is invalid or EXPIRED. "
                "Regenerate at https://github.com/settings/tokens with `gist` "
                "scope, then update gist_token in Streamlit Secrets."
            )
            return None
        if r.status_code == 404:
            _LAST_GIST_READ_ERROR = (
                f"HTTP 404 Not Found — gist_id '{gist_id}' doesn't exist. "
                "The gist may have been deleted, or gist_id is wrong in Secrets. "
                "Verify at https://gist.github.com"
            )
            return None
        if r.status_code == 403:
            # Could be rate limit OR scope issue
            remaining = r.headers.get("X-RateLimit-Remaining", "?")
            reset = r.headers.get("X-RateLimit-Reset", "?")
            _LAST_GIST_READ_ERROR = (
                f"HTTP 403 Forbidden — rate limit or token-scope issue. "
                f"Rate limit remaining: {remaining}, resets at unix {reset}. "
                f"If remaining=0, wait. Otherwise: token may not have `gist` scope; regenerate it."
            )
            return None
        if r.status_code != 200:
            _LAST_GIST_READ_ERROR = (
                f"HTTP {r.status_code} from GitHub Gist API. "
                f"Response: {r.text[:200]}"
            )
            return None
        files = r.json().get("files", {})
        target = files.get(GIST_FILENAME)
        if not target:
            # Gist exists but doesn't have our file — first save scenario.
            # This is a legitimate "empty" state, not a read failure.
            return {}
        content = target.get("content", "")
        if not content.strip():
            return {}
        return json.loads(content)
    except requests.exceptions.Timeout:
        _LAST_GIST_READ_ERROR = "Network timeout to api.github.com (>10s)"
        return None
    except requests.exceptions.ConnectionError as e:
        _LAST_GIST_READ_ERROR = f"Network connection failed: {type(e).__name__}"
        return None
    except json.JSONDecodeError as e:
        # v43.75: AUTO-RECOVERY. Previously we returned None on corruption,
        # which forced the user to manually click a reset button. They've
        # been stuck in a loop where the reset doesn't seem to stick (likely
        # they're not finding the button, or it requires re-confirmation
        # each load). The right behavior: detect corruption, auto-reset to
        # {}, log it so we can see it happened, then proceed normally.
        # Data already lost to corruption is lost; no point making the user
        # do ceremony around that fact.
        _LAST_GIST_READ_ERROR = (
            f"Gist content is not valid JSON ({e}). "
            "v43.75 auto-recovery: wiped to empty and continuing. Prior "
            "snapshots were already unreachable due to corruption."
        )
        # Attempt auto-reset. If the reset itself fails (network/auth),
        # fall back to returning None so callers don't accidentally wipe.
        try:
            reset_payload = {
                "files": {
                    GIST_FILENAME: {"content": "{}"}
                }
            }
            rr = requests.patch(
                f"{GIST_API}/{gist_id}",
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                json=reset_payload,
                timeout=15,
            )
            if rr.status_code in (200, 201):
                # Recovery succeeded. Return empty dict so subsequent saves
                # write fresh and don't think they're wiping a populated gist.
                return {}
        except Exception:
            pass
        # Auto-reset failed; preserve previous behavior of returning None
        # to protect against unintended data wipes.
        return None
    except Exception as e:
        _LAST_GIST_READ_ERROR = f"Unexpected exception: {type(e).__name__}: {e}"
        return None


def _gist_write_all(snapshots: dict) -> bool:
    """Push the entire snapshot dict back to the gist. Returns True on success."""
    token = _gist_token()
    gist_id = _gist_id()
    if not token or not gist_id:
        return False
    try:
        payload = {
            "files": {
                GIST_FILENAME: {
                    # v43.74: compact JSON (no indent, tight separators) shrinks
                    # the payload by 30-40% vs indent=2. Combined with v43.74's
                    # 500K size target, this keeps us well under Gist's 1MB
                    # ceiling even with 10 days of snapshots + outcomes.
                    "content": json.dumps(
                        snapshots, default=str, separators=(",", ":")
                    ),
                }
            }
        }
        r = requests.patch(
            f"{GIST_API}/{gist_id}",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        r.raise_for_status()
        return True
    except Exception:
        return False


# v43.48 — gist size management
# ---------------------------------------------------------------------------
# Discovered after a "Expecting ':' delimiter at char 921600" error: the
# gist grew past GitHub's effective per-file limit and got TRUNCATED mid-
# write, leaving malformed JSON. After that, every read failed → every
# write aborted (correctly, per v42i data-loss protection) → user lost
# all snapshot persistence with no automatic recovery.
#
# Two fixes:
#   1. PRUNE old snapshots BEFORE writing, capping the gist at a safe size
#   2. Detect corruption (JSONDecodeError) on read and offer recovery
# ---------------------------------------------------------------------------

# GitHub gist soft cap is around 1 MB per file before truncation/perf issues.
# We target ~700 KB to leave headroom for one write to grow past the cap
# before pruning catches it.
_GIST_SIZE_TARGET_BYTES = 500_000  # v43.74: lowered from 700K. With compact
                                     # JSON (no indent), the typical snapshot
                                     # is ~40K bytes. 500K target = ~12 days
                                     # of headroom before pruning kicks in,
                                     # well below Gist's 1MB hard limit.

# Keep at minimum this many of the most recent snapshots, regardless of size.
# Below this, even if the gist is too large, don't prune (something else is
# wrong — better to surface the error than silently delete real data).
_MIN_SNAPSHOTS_TO_KEEP = 10  # v43.74: lowered from 14. Pattern Analysis needs
                              # ~7+ snapshots for trustworthy signal; 10 leaves
                              # buffer without bloating storage.


def _prune_snapshots_for_size(snapshots: dict) -> tuple[dict, int]:
    """v43.48: Drop oldest snapshots until serialized size is under target.

    Snapshot keys are date-hour strings like '2026-06-23T19' — sortable as
    strings. We drop oldest first, keeping the most recent N at minimum.

    Returns (pruned_dict, n_dropped).
    """
    if not snapshots or len(snapshots) <= _MIN_SNAPSHOTS_TO_KEEP:
        return snapshots, 0

    serialized = json.dumps(snapshots, default=str, separators=(",", ":"))
    if len(serialized.encode("utf-8")) <= _GIST_SIZE_TARGET_BYTES:
        return snapshots, 0

    # Sort keys oldest-first. Drop oldest one at a time until under target,
    # but never go below _MIN_SNAPSHOTS_TO_KEEP.
    sorted_keys = sorted(snapshots.keys())
    pruned = dict(snapshots)
    n_dropped = 0
    while len(pruned) > _MIN_SNAPSHOTS_TO_KEEP:
        # Drop the single oldest key
        oldest = sorted_keys[n_dropped]
        del pruned[oldest]
        n_dropped += 1
        # Check size again
        if len(json.dumps(pruned, default=str, separators=(",", ":")).encode("utf-8")) <= _GIST_SIZE_TARGET_BYTES:
            break

    return pruned, n_dropped


def emergency_reset_gist() -> tuple[bool, str]:
    """v43.48: Wipe the gist back to '{}' for recovery from corruption.

    Use case: gist content is invalid JSON (truncated mid-write, manually
    edited, etc.) — all reads fail forever, blocking future saves. This
    function resets the gist to an empty dict so saves can resume.

    Returns (ok, message). On success, all prior snapshots are LOST — only
    call this when the gist is unreadable and there's no other option.
    """
    token = _gist_token()
    gist_id = _gist_id()
    if not token or not gist_id:
        return False, "Gist not configured"
    try:
        payload = {
            "files": {
                GIST_FILENAME: {"content": "{}"}
            }
        }
        r = requests.patch(
            f"{GIST_API}/{gist_id}",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        if r.status_code in (200, 201):
            return True, "Gist reset to empty {}. Future saves should now succeed."
        return False, f"Reset failed with HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"Reset exception: {type(e).__name__}: {e}"


def _save_snapshot_to_gist(snapshot_date, payload: dict) -> tuple[bool, str | None]:
    """Upsert one snapshot into the gist. Read-modify-write the dict.

    v42i: CRITICAL — if the read fails (returns None), ABORT the write
    rather than wipe existing data. Better to skip this save and try again
    next time than to nuke accumulated history.

    v43.33: returns (ok, error_message) so callers can show WHY a write
    failed (token expired, gist deleted, rate limit, etc) instead of
    silently returning False and letting the UI claim 'durable' falsely.

    v43.40: error message now includes the specific HTTP status / cause
    from _gist_read_all so the user knows exactly what to fix.

    v43.48: prune old snapshots BEFORE writing to keep gist under
    ~700 KB. Prevents the corruption-from-truncation failure mode.
    """
    if not durable_storage_configured():
        return False, "Gist not configured (gist_token/gist_id missing in secrets)"
    try:
        all_snaps = _gist_read_all()
        if all_snaps is None:
            # Surface the SPECIFIC reason — token expired, gist gone, rate
            # limit, etc — rather than the generic vague string.
            specific = last_gist_read_error() or "unknown read failure"
            return False, f"Gist read failed — write aborted to protect existing data. Reason: {specific}"
        all_snaps[str(snapshot_date)] = payload

        # v43.48: prune oldest snapshots if total size would exceed ~700 KB.
        # Without this, the gist eventually exceeds GitHub's per-file limit
        # and gets truncated mid-write → invalid JSON → all subsequent reads
        # fail → snapshots stop persisting forever (only fixed by manual reset).
        pruned, n_dropped = _prune_snapshots_for_size(all_snaps)
        if n_dropped > 0:
            all_snaps = pruned

        ok = _gist_write_all(all_snaps)
        if not ok:
            return False, "Gist write API call returned non-2xx (check token permissions, rate limit)"
        # v43.48: report pruning so the user knows old snapshots were dropped
        if n_dropped > 0:
            return True, f"OK (pruned {n_dropped} oldest snapshot{'s' if n_dropped != 1 else ''} to stay under size limit)"
        return True, None
    except Exception as e:
        return False, f"Gist save exception: {type(e).__name__}: {e}"


def _load_snapshot_from_gist(snapshot_date) -> dict | None:
    """Fetch one snapshot from the gist, or None if not found / Gist not configured."""
    if not durable_storage_configured():
        return None
    try:
        all_snaps = _gist_read_all()
        if all_snaps is None:
            return None
        return all_snaps.get(str(snapshot_date))
    except Exception:
        return None


def _list_snapshots_from_gist() -> list[str]:
    """Return list of dates in the gist. Empty list if Gist not configured
    OR if the read failed (so caller doesn't mistake a network blip for
    'no snapshots exist')."""
    if not durable_storage_configured():
        return []
    try:
        result = _gist_read_all()
        if result is None:
            return []  # Read failed — pretend no data; do NOT touch gist
        return sorted(result.keys())
    except Exception:
        return []


def _snapshot_key_for_now(snapshot_date) -> str:
    """Generate a snapshot key keyed by date + ET HOUR.

    v42: multiple snapshots per day, one per hour. This solves the early-game
    problem — a 1 PM ET game needs lineup data captured BEFORE 1 PM. A
    single end-of-day snapshot is too late for that game.

    Key format: "2026-06-08T14" = 2 PM ET on June 8.
    Saves within the same hour overwrite each other (good — you wouldn't
    intentionally save twice in 30 minutes).

    Backward-compat: load_snapshot/list_snapshots still recognize the old
    "2026-06-08" format from snapshots saved before v42.
    """
    try:
        import pytz
        et_now = datetime.now(pytz.timezone("US/Eastern"))
    except Exception:
        # Fallback: assume ET is UTC-4 (EDT)
        et_now = datetime.utcnow() - timedelta(hours=4)
    return f"{snapshot_date}T{et_now.hour:02d}"


# v43.33: module-level status of the last save attempt so callers can show
# honest "did the durable tier work?" feedback without re-running the save.
_LAST_SAVE_STATUS: dict = {
    "local": False,
    "gist": False,
    "gist_error": None,
    "key": None,
}

def last_save_status() -> dict:
    """Returns details of the most recent save_snapshot call.

    Keys:
      local:      bool — wrote to local /tmp successfully (gets wiped on restart)
      gist:       bool — wrote to GitHub Gist successfully (durable)
      gist_error: str | None — why gist failed (if it did)
      key:        str — the snapshot key the last save used
    """
    return dict(_LAST_SAVE_STATUS)


def save_snapshot(snapshot_date, matchup_df: pd.DataFrame,
                    pitcher_slate_df: pd.DataFrame,
                    snapshot_key: str | None = None,
                    app_version: str | None = None,
                    calibration_constants: dict | None = None,
                    filter_bias_metadata: dict | None = None) -> bool:
    """
    Persist a slim version of today's projections.

    v42: supports multiple snapshots per day. By default, the snapshot is
    keyed by date + ET hour (so saves in different hours create new entries
    instead of overwriting). Pass an explicit snapshot_key to override this
    (e.g., for testing or manual labeling).

    v41c: writes to BOTH GitHub Gist (durable, survives redeploys) and local
    disk (fast read for current session). Returns True if EITHER succeeded.
    The Gist write is the one that matters for backtest persistence — local
    disk gets wiped on every Streamlit Cloud container restart.

    v43.7 (data continuity fix): now stores app_version + calibration_constants
    in each snapshot. Lets the rolling aggregator label metrics by version so
    you can correctly say "v42r got 40% hit rate, v43.5 got 38%" instead of
    blending all versions together. Without this tag, every model change
    invalidates the cumulative comparison silently.

    v43.43 (reviewer-validated bias-tagging): accepts filter_bias_metadata
    dict like {"hide_started_active": True, "n_filtered": 3,
    "dropped_gamepks": [...]}. Stored in payload so the backtest aggregator
    can detect snapshots that are missing afternoon games (biased calibration
    data) and either skip them or weight them appropriately.
    """
    try:
        # v42: key by date + ET hour. Lets a 1pm and 7pm snapshot coexist.
        key = snapshot_key or _snapshot_key_for_now(snapshot_date)
        path = SNAPSHOT_DIR / f"snapshot_{key}.json"

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
                # v43.4: 'game' is REQUIRED for the v43.3 pick_score diversity
                # validation to function. Without it, _diverse_top_n's max-2-
                # per-game cap silently never fires (g is None for every row),
                # which means the top10_pick_score metric was grading a list
                # that could stack 4 hitters from one game — not what users see.
                "game",
                "is_roster_fill",  # v42: track which snapshots had real lineups
                "is_bench",        # v43.12: bench / late-swap candidate flag
                "power_score", "hr_game_pct", "hr_pa_pct",
                "matchup", "sleeper_score", "barrel_pct", "iso",
                # v41 Patch 2: pick_score + per-component decomposition
                "pick_score",
                "ps_hr_game", "ps_matchup_opp", "ps_power", "ps_pitch_hr",
                "ps_form", "ps_sleeper", "ps_lift", "ps_env",
                "ps_bonus_lineup", "ps_bonus_platoon",
                "ps_bonus_recent_hr", "ps_penalty_il",
                # v43.74: critical for Pattern Analysis to read accumulated
                # snapshots and surface researcher-framework patterns. Without
                # these, the analysis section says "no data" even with
                # snapshots saved. Adds ~10 fields × 300 hitters = ~3000
                # additional values per snapshot; compact JSON keeps it small.
                "hr_score", "grade",
                "must_have_met", "must_have_total", "must_have_pass",
                "nuclear_met", "nuclear_total", "nuclear_grade",
                "hit_game_pct", "tb_game_pct",
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
            "key": key,
            "saved_at": datetime.utcnow().isoformat(),
            # v43.7: data continuity tags — lets the rolling aggregator
            # label/filter by version. None for backward compat with
            # snapshots saved before this field was added.
            "app_version": app_version,
            "calibration_constants": calibration_constants or {},
            # v43.43: filter bias metadata. When hide_started filtered out
            # games (e.g., afternoon games on an evening reload), this records
            # how many and which ones. Backtest aggregator can use this to
            # detect biased snapshots and either skip them or note the bias.
            # None / empty dict on snapshots taken before any games started.
            "filter_bias": filter_bias_metadata or {},
            "hitters": hitter_records,
            "pitchers": pitcher_records,
        }

        # Write to BOTH tiers. Gist is the durable one; local is fast access.
        local_ok = False
        gist_ok = False
        gist_error = None
        try:
            with open(path, "w") as f:
                json.dump(payload, f, default=str)
            local_ok = True
        except Exception as e:
            local_ok = False
            # If even the local write failed, record why for diagnostics
            if gist_error is None:
                gist_error = f"local also failed: {type(e).__name__}: {e}"

        if durable_storage_configured():
            # v43.33: gist save now returns (ok, error) so we can show the
            # user WHY the durable tier failed, not just that it did.
            gist_ok, gist_error_str = _save_snapshot_to_gist(key, payload)
            if not gist_ok:
                gist_error = gist_error_str
        else:
            gist_error = "Gist not configured in Streamlit Secrets"

        # v43.33: record the granular status so the UI can be honest about
        # which tier(s) actually succeeded. The bool return below preserves
        # back-compat with existing callers.
        _LAST_SAVE_STATUS["local"] = local_ok
        _LAST_SAVE_STATUS["gist"] = gist_ok
        _LAST_SAVE_STATUS["gist_error"] = gist_error
        _LAST_SAVE_STATUS["key"] = key

        return local_ok or gist_ok
    except Exception as e:
        _LAST_SAVE_STATUS["local"] = False
        _LAST_SAVE_STATUS["gist"] = False
        _LAST_SAVE_STATUS["gist_error"] = f"save_snapshot outer exception: {type(e).__name__}: {e}"
        return False


def save_snapshot_durable(snapshot_date) -> bool:
    """True iff the LAST save_snapshot call wrote to durable storage (Gist).

    UI uses this to show "✅ Saved (durable)" vs "⚠️ Saved (this session only)".
    Lightweight check: just verifies the Gist has this date right now.
    """
    if not durable_storage_configured():
        return False
    try:
        return str(snapshot_date) in _list_snapshots_from_gist()
    except Exception:
        return False


def load_snapshot(snapshot_key) -> dict | None:
    """Load a previously saved snapshot.

    v42: snapshot_key may be either:
      - "2026-06-08T14"  (date + hour, new format)
      - "2026-06-08"     (date only, legacy or "find best snapshot for date")

    For date-only input, returns the LATEST snapshot saved for that date
    (which is what most callers want for "what was our projection for
    that day"). Use load_snapshot_at_hour() for explicit hour selection.
    """
    key = str(snapshot_key)

    # If key is a date-only format, find the latest hourly snapshot for it
    if "T" not in key:
        all_keys = list_snapshots()
        # Filter to keys matching this date (either exact match or "DATE T HH")
        matching = [k for k in all_keys if k == key or k.startswith(key + "T")]
        if not matching:
            return None
        # Use the latest (highest hour for a given date)
        key = sorted(matching)[-1]

    # Try Gist first — it's authoritative across container restarts
    if durable_storage_configured():
        try:
            snap = _load_snapshot_from_gist(key)
            if snap:
                return snap
        except Exception:
            pass
    # Fallback to local disk (may be missing post-redeploy)
    try:
        path = SNAPSHOT_DIR / f"snapshot_{key}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_snapshot_before_hour(snapshot_date, target_hour_et: int) -> dict | None:
    """Load the snapshot taken closest to (but BEFORE) a given hour on a date.

    v42: used by evaluate_hitter_projections to grade each game against the
    snapshot that was current right before first pitch. A 1pm-game evaluation
    uses a snapshot taken at 11am or 12pm if one exists. A 7pm-game evaluation
    uses a 5pm or 6pm snapshot if available, falling back to the latest
    available snapshot from earlier in the day.

    Returns None if no snapshots before target_hour exist for this date.
    """
    date_str = str(snapshot_date)
    all_keys = list_snapshots()
    eligible = []
    for k in all_keys:
        if k.startswith(date_str + "T"):
            try:
                hour = int(k.split("T")[1])
                if hour < target_hour_et:
                    eligible.append((hour, k))
            except (ValueError, IndexError):
                continue
        elif k == date_str:
            # Legacy format — assume late-night (best guess)
            eligible.append((23, k))
    if not eligible:
        return None
    # Take the latest snapshot that's still before target_hour
    eligible.sort()
    _, best_key = eligible[-1]
    return load_snapshot(best_key)


def list_snapshots() -> list[str]:
    """Return list of snapshot keys we have (date-only and date+hour formats).

    v42: keys may be "2026-06-08" (legacy) or "2026-06-08T14" (hour-keyed).
    """
    keys = set()
    # Gist tier — durable storage
    if durable_storage_configured():
        try:
            keys.update(_list_snapshots_from_gist())
        except Exception:
            pass
    # Local tier — fast access for current session
    try:
        local_keys = [
            p.stem.replace("snapshot_", "")
            for p in SNAPSHOT_DIR.glob("snapshot_*.json")
        ]
        keys.update(local_keys)
    except Exception:
        pass
    return sorted(keys)


def list_snapshot_dates() -> list[str]:
    """Return list of unique DATES we have any snapshot for (collapses hours).

    Useful for UI that wants to show 'we have snapshots for these days' rather
    than the full hourly granularity.
    """
    dates = set()
    for k in list_snapshots():
        dates.add(k.split("T")[0])
    return sorted(dates)


# ----------------------------------------------------------------------------
# Actual outcome fetcher - what actually happened on a given date
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Actual outcome fetcher - what actually happened on a given date
# ----------------------------------------------------------------------------

def auto_attach_outcomes_to_past_snapshots(max_dates: int = 14) -> dict:
    """v43.70 — Daily self-improvement loop.

    For each snapshot whose date is in the past AND that doesn't yet have
    outcomes attached, fetch the actual outcomes from MLB Stats API and
    write them back to Gist. Caps at `max_dates` distinct dates per call
    to bound runtime and API usage.

    Designed to run automatically on app load. Idempotent — re-running
    is a no-op for snapshots that already have outcomes.

    Returns: {"n_processed": int, "n_attached": int, "errors": [str, ...]}

    This is the foundation of the user-requested "model improves itself
    daily" loop. After it runs, the Pattern Analysis section sees fresh
    data and can identify new patterns.
    """
    from datetime import date, datetime
    result = {"n_processed": 0, "n_attached": 0, "errors": []}
    try:
        snaps = _gist_read_all() or {}
        if not snaps:
            return result

        today = date.today()
        # Group snapshots by date
        snaps_by_date = {}
        for k, v in snaps.items():
            if not isinstance(v, dict):
                continue
            d_str = str(k).split("T")[0]
            snaps_by_date.setdefault(d_str, []).append(k)

        # Identify which dates need outcomes attached
        candidate_dates = []
        for d_str in sorted(snaps_by_date.keys()):
            try:
                d_obj = datetime.fromisoformat(d_str).date()
            except Exception:
                continue
            # Only process dates strictly in the past (not today, not future).
            # Games on the current date may still be in progress.
            if d_obj >= today:
                continue
            # Skip if every snapshot for this date ALREADY has outcomes
            already_done = all(
                snaps.get(k, {}).get("hitter_outcomes")
                for k in snaps_by_date[d_str]
            )
            if already_done:
                continue
            candidate_dates.append(d_str)

        # Process newest-first up to max_dates
        candidate_dates = sorted(candidate_dates, reverse=True)[:max_dates]

        any_updated = False
        for d_str in candidate_dates:
            try:
                h_out = fetch_hitter_outcomes(d_str)
                p_out = fetch_pitcher_outcomes(d_str)
                if not h_out and not p_out:
                    continue
                # Convert int keys to str for JSON safety in Gist payload
                h_out_str = {str(k): v for k, v in h_out.items()}
                p_out_str = {str(k): v for k, v in p_out.items()}

                for snap_key in snaps_by_date[d_str]:
                    payload = snaps.get(snap_key) or {}
                    if not isinstance(payload, dict):
                        continue
                    # Don't overwrite existing outcomes (idempotent)
                    if payload.get("hitter_outcomes"):
                        continue
                    payload["hitter_outcomes"] = h_out_str
                    payload["pitcher_outcomes"] = p_out_str
                    payload["outcomes_attached_at"] = datetime.utcnow().isoformat() + "Z"
                    snaps[snap_key] = payload
                    result["n_attached"] += 1
                    any_updated = True
                result["n_processed"] += 1
            except Exception as e:
                result["errors"].append(f"{d_str}: {type(e).__name__}: {e}")
                continue

        # Single Gist write at the end (avoid hammering API)
        if any_updated:
            _gist_write_all(snaps)
    except Exception as e:
        result["errors"].append(f"auto_attach outer: {type(e).__name__}: {e}")
    return result


def fetch_hitter_outcomes(target_date) -> dict:
    """
    For a given date, fetch which hitters homered and their game line.
    Returns {player_id: {"hr": int, "ab": int, "h": int, "k": int, "bb": int, "rbi": int}}

    v42g BUGFIX: accepts either a date-only key ("2026-06-08") OR an hourly
    snapshot key ("2026-06-08T17"). The hour portion is stripped before
    querying MLB Stats API. Without this fix, hourly snapshot keys were
    sent to MLB with `?date=2026-06-08T17` which the API rejects, returning
    no outcomes — explaining the "No actual outcomes returned" error.
    """
    # Strip hour portion if present (v42 hourly snapshots use "DATE T HH" format)
    target_date_str = str(target_date).split("T")[0]
    try:
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={target_date_str}&hydrate=team,probablePitcher"
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
                        # v43.70 (outcome tracker expansion): capture 2B/3B/R
                        # to derive total_bases for the 2+ bases prop backtest,
                        # and runs for run-scored analysis. Existing HR/H/K/BB
                        # unchanged.
                        doubles_ = int(bat.get("doubles") or 0)
                        triples_ = int(bat.get("triples") or 0)
                        hr_ = int(bat.get("homeRuns") or 0)
                        h_ = int(bat.get("hits") or 0)
                        singles_ = max(0, h_ - doubles_ - triples_ - hr_)
                        total_bases = singles_ + 2*doubles_ + 3*triples_ + 4*hr_
                        out[pid] = {
                            "hr": hr_,
                            "ab": int(bat.get("atBats") or 0),
                            "h": h_,
                            "k": int(bat.get("strikeOuts") or 0),
                            "bb": int(bat.get("baseOnBalls") or 0),
                            "rbi": int(bat.get("rbi") or 0),
                            # v43.70 additions
                            "doubles": doubles_,
                            "triples": triples_,
                            "total_bases": total_bases,
                            "runs": int(bat.get("runs") or 0),
                            # Convenience binary flags for the analysis section
                            "homered": hr_ > 0,
                            "got_hit": h_ > 0,
                            "got_2plus_bases": total_bases >= 2,
                            "got_3plus_bases": total_bases >= 3,
                        }
            except Exception:
                continue
    return out


def fetch_pitcher_outcomes(target_date) -> dict:
    """
    For a given date, fetch what each starting pitcher actually did.
    Returns {player_id: {"ip": float, "k": int, "er": int, "bb": int, "hr": int, "h": int}}

    v42g: same date-key fix as fetch_hitter_outcomes — strip the hour portion
    so hourly snapshot keys work.
    """
    target_date_str = str(target_date).split("T")[0]
    try:
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={target_date_str}&hydrate=team"
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
            # v43.3: extract pick_score + game so we can evaluate the actual
            # ranking the app ships (top 10 by pick_score with max-2-per-game),
            # not just the raw HR Game% list. Previously we were measuring a
            # different selection than the one users actually see.
            "pick_score": h.get("pick_score"),
            "game": h.get("game"),
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
        # probabilistic forecasts. Lower = better. v43.3: thresholds aligned
        # with the v42p app.py calibration. For HR/no-HR with ~14% base rate,
        # the BLIND baseline (always predicting base rate) gets Brier ≈ 0.12.
        # Useful interpretation: <0.09 excellent, <0.11 good, <0.13 decent
        # (ranking strong but probs slightly inflated), 0.13+ needs tuning.
        # Brier measures ABSOLUTE probability calibration, NOT ranking — a
        # model can have great ranking + mediocre Brier if probs are inflated.
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
    # v43.3 CRITICAL FIX: Previously this section graded `df_played.nlargest(10,
    # "hr_game_pct")` — but the APP actually ships top 10 by `pick_score` with
    # max-2-per-game diversity. So the backtest was measuring a different list
    # than the one users see. The headline "+25.9pp edge" was for hr_game_pct,
    # not pick_score, meaning everything pick_score adds (env_boost weighting,
    # convergence bonuses, platoon adjustments) was invisible to validation.
    # Fix: grade BOTH lists. hr_game_pct top 10 kept for backward compat;
    # pick_score top 10 added so we can finally measure the product we ship.
    def _diverse_top_n(df, score_col, n=10, max_per_game=2):
        """Select top-N by score_col, capping max_per_game per game (same
        diversity rule the live app applies). Returns a DataFrame of the
        selected rows in score order."""
        if score_col not in df.columns:
            return df.head(0)
        sorted_df = df.dropna(subset=[score_col]).sort_values(
            score_col, ascending=False
        )
        picked = []
        game_counts = {}
        for _, r in sorted_df.iterrows():
            g = r.get("game") if "game" in df.columns else None
            if g and game_counts.get(g, 0) >= max_per_game:
                continue
            picked.append(r)
            if g:
                game_counts[g] = game_counts.get(g, 0) + 1
            if len(picked) >= n:
                break
        return pd.DataFrame(picked) if picked else df.head(0)

    if "hr_game_pct" in df_played.columns and df_played["hr_game_pct"].notna().any():
        # Legacy hr_game_pct top 10 — kept for backward-compat tracking
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

    # v43.3: THE ONE THAT ACTUALLY MATTERS — pick_score top 10 with diversity.
    # This grades the ranking the live app ships.
    if "pick_score" in df_played.columns and df_played["pick_score"].notna().any():
        top10_ps = _diverse_top_n(df_played, "pick_score", n=10, max_per_game=2)
        if not top10_ps.empty:
            metrics["top10_pick_score_predictions"] = [
                {
                    "name": r["player_name"],
                    "pick_score": round(r.get("pick_score", 0), 1),
                    "hr_game_pct": round(r.get("hr_game_pct", 0), 1) if pd.notna(r.get("hr_game_pct")) else None,
                    "game": r.get("game"),
                    "homered": bool(r["homered"]),
                }
                for _, r in top10_ps.iterrows()
            ]
            metrics["top10_pick_score_hit_rate"] = round(
                top10_ps["homered"].sum() / len(top10_ps) * 100, 1
            )

    # v43.3: Confirmed-starter baseline. Previously the slate baseline was
    # actual_hr_rate over ALL matched hitters (including bench appearances) —
    # too generous, made any 10 strong starters look good. Use only confirmed
    # starters as the fair comparator.
    confirmed_starters = df_played[df_played.get("actual_ab", 0) >= 3] if "actual_ab" in df_played.columns else df_played
    if not confirmed_starters.empty:
        metrics["confirmed_starter_baseline_hr_rate"] = round(
            confirmed_starters["homered"].sum() / len(confirmed_starters) * 100, 1
        )

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


# ----------------------------------------------------------------------------
# ROLLING AGGREGATE — combine multiple days of snapshot+actuals into one
# season-summary metrics dict. The single biggest signal for model trust.
# ----------------------------------------------------------------------------

def _rolling_aggregate_uncached(snapshot_dates_key: str, max_days: int) -> dict:
    """Inner implementation. snapshot_dates_key is a stable cache-key string."""
    snapshot_dates = snapshot_dates_key.split(",") if snapshot_dates_key else None
    available = list_snapshots()
    if not available:
        return {"error": "No snapshots available"}
    if snapshot_dates is None or snapshot_dates == [""]:
        snapshot_dates = sorted(available)[-max_days:]

    n_snapshots = 0
    all_top10_rows = []           # legacy hr_game_pct top 10
    all_top10_ps_rows = []        # v43.8: pick_score top 10 (the one we ship)
    all_brier = []
    per_day = []

    for sd in snapshot_dates:
        snapshot = load_snapshot(sd)
        if not snapshot:
            continue
        hitter_actuals = fetch_hitter_outcomes(sd)
        if not hitter_actuals:
            continue
        h_metrics = evaluate_hitter_projections(snapshot, hitter_actuals)
        if not h_metrics or h_metrics.get("error"):
            continue
        n_snapshots += 1

        day_summary = {
            "date": sd,
            "hitters_played": h_metrics.get("hitters_who_played", 0),
            "actual_hrs": h_metrics.get("total_actual_hrs", 0),
            "slate_hr_rate_pct": h_metrics.get("actual_hr_rate_pct", 0),
            # Legacy metric (top 10 by raw hr_game_pct, no diversity)
            "top10_hit_rate_pct": h_metrics.get("top10_hr_hit_rate", 0),
            # v43.8: pick_score top 10 hit rate (the ranking the app SHIPS).
            # May be None for snapshots saved before v43.4 (no `game` field
            # meant diversity cap silently never fired).
            "top10_pick_score_hit_rate_pct": h_metrics.get("top10_pick_score_hit_rate"),
            "brier": h_metrics.get("brier_score"),
            # v43.19 (reviewer-validated): raw counts so the dominant-
            # partition aggregator can correctly sum across days. Previously
            # only rates were stored, and the v43.18 dominant block tried
            # to read fields like top10_picks_total / slate_hrs that don't
            # exist, silently making the whole partition feature a no-op.
            "top10_picks_total": len(h_metrics.get("top10_hr_predictions") or []),
            "top10_hrs_hit": sum(
                1 for e in (h_metrics.get("top10_hr_predictions") or [])
                if e.get("homered")
            ),
            "top10_ps_picks_total": len(h_metrics.get("top10_pick_score_predictions") or []),
            "top10_ps_hrs_hit": sum(
                1 for e in (h_metrics.get("top10_pick_score_predictions") or [])
                if e.get("homered")
            ),
            # v43.7: version + key calibration constant
            "app_version": snapshot.get("app_version"),
            "ps_env_weight": (
                snapshot.get("calibration_constants", {})
                        .get("ps_weights", {})
                        .get("ps_env")
                if snapshot.get("calibration_constants") else None
            ),
        }
        per_day.append(day_summary)
        if h_metrics.get("brier_score") is not None:
            all_brier.append(h_metrics["brier_score"])

        for entry in h_metrics.get("top10_hr_predictions", []):
            all_top10_rows.append({
                "date": sd,
                "name": entry.get("name"),
                "predicted_pct": entry.get("predicted_hr_pct"),
                "homered": int(entry.get("homered", False)),
            })
        # v43.8: also accumulate pick_score top 10 picks for the new headline
        for entry in h_metrics.get("top10_pick_score_predictions", []) or []:
            all_top10_ps_rows.append({
                "date": sd,
                "name": entry.get("name"),
                "pick_score": entry.get("pick_score"),
                "homered": int(entry.get("homered", False)),
            })

    if n_snapshots == 0:
        return {"error": "No snapshots had matched actuals"}

    # Legacy hr_game_pct top 10
    top10_total = len(all_top10_rows)
    top10_hr_count = sum(r["homered"] for r in all_top10_rows)
    top10_hr_rate = (top10_hr_count / top10_total * 100) if top10_total else 0

    # v43.8: pick_score top 10 (the ranking we actually ship)
    top10_ps_total = len(all_top10_ps_rows)
    top10_ps_hr_count = sum(r["homered"] for r in all_top10_ps_rows)
    top10_ps_hit_rate = (
        (top10_ps_hr_count / top10_ps_total * 100) if top10_ps_total else 0
    )
    # Days where pick_score actually produced a top 10 (needs game field)
    days_with_ps_data = len({r["date"] for r in all_top10_ps_rows})

    slate_rates = [d["slate_hr_rate_pct"] for d in per_day if d["slate_hr_rate_pct"]]
    avg_slate_rate = sum(slate_rates) / len(slate_rates) if slate_rates else 0

    days_with_hit = 0
    days_total = 0
    by_date = {}
    for r in all_top10_rows:
        by_date.setdefault(r["date"], []).append(r["homered"])
    for d, picks in by_date.items():
        days_total += 1
        if sum(picks) > 0:
            days_with_hit += 1
    any_hit_rate = (days_with_hit / days_total * 100) if days_total else 0

    # v43.8: VERSION-DRIFT WARNING. If the aggregate spans multiple values of
    # ps_env_weight (or multiple app_versions where the weight differs), the
    # numbers blend two different models. Surface this explicitly so users
    # know whether the aggregate is apples-to-apples or contains drift.
    versions_seen = sorted(set(
        d.get("app_version") for d in per_day if d.get("app_version")
    ))
    env_weights_seen = sorted(set(
        d.get("ps_env_weight") for d in per_day
        if d.get("ps_env_weight") is not None
    ))
    untagged_days = sum(1 for d in per_day if d.get("app_version") is None)
    # v43.18 (reviewer-validated): the warning approach was a workaround.
    # The real fix is to partition the aggregate by ps_env_weight so we
    # don't blend two different models in the headline number. This block
    # computes the SAME aggregate but filtered to the dominant env_weight
    # only, and surfaces both numbers (dominant-partition AND total) so
    # users see what "apples-to-apples" looks like.
    dominant_partition = None
    dominant_edge_legacy = None
    dominant_edge_ps = None
    dominant_brier = None
    dominant_n = None
    if len(env_weights_seen) > 1:
        # Find the env_weight with the most snapshot-days
        from collections import Counter
        weight_counts = Counter(
            d.get("ps_env_weight") for d in per_day
            if d.get("ps_env_weight") is not None
        )
        if weight_counts:
            dominant_partition = weight_counts.most_common(1)[0][0]
            dom_days = [
                d for d in per_day
                if d.get("ps_env_weight") == dominant_partition
            ]
            # v43.19 (reviewer-validated fix): use the correct day_summary
            # key names. The v43.18 version read top10_picks_total / slate_hrs
            # etc. which don't exist on day_summary, so every aggregate was
            # 0 and the partition feature was silently a no-op.
            dom_top10_total = sum(d.get("top10_picks_total", 0) for d in dom_days)
            dom_top10_hr = sum(d.get("top10_hrs_hit", 0) for d in dom_days)
            dom_top10_ps_total = sum(d.get("top10_ps_picks_total", 0) for d in dom_days)
            dom_top10_ps_hr = sum(d.get("top10_ps_hrs_hit", 0) for d in dom_days)
            # Slate baseline: weight per-day rates by hitters_played
            dom_total_hitters = sum(d.get("hitters_played", 0) for d in dom_days)
            dom_total_hrs = sum(d.get("actual_hrs", 0) for d in dom_days)
            if dom_total_hitters > 0:
                dom_slate_rate = dom_total_hrs / dom_total_hitters * 100
                dom_legacy_rate = (
                    dom_top10_hr / dom_top10_total * 100 if dom_top10_total else 0
                )
                dom_ps_rate = (
                    dom_top10_ps_hr / dom_top10_ps_total * 100 if dom_top10_ps_total else 0
                )
                dominant_edge_legacy = round(dom_legacy_rate - dom_slate_rate, 1)
                dominant_edge_ps = round(dom_ps_rate - dom_slate_rate, 1) if dom_top10_ps_total else None
                dominant_n = len(dom_days)
            # Mean brier within the partition — key is "brier" not "brier_score"
            dom_brier_vals = []
            for d in dom_days:
                b = d.get("brier")   # v43.19 fix: was "brier_score"
                if b is not None:
                    dom_brier_vals.append(b)
            if dom_brier_vals:
                dominant_brier = round(sum(dom_brier_vals) / len(dom_brier_vals), 4)

    version_drift_warning = None
    if len(env_weights_seen) > 1:
        version_drift_warning = (
            f"⚠️ AGGREGATE SPANS {len(env_weights_seen)} DIFFERENT ENV WEIGHTS "
            f"({env_weights_seen}). The env reweight (v42q→v42r) changed "
            f"how heavily park/weather drives ranking. Headline edge below "
            f"blends two different models. **Use the 'Dominant partition' "
            f"numbers below for apples-to-apples comparison** (env_weight="
            f"{dominant_partition}, {dominant_n} days)."
        )
    elif untagged_days > 0:
        version_drift_warning = (
            f"ℹ️ {untagged_days} of {len(per_day)} days are from pre-v43.7 "
            f"snapshots (no version tag). Can't verify calibration consistency "
            f"for those days."
        )

    return {
        "n_snapshots": n_snapshots,
        "snapshot_dates": sorted([d["date"] for d in per_day]),
        # Legacy fields (kept for backward-compat with banner / tweets)
        "top10_picks_total": top10_total,
        "top10_hrs_hit": top10_hr_count,
        "top10_hr_rate_pct": round(top10_hr_rate, 1),
        # v43.8: pick_score fields (the metric we should be tuning against)
        "top10_pick_score_picks_total": top10_ps_total,
        "top10_pick_score_hrs_hit": top10_ps_hr_count,
        "top10_pick_score_hit_rate_pct": round(top10_ps_hit_rate, 1) if top10_ps_total else None,
        "days_with_pick_score_data": days_with_ps_data,
        "slate_baseline_hr_rate_pct": round(avg_slate_rate, 1),
        # Edges for BOTH metrics — let user compare which ranking is better
        "edge_vs_slate_pp": round(top10_hr_rate - avg_slate_rate, 1),  # legacy
        "edge_vs_slate_pp_pick_score": (
            round(top10_ps_hit_rate - avg_slate_rate, 1) if top10_ps_total else None
        ),
        "days_with_any_top10_hit": days_with_hit,
        "days_total": days_total,
        "any_hit_rate_pct": round(any_hit_rate, 1),
        "brier_mean": round(sum(all_brier) / len(all_brier), 4) if all_brier else None,
        "per_day_summary": per_day,
        # v43.8: surface version state for the banner
        "versions_seen": versions_seen,
        "env_weights_seen": env_weights_seen,
        "untagged_days": untagged_days,
        "version_drift_warning": version_drift_warning,
        # v43.18: dominant-partition numbers — apples-to-apples comparison
        # when the aggregate spans multiple env_weights
        "dominant_env_partition": dominant_partition,
        "dominant_edge_legacy_pp": dominant_edge_legacy,
        "dominant_edge_ps_pp": dominant_edge_ps,
        "dominant_brier": dominant_brier,
        "dominant_n_days": dominant_n,
    }


# Apply Streamlit caching only if streamlit is available (i.e., we're inside
# the Streamlit runtime). Avoids ImportError if backtest.py is used standalone.
if _HAVE_ST:
    _rolling_aggregate_uncached_cached = st.cache_data(ttl=1800)(_rolling_aggregate_uncached)
else:
    _rolling_aggregate_uncached_cached = _rolling_aggregate_uncached


def rolling_aggregate_hitters(snapshot_dates: list[str] | None = None,
                                max_days: int = 30) -> dict:
    """
    Aggregate hitter projection accuracy across multiple snapshots.

    Returns a dict with:
      - n_snapshots: how many days included
      - total_hitters: total hitter-game observations
      - total_hrs: total HRs that happened
      - top10_hit_rate: % of days where at least one Top 10 pick homered
      - top10_hr_rate_pct: HR rate of all Top 10 picks across days
      - slate_hr_rate_pct: HR rate of all qualified hitters (comparison baseline)
      - power_score_bands: per-band cross-day HR rate
      - hr_pct_bands: per-band cross-day HR rate
      - brier_mean: average Brier score across days
      - per_day_summary: one row per snapshot with rate + count

    If snapshot_dates is None, uses all available snapshots up to max_days.

    CACHING (v32 update): aggregation can loop through 20+ snapshots each
    fetching same-day MLB API actuals. Was recomputing on every rerender of
    the Twitter "edge stat" expander or the Backtest panel. Now cached for
    30 minutes per (snapshot_dates, max_days) key.
    """
    # Stable cache key: comma-joined sorted dates, or empty string for "use all"
    cache_key = ",".join(sorted(snapshot_dates)) if snapshot_dates else ""
    return _rolling_aggregate_uncached_cached(cache_key, max_days)
