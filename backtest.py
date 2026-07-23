"""
Backtest framework - tracks model projections vs actual outcomes.

V1: in-memory snapshot via st.session_state and snapshot-fetch on demand.
The user clicks a button to evaluate yesterday's predictions and we
fetch actual outcomes from MLB Stats API to compare.

For long-term tracking across container restarts, this same data can be
mirrored to a Gist or GitHub repo (future enhancement).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import pandas as pd
import requests


def _today_et():
    """v44.98: ET-aware today. Server clock is UTC on Streamlit Cloud; after
    8 PM ET (00:00 UTC) a naive date.today() rolls to tomorrow, making today's
    in-progress slate look like a finished past day → partial outcomes attached.
    """
    try:
        import pytz
        return datetime.now(pytz.timezone("US/Eastern")).date()
    except Exception:
        _u = datetime.now(timezone.utc)
        _off = 4 if 3 <= _u.month <= 11 else 5
        return (_u.replace(tzinfo=None) - timedelta(hours=_off)).date()


try:
    import streamlit as st
    _HAVE_ST = True
except ImportError:
    st = None  # v45.48 (review): keep the name bound so guards don't NameError
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
#      (NOTE: the gist filename keeps the legacy "dingermaven" name on purpose —
#       it is a STORAGE KEY holding all snapshot/correlation history. Renaming it
#       would orphan the entire history. App branding is LaunchCast as of v45.31.)
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
# v44.60: games excluded from outcome extraction because MLB marked them
# postponed/suspended/cancelled (so their hitters aren't graded as "no HR").
_LAST_OUTCOME_EXCLUDED_GAMES: list = []

def last_gist_read_error() -> str | None:
    """Returns the most recent gist read failure reason, or None if last read succeeded."""
    return _LAST_GIST_READ_ERROR


def _gist_read_all_uncached() -> dict | None:
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
        # v44.54 (code review #8): parse the API ENVELOPE separately from the
        # content. If r.json() (the GitHub API response wrapper) fails — e.g. a
        # proxy/GitHub returns a 200 with a non-JSON body — that is NOT gist
        # corruption, and `content` would be unbound. Handling it here means the
        # content-parse auto-wipe below can never fire with unbound `content`
        # (which previously NameError'd → _content_len=0 → wrongly wiped intact
        # data). This keeps the destructive recovery scoped ONLY to a genuine
        # json.loads(content) failure on real, bound content.
        try:
            _envelope = r.json()
        except Exception:
            _LAST_GIST_READ_ERROR = (
                "GitHub API returned a 200 with a non-JSON envelope. NOT gist "
                "corruption and NOT wiping — data is intact on GitHub, just "
                "unreadable this run."
            )
            return None
        files = _envelope.get("files", {})
        target = files.get(GIST_FILENAME)
        if not target:
            # Gist exists but doesn't have our file — first save scenario.
            # This is a legitimate "empty" state, not a read failure.
            return {}
        # v43.88 CRITICAL: GitHub truncates file content over ~1MB in API
        # responses and sets truncated=true + a raw_url that serves the FULL
        # content (up to 10MB). Without this check, a >1MB gist came back as
        # cut-off JSON → JSONDecodeError → v43.75 auto-recovery misdiagnosed
        # truncation as corruption and WIPED THE GIST. This is how accumulated
        # snapshots (e.g. July 1 with outcomes) were destroyed.
        if target.get("truncated") and target.get("raw_url"):
            rr = requests.get(
                target["raw_url"],
                headers={"Authorization": f"token {token}"},
                timeout=20,
            )
            if rr.status_code == 200:
                content = rr.text
            else:
                _LAST_GIST_READ_ERROR = (
                    f"Gist file is truncated (>1MB) and raw_url fetch failed "
                    f"with HTTP {rr.status_code}. NOT wiping — data is intact "
                    f"on GitHub, just unreadable this run."
                )
                return None
        else:
            content = target.get("content", "")
        if not content.strip():
            return {}
        # v44.54: parse the CONTENT in its own try so the auto-wipe recovery
        # only ever sees a real json.loads(content) failure with bound content.
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # v43.88 CRITICAL GUARD: if the unparseable content is LARGE
            # (≥850KB), this is almost certainly GitHub-side truncation of a
            # >1MB file, NOT corruption. The full data is intact on GitHub.
            # Auto-wiping here (the v43.75 behavior) DESTROYS real data.
            # Only auto-wipe when the content is small — a genuinely mangled
            # small file is unrecoverable anyway.
            try:
                _content_len = len(content.encode("utf-8"))
            except Exception:
                # content somehow unusable — do NOT wipe on uncertainty
                _LAST_GIST_READ_ERROR = (
                    "Gist content unreadable for length check — NOT wiping "
                    "out of caution."
                )
                return None
            if _content_len >= 850_000:
                _LAST_GIST_READ_ERROR = (
                    f"Gist content ({_content_len:,} bytes) failed to parse — "
                    f"this is truncation-shaped (file likely >1MB), NOT corruption. "
                    f"NOT wiping. Data is intact on GitHub. ({e})"
                )
                return None
            # v43.75: AUTO-RECOVERY for genuinely corrupt SMALL content.
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
    except requests.exceptions.Timeout:
        _LAST_GIST_READ_ERROR = "Network timeout to api.github.com (>10s)"
        return None
    except requests.exceptions.ConnectionError as e:
        _LAST_GIST_READ_ERROR = f"Network connection failed: {type(e).__name__}"
        return None
    except Exception as e:
        _LAST_GIST_READ_ERROR = f"Unexpected exception: {type(e).__name__}: {e}"
        return None


# v44.57 (code review #12): _gist_read_all is called ~13× per script run
# (list_snapshots, load_snapshot, saves, discovery, custom metrics) — each was
# a separate HTTP GET to GitHub, adding latency and rate-limit exposure.
# Memoize the read for the duration of a run. The cache is invalidated whenever
# _gist_write_all succeeds (so post-write reads see fresh data). Returns a deep
# copy so callers that mutate the dict (all_snaps[key] = ...) can't corrupt the
# cached copy.
_GIST_READ_CACHE: dict | None = None
_GIST_READ_CACHE_SET = False


def _gist_read_all() -> dict | None:
    global _GIST_READ_CACHE, _GIST_READ_CACHE_SET
    if _GIST_READ_CACHE_SET:
        if _GIST_READ_CACHE is None:
            return None
        # v44.66 CRITICAL FIX: return a SHALLOW copy, not deepcopy. The prior
        # deepcopy cloned the ENTIRE accumulated snapshot dict (all days, all
        # player-games) on every read — ~13 reads/run — which spiked memory and
        # got worse as the gist grew, causing the app to be killed/rebooted more
        # often over time. Callers mutate only the TOP LEVEL (all_snaps[key] =
        # payload / del all_snaps[key]) before writing back, so a shallow copy
        # fully protects the cache from those mutations at O(keys) cost instead
        # of O(entire nested structure). The nested snapshot values are treated
        # as read-only by callers, which is the existing contract.
        return dict(_GIST_READ_CACHE)
    _result = _gist_read_all_uncached()
    # Only cache successful reads. A None (failure) is NOT cached — a transient
    # error shouldn't poison every subsequent read this run; let it retry.
    if _result is not None:
        _GIST_READ_CACHE = _result
        _GIST_READ_CACHE_SET = True
        return dict(_result)
    return None


def _invalidate_gist_read_cache():
    """Clear the memoized gist read (called after any successful write)."""
    global _GIST_READ_CACHE, _GIST_READ_CACHE_SET
    _GIST_READ_CACHE = None
    _GIST_READ_CACHE_SET = False


def _gist_write_all(snapshots: dict) -> bool:
    """Push the entire snapshot dict back to the gist. Returns True on success.

    v43.88 CRITICAL: this is now the single size-enforcement choke point.
    Previously pruning only ran inside save_snapshot — but the outcome-attach
    paths (auto_attach_outcomes, auto_run_pattern_discovery) call this
    function directly and were adding ~25KB of outcomes per snapshot with NO
    pruning. Combined with v43.83's expanded columns, the file blew past
    GitHub's ~1MB write-truncation threshold → truncated JSON → (pre-v43.88)
    auto-wipe. Every write now prunes to the target first, and if the payload
    STILL exceeds the hard cap, the write is REFUSED (protecting the intact
    data already on GitHub) rather than risking a truncated write.
    """
    token = _gist_token()
    gist_id = _gist_id()
    if not token or not gist_id:
        return False
    try:
        # Enforce the size budget on every write path.
        snapshots, _n_dropped = _prune_snapshots_for_size(snapshots)
        content = json.dumps(snapshots, default=str, separators=(",", ":"))
        if len(content.encode("utf-8")) > _GIST_HARD_CAP_BYTES:
            # Even after pruning we're over the hard cap — refuse rather
            # than send a write GitHub may truncate mid-file.
            return False
        payload = {
            "files": {
                GIST_FILENAME: {
                    "content": content,
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
        # v44.57: a successful write means the memoized read is now stale —
        # invalidate so any subsequent read this run refetches fresh data.
        _invalidate_gist_read_cache()
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
_GIST_SIZE_TARGET_BYTES = 700_000  # v43.88: prune-to target. Slimmed
                                     # snapshots (compact orientation, 3dp
                                     # rounding, starters-only) run ~150-200KB
                                     # each, so this holds ~3-4 snapshots.
                                     # Raw-snapshot history is intentionally a
                                     # short window — the durable learning
                                     # memory is _pattern_history (60 days of
                                     # daily correlations, a few KB each).

_GIST_HARD_CAP_BYTES = 900_000  # v43.88: never send a write above this.
                                 # GitHub truncates gist files near ~1MB
                                 # (observed mid-write truncation at char
                                 # ~921600 in v43.48 incident). A refused
                                 # write preserves the intact data already
                                 # on GitHub; a truncated write corrupts it.

# Keep at minimum this many of the most recent snapshots when pruning to the
# soft TARGET. The HARD cap overrides this — a gist that would exceed the
# hard cap gets pruned below this floor, because losing old snapshots is
# strictly better than a truncated write destroying everything.
_MIN_SNAPSHOTS_TO_KEEP = 4  # v43.88: lowered from 10. With per-snapshot size
                             # ~150-200KB, 10 could never fit under the cap —
                             # the old floor made pruning mathematically
                             # unable to succeed, guaranteeing oversize writes.

_MAX_SNAPSHOTS_PER_DATE = 2  # v43.88: hourly auto-saves can create 5+
                              # snapshots/day. For eval + pattern learning we
                              # need at most the last couple per date (latest
                              # lineups are the most accurate). Extra hourly
                              # duplicates are the first thing pruned.


def _prune_snapshots_for_size(snapshots: dict) -> tuple[dict, int]:
    """v43.88 rewrite: bound serialized size while protecting what matters.

    Order of operations:
      1. Internal metadata keys (leading "_", e.g. _pattern_history) are
         NEVER pruned — they're small and hold the 60-day learning memory.
      2. If over target: collapse hourly duplicates — keep only the latest
         _MAX_SNAPSHOTS_PER_DATE snapshots per date, dropping older hours.
      3. If still over target: drop whole oldest snapshots, but not below
         _MIN_SNAPSHOTS_TO_KEEP.
      4. If still over the HARD cap: keep dropping oldest below the floor —
         a short history beats a truncated write that destroys everything.

    Returns (pruned_dict, n_dropped).
    """
    if not snapshots:
        return snapshots, 0

    def _size(d: dict) -> int:
        return len(json.dumps(d, default=str, separators=(",", ":")).encode("utf-8"))

    if _size(snapshots) <= _GIST_SIZE_TARGET_BYTES:
        return snapshots, 0

    protected = {k: v for k, v in snapshots.items() if str(k).startswith("_")}
    real = {k: v for k, v in snapshots.items() if not str(k).startswith("_")}
    n_dropped = 0

    # Step 2: per-date hourly collapse (drop older hours first)
    by_date: dict = {}
    for k in sorted(real.keys()):
        by_date.setdefault(str(k).split("T")[0], []).append(k)
    for d_str, keys in by_date.items():
        # keys are sorted ascending; keep the LAST _MAX_SNAPSHOTS_PER_DATE
        for k in keys[:-_MAX_SNAPSHOTS_PER_DATE] if len(keys) > _MAX_SNAPSHOTS_PER_DATE else []:
            del real[k]
            n_dropped += 1
    merged = {**real, **protected}
    if _size(merged) <= _GIST_SIZE_TARGET_BYTES:
        return merged, n_dropped

    # Steps 3+4: drop whole oldest snapshots
    sorted_keys = sorted(real.keys())
    for oldest in sorted_keys:
        current = _size({**real, **protected})
        if current <= _GIST_SIZE_TARGET_BYTES:
            break
        if len(real) <= _MIN_SNAPSHOTS_TO_KEEP and current <= _GIST_HARD_CAP_BYTES:
            break  # under hard cap and at the floor — stop
        del real[oldest]
        n_dropped += 1

    return {**real, **protected}, n_dropped


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
            _invalidate_gist_read_cache()  # v44.57: gist wiped, cache stale
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
        et_now = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=4)
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


def snapshot_hitters(payload: dict) -> list[dict]:
    """v43.88: decode hitter rows from a snapshot payload, either format.

    New format (v43.88+): payload["hitters_compact"] = {"columns": [...],
    "rows": [[...], ...]} — column-oriented to avoid repeating key names
    per hitter (~470KB/snapshot of pure key repetition in the old format).
    Legacy format: payload["hitters"] = [ {col: val, ...}, ... ].

    Returns a list of dicts in all cases (possibly empty).
    """
    if not isinstance(payload, dict):
        return []
    compact = payload.get("hitters_compact")
    if isinstance(compact, dict) and compact.get("columns"):
        cols = compact["columns"]
        return [dict(zip(cols, row)) for row in (compact.get("rows") or [])]
    legacy = payload.get("hitters")
    if isinstance(legacy, list):
        return legacy
    return []


def build_pitcher_hr_analysis() -> dict:
    """v44.79: merge pitcher snapshot records with their actual HRs-allowed
    outcomes across all snapshots, then run the correlation analysis. Returns
    the pitcher_hr_allowed_analysis result (which pitcher factors predict HRs).

    Mirrors the hitter pattern-analysis path but for the pitcher side. Reads
    from the same gist snapshots; each snapshot's "pitchers" list holds the
    factors, and "pitcher_outcomes" holds actual HRs allowed keyed by pid.
    """
    try:
        from pattern_analysis import pitcher_hr_allowed_analysis
    except Exception:
        return {}
    snaps = _gist_read_all() or {}
    if not snaps:
        return {}
    rows = []
    for key, payload in snaps.items():
        if key.startswith("_") or not isinstance(payload, dict):
            continue
        pitchers = payload.get("pitchers") or []
        p_out_raw = payload.get("pitcher_outcomes")
        if not pitchers or not p_out_raw:
            continue
        # pitcher_outcomes is a JSON string keyed by pid → {hr: n, ...}
        try:
            import json as _json
            p_out = _json.loads(p_out_raw) if isinstance(p_out_raw, str) else p_out_raw
        except Exception:
            continue
        if not isinstance(p_out, dict):
            continue
        for rec in pitchers:
            if not isinstance(rec, dict):
                continue
            pid = rec.get("player_id")
            if pid is None:
                continue
            _o = p_out.get(str(pid)) or p_out.get(pid)
            if not isinstance(_o, dict):
                continue
            _hr = _o.get("hr", _o.get("actual_hr_allowed"))
            if _hr is None:
                continue
            _r = dict(rec)
            _r["actual_hr_allowed"] = _hr
            rows.append(_r)
    if not rows:
        return {"factors": [], "n": 0, "reliable": False}
    import pandas as _pd
    return pitcher_hr_allowed_analysis(_pd.DataFrame(rows))


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
        #
        # v43.88 (size crisis fix): three changes to keep the gist under
        # GitHub's ~1MB truncation threshold:
        #   1. ROUND floats to 3dp — full float64 reprs ("12.345678901234567")
        #      were ~17 bytes each across ~50 numeric columns.
        #   2. EXCLUDE bench rows — snapshots grade what the app ships
        #      (starters + fills). Bench nearly doubled row count.
        #   3. COMPACT column orientation ("hitters_compact": {"columns",
        #      "rows"}) — records orientation repeated ~800 bytes of key
        #      names PER HITTER (~470KB of pure key repetition per snapshot).
        # Readers use snapshot_hitters() below, which handles both formats.
        def _rounded(v):
            if isinstance(v, float):
                return round(v, 3)
            return v

        hitters_compact = None
        if matchup_df is not None and not matchup_df.empty:
            # v44.70 (user: analyze ALL players league-wide, not just starters/
            # top lists). Previously bench players were dropped from the
            # snapshot entirely, so a bench hitter who actually PLAYED (late
            # swap, injury replacement, DH) was never learned from. Now we keep
            # starters + bench, each carrying is_bench so downstream can still
            # segment. The outcome merge already skips any player with no game
            # outcome, so bench players who SAT are auto-excluded — but ones who
            # PLAYED contribute real league-wide pattern data.
            #
            # SIZE GUARD: including all bench (~150 rows) could push the gist
            # write toward its prune target and cost history days. So we cap the
            # snapshot at _SNAPSHOT_ROW_CAP rows, keeping all starters plus the
            # highest-pick_score bench players (the ones most likely to matter /
            # be swapped in). This widens the sample from starters-only to
            # ~league-wide while keeping the payload bounded and deterministic.
            _df = matchup_df
            _SNAPSHOT_ROW_CAP = 420
            if len(_df) > _SNAPSHOT_ROW_CAP:
                # keep all starters; fill remaining budget with top bench by pick_score
                if "is_bench" in _df.columns:
                    _starters = _df[~_df["is_bench"].fillna(False).astype(bool)]
                    _bench = _df[_df["is_bench"].fillna(False).astype(bool)]
                    _bench_budget = max(0, _SNAPSHOT_ROW_CAP - len(_starters))
                    if _bench_budget > 0 and not _bench.empty:
                        _sort_col = "pick_score" if "pick_score" in _bench.columns else None
                        if _sort_col:
                            _bench = _bench.sort_values(_sort_col, ascending=False)
                        _bench = _bench.head(_bench_budget)
                        _df = pd.concat([_starters, _bench], ignore_index=True)
                    else:
                        _df = _starters
                else:
                    _df = _df.head(_SNAPSHOT_ROW_CAP)
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
                "matchup", "sleeper_score", "convergence_count", "barrel_pct", "iso",
                # v41 Patch 2: pick_score + per-component decomposition
                "pick_score",
                "ps_hr_game", "ps_matchup_opp", "ps_power", "ps_pitch_hr",
                "ps_form", "ps_sleeper", "ps_lift", "ps_env",
                "ps_discipline", "ps_bonus_recent_hr", "ps_bonus_platoon",
                "ps_bonus_lineup", "ps_penalty_il",  # v44.77: all pick_score parts
                # v43.74: critical for Pattern Analysis to read accumulated
                # snapshots and surface researcher-framework patterns. Without
                # these, the analysis section says "no data" even with
                # snapshots saved. Adds ~10 fields × 300 hitters = ~3000
                # additional values per snapshot; compact JSON keeps it small.
                "hr_score", "grade",
                "must_have_met", "must_have_total", "must_have_pass",
                "nuclear_met", "nuclear_total", "nuclear_grade",
                "hit_game_pct", "expected_total_bases", "tb_pa",
                # v43.83 (user-requested self-improvement): the pattern-
                # discovery loop needs the FULL predictor set to measure
                # which features actually predict outcomes. Previously only
                # barrel_pct + iso were snapshotted → correlation analysis
                # was blind to hard_hit, avg_ev, blast_pct, pull metrics,
                # xwoba, slg, etc. These are the actual candidate features
                # for the adaptive score. Compact JSON keeps the payload
                # under the Gist size budget even with these additions.
                "avg_ev", "hard_hit", "blast_pct", "blast_pct_real",
                "pull_pct", "pull_air_pct", "pulled_brl_pct",
                "fb_pct", "gb_pct", "ld_pct",
                "xwoba", "xslg", "slg", "obp", "ops",
                "k_pct", "bb_pct", "whiff_pct",
                "discipline_score", "lift_score", "matchup_opp",
                "recent_hr", "recent_hr_weighted_rate",
                "pitch_hr_score", "pitch_match_score",
                "lineup_confirmed",  # v43.93: segment confirmed-vs-projected accuracy
                "is_home", "is_day_game",  # v44.67: home/away + day/night segmentation
                # Environment for env-boosted correlations
                "env_boost", "opp_pitcher_xwoba",
                # v44.74: matchup-quality signals so we can measure whether a
                # good matchup lifts HR rate WITHIN a profile tier (user: do
                # individual matchups actually move outcomes?).
                "opp_pitcher_grade", "smash_spot",
                "dinger_score",  # v44.11: grade Dinger Score
                "power_composite",  # v44.32: grade HR+Dinger composite
                "barrel_matchup_score",  # v44.34: grade new metric
                "two_way_matchup_score",  # v44.43: grade two-way metric
                "is_moonshot_target", "is_laser_target",  # v44.18: grade power targets
                "moonshot_score", "laser_score",  # v44.78: slate-wide power-type scores
            ] if c in matchup_df.columns]
            # v45.70 (CRITICAL, structural): DERIVE the tail of the
            # whitelist from HR_CANDIDATE_FEATURES. A feature can be added
            # to the tracked-candidates list and still be silently stripped
            # here — that is exactly what happened to ctx_lift_pp (tracked
            # since v45.49, never snapshotted, so it could never appear in
            # Section G). Deriving means any future tracked feature is
            # snapshotted automatically and this class of bug cannot recur.
            try:
                from pattern_analysis import HR_CANDIDATE_FEATURES as _hcf
                for _c in _hcf:
                    if _c not in keep_cols and _c in matchup_df.columns:
                        keep_cols.append(_c)
            except Exception:
                pass
            # Build compact column-oriented storage with rounded values
            _sub = _df[keep_cols].where(_df[keep_cols].notna(), None)
            hitters_compact = {
                "columns": keep_cols,
                "rows": [
                    [_rounded(v) if v is not None and v == v else None for v in row]
                    for row in _sub.itertuples(index=False, name=None)
                ],
            }

        pitcher_records = []
        if pitcher_slate_df is not None and not pitcher_slate_df.empty:
            keep_cols = [c for c in [
                "player_id", "pitcher_name", "team", "opp",
                "test_score", "kHR", "hr_suppress",
                "proj_k", "role", "reliability",
                # v44.79: HR-vulnerability factors so we can grade whether they
                # predict HRs the pitcher actually allows. These are the pitcher
                # side of HR prediction — the foundation for folding pitcher
                # matchup into the combined metric.
                "grade", "barrel_allowed", "xwoba_allowed", "fb_allowed",
                "hard_hit_allowed", "hr_per_9", "slg_allowed",
                "whiff_pct", "k_pct", "csw_pct",
            ] if c in pitcher_slate_df.columns]
            _psub = pitcher_slate_df[keep_cols].where(
                pitcher_slate_df[keep_cols].notna(), None
            )
            pitcher_records = [
                {k: _rounded(v) if v is not None and v == v else None
                 for k, v in rec.items()}
                for rec in _psub.to_dict("records")
            ]

        payload = {
            "date": str(snapshot_date),
            "key": key,
            "saved_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
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
            # v43.88: compact column-oriented hitters. Readers must use
            # snapshot_hitters(payload) which decodes both this and the
            # legacy "hitters" records list.
            "hitters_compact": hitters_compact,
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


def list_snapshots() -> list[str]:
    """Return list of snapshot keys we have (date-only and date+hour formats).

    v42: keys may be "2026-06-08" (legacy) or "2026-06-08T14" (hour-keyed).

    v43.87: EXCLUDE keys that start with "_" (like "_pattern_history",
    "_LAST_SAVE_STATUS", etc.) — these are internal Gist-level metadata
    buckets, not snapshots. Previously they leaked through and downstream
    code that tried to parse them as dates crashed with NaT comparison
    errors (yesterday's-results banner).
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
    # v43.87: filter internal metadata keys
    return sorted(k for k in keys if not str(k).startswith("_"))


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

        today = _today_et()  # v44.98: ET-aware
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
                    # v44.66: shallow-copy the payload before mutating so we
                    # don't touch the shared cached object (the read is now a
                    # shallow copy for memory reasons). The write below
                    # invalidates the cache regardless, but this keeps the cache
                    # clean in the interim.
                    payload = dict(payload)
                    payload["hitter_outcomes"] = h_out_str
                    payload["pitcher_outcomes"] = p_out_str
                    payload["outcomes_attached_at"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"
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


# ============================================================================
# v43.83 — Daily automated pattern discovery
# ============================================================================
def auto_run_pattern_discovery() -> dict:
    """Run today's correlation analysis and append to the rolling history.

    v43.85 (bug fix): use list_snapshots()/load_snapshot() which check BOTH
    Gist AND local disk, instead of _gist_read_all() alone. Previously if
    the user's Gist was empty (fresh reset) but local disk had snapshots
    from the current session, my discovery reported "no snapshots" while
    the yesterday's-results banner (which uses load_snapshot's multi-tier
    lookup) worked fine. Now they use the same discovery path.

    v43.84: fetch outcomes on demand from MLB Stats API for any past-date
    snapshot without stored outcomes, then persist them back.

    Returns detailed diagnostic:
        {"date", "computed", "n_features", "note",
         "n_snapshots_total", "n_from_gist", "n_from_local",
         "n_with_stored_outcomes", "n_fetched_live",
         "gist_configured", "gist_error"}
    """
    from datetime import date, datetime
    result = {
        "date": str(date.today()), "computed": False,
        "n_features": 0, "note": "",
        "n_snapshots_total": 0,
        "n_from_gist": 0, "n_from_local": 0,
        "n_with_stored_outcomes": 0, "n_fetched_live": 0,
        "gist_configured": durable_storage_configured(),
        "gist_error": None,
    }
    try:
        # ------- Enumerate snapshots from BOTH tiers -------
        try:
            gist_keys = _list_snapshots_from_gist() if result["gist_configured"] else []
            result["n_from_gist"] = len(gist_keys)
        except Exception as ge:
            gist_keys = []
            result["gist_error"] = f"{type(ge).__name__}: {ge}"
        # If Gist read failed, capture the reason
        if result["gist_configured"] and not gist_keys:
            _err = last_gist_read_error()
            if _err:
                result["gist_error"] = _err

        # Local tier
        try:
            local_keys = []
            if SNAPSHOT_DIR.exists():
                local_keys = [
                    p.stem.replace("snapshot_", "")
                    for p in SNAPSHOT_DIR.glob("snapshot_*.json")
                ]
            result["n_from_local"] = len(local_keys)
        except Exception:
            local_keys = []

        all_keys = sorted(set(gist_keys) | set(local_keys))
        # Filter out the special internal keys (history bucket, save-status, etc.)
        all_keys = [k for k in all_keys if not str(k).startswith("_")]
        result["n_snapshots_total"] = len(all_keys)
        if not all_keys:
            if result["gist_error"]:
                result["note"] = (
                    f"No snapshots found. Gist error: {result['gist_error']}. "
                    f"Local disk: 0 files. Save today's slate to start "
                    f"accumulating history."
                )
            else:
                result["note"] = (
                    f"No snapshots found (Gist configured: {result['gist_configured']}, "
                    f"local: 0 files). Once you run a slate, this loop will "
                    f"start accumulating correlation history on subsequent days."
                )
            return result

        today = _today_et()  # v44.98: ET-aware
        # Load each snapshot payload (multi-tier via load_snapshot)
        # and separate has-outcomes vs needs-fetch (past dates)
        with_stored_outcomes = {}
        needs_live_fetch = {}  # date_str -> list[snap_key]
        skipped_future = 0
        for k in all_keys:
            d_str = str(k).split("T")[0]
            try:
                d_obj = datetime.fromisoformat(d_str).date()
            except Exception:
                continue
            payload = load_snapshot(k)
            if not isinstance(payload, dict):
                continue
            if payload.get("hitter_outcomes"):
                with_stored_outcomes[k] = payload
            elif d_obj < today:
                needs_live_fetch.setdefault(d_str, []).append(k)
            else:
                # today or future — games not final, can't fetch outcomes yet
                skipped_future += 1

        result["n_with_stored_outcomes"] = len(with_stored_outcomes)

        # Fetch outcomes for past-date snapshots that don't have them stored
        enriched = dict(with_stored_outcomes)
        # Track newly-fetched-and-attached-to-Gist snapshots for a single write
        gist_updates = {}
        for d_str, snap_keys in needs_live_fetch.items():
            try:
                h_out = fetch_hitter_outcomes(d_str)
                if not h_out:
                    continue
                h_out_str = {str(pid): row for pid, row in h_out.items()}
                for snap_key in snap_keys:
                    payload = load_snapshot(snap_key) or {}
                    payload = dict(payload)
                    payload["hitter_outcomes"] = h_out_str
                    payload["outcomes_attached_at"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"
                    enriched[snap_key] = payload
                    gist_updates[snap_key] = payload
                    result["n_fetched_live"] += 1
            except Exception:
                continue

        # Persist newly-attached outcomes back to Gist so tomorrow's run finds them
        # (only if Gist is configured — no-op otherwise, which is fine for this session)
        if gist_updates and result["gist_configured"]:
            try:
                snaps = _gist_read_all() or {}
                for k, payload in gist_updates.items():
                    snaps[k] = payload
                _gist_write_all(snaps)
            except Exception:
                pass

        if not enriched:
            _future_note = (
                f" ({skipped_future} snapshot(s) are today/future — games not final yet)"
                if skipped_future else ""
            )
            result["note"] = (
                f"Found {result['n_snapshots_total']} snapshot(s){_future_note} "
                f"but none had outcomes fetchable from MLB Stats API. "
                f"Try again later — some game results may not be posted yet."
            )
            return result

        # Compute correlations
        from pattern_analysis import (
            merge_snapshots_with_outcomes,
            compute_daily_correlations,
        )
        merged = merge_snapshots_with_outcomes(enriched)
        if merged.empty:
            result["note"] = (
                f"Merged {len(enriched)} snapshot(s) but yielded no player-game rows. "
                f"Older snapshots may be missing the feature columns v43.83 needs "
                f"(barrel_pct, iso, avg_ev, blast_pct, etc.). Newer snapshots will "
                f"populate correlation history."
            )
            return result

        # v43.96 (user-prompted redesign): bank ONE entry PER SLATE DATE,
        # not one pooled entry per run-date. Two wins:
        #   1. BACKFILL — every slate with outcomes still in storage gets
        #      its history entry (re)computed on every run. Days lost to
        #      the July-2 storage wipe are recovered automatically as long
        #      as their snapshots exist; days never banked (like July 3's
        #      empty morning) get banked the moment their data appears.
        #   2. STATISTICS — pooled-cumulative entries were autocorrelated
        #      (each contained all prior data), artificially deflating the
        #      std in Section G and inflating "reliability." Per-slate
        #      entries are independent observations — the correct unit.
        # Entries for dates older than raw-snapshot retention keep their
        # last banked value untouched, so history still outlives the
        # short raw window by design.
        new_entries = []
        for d_str, day_df in merged.groupby("snapshot_date"):
            day_corrs = compute_daily_correlations(day_df, "homered")
            if day_corrs.get("correlations"):
                day_corrs["date_computed"] = str(d_str)  # slate date, not run date
                day_corrs["basis"] = "slate"
                new_entries.append(day_corrs)

        # v44.14 — PARALLEL PROP LOOPS: compute the same per-slate feature
        # correlations against HITS (got_hit) and 2+ TOTAL BASES
        # (got_2plus_bases), banked under separate history keys. This lets the
        # app learn which stats predict a HIT or an extra-base day — often a
        # better play than a HR, since a hitter reaches base far more often
        # than he homers. Same independent-per-slate structure as the HR loop.
        _prop_loops = {
            "_pattern_history_hits": "got_hit",
            "_pattern_history_tb": "got_2plus_bases",
        }
        _prop_new_entries = {k: [] for k in _prop_loops}
        for _hist_key, _outcome in _prop_loops.items():
            if _outcome not in merged.columns:
                continue
            for d_str, day_df in merged.groupby("snapshot_date"):
                _c = compute_daily_correlations(day_df, _outcome)
                if _c.get("correlations"):
                    _c["date_computed"] = str(d_str)
                    _c["basis"] = "slate"
                    _c["outcome"] = _outcome
                    _prop_new_entries[_hist_key].append(_c)

        if not new_entries:
            result["note"] = (
                f"Merged {len(merged)} rows but no slate-date had ≥30 valid "
                f"pairs for any feature. Accumulate more slates."
            )
            return result

        today_corrs = new_entries[-1]  # newest slate, for result reporting

        # Persist: replace any existing entries for recomputed dates, keep rest
        if result["gist_configured"]:
            try:
                snaps = _gist_read_all() or {}
                history = snaps.get("_pattern_history") or []
                if not isinstance(history, list):
                    history = []
                _recomputed = {e["date_computed"] for e in new_entries}
                history = [
                    h for h in history
                    if h.get("date_computed") not in _recomputed
                ]
                history.extend(new_entries)
                history = sorted(history, key=lambda h: h.get("date_computed", ""))
                history = history[-60:]
                snaps["_pattern_history"] = history
                # v44.14: persist the parallel prop loops the same way
                for _hk, _entries in _prop_new_entries.items():
                    if not _entries:
                        continue
                    _ph = snaps.get(_hk) or []
                    if not isinstance(_ph, list):
                        _ph = []
                    _rec = {e["date_computed"] for e in _entries}
                    _ph = [h for h in _ph if h.get("date_computed") not in _rec]
                    _ph.extend(_entries)
                    _ph = sorted(_ph, key=lambda h: h.get("date_computed", ""))[-60:]
                    snaps[_hk] = _ph
                _gist_write_all(snaps)
                result["n_history_days"] = len(history)
            except Exception:
                # Non-fatal — session still has computed entries
                pass
        else:
            # Gist not configured — stash in-session so at least this run's
            # analysis is visible in the UI
            try:
                import streamlit as st  # noqa
                if hasattr(st, "session_state"):
                    _hist = st.session_state.get("_pattern_history_local", [])
                    _rec = {e["date_computed"] for e in new_entries}
                    _hist = [h for h in _hist if h.get("date_computed") not in _rec]
                    _hist.extend(new_entries)
                    _hist = sorted(_hist, key=lambda h: h.get("date_computed", ""))[-60:]
                    st.session_state["_pattern_history_local"] = _hist
                    # v44.14: parallel prop loops in session fallback
                    for _hk, _entries in _prop_new_entries.items():
                        if not _entries:
                            continue
                        _ph = st.session_state.get(f"{_hk}_local", [])
                        _rec = {e["date_computed"] for e in _entries}
                        _ph = [h for h in _ph if h.get("date_computed") not in _rec]
                        _ph.extend(_entries)
                        _ph = sorted(_ph, key=lambda h: h.get("date_computed", ""))[-60:]
                        st.session_state[f"{_hk}_local"] = _ph
            except Exception:
                pass

        result["computed"] = True
        result["n_features"] = len(today_corrs["correlations"])
        result["n_samples"] = today_corrs["n_samples"]
        _live_note = (
            f", fetched {result['n_fetched_live']} live"
            if result['n_fetched_live'] > 0 else ""
        )
        _durability = (
            "persisted to Gist"
            if result["gist_configured"]
            else "⚠️ session-only (Gist not configured)"
        )
        result["note"] = (
            f"Banked/updated {len(new_entries)} slate-day correlation "
            f"entr{'y' if len(new_entries) == 1 else 'ies'} "
            f"({', '.join(e['date_computed'] for e in new_entries)}) from "
            f"{len(enriched)} snapshot(s) "
            f"({result['n_with_stored_outcomes']} stored{_live_note}); "
            f"{result['n_features']} features tracked — {_durability}. "
            f"History: {result.get('n_history_days', '?')} day(s)."
        )
    except Exception as e:
        result["note"] = f"Pattern discovery failed: {type(e).__name__}: {e}"
    return result


# ============================================================================
# v44.08 — user-defined custom metrics (saved grade formulas)
# ============================================================================
# Lets the user name and SAVE a custom weighted-metric formula from the
# grade builder, so it persists across sessions and can be re-applied. Stored
# under "_custom_metrics" in the Gist — the leading underscore makes it
# pruning-exempt (see _prune_snapshots_for_size), so a saved formula is never
# dropped to make room for snapshots. Formulas are tiny (a name + a
# {column: weight} dict), so dozens fit in a few KB.
def save_custom_metric(name: str, weights: dict) -> bool:
    """Persist a named custom-metric formula. Returns True on success.

    Args:
        name: user-chosen label (e.g. "My Power Blend").
        weights: {column_name: weight} the builder used.
    """
    if not name or not isinstance(weights, dict) or not weights:
        return False
    try:
        entry = {
            "name": str(name)[:60],
            "weights": {str(k): float(v) for k, v in weights.items()},
            "saved_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        }
        if durable_storage_configured():
            snaps = _gist_read_all() or {}
            metrics = snaps.get("_custom_metrics") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            metrics[entry["name"]] = entry
            # cap at 25 saved formulas to bound size
            if len(metrics) > 25:
                # drop oldest by saved_at
                ordered = sorted(metrics.values(), key=lambda e: e.get("saved_at", ""))
                for old in ordered[:len(metrics) - 25]:
                    metrics.pop(old["name"], None)
            snaps["_custom_metrics"] = metrics
            return _gist_write_all(snaps)
        # session fallback
        import streamlit as st  # noqa
        if hasattr(st, "session_state"):
            m = st.session_state.get("_custom_metrics_local", {})
            m[entry["name"]] = entry
            st.session_state["_custom_metrics_local"] = m
            return True
    except Exception:
        pass
    return False


def load_custom_metrics() -> dict:
    """Return all saved custom-metric formulas as {name: entry}. Empty if none."""
    try:
        if durable_storage_configured():
            snaps = _gist_read_all() or {}
            metrics = snaps.get("_custom_metrics") or {}
            if isinstance(metrics, dict) and metrics:
                return metrics
    except Exception:
        pass
    try:
        import streamlit as st  # noqa
        if hasattr(st, "session_state"):
            return st.session_state.get("_custom_metrics_local", {}) or {}
    except Exception:
        pass
    return {}


def delete_custom_metric(name: str) -> bool:
    """Remove a saved custom-metric formula by name."""
    try:
        if durable_storage_configured():
            snaps = _gist_read_all() or {}
            metrics = snaps.get("_custom_metrics") or {}
            if name in metrics:
                metrics.pop(name)
                snaps["_custom_metrics"] = metrics
                return _gist_write_all(snaps)
        import streamlit as st  # noqa
        if hasattr(st, "session_state"):
            m = st.session_state.get("_custom_metrics_local", {})
            if name in m:
                m.pop(name)
                st.session_state["_custom_metrics_local"] = m
                return True
    except Exception:
        pass
    return False


def load_pattern_history() -> list:
    """Load the accumulated daily correlation history.

    v43.85: check Gist first (durable); fall back to session state.
    Returns list of dicts (see compute_daily_correlations for schema),
    ordered by date. Empty list if no history yet.
    """
    try:
        if durable_storage_configured():
            snaps = _gist_read_all() or {}
            history = snaps.get("_pattern_history") or []
            if isinstance(history, list) and history:
                return history
    except Exception:
        pass
    # Fall back to session state (for users without Gist configured)
    try:
        import streamlit as st  # noqa
        if hasattr(st, "session_state"):
            _hist = st.session_state.get("_pattern_history_local")
            if isinstance(_hist, list):
                return _hist
    except Exception:
        pass
    return []


def load_prop_pattern_history(prop: str = "hr") -> list:
    """v44.14: load the accumulated per-slate correlation history for a prop.

    prop: "hr" (homered), "hits" (got_hit), or "tb" (got_2plus_bases).
    Returns list of per-slate correlation dicts ordered by date; [] if none.
    """
    _key = {
        "hr": "_pattern_history",
        "hits": "_pattern_history_hits",
        "tb": "_pattern_history_tb",
    }.get(prop, "_pattern_history")
    try:
        if durable_storage_configured():
            snaps = _gist_read_all() or {}
            history = snaps.get(_key) or []
            if isinstance(history, list) and history:
                return history
    except Exception:
        pass
    try:
        import streamlit as st  # noqa
        if hasattr(st, "session_state"):
            _local = st.session_state.get(f"{_key}_local")
            if isinstance(_local, list):
                return _local
    except Exception:
        pass
    return []


def _extract_all_from_feeds(target_date) -> dict:
    """v44.55 (code review #11): fetch every game's feed/live for a date ONCE,
    in parallel, and extract BOTH batting and pitching lines in a single pass.

    Previously fetch_hitter_outcomes and fetch_pitcher_outcomes each fetched
    the SAME feed/live per game, sequentially — ~15 games × 2 fetchers × 1-2s
    = 30-60s, called inside auto-eval + back-fill + pattern discovery. This
    consolidates to one fetch per game, parallelized (15 workers), and cached
    by date (past dates are immutable once final, so a 1-hour TTL is safe).

    Returns {"hitters": {pid: {...}}, "pitchers": {pid: {...}}}.
    """
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
        return {"hitters": {}, "pitchers": {}}

    game_pks = []
    _excluded_status = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            _pk = g.get("gamePk")
            if not _pk:
                continue
            # v44.87 (user: All-Star break shouldn't skew the data). Only grade
            # REGULAR-SEASON games. gameType "R" = regular season; "A" = All-Star,
            # "S" = spring, "E"/"F"/"D"/"L"/"W" = exhibition/postseason. During the
            # ASB (and the ASG on ~July 14), the schedule feed returns the All-Star
            # Game — without this filter its HRs would be graded as outcomes vs our
            # predictions, corrupting pattern analysis + backtests. The season-stat
            # fetch already filters gameType=R; this outcome path was missing it.
            _game_type = str(g.get("gameType") or "").upper()
            if _game_type and _game_type != "R":
                _excluded_status.append((_pk, f"gameType={_game_type}"))
                continue
            # v44.60 (user: don't let postponed/cancelled games count as
            # "didn't homer"). Only pull outcomes from games MLB marks Final/
            # completed. A postponed or suspended game has no valid full box
            # score, so grading a hitter from it would either miss them (fine)
            # or record a partial/false negative (bad). Gate on game status.
            _status = (g.get("status") or {})
            _coded = str(_status.get("codedGameState") or "").upper()  # F=Final, D=postponed, etc.
            _abstract = str(_status.get("abstractGameState") or "").upper()
            _detailed = str(_status.get("detailedState") or "").lower()
            _is_final = (_coded == "F") or (_abstract == "FINAL") or ("final" in _detailed and "postpon" not in _detailed)
            _is_bad = any(w in _detailed for w in ("postpon", "cancel", "suspend")) or _coded in ("D", "C", "U", "T")
            if _is_bad or not _is_final:
                _excluded_status.append((_pk, _detailed or _coded))
                continue
            game_pks.append(_pk)
    # Stash which games were excluded so it can be surfaced if needed.
    global _LAST_OUTCOME_EXCLUDED_GAMES
    try:
        _LAST_OUTCOME_EXCLUDED_GAMES = _excluded_status
    except Exception:
        pass

    def _fetch_one(game_pk):
        try:
            box_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
            br = requests.get(box_url, headers=HEADERS, timeout=15)
            br.raise_for_status()
            return br.json()
        except Exception:
            return None

    boxes = []
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=15) as ex:
            boxes = list(ex.map(_fetch_one, game_pks))
    except Exception:
        # Fallback to sequential if threading is unavailable for any reason
        boxes = [_fetch_one(pk) for pk in game_pks]

    hitters, pitchers = {}, {}
    for box in boxes:
        if not box:
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
                    stats = (player.get("stats") or {})
                    # ----- batting -----
                    bat = stats.get("batting") or {}
                    if bat:
                        doubles_ = int(bat.get("doubles") or 0)
                        triples_ = int(bat.get("triples") or 0)
                        hr_ = int(bat.get("homeRuns") or 0)
                        h_ = int(bat.get("hits") or 0)
                        singles_ = max(0, h_ - doubles_ - triples_ - hr_)
                        total_bases = singles_ + 2*doubles_ + 3*triples_ + 4*hr_
                        hitters[pid] = {
                            "hr": hr_, "ab": int(bat.get("atBats") or 0), "h": h_,
                            "k": int(bat.get("strikeOuts") or 0),
                            "bb": int(bat.get("baseOnBalls") or 0),
                            "rbi": int(bat.get("rbi") or 0),
                            "doubles": doubles_, "triples": triples_,
                            "total_bases": total_bases,
                            "runs": int(bat.get("runs") or 0),
                            "homered": hr_ > 0, "got_hit": h_ > 0,
                            "got_2plus_bases": total_bases >= 2,
                            "got_3plus_bases": total_bases >= 3,
                        }
                    # ----- pitching (starters only) -----
                    pit = stats.get("pitching") or {}
                    if pit and pit.get("gamesStarted", 0):
                        try:
                            ip_str = str(pit.get("inningsPitched") or "0.0")
                            whole, _, frac = ip_str.partition(".")
                            ip = float(whole) + (float(frac or 0) / 3.0)
                        except Exception:
                            ip = 0.0
                        pitchers[pid] = {
                            "ip": ip, "k": int(pit.get("strikeOuts") or 0),
                            "er": int(pit.get("earnedRuns") or 0),
                            "bb": int(pit.get("baseOnBalls") or 0),
                            "hr": int(pit.get("homeRuns") or 0),
                            "h": int(pit.get("hits") or 0),
                        }
        except Exception:
            continue
    return {"hitters": hitters, "pitchers": pitchers}


# v44.55: cache the unified feed extraction by date. Past dates are immutable
# once games are final, so a 1-hour TTL is safe and eliminates the repeated
# 30-60s refetch across auto-eval + back-fill + pattern discovery in one run.
if st is not None and hasattr(st, "cache_data"):
    _extract_all_from_feeds = st.cache_data(ttl=3600)(_extract_all_from_feeds)


def fetch_hitter_outcomes(target_date) -> dict:
    """
    For a given date, fetch which hitters homered and their game line.
    Returns {player_id: {"hr": int, "ab": int, "h": int, "k": int, "bb": int, "rbi": int}}

    v44.55: now delegates to the unified cached+parallel feed extractor so the
    same game feeds aren't fetched twice (once here, once for pitchers).
    """
    return _extract_all_from_feeds(target_date).get("hitters", {})


def fetch_pitcher_outcomes(target_date) -> dict:
    """
    For a given date, fetch what each starting pitcher actually did.
    Returns {player_id: {"ip": float, "k": int, "er": int, "bb": int, "hr": int, "h": int}}

    v44.55: delegates to the unified cached+parallel feed extractor (shares the
    same fetched feeds with fetch_hitter_outcomes — no double fetch).
    """
    return _extract_all_from_feeds(target_date).get("pitchers", {})


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

    hitters = snapshot_hitters(snapshot)  # v43.88: handles both formats
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

    # v44.45 (user insight: don't count starts cut short by injury/ejection —
    # those aren't projection failures). A starter projected for a full outing
    # who records < 3 IP almost always left early (injury, ejection, rain,
    # position-player mop-up). Grade K accuracy on COMPLETED-ish starts only
    # (>= 3 IP), but report how many were excluded so it's transparent.
    _full = df[pd.to_numeric(df["actual_ip"], errors="coerce") >= 3.0].copy()
    _n_short = len(df) - len(_full)
    _grade_df = _full if len(_full) >= 3 else df  # fall back if too few

    metrics = {
        "total_pitchers_matched": len(df),
        "pitchers_graded": len(_grade_df),
        "short_starts_excluded": _n_short,
        "k_projection_rmse": round(((_grade_df["k_error"] ** 2).mean()) ** 0.5, 2),
        "k_projection_bias": round(_grade_df["k_error"].mean(), 2),
    }

    # K accuracy detail (full list, but flag the short ones)
    df_sorted = df.sort_values("proj_k", ascending=False)
    metrics["k_projections_detail"] = [
        {
            "name": r["pitcher_name"],
            "projected_k": round(r["proj_k"], 1),
            "actual_k": int(r["actual_k"]),
            "actual_ip": round(r["actual_ip"], 1),
            "diff": round(r["k_error"], 1),
            "short_start": bool(pd.to_numeric(r["actual_ip"], errors="coerce") < 3.0),
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
        # v44.57 (code review #13): prefer the outcomes already stored on the
        # snapshot (v43.70+ back-fills hitter_outcomes) over a live refetch.
        # Faster and works even when the MLB API is flaky. Fall back to a live
        # fetch only when the snapshot doesn't carry outcomes yet.
        hitter_actuals = None
        _stored = snapshot.get("hitter_outcomes") if isinstance(snapshot, dict) else None
        if _stored:
            try:
                # stored keys are strings; evaluate_* expects int player_ids
                hitter_actuals = {}
                for _pid, _row in _stored.items():
                    try:
                        hitter_actuals[int(_pid)] = _row
                    except (TypeError, ValueError):
                        hitter_actuals[_pid] = _row
            except Exception:
                hitter_actuals = None
        if not hitter_actuals:
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
