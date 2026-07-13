"""Column schema registry — single source of truth for canonical column names.

v45.00 (external review): the app has repeatedly shipped bugs where a column
was renamed in one place but read by its old name elsewhere, and the
`if col in df.columns:` guard pattern silently skipped the missing column
instead of erroring. Confirmed incidents: env_mult vs env_boost (v44.99),
barrel_allowed (v43.15), fb_allowed, handedness splits. This registry gives:

  1. CANONICAL_COLUMNS   — the authoritative name for each concept.
  2. COLUMN_ALIASES      — historical/alternate names → canonical.
  3. resolve_column()    — map any known alias to its canonical name.
  4. assert_schema()     — check a finalized frame for expected columns and
                           report drift as a diagnostic (fail-open: never
                           crashes the app, just surfaces the problem).

The point is to make name drift VISIBLE at the point it happens, rather than
letting a gate go silently dead. This is a detection tool, not a rename tool —
it does not mutate frames.
"""

from __future__ import annotations


# Canonical name → short description (the authoritative column name per concept).
# These are the names the scoring + display code should read.
CANONICAL_COLUMNS = {
    # identity / lineup
    "player_id": "MLBAM player id",
    "player_name": "hitter display name",
    "team": "hitter team abbr",
    "game": "game label (away @ home)",
    "is_bench": "True if hitter is a bench/roster-fill row",
    "lineup_confirmed": "True if in the confirmed starting lineup",
    "bats": "batter handedness (L/R/S)",
    # environment
    "env_boost": "park × weather × pull-wind HR multiplier",
    # core hitter power
    "hr_game_pct": "modeled HR probability for this game (%)",
    "barrel_pct": "hitter barrel rate",
    "pulled_brl_pct": "hitter pulled-barrel rate",
    "avg_ev": "hitter average exit velocity",
    "hard_hit": "hitter hard-hit rate",
    "iso": "hitter isolated power",
    "blast_pct": "hitter blast rate (bat-tracking; may be median-imputed)",
    "blast_pct_real": "hitter blast rate, un-imputed (NaN if no data)",
    "xwoba": "hitter expected wOBA",
    "xslg": "hitter expected SLG",
    # composite scores
    "hr_score": "primary matchup-aware HR composite",
    "pick_score": "shipped ranking composite",
    "dinger_score": "raw-power composite",
    "power_score": "power composite",
    "lift_score": "value/leverage lift",
    "sleeper_score": "percentile-differential upside",
    "matchup_opp": "matchup opportunity score",
    "grade": "hitter letter grade (A+/A/B+/...)",
    "smash_spot": "smash-tier flag string",
    # pitcher matchup (opp = opposing pitcher, from batter's perspective)
    "opp_pitcher": "opposing pitcher name",
    "opp_pitcher_grade": "opposing pitcher grade (EXPLOIT+/EXPLOIT/.../ELITE)",
    "opp_pitcher_hr9": "opposing pitcher HR/9 allowed",
    "opp_pitcher_barrel_allowed": "opposing pitcher barrel% allowed",
    "opp_pitcher_fb_allowed": "opposing pitcher fly-ball% allowed",
    "opp_pitcher_xwoba_allowed": "opposing pitcher xwOBA allowed",
}


# Historical / alternate names → canonical name. When a read finds one of these,
# resolve_column() maps it back so old call sites keep working while surfacing
# that the name is non-canonical.
COLUMN_ALIASES = {
    "env_mult": "env_boost",          # v44.99 drift
    "environment_boost": "env_boost",
    "barrel_allowed": "opp_pitcher_barrel_allowed",   # v43.15 drift
    "fb_allowed": "opp_pitcher_fb_allowed",
    "xwoba_allowed": "opp_pitcher_xwoba_allowed",
    "pitcher_grade": "opp_pitcher_grade",
    "hr_prob": "hr_game_pct",
    "bench": "is_bench",
}


def resolve_column(name: str) -> str:
    """Return the canonical name for a column, mapping known aliases.

    Unknown names pass through unchanged (so this is safe to call on any
    column). Use this when reading a column whose name may have drifted.
    """
    return COLUMN_ALIASES.get(name, name)


def find_column(df, name: str):
    """Return the actual column in df matching `name` or any of its aliases,
    or None if absent. Lets a reader accept both canonical and legacy names.
    """
    if name in df.columns:
        return name
    canonical = resolve_column(name)
    if canonical in df.columns:
        return canonical
    # reverse: caller asked for canonical but frame still has an alias
    for alias, canon in COLUMN_ALIASES.items():
        if canon == canonical and alias in df.columns:
            return alias
    return None


# Columns that MUST exist on the finalized combined_all / combined_picks frame
# for the core pick + display pipeline to work. Missing any of these means a
# gate or display is silently dead — exactly the bug class this registry exists
# to catch.
REQUIRED_ON_COMBINED = [
    "player_id", "player_name", "team", "is_bench",
    "hr_game_pct", "hr_score", "env_boost", "grade",
    "opp_pitcher_grade",
]


def assert_schema(df, required=None, frame_name="combined"):
    """Check `df` for required columns; return a list of problems (drift report).

    Fail-open: never raises. For each required column that's absent, it checks
    whether a known ALIAS is present instead (name drift) vs the column being
    entirely missing (fetch/enrichment gap), and reports which. The caller
    surfaces the returned messages as diagnostics.

    Returns: list[str] of human-readable problem descriptions (empty = clean).
    """
    problems = []
    if df is None:
        return [f"{frame_name}: frame is None"]
    try:
        cols = set(df.columns)
    except Exception:
        return [f"{frame_name}: not a DataFrame-like object"]

    for col in (required or REQUIRED_ON_COMBINED):
        if col in cols:
            continue
        # Is a legacy alias present instead? That's name drift.
        drift_alias = None
        for alias, canon in COLUMN_ALIASES.items():
            if canon == col and alias in cols:
                drift_alias = alias
                break
        if drift_alias:
            problems.append(
                f"{frame_name}: expected '{col}' but found legacy alias "
                f"'{drift_alias}' — a reader/gate may be silently dead. "
                f"Rename to '{col}'."
            )
        else:
            problems.append(
                f"{frame_name}: required column '{col}' is MISSING "
                f"(fetch/enrichment gap — dependent gates/scores will be blank)."
            )
    return problems
