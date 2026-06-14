"""
weather.py
==========
Pulls game-time weather from Open-Meteo and converts it into an HR-impact
multiplier based on:
  - wind speed
  - wind direction relative to park's CF bearing (out / in / cross)
  - temperature (warm air = ball travels further)
  - humidity & air pressure (lower density = more carry)
  - indoor/retractable roof (neutralizes weather effects)

ARCHITECTURE (June 2026, v36):
  Open-Meteo is the single weather provider. NWS (api.weather.gov) was
  tried in v35 as a primary with Open-Meteo fallback, but NWS returns
  403 "Host not in allowlist" from many cloud provider IPs (including
  Streamlit Cloud), making it unreliable as a primary provider.

  Open-Meteo:
    - Free, no API key, CDN-backed (works from any cloud platform)
    - Supports up to 1000 locations in a SINGLE request via comma-separated
      coordinate lists. Means one API call for entire slate = no rate limit
      problems even on cold cache.
    - 10 req/min limit only matters if we burst many single-location calls
    - Free tier: 10,000 calls/day

  Strategy:
    - prefetch_weather_batch(coords_list, when) — one call for all games,
      warms cache for every (lat, lon, hour) tuple in the slate
    - fetch_weather(lat, lon, when) — hits warmed cache, falls back to
      single-location API call if cache miss (e.g., schedule changes
      mid-page-render)
    - Diagnostic logging captures actual failure reasons when they happen

Open-Meteo docs: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import requests
import streamlit as st


# Module-level diagnostics — captures last error for surfacing in UI
_LAST_ERROR: str | None = None
_LAST_SOURCE: str | None = None


def get_last_weather_error() -> str | None:
    """Return the last weather fetch error (or None if last fetch succeeded)."""
    return _LAST_ERROR


def _normalize_target_dt(when) -> datetime:
    """Normalize various time inputs to a naive datetime rounded to the hour."""
    if when is None:
        target_dt = datetime.now()
    elif isinstance(when, str):
        try:
            cleaned = when.replace("Z", "+00:00")
            target_dt = datetime.fromisoformat(cleaned)
            if target_dt.tzinfo is not None:
                target_dt = target_dt.replace(tzinfo=None)
        except (ValueError, TypeError):
            target_dt = datetime.now()
    elif hasattr(when, "strftime"):
        target_dt = when
        if hasattr(target_dt, "tz_localize") and getattr(target_dt, "tz", None) is not None:
            try:
                target_dt = target_dt.tz_convert(None)
            except Exception:
                pass
    else:
        target_dt = datetime.now()

    # Round to hour for cache normalization (Open-Meteo only returns hourly anyway)
    try:
        if hasattr(target_dt, "replace"):
            target_dt = target_dt.replace(minute=0, second=0, microsecond=0)
    except Exception:
        pass
    return target_dt


def _parse_om_response_for_hour(response: dict, target_dt: datetime) -> dict:
    """Extract weather at target hour from an Open-Meteo single-location response."""
    hourly = response.get("hourly", {})
    times = hourly.get("time") or []
    if not times:
        return {}

    target_naive = target_dt.replace(tzinfo=None) if target_dt.tzinfo else target_dt
    try:
        dts = [datetime.fromisoformat(t) for t in times]
        idx = min(range(len(dts)),
                   key=lambda i: abs((dts[i] - target_naive).total_seconds()))
    except Exception:
        idx = min(12, len(times) - 1)

    def _safe(key):
        arr = hourly.get(key) or []
        if idx < len(arr):
            v = arr[idx]
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        return None

    return {
        "temp_f": _safe("temperature_2m"),
        "wind_mph": _safe("wind_speed_10m"),
        "wind_dir_deg": _safe("wind_direction_10m"),
        "humidity": _safe("relative_humidity_2m"),
        "pressure_hpa": _safe("surface_pressure"),
        "precip_prob": _safe("precipitation_probability"),
        "_source": "OpenMeteo",
    }


# ============================================================================
# WTTR.IN FALLBACK (v42a) — second free weather provider for Open-Meteo 429s
# ============================================================================
# Streamlit Cloud's outbound IPs are shared across many apps. Open-Meteo's
# free tier caps at 10,000 calls/day per IP, so we frequently hit 429 errors
# not from our own usage but from neighbor apps on the same egress IP. wttr.in
# is a separate provider (no API key, no per-IP daily cap published, accepts
# lat/lon directly) — when Open-Meteo fails we try wttr.in to fill the gap.
#
# wttr.in JSON shape (?format=j1):
#   data["weather"][N]["hourly"][H] gives forecast at 3-hour intervals
#   data["weather"][N]["date"] is the day
#   hourly fields: time (0/300/600/900/1200/1500/1800/2100), tempF,
#                  windspeedMiles, winddirDegree, humidity, pressure,
#                  chanceofrain
#
# This is a graceful-degrade fallback, not a primary replacement. wttr.in
# only returns 3-hour intervals (vs Open-Meteo's hourly), so we match to
# the nearest 3-hour slot.
# ============================================================================
def _parse_wttr_response_for_hour(response: dict, target_dt: datetime) -> dict:
    """Extract weather at target hour from a wttr.in ?format=j1 response."""
    weather_days = response.get("weather", [])
    if not weather_days:
        return {}

    target_naive = target_dt.replace(tzinfo=None) if target_dt.tzinfo else target_dt
    target_date_str = target_naive.strftime("%Y-%m-%d")

    # Find the day matching our target date; fall back to first day
    day = None
    for d in weather_days:
        if d.get("date") == target_date_str:
            day = d
            break
    if day is None:
        day = weather_days[0]

    hourly = day.get("hourly", []) or []
    if not hourly:
        return {}

    # wttr.in time strings are like "0", "300", "600", ..., "2100" (3-hour intervals)
    target_minutes = target_naive.hour * 60 + target_naive.minute

    def _time_to_minutes(t_str):
        try:
            t_int = int(t_str)
            return (t_int // 100) * 60 + (t_int % 100)
        except (TypeError, ValueError):
            return 0

    best_idx = 0
    best_delta = float("inf")
    for i, h in enumerate(hourly):
        h_min = _time_to_minutes(h.get("time", "0"))
        delta = abs(h_min - target_minutes)
        if delta < best_delta:
            best_delta = delta
            best_idx = i

    h = hourly[best_idx]

    def _f(key):
        v = h.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "temp_f": _f("tempF"),
        "wind_mph": _f("windspeedMiles"),
        "wind_dir_deg": _f("winddirDegree"),
        "humidity": _f("humidity"),
        # wttr.in pressure is in mb (hPa equivalent for our purposes)
        "pressure_hpa": _f("pressure"),
        "precip_prob": _f("chanceofrain"),
        "_source": "wttr.in",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_single_wttr(lat: float, lon: float, target_dt_str: str) -> dict:
    """Cached wttr.in single-location fetch. Used as Open-Meteo fallback.

    Returns same dict shape as _fetch_single_om() so callers can swap.
    Empty dict on failure — caller falls back to neutral park-only weather.
    """
    global _LAST_ERROR, _LAST_SOURCE
    target_dt = datetime.fromisoformat(target_dt_str)
    url = f"https://wttr.in/{lat},{lon}?format=j1"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            _LAST_ERROR = f"wttr.in HTTP {r.status_code}: {r.text[:120]}"
            return {}
        data = r.json()
        result = _parse_wttr_response_for_hour(data, target_dt)
        if result.get("temp_f") is None:
            _LAST_ERROR = "wttr.in returned no hourly data for target time"
            return {}
        _LAST_ERROR = None
        _LAST_SOURCE = "wttr.in-fallback"
        return result
    except requests.exceptions.Timeout:
        _LAST_ERROR = "wttr.in request timed out (12s)"
        return {}
    except Exception as e:
        _LAST_ERROR = f"wttr.in exception: {type(e).__name__}: {str(e)[:120]}"
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_single_om(lat: float, lon: float, target_dt_str: str) -> dict:
    """
    Cached single-location Open-Meteo fetch. Cache key is (lat, lon, hour-iso).

    Public callers should use fetch_weather() which normalizes inputs first.
    """
    global _LAST_ERROR, _LAST_SOURCE
    target_dt = datetime.fromisoformat(target_dt_str)
    iso = target_dt.strftime("%Y-%m-%d")
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,"
        "wind_speed_10m,wind_direction_10m,surface_pressure"
        f"&start_date={iso}&end_date={iso}"
        # v42e: timezone=GMT (was =auto). _normalize_target_dt produces a
        # UTC-naive datetime; matching it against a venue-local hourly array
        # was systematically off by 4 hours (East Coast) to 7 hours (West
        # Coast). For late West Coast games the local-time array can also
        # fall OUTSIDE the requested start_date/end_date entirely → empty
        # weather. With timezone=GMT both sides are now in UTC.
        "&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=GMT"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 429:
            _LAST_ERROR = "Open-Meteo rate limited (429). Try again in 60s."
            return {}
        if r.status_code != 200:
            _LAST_ERROR = f"Open-Meteo HTTP {r.status_code}: {r.text[:120]}"
            return {}
        data = r.json()
        result = _parse_om_response_for_hour(data, target_dt)
        if result.get("temp_f") is None:
            _LAST_ERROR = "Open-Meteo returned no hourly data for target time"
            return {}
        _LAST_ERROR = None
        _LAST_SOURCE = "OpenMeteo-single"
        return result
    except requests.exceptions.Timeout:
        _LAST_ERROR = "Open-Meteo request timed out (10s)"
        return {}
    except Exception as e:
        _LAST_ERROR = f"Open-Meteo exception: {type(e).__name__}: {str(e)[:120]}"
        return {}


def fetch_weather(lat: float, lon: float, when) -> dict:
    """
    Return weather forecast nearest to `when` for the given coords.

    v42a: tries Open-Meteo first; if it fails (429 rate limit, timeout, etc.)
    falls back to wttr.in. Returns the first non-empty result. If BOTH fail,
    returns {} (caller will use neutral park-only env_boost).

    `when` accepts datetime, pd.Timestamp, ISO string, or None.
    Cache key is normalized to (rounded lat/lon, hour-of-day).
    """
    if lat is None or lon is None:
        return {}
    try:
        lat = round(float(lat), 3)
        lon = round(float(lon), 3)
    except (TypeError, ValueError):
        return {}
    target_dt = _normalize_target_dt(when)
    target_iso = target_dt.isoformat()

    # Try Open-Meteo first (primary source — better accuracy, hourly resolution)
    result = _fetch_single_om(lat, lon, target_iso)
    if result and result.get("temp_f") is not None:
        return result

    # Fallback: wttr.in (no API key, no daily quota — works when Open-Meteo 429s)
    result = _fetch_single_wttr(lat, lon, target_iso)
    if result and result.get("temp_f") is not None:
        return result

    # Both failed — return empty so caller uses neutral env
    return {}


def _batch_fallback_via_wttr(unique_keys: list) -> int:
    """When Open-Meteo batch fails, try wttr.in per-venue to salvage data.

    Iterates each (lat, lon, target_dt) tuple, calls _fetch_single_wttr,
    and primes _BATCH_CACHE on success. Returns count of venues filled.
    wttr.in handles ~16 calls fine since it's per-venue, no per-IP cap.
    """
    n_success = 0
    for (lat_r, lon_r, target_dt) in unique_keys:
        try:
            result = _fetch_single_wttr(lat_r, lon_r, target_dt.isoformat())
            if result and result.get("temp_f") is not None:
                _BATCH_CACHE[(lat_r, lon_r, target_dt.isoformat())] = result
                n_success += 1
        except Exception:
            continue
    return n_success


def prefetch_weather_batch(coords_when_list: list[tuple[float, float, object]]) -> dict:
    """
    Fetch weather for many (lat, lon, when) tuples in a SINGLE Open-Meteo call,
    then prime the per-location cache so subsequent fetch_weather() calls are
    instant cache hits.

    Returns a summary dict: {n_locations, n_success, n_cached, source}.

    This is the recommended way to load weather for an entire slate. Call it
    once near the top of the page, BEFORE the per-game loop that calls
    fetch_weather. One HTTP request for the whole slate instead of 16.
    """
    global _LAST_ERROR, _LAST_SOURCE
    if not coords_when_list:
        return {"n_locations": 0, "n_success": 0}

    # Deduplicate by (rounded lat, rounded lon, rounded hour)
    seen = {}
    for lat, lon, when in coords_when_list:
        if lat is None or lon is None:
            continue
        try:
            lat_r = round(float(lat), 3)
            lon_r = round(float(lon), 3)
        except (TypeError, ValueError):
            continue
        target_dt = _normalize_target_dt(when)
        key = (lat_r, lon_r, target_dt.isoformat())
        seen[key] = (lat_r, lon_r, target_dt)

    if not seen:
        return {"n_locations": 0, "n_success": 0}

    unique_keys = list(seen.values())
    n_locations = len(unique_keys)

    # Open-Meteo multi-location: comma-separated lat/lon lists. All locations
    # share the same date range (we use widest range across all targets).
    # Since all MLB games typically happen on the same calendar date, this is fine.
    lats_str = ",".join(f"{k[0]:.3f}" for k in unique_keys)
    lons_str = ",".join(f"{k[1]:.3f}" for k in unique_keys)
    # Pick the earliest and latest dates across all targets
    all_dates = [k[2].strftime("%Y-%m-%d") for k in unique_keys]
    start_date = min(all_dates)
    end_date = max(all_dates)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats_str}&longitude={lons_str}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,"
        "wind_speed_10m,wind_direction_10m,surface_pressure"
        f"&start_date={start_date}&end_date={end_date}"
        # v42e: timezone=GMT (was =auto). Match the UTC-naive target_dt
        # produced by _normalize_target_dt. See _fetch_single_om for full
        # explanation. Without this, batch results were systematically off
        # by 4-7 hours per venue.
        "&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=GMT"
    )

    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 429:
            _LAST_ERROR = "Open-Meteo rate limited (429) on batch prefetch"
            # v42a: try wttr.in per-location to salvage the slate
            n_wttr = _batch_fallback_via_wttr(unique_keys)
            if n_wttr > 0:
                _LAST_SOURCE = "wttr.in-batch-fallback"
                return {
                    "n_locations": n_locations, "n_success": n_wttr,
                    "source": "wttr.in", "note": "Open-Meteo 429, wttr.in fallback used",
                }
            return {"n_locations": n_locations, "n_success": 0, "error": _LAST_ERROR}
        if r.status_code != 200:
            _LAST_ERROR = f"Open-Meteo batch HTTP {r.status_code}: {r.text[:120]}"
            # v42a: try wttr.in per-location as fallback
            n_wttr = _batch_fallback_via_wttr(unique_keys)
            if n_wttr > 0:
                _LAST_SOURCE = "wttr.in-batch-fallback"
                return {
                    "n_locations": n_locations, "n_success": n_wttr,
                    "source": "wttr.in", "note": f"Open-Meteo HTTP {r.status_code}, wttr.in fallback used",
                }
            return {"n_locations": n_locations, "n_success": 0, "error": _LAST_ERROR}
        data = r.json()
    except requests.exceptions.Timeout:
        _LAST_ERROR = "Open-Meteo batch request timed out (20s)"
        n_wttr = _batch_fallback_via_wttr(unique_keys)
        if n_wttr > 0:
            _LAST_SOURCE = "wttr.in-batch-fallback"
            return {
                "n_locations": n_locations, "n_success": n_wttr,
                "source": "wttr.in", "note": "Open-Meteo timeout, wttr.in fallback used",
            }
        return {"n_locations": n_locations, "n_success": 0, "error": _LAST_ERROR}
    except Exception as e:
        _LAST_ERROR = f"Open-Meteo batch exception: {type(e).__name__}: {str(e)[:120]}"
        return {"n_locations": n_locations, "n_success": 0, "error": _LAST_ERROR}

    # Open-Meteo returns a LIST when multiple locations were requested,
    # or a single object for one location.
    locations = data if isinstance(data, list) else [data]

    # Prime the cache for each location
    n_success = 0
    for i, (lat_r, lon_r, target_dt) in enumerate(unique_keys):
        if i >= len(locations):
            break
        loc_data = locations[i]
        if not loc_data:
            continue
        result = _parse_om_response_for_hour(loc_data, target_dt)
        if result.get("temp_f") is not None:
            # Manually populate the cache for this (lat, lon, target_dt) key.
            # Streamlit's @st.cache_data uses argument values as the cache key,
            # so calling _fetch_single_om with these exact args will populate
            # the cache. But we already HAVE the data — instead, store it in
            # a module-level dict that fetch_weather can check before hitting
            # the single-location API.
            _BATCH_CACHE[(lat_r, lon_r, target_dt.isoformat())] = result
            n_success += 1

    if n_success > 0:
        _LAST_ERROR = None
        _LAST_SOURCE = "OpenMeteo-batch"
    return {
        "n_locations": n_locations,
        "n_success": n_success,
        "source": _LAST_SOURCE,
    }


# In-memory batch cache populated by prefetch_weather_batch. Reset on app reboot,
# but during a single page render this holds the entire slate's weather after
# one batch call. fetch_weather checks here first before hitting the API.
_BATCH_CACHE: dict[tuple, dict] = {}


def _fetch_with_batch_cache_check(lat: float, lon: float, when) -> dict:
    """fetch_weather that checks _BATCH_CACHE before going to API."""
    if lat is None or lon is None:
        return {}
    try:
        lat = round(float(lat), 3)
        lon = round(float(lon), 3)
    except (TypeError, ValueError):
        return {}
    target_dt = _normalize_target_dt(when)
    key = (lat, lon, target_dt.isoformat())
    if key in _BATCH_CACHE:
        return _BATCH_CACHE[key]
    # Fall through to cached single-location API call
    return _fetch_single_om(lat, lon, target_dt.isoformat())


# Override the public name to use the batch-aware version
fetch_weather = _fetch_with_batch_cache_check


def wind_component_out(wind_dir_deg: float, cf_bearing_deg: float) -> float:
    """
    Returns the cosine of the angle between wind direction and the line from
    home plate to CF. Range: -1 (wind blowing IN from CF toward home) to
    +1 (wind blowing OUT toward CF).

    Note: meteorological wind_direction is the direction wind is COMING FROM.
    So wind_dir = 180 (south) means wind is blowing TO the north (0°).
    We flip 180° to get "blowing toward" direction.
    """
    if wind_dir_deg is None or cf_bearing_deg is None:
        return 0.0
    blowing_toward = (wind_dir_deg + 180) % 360
    angle_diff = math.radians(blowing_toward - cf_bearing_deg)
    return math.cos(angle_diff)


def hr_multiplier(weather: dict, park: dict, skip_wind: bool = False,
                    real_roof_closed: bool | None = None) -> tuple[float, str]:
    """
    Combine weather + park into a single HR multiplier (1.0 = neutral).
    Returns (multiplier, plain-English summary).

    v42f: real_roof_closed parameter added. When the caller has MLB ground
    truth about the retractable roof state, pass it. None = unknown (use the
    temperature heuristic). True = closed → neutral. False = confirmed OPEN
    → ignore the temperature heuristic and apply full weather effects.

    Heuristics calibrated to public HR/weather studies:
      - Each 10°F above 70°F:  +3% HR rate
      - Each 10°F below 70°F:  -4% HR rate (cold ball deadens more than warm helps)
      - Each 1 mph net wind out:    +1% HR rate
      - Each 1 mph net wind in:     -1% HR rate (capped at -25%)
      - Indoors/closed dome:        weather effects zeroed out
      - Retractable roof:           weather effects halved (open ~50% of time)
      - High humidity (>70%):       -2% (humid air is slightly heavier in baseball range)
      - Low pressure (<1010 hPa):   +2% (storm fronts let balls carry)
      - High pressure (>1020 hPa):  -2% (cold/clear/dense air suppresses)
      - Heavy rain (>70% chance):   -10% (wet ball, possible delay)
    """
    if not weather or weather.get("error"):
        return 1.0, "Weather unavailable"

    roof = park.get("roof", "open")
    if roof == "dome":
        return 1.0, "Indoor — weather neutral"

    summary = []
    mult = 1.0

    # Determine roof_factor. v42f: MLB ground truth (real_roof_closed) overrides
    # the temperature heuristic. Previous bug: when MLB confirmed the roof was
    # OPEN but the day was cold (<60°F) or hot (>88°F), the heuristic guessed
    # "likely closed" and returned neutral — wiping real wind/temp effects on
    # confirmed-open games.
    _temp_check = weather.get("temp_f")
    if roof == "retractable":
        if real_roof_closed is True:
            # MLB says CLOSED — neutralize
            return 1.0, "🏟️ Retractable roof CLOSED (MLB-confirmed) — indoor neutral"
        elif real_roof_closed is False:
            # MLB says OPEN — full weather effects, skip the temp heuristic
            roof_factor = 1.0
        elif _temp_check is None:
            roof_factor = 0.7  # unknown temp + unknown roof, modest dampener
        elif _temp_check < 60 or _temp_check > 88:
            # Roof likely CLOSED based on temp — return neutral
            return 1.0, f"🏟️ Retractable roof likely CLOSED ({_temp_check:.0f}°F) — indoor conditions"
        else:
            roof_factor = 1.0  # roof likely open based on comfortable temp
    else:
        roof_factor = 1.0

    # Temperature - asymmetric (cold suppresses more than warm boosts).
    # ALTITUDE DAMPENING: park HR factors for Coors Field (1.21) and Sutter
    # Health Park already embed thousands of historical games' worth of
    # weather and altitude effects. Adding a full temperature multiplier on
    # top of those park factors double-counts the temperature contribution.
    # Solution: halve the temperature effect at high-altitude parks.
    HIGH_ALTITUDE_PARKS = {"Coors Field", "Las Vegas Ballpark"}
    # v43.13 (reviewer-validated): Sutter Health Park is ~30 ft elevation
    # (West Sacramento) — NOT high altitude. It was incorrectly included
    # in this set with a comment claiming HR factor 1.13 to "embed altitude
    # like Coors," but the actual PARKS value is 95 (suppressive). The
    # 0.5× temperature and 0.6× wind dampeners that fire here are meant
    # for genuine altitude (Coors at 5,200 ft, Las Vegas at 2,030 ft).
    # Apply them to a sea-level park = under-counting weather effects there.
    altitude_dampener = 0.5 if park.get("name") in HIGH_ALTITUDE_PARKS else 1.0
    temp = weather.get("temp_f")
    if temp is not None:
        if temp >= 70:
            t_eff = (temp - 70) / 10 * 0.03 * altitude_dampener
        else:
            t_eff = (temp - 70) / 10 * 0.04 * altitude_dampener  # stronger cold penalty
        mult *= (1 + t_eff * roof_factor)
        if temp >= 85:
            summary.append(f"🌡️ {temp:.0f}°F (carries well)")
        elif temp >= 75:
            summary.append(f"🌡️ {temp:.0f}°F")
        elif temp <= 45:
            summary.append(f"🥶 {temp:.0f}°F (dead ball)")
        elif temp <= 55:
            summary.append(f"❄️ {temp:.0f}°F (suppresses HRs)")
        else:
            summary.append(f"{temp:.0f}°F")

    # Wind
    # When skip_wind=True (called from per-hitter loop with pull-side wind
    # already applied separately), we skip wind here to avoid double-counting.
    # The pull-side wind multiplier from park_factors.wind_pull_side_multiplier
    # captures both directional and pull-side wind effects per handedness.
    #
    # ALTITUDE DAMPENING for wind (June 2026): Coors and Sutter Health park
    # factors (1.21, 1.13) already embed historical out-blowing wind effects
    # from decades of games at altitude. Adding the full real-time wind on top
    # produces a double-count on already HR-friendly conditions. Apply a 0.6×
    # dampener at altitude parks to reduce the wind multiplier without
    # eliminating real-time signal entirely.
    wind_dampener = 0.6 if park.get("name") in HIGH_ALTITUDE_PARKS else 1.0
    wind_mph = weather.get("wind_mph", 0)
    wind_dir = weather.get("wind_dir_deg")
    if wind_mph and wind_dir is not None and not skip_wind:
        component = wind_component_out(wind_dir, park.get("cf_bearing", 0))
        net = component * wind_mph
        wind_eff = max(-0.25, net * 0.01) * wind_dampener
        mult *= (1 + wind_eff * roof_factor)
        if net >= 8:
            summary.append(f"💨 {wind_mph:.0f}mph OUT (huge HR boost)")
        elif net >= 4:
            summary.append(f"💨 {wind_mph:.0f}mph out")
        elif net <= -8:
            summary.append(f"🌬️ {wind_mph:.0f}mph IN (kills flyballs)")
        elif net <= -4:
            summary.append(f"🌬️ {wind_mph:.0f}mph in")
        elif wind_mph >= 5:
            summary.append(f"{wind_mph:.0f}mph cross")

    # Humidity - heavier air at high humidity in normal temp ranges
    humidity = weather.get("humidity")
    if humidity is not None and humidity > 70:
        mult *= (1 - 0.02 * roof_factor)
        if humidity > 85:
            summary.append(f"💦 {humidity:.0f}% humid")

    # Pressure - low pressure = ball carries (storm fronts, etc)
    pressure = weather.get("pressure_hpa")
    if pressure is not None:
        if pressure < 1010:
            mult *= (1 + 0.02 * roof_factor)
            summary.append(f"⬇️ Low pressure {pressure:.0f}hPa (carries)")
        elif pressure > 1020:
            mult *= (1 - 0.02 * roof_factor)

    # Precipitation - wet ball + likely delay = HRs suppressed
    # Heavy rain has multiple suppressing effects:
    # - Wet ball doesn't carry (less spin retention, more drag)
    # - Cold/damp conditions
    # - Reduced exit velocity from rain on contact
    # - PPD risk means projections shouldn't pile up green if game won't be played
    pp = weather.get("precip_prob", 0)
    if pp is not None:
        if pp >= 80:
            # Likely PPD or delay - strong suppression
            mult *= (1 - 0.25 * roof_factor)
            summary.append(f"🌧️ {pp:.0f}% RAIN (likely delay/PPD)")
        elif pp >= 60:
            mult *= (1 - 0.15 * roof_factor)
            summary.append(f"☔ {pp:.0f}% rain (heavy)")
        elif pp >= 40:
            mult *= (1 - 0.08 * roof_factor)
            summary.append(f"🌧️ {pp:.0f}% rain")
        elif pp >= 25:
            mult *= (1 - 0.03 * roof_factor)
            summary.append(f"💧 {pp:.0f}% drizzle risk")

    if roof == "retractable" and _temp_check is not None and 60 <= _temp_check <= 88:
        summary.append(f"(retractable, likely open at {_temp_check:.0f}°F)")
    elif roof == "retractable":
        summary.append("(retractable, status uncertain)")

    # Sanity bounds
    mult = max(0.55, min(1.45, mult))

    return round(mult, 3), " · ".join(summary) if summary else "Neutral"
