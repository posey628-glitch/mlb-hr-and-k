"""
weather.py
==========
Pulls game-time weather from Open-Meteo (free, no API key, no signup) and
converts it into an HR-impact multiplier based on:
  - wind speed
  - wind direction relative to park's CF bearing (out / in / cross)
  - temperature (warm air = ball travels further)
  - humidity & air pressure (lower density = more carry)
  - indoor/retractable roof (neutralizes weather effects)

Open-Meteo docs: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import requests
import streamlit as st


@st.cache_data(ttl=1800)
def fetch_weather(lat: float, lon: float, when) -> dict:
    """Return weather forecast nearest to `when` for the given coords.

    `when` accepts datetime, pd.Timestamp, ISO string, or None.
    """
    if lat is None or lon is None:
        return {}

    # Normalize `when` to a datetime
    if when is None:
        target_dt = datetime.now()
    elif isinstance(when, str):
        try:
            # Handle timezone suffixes by stripping them for simplicity
            cleaned = when.replace("Z", "+00:00")
            target_dt = datetime.fromisoformat(cleaned)
            # Strip timezone for naive comparison later
            if target_dt.tzinfo is not None:
                target_dt = target_dt.replace(tzinfo=None)
        except (ValueError, TypeError):
            target_dt = datetime.now()
    elif hasattr(when, "strftime"):
        # datetime or pd.Timestamp
        target_dt = when
        if hasattr(target_dt, "tz_localize") and getattr(target_dt, "tz", None) is not None:
            try:
                target_dt = target_dt.tz_convert(None)
            except Exception:
                pass
    else:
        target_dt = datetime.now()

    iso = target_dt.strftime("%Y-%m-%d")
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,"
        "wind_speed_10m,wind_direction_10m,surface_pressure"
        f"&start_date={iso}&end_date={iso}"
        "&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": str(e)}

    hourly = data.get("hourly", {})
    if not hourly.get("time"):
        return {}

    # Find the hour nearest target time
    try:
        times = [datetime.fromisoformat(t) for t in hourly["time"]]
        target = target_dt.replace(tzinfo=None) if hasattr(target_dt, "replace") else target_dt
        # Make sure target has no tzinfo for comparison
        if hasattr(target, "tzinfo") and target.tzinfo is not None:
            target = target.replace(tzinfo=None)
        idx = min(range(len(times)), key=lambda i: abs((times[i] - target).total_seconds()))
    except Exception:
        # Fall back to midday hour
        idx = min(12, len(hourly["time"]) - 1)

    return {
        "temp_f":       hourly["temperature_2m"][idx],
        "humidity":     hourly["relative_humidity_2m"][idx],
        "precip_prob":  hourly["precipitation_probability"][idx],
        "wind_mph":     hourly["wind_speed_10m"][idx],
        "wind_dir_deg": hourly["wind_direction_10m"][idx],
        "pressure_hpa": hourly["surface_pressure"][idx],
        "time": hourly["time"][idx],
    }


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


def hr_multiplier(weather: dict, park: dict) -> tuple[float, str]:
    """
    Combine weather + park into a single HR multiplier (1.0 = neutral).
    Returns (multiplier, plain-English summary).

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
    roof_factor = 0.5 if roof == "retractable" else 1.0

    # Temperature - asymmetric (cold suppresses more than warm boosts)
    temp = weather.get("temp_f")
    if temp is not None:
        if temp >= 70:
            t_eff = (temp - 70) / 10 * 0.03
        else:
            t_eff = (temp - 70) / 10 * 0.04  # stronger cold penalty
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
    wind_mph = weather.get("wind_mph", 0)
    wind_dir = weather.get("wind_dir_deg")
    if wind_mph and wind_dir is not None:
        component = wind_component_out(wind_dir, park.get("cf_bearing", 0))
        net = component * wind_mph
        wind_eff = max(-0.25, net * 0.01)
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

    if roof == "retractable":
        summary.append("(retractable roof: 50% weather effect)")

    # Sanity bounds
    mult = max(0.55, min(1.45, mult))

    return round(mult, 3), " · ".join(summary) if summary else "Neutral"
