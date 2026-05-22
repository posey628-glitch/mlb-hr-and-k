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
def fetch_weather(lat: float, lon: float, when: datetime) -> dict:
    """Return weather forecast nearest to `when` for the given coords."""
    if lat is None or lon is None:
        return {}

    iso = when.strftime("%Y-%m-%d")
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

    # Find the hour nearest game time
    times = [datetime.fromisoformat(t) for t in hourly["time"]]
    target = when.replace(tzinfo=None)
    idx = min(range(len(times)), key=lambda i: abs((times[i] - target).total_seconds()))

    return {
        "temp_f":       hourly["temperature_2m"][idx],
        "humidity":     hourly["relative_humidity_2m"][idx],
        "precip_prob":  hourly["precipitation_probability"][idx],
        "wind_mph":     hourly["wind_speed_10m"][idx],
        "wind_dir_deg": hourly["wind_direction_10m"][idx],
        "pressure_hpa": hourly["surface_pressure"][idx],
        "time": times[idx].isoformat(),
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
      - Each 1 mph net out:    +1% HR rate
      - Each 1 mph net in:     -1% HR rate (capped at -25%)
      - Indoors/closed roof:   weather effects zeroed out
      - High humidity:         marginal (-1% per 20% above 60%)
    """
    if not weather or weather.get("error"):
        return 1.0, "Weather unavailable"

    roof = park.get("roof", "open")
    if roof == "dome":
        return 1.0, "Indoor — weather neutral"

    summary = []
    mult = 1.0

    # Temperature
    temp = weather.get("temp_f")
    if temp is not None:
        t_eff = (temp - 70) / 10 * 0.03
        mult *= (1 + t_eff)
        if temp >= 80:
            summary.append(f"🌡️ {temp:.0f}°F (carries well)")
        elif temp <= 55:
            summary.append(f"🥶 {temp:.0f}°F (ball deadens)")
        else:
            summary.append(f"{temp:.0f}°F")

    # Wind
    wind_mph = weather.get("wind_mph", 0)
    wind_dir = weather.get("wind_dir_deg")
    if wind_mph and wind_dir is not None and roof != "retractable":
        # Treat retractable as 50% weather effect on average
        component = wind_component_out(wind_dir, park.get("cf_bearing", 0))
        net = component * wind_mph
        wind_eff = max(-0.25, net * 0.01)
        mult *= (1 + wind_eff)
        if net >= 8:
            summary.append(f"💨 {wind_mph:.0f}mph OUT (huge HR boost)")
        elif net >= 4:
            summary.append(f"💨 {wind_mph:.0f}mph out")
        elif net <= -8:
            summary.append(f"🌬️ {wind_mph:.0f}mph IN (kills flyballs)")
        elif net <= -4:
            summary.append(f"🌬️ {wind_mph:.0f}mph in")
        else:
            summary.append(f"{wind_mph:.0f}mph cross")
    elif roof == "retractable":
        summary.append("Retractable roof (assume neutral)")

    # Precipitation
    pp = weather.get("precip_prob", 0)
    if pp and pp >= 50:
        summary.append(f"☔ {pp:.0f}% rain")

    return round(mult, 3), " · ".join(summary) if summary else "Neutral"
