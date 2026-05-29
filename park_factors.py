"""
park_factors.py
================
Static MLB park factors plus venue geographic data needed for weather/wind
calculations.

HR_FACTOR: Multiplier on league-average HR rate at this park (100 = neutral).
  Source: Baseball Savant Statcast Park Factors (3-year rolling, 2022-2024).
  Update annually from:
    https://baseballsavant.mlb.com/leaderboard/statcast-park-factors

HR_FACTOR_L: HR factor specifically for LEFT-HANDED batters (100 = neutral).
HR_FACTOR_R: HR factor specifically for RIGHT-HANDED batters (100 = neutral).
  These reflect park asymmetry. Yankee Stadium's short RF = big LHB boost
  but only modest RHB boost. Petco's deep LF suppresses RHB pull-power.
  Source: Statcast park factors by handedness.

RUNS_FACTOR: Same idea for total runs (100 = neutral).

CF_BEARING_DEG: Compass direction from home plate toward center field.
  Used to determine if wind is blowing out (additive to HR) or in (suppressive).
  Source: stadium orientation diagrams.

LF_BEARING / RF_BEARING: bearing toward LF and RF foul poles. Used to
  compute hand-aware wind effects (LHB pull = RF, RHB pull = LF).

LAT/LON: Used to fetch weather from Open-Meteo (no API key needed).
"""

import math

# Keys are MLB Stats API venue names (must match exactly what comes from
# the schedule endpoint). When in doubt, print game['venue'] and add it here.
PARKS = {
    "Coors Field":           {"hr_factor": 121, "hr_factor_L": 119, "hr_factor_R": 122, "runs_factor": 118, "cf_bearing": 0,   "lat": 39.756, "lon": -104.994, "roof": "open"},
    "Great American Ball Park":{"hr_factor": 115, "hr_factor_L": 118, "hr_factor_R": 113, "runs_factor": 107, "cf_bearing": 30,  "lat": 39.097, "lon": -84.507,  "roof": "open"},
    "Yankee Stadium":        {"hr_factor": 113, "hr_factor_L": 125, "hr_factor_R": 105, "runs_factor": 104, "cf_bearing": 22,  "lat": 40.829, "lon": -73.926,  "roof": "open"},
    "Citizens Bank Park":    {"hr_factor": 110, "hr_factor_L": 113, "hr_factor_R": 108, "runs_factor": 103, "cf_bearing": 18,  "lat": 39.906, "lon": -75.166,  "roof": "open"},
    "Globe Life Field":      {"hr_factor": 108, "hr_factor_L": 109, "hr_factor_R": 107, "runs_factor": 102, "cf_bearing": 11,  "lat": 32.747, "lon": -97.083,  "roof": "retractable"},
    "Fenway Park":           {"hr_factor": 107, "hr_factor_L": 96,  "hr_factor_R": 115, "runs_factor": 108, "cf_bearing": 50,  "lat": 42.346, "lon": -71.097,  "roof": "open"},  # Green Monster boosts RHB
    "Wrigley Field":         {"hr_factor": 105, "hr_factor_L": 105, "hr_factor_R": 105, "runs_factor": 104, "cf_bearing": 38,  "lat": 41.948, "lon": -87.655,  "roof": "open"},
    "Chase Field":           {"hr_factor": 104, "hr_factor_L": 105, "hr_factor_R": 103, "runs_factor": 101, "cf_bearing": 23,  "lat": 33.445, "lon": -112.067, "roof": "retractable"},
    "Rogers Centre":         {"hr_factor": 103, "hr_factor_L": 103, "hr_factor_R": 103, "runs_factor": 100, "cf_bearing": 0,   "lat": 43.641, "lon": -79.389,  "roof": "retractable"},
    "Daikin Park":           {"hr_factor": 102, "hr_factor_L": 99,  "hr_factor_R": 104, "runs_factor": 101, "cf_bearing": 0,   "lat": 29.757, "lon": -95.355,  "roof": "retractable"},  # Crawford Boxes boost RHB
    "Minute Maid Park":      {"hr_factor": 102, "hr_factor_L": 99,  "hr_factor_R": 104, "runs_factor": 101, "cf_bearing": 0,   "lat": 29.757, "lon": -95.355,  "roof": "retractable"},
    "Truist Park":           {"hr_factor": 101, "hr_factor_L": 101, "hr_factor_R": 101, "runs_factor": 99,  "cf_bearing": 14,  "lat": 33.890, "lon": -84.468,  "roof": "open"},
    "Nationals Park":        {"hr_factor": 100, "hr_factor_L": 100, "hr_factor_R": 100, "runs_factor": 99,  "cf_bearing": 19,  "lat": 38.873, "lon": -77.008,  "roof": "open"},
    "Target Field":          {"hr_factor": 100, "hr_factor_L": 99,  "hr_factor_R": 101, "runs_factor": 99,  "cf_bearing": 26,  "lat": 44.982, "lon": -93.278,  "roof": "open"},
    "Busch Stadium":         {"hr_factor": 99,  "hr_factor_L": 99,  "hr_factor_R": 99,  "runs_factor": 98,  "cf_bearing": 13,  "lat": 38.622, "lon": -90.193,  "roof": "open"},
    "Citi Field":            {"hr_factor": 99,  "hr_factor_L": 98,  "hr_factor_R": 99,  "runs_factor": 97,  "cf_bearing": 24,  "lat": 40.757, "lon": -73.846,  "roof": "open"},
    "American Family Field": {"hr_factor": 99,  "hr_factor_L": 100, "hr_factor_R": 98,  "runs_factor": 100, "cf_bearing": 14,  "lat": 43.028, "lon": -87.971,  "roof": "retractable"},
    "Progressive Field":     {"hr_factor": 98,  "hr_factor_L": 98,  "hr_factor_R": 98,  "runs_factor": 98,  "cf_bearing": 25,  "lat": 41.495, "lon": -81.685,  "roof": "open"},
    "PNC Park":              {"hr_factor": 98,  "hr_factor_L": 92,  "hr_factor_R": 102, "runs_factor": 99,  "cf_bearing": 117, "lat": 40.447, "lon": -80.006,  "roof": "open"},  # Deep RF suppresses LHB
    "Angel Stadium":         {"hr_factor": 97,  "hr_factor_L": 97,  "hr_factor_R": 97,  "runs_factor": 96,  "cf_bearing": 50,  "lat": 33.800, "lon": -117.883, "roof": "open"},
    "Dodger Stadium":        {"hr_factor": 97,  "hr_factor_L": 97,  "hr_factor_R": 97,  "runs_factor": 97,  "cf_bearing": 22,  "lat": 34.073, "lon": -118.240, "roof": "open"},
    "Kauffman Stadium":      {"hr_factor": 96,  "hr_factor_L": 96,  "hr_factor_R": 96,  "runs_factor": 99,  "cf_bearing": 5,   "lat": 39.051, "lon": -94.480,  "roof": "open"},
    "Comerica Park":         {"hr_factor": 95,  "hr_factor_L": 96,  "hr_factor_R": 94,  "runs_factor": 97,  "cf_bearing": 39,  "lat": 42.339, "lon": -83.048,  "roof": "open"},
    "loanDepot park":        {"hr_factor": 95,  "hr_factor_L": 95,  "hr_factor_R": 95,  "runs_factor": 96,  "cf_bearing": 36,  "lat": 25.778, "lon": -80.220,  "roof": "retractable"},
    "Petco Park":            {"hr_factor": 94,  "hr_factor_L": 95,  "hr_factor_R": 93,  "runs_factor": 95,  "cf_bearing": 14,  "lat": 32.707, "lon": -117.157, "roof": "open"},  # Deep LF
    "T-Mobile Park":         {"hr_factor": 93,  "hr_factor_L": 93,  "hr_factor_R": 93,  "runs_factor": 94,  "cf_bearing": 30,  "lat": 47.591, "lon": -122.332, "roof": "retractable"},
    "Sutter Health Park":    {"hr_factor": 95,  "hr_factor_L": 95,  "hr_factor_R": 95,  "runs_factor": 98,  "cf_bearing": 75,  "lat": 38.580, "lon": -121.513, "roof": "open"},
    "Oakland Coliseum":      {"hr_factor": 92,  "hr_factor_L": 92,  "hr_factor_R": 92,  "runs_factor": 93,  "cf_bearing": 60,  "lat": 37.752, "lon": -122.201, "roof": "open"},
    "Oracle Park":           {"hr_factor": 88,  "hr_factor_L": 82,  "hr_factor_R": 92,  "runs_factor": 92,  "cf_bearing": 99,  "lat": 37.779, "lon": -122.389, "roof": "open"},  # Triples Alley kills LHB
    "Tropicana Field":       {"hr_factor": 96,  "hr_factor_L": 96,  "hr_factor_R": 96,  "runs_factor": 96,  "cf_bearing": 45,  "lat": 27.768, "lon": -82.653,  "roof": "dome"},
    "George M. Steinbrenner Field": {"hr_factor": 100, "hr_factor_L": 100, "hr_factor_R": 100, "runs_factor": 100, "cf_bearing": 0, "lat": 27.980, "lon": -82.507, "roof": "open"},
    "Camden Yards":          {"hr_factor": 99,  "hr_factor_L": 102, "hr_factor_R": 97,  "runs_factor": 99,  "cf_bearing": 18,  "lat": 39.284, "lon": -76.622,  "roof": "open"},
    "Oriole Park at Camden Yards": {"hr_factor": 99, "hr_factor_L": 102, "hr_factor_R": 97, "runs_factor": 99, "cf_bearing": 18, "lat": 39.284, "lon": -76.622, "roof": "open"},
    # White Sox park — was "Guaranteed Rate Field", renamed "Rate Field" Jan 2025.
    # MLB Stats API venue.name returns "Rate Field" now. Include both for safety.
    "Rate Field":            {"hr_factor": 105, "hr_factor_L": 103, "hr_factor_R": 107, "runs_factor": 102, "cf_bearing": 35, "lat": 41.830, "lon": -87.634, "roof": "open"},
    "Guaranteed Rate Field": {"hr_factor": 105, "hr_factor_L": 103, "hr_factor_R": 107, "runs_factor": 102, "cf_bearing": 35, "lat": 41.830, "lon": -87.634, "roof": "open"},
}


def get_park(venue_name: str) -> dict:
    """Return park dict, falling back to neutral defaults if unknown."""
    if venue_name in PARKS:
        return PARKS[venue_name]
    # Try fuzzy match on the first word
    for k, v in PARKS.items():
        if venue_name and venue_name.split()[0].lower() in k.lower():
            return v
    return {
        "hr_factor": 100, "hr_factor_L": 100, "hr_factor_R": 100,
        "runs_factor": 100, "cf_bearing": 0,
        "lat": None, "lon": None, "roof": "open", "unknown": True,
    }


def get_park_hand_factor(venue_name: str, bats: str) -> float:
    """
    Return HR factor multiplier (1.0 = neutral) for a hitter's handedness.
    Switch hitters use the better side based on opposing pitcher hand (handled elsewhere).
    """
    park = get_park(venue_name)
    if bats == "L":
        factor = park.get("hr_factor_L", park.get("hr_factor", 100))
    elif bats == "R":
        factor = park.get("hr_factor_R", park.get("hr_factor", 100))
    else:
        factor = park.get("hr_factor", 100)
    return factor / 100.0


def wind_pull_side_multiplier(venue_name: str, bats: str,
                               wind_mph: float | None,
                               wind_dir_deg: float | None,
                               temp_f: float | None = None,
                               real_roof_closed: bool | None = None) -> tuple[float, str]:
    """
    Compute a multiplier for HR based on wind blowing toward the hitter's pull side.

    A LHB's pull side = RF (roughly cf_bearing + 35°).
    A RHB's pull side = LF (roughly cf_bearing - 35°).

    If wind is blowing FROM home plate toward the pull side, multiplier > 1.
    If wind is blowing INTO home plate from the pull side, multiplier < 1.

    Returns (multiplier, summary_string).

    Roof handling priority order (most reliable first):
    1. real_roof_closed=True (from MLB game feed) → no wind effect, period
    2. real_roof_closed=False → full wind effect (roof confirmed open)
    3. roof type is permanent dome → no wind effect ever
    4. roof type is retractable + no real status → temperature heuristic:
        - <60°F or >88°F → likely closed → no wind effect
        - 60-88°F → likely open → full effect
        - no temp data → 70% dampener
    5. open roof → full wind effect
    """
    if wind_mph is None or wind_dir_deg is None or wind_mph < 3:
        return 1.0, ""

    park = get_park(venue_name)
    cf_bearing = park.get("cf_bearing", 0)
    roof = park.get("roof", "open")

    # Priority 1+2: real status from MLB feed
    if real_roof_closed is True:
        return 1.0, "🏟️ Roof closed (MLB-confirmed) — no wind effect"
    if real_roof_closed is False:
        # Roof confirmed open → full effect regardless of roof type
        roof_factor = 1.0
        roof_note_text = " (roof confirmed OPEN)" if roof == "retractable" else ""
    elif roof == "dome":
        # Priority 3: permanent dome
        return 1.0, ""
    elif roof == "retractable":
        # Priority 4: temperature heuristic since MLB hasn't reported status yet
        if temp_f is None:
            roof_factor = 0.7
            roof_note_text = " (retractable, roof status unknown)"
        elif temp_f < 60 or temp_f > 88:
            return 1.0, f"🏟️ Retractable roof likely CLOSED ({temp_f:.0f}°F) — no wind effect"
        else:
            roof_factor = 1.0
            roof_note_text = " (retractable, likely open)"
    else:
        # Priority 5: regular open-air park
        roof_factor = 1.0
        roof_note_text = ""

    # Pull-side bearing (where the hitter pulls the ball)
    if bats == "L":
        pull_bearing = (cf_bearing + 35) % 360  # RF
        side = "RF"
    elif bats == "R":
        pull_bearing = (cf_bearing - 35) % 360  # LF
        side = "LF"
    else:
        # Switch hitter or unknown - skip the side effect
        return 1.0, ""

    # Wind direction in meteorology is the direction wind is COMING FROM.
    # We want the direction wind is BLOWING TOWARD = (wind_dir_deg + 180) % 360.
    wind_to = (wind_dir_deg + 180) % 360

    # Angular distance between wind direction and pull-side bearing
    diff = abs(((wind_to - pull_bearing + 180) % 360) - 180)

    if diff < 30:  # wind blowing directly to pull side
        # Strong tailwind effect, scales with mph
        boost = min(0.15, wind_mph * 0.012) * roof_factor
        mult = 1 + boost
        return mult, f"💨 Wind to {side} ({wind_mph:.0f}mph) — HR boost{roof_note_text}"
    elif diff < 60:
        # Partial tailwind
        boost = min(0.08, wind_mph * 0.006) * roof_factor
        mult = 1 + boost
        return mult, f"💨 Wind partially to {side} — small HR boost{roof_note_text}"
    elif diff > 150:  # wind blowing AGAINST pull side
        suppress = min(0.12, wind_mph * 0.010) * roof_factor
        mult = 1 - suppress
        return mult, f"💨 Wind into {side} ({wind_mph:.0f}mph) — HR suppress{roof_note_text}"
    elif diff > 120:
        suppress = min(0.06, wind_mph * 0.005) * roof_factor
        mult = 1 - suppress
        return mult, f"💨 Wind partially into {side} — small HR suppress{roof_note_text}"

    return 1.0, ""


def park_k_factor(venue_name: str) -> float:
    """
    Return a K-rate multiplier for a park.
    """
    park = get_park(venue_name)
    hr_f = park.get("hr_factor", 100)
    k_factor = 1.0 - (hr_f - 100) * 0.0007
    return round(max(0.95, min(1.05, k_factor)), 3)
