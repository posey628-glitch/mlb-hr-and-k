"""
park_factors.py
================
Static MLB park factors plus venue geographic data needed for weather/wind
calculations.

HR_FACTOR: Multiplier on league-average HR rate at this park (100 = neutral).
  Source: Baseball Savant Statcast Park Factors (3-year rolling, 2022-2024).
  Update annually from:
    https://baseballsavant.mlb.com/leaderboard/statcast-park-factors

RUNS_FACTOR: Same idea for total runs (100 = neutral).

CF_BEARING_DEG: Compass direction from home plate toward center field.
  Used to determine if wind is blowing out (additive to HR) or in (suppressive).
  Source: stadium orientation diagrams.

LAT/LON: Used to fetch weather from Open-Meteo (no API key needed).
"""

# Keys are MLB Stats API venue names (must match exactly what comes from
# the schedule endpoint). When in doubt, print game['venue'] and add it here.
PARKS = {
    "Coors Field":           {"hr_factor": 121, "runs_factor": 118, "cf_bearing": 0,   "lat": 39.756, "lon": -104.994, "roof": "open"},
    "Great American Ball Park":{"hr_factor": 115, "runs_factor": 107, "cf_bearing": 30,  "lat": 39.097, "lon": -84.507,  "roof": "open"},
    "Yankee Stadium":        {"hr_factor": 113, "runs_factor": 104, "cf_bearing": 22,  "lat": 40.829, "lon": -73.926,  "roof": "open"},
    "Citizens Bank Park":    {"hr_factor": 110, "runs_factor": 103, "cf_bearing": 18,  "lat": 39.906, "lon": -75.166,  "roof": "open"},
    "Globe Life Field":      {"hr_factor": 108, "runs_factor": 102, "cf_bearing": 11,  "lat": 32.747, "lon": -97.083,  "roof": "retractable"},
    "Fenway Park":           {"hr_factor": 107, "runs_factor": 108, "cf_bearing": 50,  "lat": 42.346, "lon": -71.097,  "roof": "open"},
    "Wrigley Field":         {"hr_factor": 105, "runs_factor": 104, "cf_bearing": 38,  "lat": 41.948, "lon": -87.655,  "roof": "open"},
    "Chase Field":           {"hr_factor": 104, "runs_factor": 101, "cf_bearing": 23,  "lat": 33.445, "lon": -112.067, "roof": "retractable"},
    "Rogers Centre":         {"hr_factor": 103, "runs_factor": 100, "cf_bearing": 0,   "lat": 43.641, "lon": -79.389,  "roof": "retractable"},
    "Daikin Park":           {"hr_factor": 102, "runs_factor": 101, "cf_bearing": 0,   "lat": 29.757, "lon": -95.355,  "roof": "retractable"},  # formerly Minute Maid
    "Minute Maid Park":      {"hr_factor": 102, "runs_factor": 101, "cf_bearing": 0,   "lat": 29.757, "lon": -95.355,  "roof": "retractable"},
    "Truist Park":           {"hr_factor": 101, "runs_factor": 99,  "cf_bearing": 14,  "lat": 33.890, "lon": -84.468,  "roof": "open"},
    "Nationals Park":        {"hr_factor": 100, "runs_factor": 99,  "cf_bearing": 19,  "lat": 38.873, "lon": -77.008,  "roof": "open"},
    "Target Field":          {"hr_factor": 100, "runs_factor": 99,  "cf_bearing": 26,  "lat": 44.982, "lon": -93.278,  "roof": "open"},
    "Busch Stadium":         {"hr_factor": 99,  "runs_factor": 98,  "cf_bearing": 13,  "lat": 38.622, "lon": -90.193,  "roof": "open"},
    "Citi Field":            {"hr_factor": 99,  "runs_factor": 97,  "cf_bearing": 24,  "lat": 40.757, "lon": -73.846,  "roof": "open"},
    "American Family Field": {"hr_factor": 99,  "runs_factor": 100, "cf_bearing": 14,  "lat": 43.028, "lon": -87.971,  "roof": "retractable"},
    "Progressive Field":     {"hr_factor": 98,  "runs_factor": 98,  "cf_bearing": 25,  "lat": 41.495, "lon": -81.685,  "roof": "open"},
    "PNC Park":              {"hr_factor": 98,  "runs_factor": 99,  "cf_bearing": 117, "lat": 40.447, "lon": -80.006,  "roof": "open"},
    "Angel Stadium":         {"hr_factor": 97,  "runs_factor": 96,  "cf_bearing": 50,  "lat": 33.800, "lon": -117.883, "roof": "open"},
    "Dodger Stadium":        {"hr_factor": 97,  "runs_factor": 97,  "cf_bearing": 22,  "lat": 34.073, "lon": -118.240, "roof": "open"},
    "Kauffman Stadium":      {"hr_factor": 96,  "runs_factor": 99,  "cf_bearing": 5,   "lat": 39.051, "lon": -94.480,  "roof": "open"},
    "Comerica Park":         {"hr_factor": 95,  "runs_factor": 97,  "cf_bearing": 39,  "lat": 42.339, "lon": -83.048,  "roof": "open"},
    "loanDepot park":        {"hr_factor": 95,  "runs_factor": 96,  "cf_bearing": 36,  "lat": 25.778, "lon": -80.220,  "roof": "retractable"},
    "Petco Park":            {"hr_factor": 94,  "runs_factor": 95,  "cf_bearing": 14,  "lat": 32.707, "lon": -117.157, "roof": "open"},
    "T-Mobile Park":         {"hr_factor": 93,  "runs_factor": 94,  "cf_bearing": 30,  "lat": 47.591, "lon": -122.332, "roof": "retractable"},
    "Sutter Health Park":    {"hr_factor": 95,  "runs_factor": 98,  "cf_bearing": 75,  "lat": 38.580, "lon": -121.513, "roof": "open"},  # A's 2025-2027 temp home
    "Oakland Coliseum":      {"hr_factor": 92,  "runs_factor": 93,  "cf_bearing": 60,  "lat": 37.752, "lon": -122.201, "roof": "open"},
    "Oracle Park":           {"hr_factor": 88,  "runs_factor": 92,  "cf_bearing": 99,  "lat": 37.779, "lon": -122.389, "roof": "open"},
    "Tropicana Field":       {"hr_factor": 96,  "runs_factor": 96,  "cf_bearing": 45,  "lat": 27.768, "lon": -82.653,  "roof": "dome"},
    "George M. Steinbrenner Field": {"hr_factor": 100, "runs_factor": 100, "cf_bearing": 0, "lat": 27.980, "lon": -82.507, "roof": "open"},  # Rays 2025 temp home
    "Camden Yards":          {"hr_factor": 99,  "runs_factor": 99,  "cf_bearing": 18,  "lat": 39.284, "lon": -76.622,  "roof": "open"},
    "Oriole Park at Camden Yards": {"hr_factor": 99, "runs_factor": 99, "cf_bearing": 18, "lat": 39.284, "lon": -76.622, "roof": "open"},
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
        "hr_factor": 100, "runs_factor": 100, "cf_bearing": 0,
        "lat": None, "lon": None, "roof": "open", "unknown": True,
    }
