"""
osrm_routing.py
---------------
Adds REAL ROAD routing on top of route_optimization.optimize_route().

- Visit ORDER  : priority-weighted heuristic (priority_routing.py), so
                 urgent victims come first.
- Road PATH    : OSRM public demo server (free, no API key, no credits).
                 If it is unreachable, we fall back to straight lines.

Note: router.project-osrm.org is a shared demo server meant for light use
(fine for a college demo). For heavy use, self-host OSRM with Docker.
"""

import requests

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"


def get_road_route(points, timeout=10):
    """
    points: list of (lat, lon) in visit order, start point first.
    Returns dict {path, distance_km, duration_min, source}
      path   = list of [lat, lon] following real roads
      source = "osrm" or "straight-line" (fallback)
    """
    if len(points) < 2:
        return {"path": [list(p) for p in points], "distance_km": 0.0,
                "duration_min": 0.0, "source": "straight-line"}

    # OSRM wants lon,lat order
    coords = ";".join(f"{lon},{lat}" for lat, lon in points)
    url = f"{OSRM_URL}/{coords}"
    try:
        resp = requests.get(
            url,
            params={"overview": "full", "geometries": "geojson"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok":
            raise ValueError(data.get("message", "OSRM error"))
        route = data["routes"][0]
        # GeoJSON is [lon, lat] -> flip to [lat, lon] for Folium
        path = [[lat, lon] for lon, lat in route["geometry"]["coordinates"]]
        return {
            "path": path,
            "distance_km": round(route["distance"] / 1000, 2),
            "duration_min": round(route["duration"] / 60, 1),
            "source": "osrm",
        }
    except Exception:
        return {
            "path": [list(p) for p in points],
            "distance_km": None,
            "duration_min": None,
            "source": "straight-line",
        }