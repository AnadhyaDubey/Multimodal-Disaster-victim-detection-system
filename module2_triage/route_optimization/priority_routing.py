"""
route_optimization.py
----------------------
Rescue-routing optimization: given a rescue team's starting location and a
list of triaged victims (each with lat/lon and a priority/urgency score),
produce a visit order that balances travel distance against rescue urgency.

This is the dashboard version of the "stretch goal" rescue-routing feature
described in the project plan (module2_triage/route_optimization/) —
a TSP-with-priority-weights heuristic, kept dependency-light (pure Python,
no external solver) so it runs anywhere.

Approach: priority-weighted nearest-neighbor construction.
  - At each step, score every unvisited victim by:
        cost = distance_km / urgency_weight
    so a closer OR more urgent victim scores lower (better) and gets
    visited sooner. This is a heuristic, not an exact TSP solver — it's
    deliberately simple and explainable, matching the "rule-based,
    explainable baseline" style already used elsewhere in this project
    (see module2_triage/baseline/rule_based_triage.py).
  - Repeats until every victim has been assigned a stop number.

A true Dijkstra/TSP-with-priority-weights solver (as named in the project
plan) would need a road network graph; this haversine-distance heuristic is
the straight-line approximation appropriate for a demo without a routing
API/account.
"""

import math

URGENCY_WEIGHT = {
    "critical": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def optimize_route(start_lat, start_lon, victims):
    """
    victims: list of dicts, each with at least 'victim_id', 'lat', 'lon',
        and 'urgency' (critical/high/medium/low).

    Returns: list of dicts (one per victim, in visit order) each augmented
        with 'stop_number', 'distance_from_prev_km', 'cumulative_km'.
    """
    remaining = list(victims)
    route = []
    cur_lat, cur_lon = start_lat, start_lon
    cumulative = 0.0

    while remaining:
        best_idx, best_cost, best_dist = None, None, None
        for i, v in enumerate(remaining):
            dist = haversine_km(cur_lat, cur_lon, v["lat"], v["lon"])
            weight = URGENCY_WEIGHT.get(v.get("urgency", "medium"), 2.0)
            cost = dist / weight  # lower is better: closer AND/OR more urgent
            if best_cost is None or cost < best_cost:
                best_idx, best_cost, best_dist = i, cost, dist

        chosen = remaining.pop(best_idx)
        cumulative += best_dist
        route.append({
            **chosen,
            "stop_number": len(route) + 1,
            "distance_from_prev_km": round(best_dist, 2),
            "cumulative_km": round(cumulative, 2),
        })
        cur_lat, cur_lon = chosen["lat"], chosen["lon"]

    return route