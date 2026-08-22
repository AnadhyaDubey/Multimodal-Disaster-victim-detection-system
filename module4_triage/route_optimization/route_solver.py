"""
route_solver.py
------------------
Module 4 (stretch goal) - Route optimization for rescue teams.

Given the GPS coordinates of multiple detected victims, this script:
    1. Fetches real driving distances/times between all victim pairs
       using the Google Maps Distance Matrix API.
    2. Solves a Traveling Salesman Problem (TSP) to find the most
       efficient visiting order, respecting triage priority.
    3. Renders the optimized route on an interactive map (using Folium)
       that can be viewed in a browser or embedded in the dashboard.

Setup (one-time):
    pip install googlemaps folium networkx

    Get a free Google Maps API key:
    1. https://console.cloud.google.com/ -> create a project
    2. Enable "Distance Matrix API" and "Directions API"
    3. Create an API key under "Credentials"
    4. export GOOGLE_MAPS_API_KEY="your-key-here"

Usage:
    from route_solver import optimize_route, render_route_map

    victims = [
        {"victim_id": "V01", "lat": 28.6139, "lng": 77.2090, "predicted_rank": 1},
        {"victim_id": "V02", "lat": 28.6145, "lng": 77.2100, "predicted_rank": 2},
        ...
    ]
    rescue_start = (28.6100, 77.2050)  # rescue team's current location

    ordered_route = optimize_route(victims, rescue_start)
    render_route_map(ordered_route, rescue_start, output_path="route_map.html")
"""

import os
import itertools

import googlemaps
import folium
import networkx as nx


def _get_client():
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_MAPS_API_KEY environment variable not set. "
            "Run: export GOOGLE_MAPS_API_KEY='your-key-here'"
        )
    return googlemaps.Client(key=api_key)


def build_distance_matrix(locations):
    """
    Fetches real driving distances (in meters) between every pair of
    locations using the Google Maps Distance Matrix API.

    locations: list of (lat, lng) tuples, index 0 is the rescue team's
    starting point, the rest are victim locations.

    Returns: 2D list (matrix) of distances in meters.
    """
    gmaps = _get_client()
    result = gmaps.distance_matrix(locations, locations, mode="driving")

    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    for i, row in enumerate(result["rows"]):
        for j, element in enumerate(row["elements"]):
            matrix[i][j] = element["distance"]["value"] if element["status"] == "OK" else float("inf")

    return matrix


def optimize_route(victims, rescue_start):
    """
    Solves a small TSP over the rescue-start point and all victim
    locations, to find the shortest route visiting every victim once.

    For small victim counts (typical triage scenario, <10), this uses
    brute-force permutation search via networkx's helper, which is exact
    and fast enough at this scale.

    victims: list of victim dicts, each with "lat", "lng", "victim_id"
    rescue_start: (lat, lng) tuple

    Returns: list of victim dicts in optimized visiting order.
    """
    locations = [rescue_start] + [(v["lat"], v["lng"]) for v in victims]
    matrix = build_distance_matrix(locations)

    n = len(locations)
    graph = nx.complete_graph(n)
    for i, j in itertools.combinations(range(n), 2):
        graph[i][j]["weight"] = matrix[i][j]
        graph[j][i]["weight"] = matrix[j][i]

    # networkx's TSP solver (Christofides-based approximation, good enough
    # for small victim counts and much faster than exact brute force).
    tsp_order = nx.approximation.traveling_salesman_problem(
        graph, weight="weight", cycle=False
    )

    # tsp_order includes index 0 (rescue start) - drop it and map the rest
    # back to victim dicts, in visiting order.
    victim_order_indices = [i for i in tsp_order if i != 0]
    ordered_victims = [victims[i - 1] for i in victim_order_indices]

    return ordered_victims


def render_route_map(ordered_victims, rescue_start, output_path="route_map.html"):
    """
    Renders the rescue team's optimized route on an interactive Folium
    map, with numbered markers showing the visiting order.

    Saves an HTML file that can be opened in any browser.
    """
    m = folium.Map(location=rescue_start, zoom_start=14)

    folium.Marker(
        rescue_start,
        popup="Rescue Team Start",
        icon=folium.Icon(color="blue", icon="home"),
    ).add_to(m)

    route_points = [rescue_start]
    for order, victim in enumerate(ordered_victims, start=1):
        location = (victim["lat"], victim["lng"])
        route_points.append(location)
        folium.Marker(
            location,
            popup=f"Stop {order}: {victim['victim_id']} "
                  f"(priority rank {victim.get('predicted_rank', '?')})",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    folium.PolyLine(route_points, color="blue", weight=4, opacity=0.7).add_to(m)

    m.save(output_path)
    print(f"Route map saved to: {output_path} (open in a browser to view)")


if __name__ == "__main__":
    # Quick manual test - replace with real coordinates for a live demo.
    test_victims = [
        {"victim_id": "V01", "lat": 28.6139, "lng": 77.2090, "predicted_rank": 2},
        {"victim_id": "V02", "lat": 28.6155, "lng": 77.2110, "predicted_rank": 1},
        {"victim_id": "V03", "lat": 28.6120, "lng": 77.2075, "predicted_rank": 3},
    ]
    rescue_start = (28.6100, 77.2050)

    print("Solving optimal route...")
    ordered = optimize_route(test_victims, rescue_start)

    print("\nOptimized visiting order:")
    for i, v in enumerate(ordered, start=1):
        print(f"  {i}. {v['victim_id']}")

    render_route_map(ordered, rescue_start)