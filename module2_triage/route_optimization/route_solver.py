import argparse
import itertools
import json
import math

import folium
import requests


def haversine_km(a, b):
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def build_distance_matrix_demo(coords):
    n = len(coords)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = haversine_km(coords[i], coords[j])
    return matrix


def build_distance_matrix_osrm(coords, timeout=10):
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    url = f"http://router.project-osrm.org/table/v1/driving/{coord_str}?annotations=distance"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM table request failed: {data}")
    return [[d / 1000.0 for d in row] for row in data["distances"]]


def get_route_geometry_osrm(coord_a, coord_b, timeout=10):
    url = (f"http://router.project-osrm.org/route/v1/driving/"
           f"{coord_a[1]},{coord_a[0]};{coord_b[1]},{coord_b[0]}?overview=full&geometries=geojson")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        return [coord_a, coord_b]
    coords = data["routes"][0]["geometry"]["coordinates"]
    return [(lat, lon) for lon, lat in coords]


def solve_tsp_bruteforce(distance_matrix, start=0):
    n = len(distance_matrix)
    others = [i for i in range(n) if i != start]
    best_order, best_dist = None, float("inf")
    for perm in itertools.permutations(others):
        order = [start] + list(perm)
        dist = sum(distance_matrix[order[i]][order[i + 1]] for i in range(len(order) - 1))
        if dist < best_dist:
            best_dist, best_order = dist, order
    return best_order, best_dist


def solve_tsp_nearest_neighbor(distance_matrix, start=0):
    n = len(distance_matrix)
    visited = [start]
    unvisited = set(range(n)) - {start}
    total_dist = 0.0
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: distance_matrix[current][j])
        total_dist += distance_matrix[current][nxt]
        visited.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return visited, total_dist


def solve_tsp(distance_matrix, start=0):
    n = len(distance_matrix)
    if n <= 8:
        return solve_tsp_bruteforce(distance_matrix, start)
    print(f"({n} locations > 8 -- using nearest-neighbor heuristic, not exact optimum)")
    return solve_tsp_nearest_neighbor(distance_matrix, start)


def build_map(victims, order, coords, use_real_geometry=True, out_path="route_map.html"):
    center_lat = sum(c[0] for c in coords) / len(coords)
    center_lon = sum(c[1] for c in coords) / len(coords)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    for rank, idx in enumerate(order):
        v = victims[idx]
        folium.Marker(
            coords[idx],
            popup=f"Stop {rank + 1}: {v.get('victim_id', idx)}",
            icon=folium.Icon(color="red" if rank == 0 else "blue", icon="info-sign"),
        ).add_to(m)

    for i in range(len(order) - 1):
        a_idx, b_idx = order[i], order[i + 1]
        if use_real_geometry:
            try:
                path = get_route_geometry_osrm(coords[a_idx], coords[b_idx])
            except Exception as e:
                print(f"  (route geometry fetch failed for segment {i}, using straight line: {e})")
                path = [coords[a_idx], coords[b_idx]]
        else:
            path = [coords[a_idx], coords[b_idx]]
        folium.PolyLine(path, color="blue", weight=4, opacity=0.7).add_to(m)

    m.save(out_path)
    return out_path


DEMO_VICTIMS = [
    {"victim_id": "V01", "lat": 28.6139, "lon": 77.2090},
    {"victim_id": "V02", "lat": 28.6304, "lon": 77.2177},
    {"victim_id": "V03", "lat": 28.5921, "lon": 77.2290},
    {"victim_id": "V04", "lat": 28.6100, "lon": 77.1980},
    {"victim_id": "V05", "lat": 28.6015, "lon": 77.2350},
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--victims", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", type=str, default="route_map.html")
    args = parser.parse_args()

    if args.demo:
        victims = DEMO_VICTIMS
        coords = [(v["lat"], v["lon"]) for v in victims]
        distance_matrix = build_distance_matrix_demo(coords)
        use_real_geometry = False
    else:
        assert args.victims, "Provide --victims victims.json (or use --demo)"
        with open(args.victims) as f:
            victims = json.load(f)
        coords = [(v["lat"], v["lon"]) for v in victims]
        distance_matrix = build_distance_matrix_osrm(coords)
        use_real_geometry = True

    order, total_dist = solve_tsp(distance_matrix, start=args.start)
    route_labels = " -> ".join(victims[i]["victim_id"] for i in order)
    print(f"\nOptimal route: {route_labels}")
    print(f"Total distance: {total_dist:.2f} km")

    out_path = build_map(victims, order, coords, use_real_geometry=use_real_geometry, out_path=args.out)
    print(f"Map saved: {out_path}")