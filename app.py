from flask import Flask, render_template, request, jsonify
import pandas as pd
from geopy.distance import geodesic
import osmnx as ox
import networkx as nx
import numpy as np
import random
import re
from collections import defaultdict

app = Flask(__name__)

# ACO Algorithm Implementation
def aco_shortest_path(G, start_node, end_node, iterations=30, ants=20, alpha=1.0, beta=2.0, evaporation=0.5):
    pheromones = defaultdict(lambda: 1.0)
    best_path = None
    best_length = float('inf')
    for _ in range(iterations):
        ant_paths = []
        for _ in range(ants):
            current_node = start_node
            path = [current_node]
            visited = set([current_node])
            while current_node != end_node:
                neighbors = [n for n in G.neighbors(current_node) if n not in visited]
                if not neighbors:
                    break
                probabilities = []
                for neighbor in neighbors:
                    
                    edge_data = G.get_edge_data(current_node, neighbor)
                    if isinstance(edge_data, dict):
                        edge_data = list(edge_data.values())[0]
                    distance = edge_data['length']
                    pheromone = pheromones[(current_node, neighbor)]
                    prob = (pheromone ** alpha) * ((1.0 / distance) ** beta)
                    probabilities.append(prob)
                total = sum(probabilities)
                if total == 0:
                    break
                probabilities = [p/total for p in probabilities]
                next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
                path.append(next_node)
                visited.add(next_node)
                current_node = next_node
            if path[-1] == end_node:
                path_length = sum(
                    list(G.get_edge_data(u, v).values())[0]['length']
                    for u, v in zip(path[:-1], path[1:])
                )
                ant_paths.append((path, path_length))
                if path_length < best_length:
                    best_path = path
                    best_length = path_length
        for u, v in list(pheromones.keys()):
            pheromones[(u, v)] *= (1 - evaporation)
        for path, length in ant_paths:
            deposit = 1.0 / length
            for u, v in zip(path[:-1], path[1:]):
                pheromones[(u, v)] += deposit
    if best_path is None:
        return nx.shortest_path(G, start_node, end_node, weight='length')
    return best_path

def normalize_name(name):
    return re.sub(r'[^a-z0-9]', '', str(name).strip().lower())

# Load hospital data (first 1000 points)
df = pd.read_csv("Minor_Datset.csv").dropna(subset=["Latitude", "Longitude"]).head(1000)
hospitals = df.to_dict(orient='records')

@app.route('/')
def index():
    return render_template("index.html", hospitals=hospitals)

@app.route('/plan_route', methods=['POST'])
def plan_route():
    data = request.get_json()
    print(f"Received data: {data}")
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        current_lat = float(data.get('current_latitude', 0))
        current_lng = float(data.get('current_longitude', 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid coordinates provided"}), 400

    selected_hospital_name = data.get('hospital_name', '') or data.get('hospitalName', '') or data.get('hospital', '')
    user_location = (current_lat, current_lng)
    print(f"Selected hospital: '{selected_hospital_name}'")

    # Hospital matching
    target_hospital = None
    if selected_hospital_name and selected_hospital_name.strip():
        normalized_selected = normalize_name(selected_hospital_name)
        print(f"Normalized selected name: '{normalized_selected}'")
        # Exact match
        for hospital in hospitals:
            hospital_name = hospital.get("Name", "")
            normalized_hospital = normalize_name(hospital_name)
            if normalized_hospital == normalized_selected:
                print(f"EXACT MATCH FOUND: {hospital_name}")
                target_hospital = hospital
                break
        # Partial match if exact not found
        if not target_hospital:
            for hospital in hospitals:
                hospital_name = hospital.get("Name", "")
                normalized_hospital = normalize_name(hospital_name)
                if normalized_selected in normalized_hospital or normalized_hospital in normalized_selected:
                    print(f"PARTIAL MATCH FOUND: {hospital_name}")
                    target_hospital = hospital
                    break

    # Fallback to nearest hospital
    if not target_hospital:
        print("No match found, finding nearest hospital")
        min_distance = float('inf')
        for hospital in hospitals:
            hospital_location = (hospital['Latitude'], hospital['Longitude'])
            dist = geodesic(user_location, hospital_location).km
            if dist < min_distance:
                min_distance = dist
                target_hospital = hospital
        print(f"Nearest hospital: {target_hospital.get('Name', 'Unknown')}")

    # Load road network and find route
    try:
        G = ox.graph_from_point(user_location, dist=20000, network_type='drive', simplify=True)
        orig_node = ox.distance.nearest_nodes(G, current_lng, current_lat)
        dest_node = ox.distance.nearest_nodes(G, target_hospital["Longitude"], target_hospital["Latitude"])

        # Try ACO, fallback to Dijkstra if needed
        try:
            route = aco_shortest_path(G, orig_node, dest_node)
        except Exception as e:
            print(f"ACO failed: {e}, using Dijkstra")
            route = nx.shortest_path(G, orig_node, dest_node, weight='length')

        # Route coordinates
        route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]

        # Path length using road network
        path_length_m = sum(
            list(G.get_edge_data(u, v).values())[0]['length']
            for u, v in zip(route[:-1], route[1:])
        )
        path_length_km = round(path_length_m / 1000, 2)
        time_min = round((path_length_km / 40) * 60, 1)  

        print(f"Route calculated: {len(route_coords)} points, {path_length_km} km, {time_min} min")

        return jsonify({
            "user_location": {"latitude": current_lat, "longitude": current_lng},
            "target_hospital": {
                "name": target_hospital.get("Name", "Selected Hospital"),
                "lat": target_hospital["Latitude"],
                "lng": target_hospital["Longitude"]
            },
            "route_coords": route_coords,
            "distance_km": path_length_km,
            "time_min": time_min,
            "matched": target_hospital is not None
        })
    except Exception as e:
        print(f"Error calculating route: {str(e)}")
        return jsonify({
            "error": f"Could not calculate route: {str(e)}",
            "user_location": {"latitude": current_lat, "longitude": current_lng},
            "target_hospital": {
                "name": target_hospital.get("Name", "Selected Hospital"),
                "lat": target_hospital["Latitude"],
                "lng": target_hospital["Longitude"]
            },
            "route_coords": [],
            "distance_km": None,
            "time_min": None,
            "matched": False
        })

if __name__ == '__main__':
    app.run(debug=True)
