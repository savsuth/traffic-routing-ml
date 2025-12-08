"""
Time-Dependent A* Algorithm
Uses ML predictions for dynamic edge weights.
"""

import heapq
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeDependentAStar:


    def __init__(self, graph_path: str, models_dir: str = "data/models"):
        """Initialize router with graph and ML models."""
        logger.info("Loading graph...")
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)
        logger.info(f"✓ Loaded graph with {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")

        # Load ML models
        self.models = {}
        models_dir = Path(models_dir)

        for model_file in models_dir.glob("model_*.pkl"):
            road_type = model_file.stem.replace("model_", "")
            with open(model_file, 'rb') as f:
                self.models[road_type] = pickle.load(f)
            logger.info(f"✓ Loaded {road_type} model")


        feature_file = Path("data/processed/feature_columns.txt")
        if not feature_file.exists():
            feature_file = models_dir / "feature_columns.txt"

        with open(feature_file, 'r') as f:
            self.feature_cols = [line.strip() for line in f.readlines()]

    def find_route(
            self,
            start_node: int,
            goal_node: int,
            departure_time: datetime,
            weather: str = 'sunny',
            temperature: float = 70.0
    ) -> Dict:

        logger.info(f"Finding route from {start_node} to {goal_node}")
        logger.info(f"Departure: {departure_time}")

        # Priority queue: (f_score, node, current_time, g_score)
        open_set = [(0, start_node, departure_time, 0)]
        came_from = {}
        g_score = {start_node: 0}
        time_at_node = {start_node: departure_time}

        nodes_explored = 0

        while open_set:
            _, current, current_time, current_g = heapq.heappop(open_set)

            nodes_explored += 1

            # Goal reached
            if current == goal_node:
                route = self._reconstruct_path(came_from, current)
                total_time = g_score[goal_node]
                arrival_time = time_at_node[goal_node]

                # Calculate total distance
                total_distance = sum(
                    self.G[route[i]][route[i + 1]][0].get('length', 0)
                    for i in range(len(route) - 1)
                )

                logger.info(f"✓ Route found! Explored {nodes_explored} nodes")

                return {
                    'route': route,
                    'total_time_sec': total_time,
                    'total_time_min': total_time / 60,
                    'total_distance_m': total_distance,
                    'total_distance_km': total_distance / 1000,
                    'departure_time': departure_time,
                    'arrival_time': arrival_time,
                    'nodes_explored': nodes_explored
                }

            # Skip if we've found a better path to this node
            if current in g_score and current_g > g_score[current]:
                continue

            # Explore neighbors
            for neighbor in self.G.neighbors(current):
                # Get edge data
                edge_data = self.G[current][neighbor][0]

                # Predict travel time for this edge
                predicted_time = self._predict_travel_time(
                    edge_data,
                    current_time,
                    weather,
                    temperature
                )

                tentative_g = current_g + predicted_time

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    # Calculate arrival time at neighbor
                    arrival_time = current_time + timedelta(seconds=predicted_time)
                    time_at_node[neighbor] = arrival_time

                    # Heuristic: straight-line distance / average speed
                    h_score = self._heuristic(neighbor, goal_node)
                    f_score = tentative_g + h_score

                    heapq.heappush(open_set, (f_score, neighbor, arrival_time, tentative_g))

        logger.warning("No route found!")
        return None

    def _predict_travel_time(
            self,
            edge_data: Dict,
            arrival_time: datetime,
            weather: str,
            temperature: float
    ) -> float:
        """Predict travel time for an edge using ML model."""

        # Extract edge features
        highway_type = edge_data.get('highway_type', 'unknown')
        if isinstance(highway_type, list):
            highway_type = highway_type[0]

        speed_limit = edge_data.get('speed_limit', 30)
        length = edge_data.get('length', 100)

        # Determine which model to use
        if highway_type in ['motorway', 'trunk']:
            model_key = 'highway'
        elif highway_type in ['primary', 'secondary']:
            model_key = 'major'
        else:
            model_key = 'minor'

        if model_key not in self.models:
            # Fallback to free-flow time
            return edge_data.get('free_flow_time', length / 13.41)

        # Create feature vector
        features = self._create_features(
            arrival_time,
            highway_type,
            speed_limit,
            weather,
            temperature
        )

        # Predict
        model = self.models[model_key]
        predicted_time = model.predict([features])[0]

        # Sanity check
        min_time = length / 30  # Max 30 m/s
        max_time = length / 1  # Min 1 m/s (gridlock)
        predicted_time = np.clip(predicted_time, min_time, max_time)

        return predicted_time

    def _create_features(
            self,
            timestamp: datetime,
            highway_type: str,
            speed_limit: float,
            weather: str,
            temperature: float
    ) -> List[float]:
        """Create feature vector for ML prediction."""

        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        is_weekend = day_of_week >= 5

        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        # Rush hour flags
        is_morning_rush = 1 if 7 <= hour <= 9 else 0
        is_evening_rush = 1 if 16 <= hour <= 19 else 0
        is_night = 1 if hour >= 22 or hour <= 5 else 0
        is_rush_hour = 1 if is_morning_rush or is_evening_rush else 0

        # Weather
        is_rainy = 1 if weather == 'rainy' else 0
        is_snowy = 1 if weather == 'snowy' else 0

        # Road hierarchy
        road_hierarchy_map = {
            'motorway': 5, 'trunk': 4, 'primary': 3,
            'secondary': 2, 'tertiary': 1, 'residential': 0,
            'service': 0, 'unclassified': 1
        }
        road_hierarchy = road_hierarchy_map.get(highway_type, 1)

        # Build feature vector (order must match training!)
        features = [
            hour, hour_sin, hour_cos,
            day_of_week, day_sin, day_cos,
            month, int(is_weekend),
            is_morning_rush, is_evening_rush, is_night, is_rush_hour,
            temperature, is_rainy, is_snowy,
            0,  # event_nearby (assume no event)
            road_hierarchy, speed_limit,
            3000.0,  # distance_to_downtown (placeholder)
            0,  # is_downtown
            0  # is_suburbs
        ]

        return features

    def _heuristic(self, node1: int, node2: int) -> float:
        """Heuristic function: straight-line distance / average speed."""
        if node1 not in self.G.nodes or node2 not in self.G.nodes:
            return 0

        n1_data = self.G.nodes[node1]
        n2_data = self.G.nodes[node2]

        # Haversine distance
        lat1, lon1 = n1_data['y'], n1_data['x']
        lat2, lon2 = n2_data['y'], n2_data['x']

        R = 6371000  # Earth radius in meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)

        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c

        # Estimate time: distance / average speed (13.41 m/s = 30 mph)
        estimated_time = distance / 13.41

        return estimated_time

    def _reconstruct_path(self, came_from: Dict, current: int) -> List[int]:
        """Reconstruct path from came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def get_route_details(self, route: List[int]) -> List[Dict]:
        """Get detailed information about each segment of the route."""
        details = []

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            edge_data = self.G[u][v][0]

            u_data = self.G.nodes[u]
            v_data = self.G.nodes[v]

            details.append({
                'from_node': u,
                'to_node': v,
                'from_coords': (u_data['y'], u_data['x']),
                'to_coords': (v_data['y'], v_data['x']),
                'length_m': edge_data.get('length', 0),
                'road_type': edge_data.get('highway_type', 'unknown'),
                'speed_limit': edge_data.get('speed_limit', 30)
            })

        return details


def main():
    """Test the time-dependent A* router."""

    # Find latest graph
    graph_files = list(Path("data/raw").glob("road_network_*.pkl"))
    if not graph_files:
        raise FileNotFoundError("No graph file found!")

    latest_graph = max(graph_files, key=lambda p: p.stat().st_mtime)

    # Initialize router
    router = TimeDependentAStar(str(latest_graph))

    # Get random start and goal nodes
    nodes = list(router.G.nodes())
    start_node = nodes[100]
    goal_node = nodes[500]

    print(f"\n{'=' * 60}")
    print(f"Testing Time-Dependent A*")
    print(f"{'=' * 60}")
    print(f"Start node: {start_node}")
    print(f"Goal node: {goal_node}")

    # Test at different times
    scenarios = [
        (datetime(2024, 12, 7, 8, 0), "Morning rush hour"),
        (datetime(2024, 12, 7, 14, 0), "Afternoon (off-peak)"),
        (datetime(2024, 12, 7, 17, 30), "Evening rush hour"),
    ]

    for departure_time, description in scenarios:
        print(f"\n--- {description} ---")

        result = router.find_route(
            start_node=start_node,
            goal_node=goal_node,
            departure_time=departure_time,
            weather='sunny',
            temperature=70.0
        )

        if result:
            print(f"Route length: {len(result['route'])} nodes")
            print(f"Distance: {result['total_distance_km']:.2f} km")
            print(f"Estimated time: {result['total_time_min']:.1f} minutes")
            print(f"Arrival: {result['arrival_time'].strftime('%I:%M %p')}")

    print(f"\n{'=' * 60}")
    print("✓ A* implementation complete!")
    print("✓ Project is now functional!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()