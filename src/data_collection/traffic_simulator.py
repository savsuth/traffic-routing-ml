"""
Traffic Data Simulator
Generates realistic traffic patterns for training ML models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficSimulator:
    """Generate realistic traffic data based on road network."""

    def __init__(self, graph_path: str, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)
        logger.info(f"✓ Loaded {len(self.G.edges)} edges")

    def generate_traffic_data(
            self,
            start_date: str = "2024-06-01",
            num_days: int = 180,
            hours_per_day: int = 24
    ) -> pd.DataFrame:
        """Generate traffic data for all edges over time period."""

        logger.info(f"Generating {num_days} days of traffic data...")

        start = datetime.strptime(start_date, "%Y-%m-%d")
        all_data = []

        # Progress bar for days
        for day in tqdm(range(num_days), desc="Generating days"):
            current_date = start + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5

            # Determine if there's an event today (5% chance)
            has_event = np.random.random() < 0.05

            # Weather for the day
            weather = self._generate_weather(current_date)

            for hour in range(hours_per_day):
                timestamp = current_date + timedelta(hours=hour)

                # Sample edges (process all would be too slow)
                # In production, you'd process all edges
                edge_sample = list(self.G.edges(data=True, keys=True))[:5000]

                for u, v, key, data in edge_sample:
                    edge_id = f"{u}_{v}_{key}"
                    length = data.get('length', 100)
                    highway_type = data.get('highway_type', 'unknown')
                    speed_limit = data.get('speed_limit', 30)

                    # Calculate traffic speed based on conditions
                    speed = self._calculate_speed(
                        hour=hour,
                        is_weekend=is_weekend,
                        highway_type=highway_type,
                        speed_limit=speed_limit,
                        weather=weather,
                        has_event=has_event,
                        distance_to_downtown=self._distance_to_downtown(u)
                    )

                    travel_time = (length / (speed * 0.44704)) if speed > 0 else 1000

                    # Create record
                    record = {
                        'edge_id': edge_id,
                        'timestamp': timestamp,
                        'speed_mph': round(speed, 2),
                        'travel_time_sec': round(travel_time, 2),
                        'hour': hour,
                        'day_of_week': current_date.weekday(),
                        'is_weekend': is_weekend,
                        'is_rush_hour': self._is_rush_hour(hour),
                        'weather': weather['condition'],
                        'temperature': weather['temp'],
                        'event_nearby': has_event and self._distance_to_downtown(u) < 2000,
                        'road_type': highway_type,
                        'speed_limit': speed_limit,
                        'distance_to_downtown': round(self._distance_to_downtown(u), 2)
                    }

                    all_data.append(record)

        df = pd.DataFrame(all_data)
        logger.info(f"✓ Generated {len(df):,} records")

        return df

    def _calculate_speed(
            self,
            hour: int,
            is_weekend: bool,
            highway_type: str,
            speed_limit: float,
            weather: dict,
            has_event: bool,
            distance_to_downtown: float
    ) -> float:
        """Calculate realistic speed based on conditions."""

        # Start with speed limit
        speed = speed_limit

        # Rush hour impact (7-9 AM, 4-7 PM on weekdays)
        if not is_weekend and self._is_rush_hour(hour):
            if highway_type in ['motorway', 'trunk', 'primary']:
                speed *= np.random.uniform(0.5, 0.7)  # 30-50% slower
            else:
                speed *= np.random.uniform(0.7, 0.85)  # 15-30% slower

        # Weekend traffic (generally better)
        if is_weekend:
            speed *= np.random.uniform(1.0, 1.2)

        # Late night (very light traffic)
        if hour >= 23 or hour <= 5:
            speed *= np.random.uniform(1.1, 1.3)

        # Weather impact
        if weather['condition'] == 'rainy':
            speed *= np.random.uniform(0.75, 0.85)
        elif weather['condition'] == 'snowy':
            speed *= np.random.uniform(0.6, 0.75)

        # Event impact (if near downtown)
        if has_event and distance_to_downtown < 2000:
            speed *= np.random.uniform(0.6, 0.8)

        # Road type base adjustment
        if highway_type == 'residential':
            speed *= np.random.uniform(0.9, 1.0)

        # Add random noise (±10%)
        speed *= np.random.uniform(0.9, 1.1)

        # Ensure reasonable bounds
        speed = max(5, min(speed, speed_limit * 1.2))

        return speed

    def _is_rush_hour(self, hour: int) -> bool:
        """Check if hour is rush hour."""
        return (7 <= hour <= 9) or (16 <= hour <= 19)

    def _generate_weather(self, date: datetime) -> dict:
        """Generate weather for a given day."""
        # Simple seasonal logic
        month = date.month

        if month in [12, 1, 2]:  # Winter
            temp = np.random.uniform(25, 45)
            condition = np.random.choice(['sunny', 'rainy', 'snowy'], p=[0.4, 0.2, 0.4])
        elif month in [6, 7, 8]:  # Summer
            temp = np.random.uniform(70, 90)
            condition = np.random.choice(['sunny', 'rainy'], p=[0.7, 0.3])
        else:  # Spring/Fall
            temp = np.random.uniform(50, 70)
            condition = np.random.choice(['sunny', 'rainy'], p=[0.6, 0.4])

        return {'temp': round(temp, 1), 'condition': condition}

    def _distance_to_downtown(self, node_id) -> float:
        """Calculate approximate distance to downtown Boston."""
        # Downtown Boston coords: (42.3601, -71.0589)
        if node_id not in self.G.nodes:
            return 5000

        node_data = self.G.nodes[node_id]
        lat, lon = node_data['y'], node_data['x']

        # Simple Euclidean distance (good enough for simulation)
        # 1 degree ≈ 111 km
        dx = (lon - (-71.0589)) * 111000 * np.cos(np.radians(42.3601))
        dy = (lat - 42.3601) * 111000

        return np.sqrt(dx ** 2 + dy ** 2)

    def save_data(self, df: pd.DataFrame, filename: str = "traffic_training_data.csv"):
        """Save generated data."""
        output_path = self.output_dir / filename

        logger.info(f"Saving to {output_path}...")
        df.to_csv(output_path, index=False)

        # Save summary statistics
        summary = {
            'total_records': len(df),
            'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'unique_edges': df['edge_id'].nunique(),
            'avg_speed': df['speed_mph'].mean(),
            'avg_travel_time': df['travel_time_sec'].mean(),
        }

        summary_path = self.output_dir / "traffic_summary.txt"
        with open(summary_path, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        logger.info(f"✓ Saved {len(df):,} records")
        logger.info(f"✓ Summary saved to {summary_path}")


def main():
    """Generate traffic simulation data."""

    # Find the most recent graph file
    graph_files = list(Path("data/raw").glob("road_network_*.pkl"))
    if not graph_files:
        raise FileNotFoundError("No graph file found! Run osm_downloader.py first.")

    latest_graph = max(graph_files, key=lambda p: p.stat().st_mtime)

    # Create simulator
    simulator = TrafficSimulator(str(latest_graph))

    # Generate 6 months of data
    df = simulator.generate_traffic_data(
        start_date="2024-06-01",
        num_days=180,  # 6 months
        hours_per_day=24
    )

    # Save
    simulator.save_data(df)

    print("\n=== Data Generation Complete ===")
    print(f"Total records: {len(df):,}")
    print(f"File size: ~{len(df) * 150 / 1_000_000:.1f} MB")
    print("\n✓ Ready for Phase 3: Feature Engineering!")


if __name__ == "__main__":
    main()
