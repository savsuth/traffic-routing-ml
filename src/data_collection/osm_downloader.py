"""
OSM Road Network Downloader
Downloads and processes OpenStreetMap road network data for traffic routing.
"""

import osmnx as ox
import networkx as nx
import pickle
from pathlib import Path
from typing import Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OSMDownloader:
    """Download and process road network from OpenStreetMap."""

    def __init__(self, city_name: str, data_dir: str = "data/raw"):
        self.city_name = city_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_network(self, network_type: str = "drive", simplify: bool = True) -> nx.MultiDiGraph:
        """Download road network from OpenStreetMap."""
        logger.info(f"Downloading {network_type} network for {self.city_name}...")

        try:
            G = ox.graph_from_place(
                self.city_name,
                network_type=network_type,
                simplify=simplify
            )

            logger.info(f"✓ Downloaded {len(G.nodes)} nodes and {len(G.edges)} edges")
            return G

        except Exception as e:
            logger.error(f"Failed to download network: {e}")
            raise

    def add_edge_attributes(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Add useful attributes to edges for routing."""
        logger.info("Adding edge attributes...")

        for u, v, key, data in G.edges(keys=True, data=True):
            length = data.get('length', 100)

            if 'maxspeed' in data:
                maxspeed = data['maxspeed']
                if isinstance(maxspeed, list):
                    maxspeed = maxspeed[0]
                try:
                    speed_limit = float(str(maxspeed).split()[0])
                except:
                    speed_limit = self._estimate_speed_limit(data.get('highway', 'residential'))
            else:
                speed_limit = self._estimate_speed_limit(data.get('highway', 'residential'))

            speed_ms = speed_limit * 0.44704
            travel_time = length / speed_ms if speed_ms > 0 else length / 13.41

            G[u][v][key]['speed_limit'] = speed_limit
            G[u][v][key]['free_flow_time'] = travel_time
            G[u][v][key]['highway_type'] = data.get('highway', 'unknown')

        logger.info("✓ Edge attributes added")
        return G

    def _estimate_speed_limit(self, highway_type: str) -> float:
        """Estimate speed limit based on road type."""
        speed_map = {
            'motorway': 65,
            'trunk': 55,
            'primary': 45,
            'secondary': 35,
            'tertiary': 30,
            'residential': 25,
            'service': 15,
            'unclassified': 25,
        }

        if isinstance(highway_type, list):
            highway_type = highway_type[0]

        return speed_map.get(highway_type, 30)

    def save_network(self, G: nx.MultiDiGraph, prefix: str = "road_network") -> Tuple[Path, Path]:
        """Save network in multiple formats."""
        logger.info("Saving network...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        graphml_path = self.data_dir / f"{prefix}_{timestamp}.graphml"
        ox.save_graphml(G, graphml_path)

        pickle_path = self.data_dir / f"{prefix}_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"✓ Saved to {graphml_path}")
        logger.info(f"✓ Saved to {pickle_path}")

        return graphml_path, pickle_path

    def get_network_stats(self, G: nx.MultiDiGraph) -> dict:
        """Get basic statistics about the network."""
        stats = {
            'num_nodes': len(G.nodes),
            'num_edges': len(G.edges),
            'avg_node_degree': sum(dict(G.degree()).values()) / len(G.nodes),
        }

        highway_types = {}
        for u, v, data in G.edges(data=True):
            htype = data.get('highway', 'unknown')
            if isinstance(htype, list):
                htype = htype[0]
            highway_types[htype] = highway_types.get(htype, 0) + 1

        stats['highway_type_distribution'] = highway_types
        return stats

    def visualize_network(self, G: nx.MultiDiGraph, output_file: str = "network_map.html"):
        """Create interactive map visualization."""
        logger.info("Creating visualization...")

        try:
            import folium

            center_node = list(G.nodes(data=True))[0]
            center_lat = center_node[1]['y']
            center_lon = center_node[1]['x']

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=13,
                tiles='OpenStreetMap'
            )

            edge_sample = list(G.edges(data=True))[:1000]
            for u, v, data in edge_sample:
                if 'geometry' in data:
                    coords = [(point[1], point[0]) for point in data['geometry'].coords]
                else:
                    u_data = G.nodes[u]
                    v_data = G.nodes[v]
                    coords = [(u_data['y'], u_data['x']), (v_data['y'], v_data['x'])]

                folium.PolyLine(coords, color='blue', weight=2, opacity=0.6).add_to(m)

            output_path = self.data_dir / output_file
            m.save(str(output_path))
            logger.info(f"✓ Map saved to {output_path}")

        except ImportError:
            logger.warning("folium not installed, skipping visualization")


def main():
    """Main function to download and process network."""

    downloader = OSMDownloader("Boston, Massachusetts, USA")

    G = downloader.download_network(network_type='drive')
    G = downloader.add_edge_attributes(G)

    stats = downloader.get_network_stats(G)
    print("\n=== Network Statistics ===")
    for key, value in stats.items():
        if key != 'highway_type_distribution':
            print(f"{key}: {value}")

    print("\n=== Road Type Distribution ===")
    for road_type, count in sorted(
            stats['highway_type_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
    )[:10]:
        print(f"{road_type}: {count}")

    downloader.save_network(G)
    downloader.visualize_network(G)

    print("\n✓ Download complete!")


if __name__ == "__main__":
    main()