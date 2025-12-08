import streamlit as st
import folium
from streamlit_folium import st_folium
from pathlib import Path
import pickle

st.title("Quick Map Test")

# Load graph
graph_file = max(Path("data/raw").glob("road_network_*.pkl"))
with open(graph_file, 'rb') as f:
    G = pickle.load(f)

st.write(f"Graph loaded: {len(G.nodes)} nodes")

# Create simple map
nodes_list = list(G.nodes(data=True))
center = nodes_list[0]
m = folium.Map(
    location=[center[1]['y'], center[1]['x']],
    zoom_start=13
)

# Add a test marker
folium.Marker(
    [center[1]['y'], center[1]['x']],
    popup="Test Marker",
    icon=folium.Icon(color='red')
).add_to(m)

st.write("Click on the map below:")
map_data = st_folium(m, width=700, height=500)

st.write("Debug - Map data:")
st.write(map_data)

if map_data and map_data.get('last_clicked'):
    st.success(f"You clicked: {map_data['last_clicked']}")
else:
    st.info("No clicks detected yet")
