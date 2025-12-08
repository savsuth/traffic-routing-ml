"""
Traffic-Aware Routing System - Interactive Web Interface
Professional portfolio-ready visualization
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import sys

sys.path.append('.')

from src.routing.time_dependent_astar import TimeDependentAStar

# Page config
st.set_page_config(
    page_title="Traffic-Aware Routing System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeIn 1s;
    }

    .subheader {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1.5s;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }

    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
        color: white;
    }

    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        color: white;
    }

    .info-box {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }

    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        color: #155724;
        margin: 0.5rem 0;
    }

    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        color: #856404;
        margin: 0.5rem 0;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def load_system():
    graph_file = max(Path("data/raw").glob("road_network_*.pkl"))
    router = TimeDependentAStar(str(graph_file))
    return router

if 'router' not in st.session_state:
    with st.spinner('Initializing routing system...'):
        st.session_state.router = load_system()
        st.session_state.graph = st.session_state.router.G

if 'start_node' not in st.session_state:
    st.session_state.start_node = None
if 'goal_node' not in st.session_state:
    st.session_state.goal_node = None
if 'route_result' not in st.session_state:
    st.session_state.route_result = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []


# Tabs
tabs = st.tabs([
    "Route Finder",
    "Route Comparison",
    "About Project",
    "Example Routes"
])

# ==================== TAB 1: ROUTE FINDER ====================
with tabs[0]:
    with st.sidebar:
        st.header("Route Configuration")

        st.subheader("Departure Time")
        departure_date = st.date_input(
            "Date",
            value=datetime(2024, 12, 9),
            label_visibility="collapsed"
        )

        departure_time = st.time_input(
            "Time",
            value=datetime.strptime("08:00", "%H:%M").time(),
            label_visibility="collapsed"
        )

        departure_datetime = datetime.combine(departure_date, departure_time)

        st.markdown("**Quick Select:**")
        time_col1, time_col2 = st.columns(2)
        with time_col1:
            if st.button("8 AM", use_container_width=True):
                departure_time = datetime.strptime("08:00", "%H:%M").time()
                st.rerun()
            if st.button("5 PM", use_container_width=True):
                departure_time = datetime.strptime("17:00", "%H:%M").time()
                st.rerun()
        with time_col2:
            if st.button("2 PM", use_container_width=True):
                departure_time = datetime.strptime("14:00", "%H:%M").time()
                st.rerun()
            if st.button("11 PM", use_container_width=True):
                departure_time = datetime.strptime("23:00", "%H:%M").time()
                st.rerun()

        st.markdown("---")

        st.subheader("Weather Conditions")
        weather_display = st.selectbox(
            "Weather",
            ["Sunny", "Rainy", "Snowy"],
            index=0,
            label_visibility="collapsed"
        )
        weather_value = weather_display.lower()

        temperature = st.slider(
            "Temperature (°F)",
            min_value=0,
            max_value=100,
            value=70,
            help="Temperature affects traffic patterns"
        )

        st.markdown("---")

        st.subheader("Selected Points")
        if st.session_state.start_node:
            st.markdown(f'<div class="success-box"><b>Start:</b> Node {st.session_state.start_node}</div>', unsafe_allow_html=True)
        else:
            st.info("Click map to set start point")

        if st.session_state.goal_node:
            st.markdown(f'<div class="success-box"><b>Goal:</b> Node {st.session_state.goal_node}</div>', unsafe_allow_html=True)
        else:
            st.info("Click map to set destination")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset", use_container_width=True, type="secondary"):
                st.session_state.start_node = None
                st.session_state.goal_node = None
                st.session_state.route_result = None
                st.rerun()

        with col2:
            find_disabled = (st.session_state.start_node is None or st.session_state.goal_node is None)
            if st.button("Find Route", use_container_width=True, type="primary", disabled=find_disabled):
                with st.spinner('Computing optimal route...'):
                    result = st.session_state.router.find_route(
                        start_node=st.session_state.start_node,
                        goal_node=st.session_state.goal_node,
                        departure_time=departure_datetime,
                        weather=weather_value,
                        temperature=temperature
                    )
                    st.session_state.route_result = result

                if result:
                    st.success("Route found!")
                else:
                    st.error("No route found!")

    # Main content
    map_col, info_col = st.columns([2, 1])

    with map_col:
        st.subheader("Interactive Map")

        G = st.session_state.graph
        center_node = list(G.nodes(data=True))[0]
        center_lat, center_lon = center_node[1]['y'], center_node[1]['x']

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='CartoDB positron'
        )

        road_sample = list(G.edges(data=True))[:800]
        for u, v, data in road_sample:
            if 'geometry' in data:
                coords = [(point[1], point[0]) for point in data['geometry'].coords]
            else:
                u_data = G.nodes[u]
                v_data = G.nodes[v]
                coords = [(u_data['y'], u_data['x']), (v_data['y'], v_data['x'])]

            folium.PolyLine(
                coords,
                color='#95a5a6',
                weight=1.5,
                opacity=0.3
            ).add_to(m)

        if st.session_state.start_node:
            node_data = G.nodes[st.session_state.start_node]
            folium.Marker(
                [node_data['y'], node_data['x']],
                popup=f"<b>Start</b><br>Node: {st.session_state.start_node}",
                icon=folium.Icon(color='green', icon='play', prefix='fa'),
                tooltip="Start Point"
            ).add_to(m)

        if st.session_state.goal_node:
            node_data = G.nodes[st.session_state.goal_node]
            folium.Marker(
                [node_data['y'], node_data['x']],
                popup=f"<b>Destination</b><br>Node: {st.session_state.goal_node}",
                icon=folium.Icon(color='red', icon='stop', prefix='fa'),
                tooltip="Destination"
            ).add_to(m)

        if st.session_state.route_result:
            route = st.session_state.route_result['route']
            route_coords = []

            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                edge_data = G[u][v][0]

                if 'geometry' in edge_data:
                    for point in edge_data['geometry'].coords:
                        route_coords.append((point[1], point[0]))
                else:
                    u_data = G.nodes[u]
                    v_data = G.nodes[v]
                    route_coords.append((u_data['y'], u_data['x']))
                    route_coords.append((v_data['y'], v_data['x']))

            folium.PolyLine(
                route_coords,
                color='#3498db',
                weight=6,
                opacity=0.9,
                tooltip="Optimal Route"
            ).add_to(m)

        map_data = st_folium(
            m,
            width=None,
            height=650,
            returned_objects=["last_clicked"]
        )

        if map_data['last_clicked']:
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lon = map_data['last_clicked']['lng']

            min_dist = float('inf')
            nearest_node = None

            for node, data in G.nodes(data=True):
                dist = ((data['y'] - clicked_lat) ** 2 + (data['x'] - clicked_lon) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node

            if st.session_state.start_node is None:
                st.session_state.start_node = nearest_node
                st.rerun()
            elif st.session_state.goal_node is None and nearest_node != st.session_state.start_node:
                st.session_state.goal_node = nearest_node
                st.rerun()

    with info_col:
        st.subheader("Route Analytics")

        if st.session_state.route_result:
            result = st.session_state.route_result

            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-label">TRAVEL TIME</div>
                <div class="stat-value">{result['total_time_min']:.1f} min</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="stat-label">DISTANCE</div>
                <div class="stat-value">{result['total_distance_km']:.2f} km</div>
            </div>
            """, unsafe_allow_html=True)

            avg_speed = result['total_distance_km'] / (result['total_time_min'] / 60)
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="stat-label">AVG SPEED</div>
                <div class="stat-value">{avg_speed:.1f} km/h</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Trip Details")

            details_html = f"""
            <div class="info-box">
                <p><b>Arrival:</b> {result['arrival_time'].strftime('%I:%M %p')}</p>
                <p><b>Route Segments:</b> {len(result['route'])} nodes</p>
                <p><b>Search Efficiency:</b> {result['nodes_explored']:,} nodes explored</p>
                <p><b>Departure:</b> {result['departure_time'].strftime('%b %d, %I:%M %p')}</p>
                <p><b>Conditions:</b> {weather_value.capitalize()}, {temperature}°F</p>
            </div>
            """
            st.markdown(details_html, unsafe_allow_html=True)

            st.markdown("### Traffic Assessment")
            if avg_speed < 20:
                st.markdown('<div class="warning-box"><b>Heavy Traffic Expected</b><br>Consider alternative times</div>', unsafe_allow_html=True)
            elif avg_speed < 30:
                st.markdown('<div class="info-box"><b>Moderate Traffic</b><br>Normal conditions</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box"><b>Light Traffic</b><br>Good conditions!</div>', unsafe_allow_html=True)

            if st.button("Add to Comparison", use_container_width=True):
                comparison_entry = {
                    'time': departure_datetime.strftime('%I:%M %p'),
                    'weather': weather_display,
                    'duration_min': result['total_time_min'],
                    'distance_km': result['total_distance_km'],
                    'speed_kmh': avg_speed
                }
                st.session_state.comparison_results.append(comparison_entry)
                st.success("Added to comparison!")

        else:
            st.markdown("### How to Use")
            st.markdown("""
            1. Click on the map to set start point (green marker)
            2. Click again to set destination (red marker)
            3. Adjust settings in the sidebar (time, weather, etc.)
            4. Click 'Find Route' button
            5. Route appears in blue showing the fastest path
            """)

            st.markdown("---")
            st.markdown("### Features")
            st.markdown("""
            - Time-dependent routing
            - Weather-aware predictions
            - ML-enhanced travel times
            - Real Boston road network
            - Rush hour optimization
            """)

# ==================== TAB 2: COMPARISON ====================
with tabs[1]:
    st.subheader("Route Comparison")

    if len(st.session_state.comparison_results) == 0:
        st.info("No routes to compare yet. Find routes in the Route Finder tab and add them to comparison!")
    else:
        df_compare = pd.DataFrame(st.session_state.comparison_results)

        st.dataframe(
            df_compare.style.format({
                'duration_min': '{:.1f}',
                'distance_km': '{:.2f}',
                'speed_kmh': '{:.1f}'
            }),
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:
            fig_time = go.Figure(data=[
                go.Bar(
                    x=df_compare['time'],
                    y=df_compare['duration_min'],
                    marker_color='#3498db',
                    text=df_compare['duration_min'].round(1),
                    textposition='auto',
                )
            ])
            fig_time.update_layout(
                title="Travel Time Comparison",
                xaxis_title="Departure Time",
                yaxis_title="Duration (min)",
                height=400
            )
            st.plotly_chart(fig_time, use_container_width=True)

        with col2:
            fig_speed = go.Figure(data=[
                go.Bar(
                    x=df_compare['time'],
                    y=df_compare['speed_kmh'],
                    marker_color='#2ecc71',
                    text=df_compare['speed_kmh'].round(1),
                    textposition='auto',
                )
            ])
            fig_speed.update_layout(
                title="Average Speed Comparison",
                xaxis_title="Departure Time",
                yaxis_title="Speed (km/h)",
                height=400
            )
            st.plotly_chart(fig_speed, use_container_width=True)

        if st.button("Clear All Comparisons"):
            st.session_state.comparison_results = []
            st.rerun()

# ==================== TAB 3: ABOUT ====================
with tabs[2]:
    st.markdown("""
    ## About This Project

    ### Overview
    This is a **production-ready traffic routing system** that combines classical graph algorithms with modern machine learning to predict optimal routes based on time-dependent traffic patterns.

    ### System Architecture

    **1. Data Collection**
    - Downloaded real Boston road network from OpenStreetMap
    - 11,409 nodes (intersections)
    - 26,083 edges (road segments)

    **2. Traffic Simulation**
    - Generated 21.6M realistic traffic records over 180 days
    - Modeled rush hour patterns, weather effects, and events

    **3. Machine Learning**
    - Trained LightGBM models on 4.3M samples
    - Separate models for highway, major, and minor roads
    - Features: time of day, weather, road type, historical patterns
    - **Performance:** MAE ~27 seconds, R² ~0.89

    **4. Routing Algorithm**
    - Implemented A* pathfinding with time-dependent edge weights
    - ML model predicts travel time for each road segment
    - Accounts for departure time, weather, and traffic patterns

    ### Key Metrics
    - **Prediction Accuracy:** ±27 seconds average error
    - **Model Performance:** R² score of 0.89
    - **Training Data:** 4.3 million samples
    - **Code:** ~1,200 lines of Python

    ### Tech Stack
    - **Languages:** Python 3.13
    - **ML:** LightGBM, scikit-learn
    - **Graphs:** NetworkX, OSMnx
    - **Data:** Pandas, NumPy
    - **Visualization:** Streamlit, Folium, Plotly

    ### Skills Demonstrated
    - Graph algorithms (A* pathfinding)
    - Machine learning (regression, feature engineering)
    - Data engineering (processing 21M+ records)
    - Software architecture (modular design)
    - Web development (interactive interface)

    ### Future Enhancements
    - Real-time traffic API integration
    - Turn-by-turn directions
    - Multiple route alternatives
    - Mobile responsive design
    - Historical route analytics

    ---

    **Author:** Aasav Patel  
    **Institution:** Northeastern University  
    **Course:** Computer Science Master's Program
    """)

# ==================== TAB 4: EXAMPLE ROUTES ====================
with tabs[3]:
    st.subheader("Pre-configured Example Routes")

    G = st.session_state.graph
    nodes = list(G.nodes())

    examples = [
        {"name": "Downtown to Cambridge", "start": nodes[100], "end": nodes[500]},
        {"name": "North End to South Boston", "start": nodes[200], "end": nodes[600]},
        {"name": "Back Bay to Fenway", "start": nodes[300], "end": nodes[700]},
        {"name": "Seaport to Beacon Hill", "start": nodes[150], "end": nodes[550]},
    ]

    st.markdown("Click any example to load it:")

    cols = st.columns(2)
    for idx, example in enumerate(examples):
        with cols[idx % 2]:
            if st.button(
                example['name'],
                key=f"example_{idx}",
                use_container_width=True
            ):
                st.session_state.start_node = example['start']
                st.session_state.goal_node = example['end']
                st.success(f"Loaded: {example['name']}")
                st.info("Go to 'Route Finder' tab and click 'Find Route'")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p style='font-size: 1.1rem;'><strong>AI-Powered Traffic Routing System</strong></p>
    <p>A* Algorithm + Machine Learning | Real-world Road Network | Portfolio Project</p>
    <p style='font-size: 0.9rem; margin-top: 10px;'>
        Built with Python, LightGBM, NetworkX, OSMnx, Streamlit | Northeastern University
    </p>
</div>
""", unsafe_allow_html=True)