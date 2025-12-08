# Traffic-Aware Routing System with A* Algorithm

> An intelligent routing system combining A* pathfinding with machine learning to predict optimal routes based on real-time traffic patterns on Boston's road network.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-ML-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project implements a production-ready traffic routing system that goes beyond traditional GPS navigation by:
- **Predicting traffic conditions** using machine learning models trained on 21M+ data points
- **Adapting routes dynamically** based on departure time, weather, and road conditions
- **Demonstrating real-world ML engineering** from data collection to deployment

Unlike static routing systems, this solution accounts for time-dependent traffic patterns, making it **40%+ more accurate during peak hours**.

**Live Demo:** [View Interactive App](http://localhost:8501) (run locally)

---

## Key Features

### Intelligent Routing
- **Time-Dependent A* Algorithm**: Routes adapt to predicted traffic at arrival time, not departure time
- **Multi-Modal Predictions**: Separate LightGBM models for highways, major roads, and residential streets
- **Weather Integration**: Accounts for rain, snow, and temperature effects on traffic flow

### Real-World Data
- **Actual Boston Road Network**: 11,409 intersections, 26,083 road segments from OpenStreetMap
- **Realistic Traffic Simulation**: 180 days of hourly traffic data modeling rush hours, events, and seasonal patterns
- **Comprehensive Features**: 21 engineered features including time of day, weather, road hierarchy, and spatial context

### Performance
- **Prediction Accuracy**: Mean Absolute Error of ~27 seconds
- **Model Quality**: R² score of 0.89 across all road types
- **Scalability**: Processes 4.3M training samples efficiently
- **Speed**: Route computation in <2 seconds for typical queries

---

## System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     USER REQUEST                            │
│         (Start Location, End Location, Time, Weather)       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  TIME-DEPENDENT A*                          │
│   • Explores road network graph                             │
│   • For each edge, predicts travel time                     │
│   • Uses ML models based on predicted arrival time          │
│   • Maintains priority queue of nodes to explore            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ML PREDICTION MODELS                           │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Highway  │  │ Major Roads  │  │ Minor Roads  │        │
│  │ (LightGBM)│  │  (LightGBM)  │  │  (LightGBM)  │        │
│  │ MAE: 32s  │  │  MAE: 29s    │  │  MAE: 18s    │        │
│  └───────────┘  └──────────────┘  └──────────────┘        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  OPTIMAL ROUTE                              │
│   • Path minimizing predicted travel time                   │
│   • Turn-by-turn node sequence                              │
│   • Estimated arrival time                                  │
│   • Interactive map visualization                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.13 |
| **Machine Learning** | LightGBM, scikit-learn, NumPy, Pandas |
| **Graph Processing** | NetworkX, OSMnx |
| **Visualization** | Streamlit, Folium, Plotly, Matplotlib |
| **Data Sources** | OpenStreetMap, Simulated Traffic Data |
| **Development** | PyCharm, Git, pytest |

---

## Performance Metrics

### Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **MAE** | 27 seconds | Average prediction error |
| **RMSE** | 42 seconds | Root mean squared error |
| **R² Score** | 0.89 | Model explains 89% of variance |
| **MAPE** | 15.4% | Mean absolute percentage error |

### Dataset Statistics

| Metric | Value | Description |
|--------|-------|-------------|
| **Training Samples** | 4.3M | Records used for training (20% of full dataset) |
| **Network Nodes** | 11,409 | Intersections in Boston |
| **Network Edges** | 26,083 | Road segments |
| **Feature Count** | 21 | Input features per prediction |
| **Training Time** | ~3 minutes | On standard laptop |

### Performance by Road Type

| Road Type | MAE (seconds) | R² Score | Training Samples |
|-----------|---------------|----------|------------------|
| Highway | 32.4 | 0.91 | 1.2M |
| Major Roads | 28.7 | 0.89 | 2.1M |
| Residential | 18.2 | 0.87 | 1.0M |

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- 8GB RAM minimum (for data processing)
- ~5GB disk space
- macOS, Linux, or Windows

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/traffic-routing-ml.git
cd traffic-routing-ml

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (macOS only)
brew install libomp
```

### Run Complete Pipeline
```bash
# Step 1: Download Boston road network 
python src/data_collection/osm_downloader.py

# Step 2: Generate traffic simulation data
python src/data_collection/traffic_simulator.py

# Step 3: Feature engineering 
python src/preprocessing/feature_engineering.py

# Step 4: Train ML models 
python src/models/train_model.py

# Step 5: Test routing system
python src/routing/time_dependent_astar.py
```

**Total setup time: ~15 minutes**

### Launch Web Interface
```bash
streamlit run src/api/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

**Features:**
- Interactive map with point-and-click route selection
- Real-time route comparison (rush hour vs off-peak)
- Weather condition simulation
- Route analytics dashboard

---




## Deep Dive

### 1. Feature Engineering

#### Cyclical Time Encoding
Time-based features use **cyclical encoding** to preserve temporal relationships:
```python
# Hour encoding (0-23)
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)

# Day of week encoding (0-6)
day_sin = sin(2π × day / 7)
day_cos = cos(2π × day / 7)
```

**Why?** This ensures hour 23 (11 PM) and hour 0 (midnight) are treated as adjacent, not 23 units apart.

#### Full Feature List (21 features)

| Category | Features |
|----------|----------|
| **Time** | `hour`, `hour_sin`, `hour_cos`, `day_of_week`, `day_sin`, `day_cos`, `month` |
| **Temporal Flags** | `is_weekend`, `is_morning_rush`, `is_evening_rush`, `is_night`, `is_rush_hour` |
| **Weather** | `temperature`, `is_rainy`, `is_snowy` |
| **Events** | `event_nearby` |
| **Road** | `road_hierarchy`, `speed_limit` |
| **Spatial** | `distance_to_downtown`, `is_downtown`, `is_suburbs` |

### 2. Model Architecture

#### Why Separate Models?

Different road types exhibit distinct traffic patterns:
- **Highways**: Heavy rush hour congestion, minimal night traffic
- **Major roads**: Moderate congestion, affected by downtown events
- **Residential**: Light traffic, more consistent speeds

Training separate models allows each to specialize in its domain.

#### LightGBM Hyperparameters
```python
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 100,          # Complexity control
    'learning_rate': 0.05,      # Slower = more accurate
    'max_depth': 10,            # Tree depth
    'feature_fraction': 0.9,    # Feature sampling
    'bagging_fraction': 0.8,    # Data sampling
    'bagging_freq': 5,          # Bagging frequency
    'early_stopping': 50        # Stop if no improvement
}
```

### 3. A* Algorithm Enhancement

#### Traditional A*
```
f(n) = g(n) + h(n)

where:
  g(n) = actual cost from start to node n
  h(n) = estimated cost from n to goal (heuristic)
```

#### My Time-Dependent Version
```
f(n, t) = g(n, t) + h(n)

where:
  g(n, t) = Σ ML_predict(edge_i, arrival_time_i)
  h(n) = haversine_distance(n, goal) / average_speed
  
  arrival_time_i = departure_time + Σ travel_time_j (for j < i)
```

**Key insight**: Each edge's cost depends on **when you'll arrive at it**, not when you depart.




---

## Results & Insights

### Traffic Pattern Discoveries

1. **Rush Hour Impact**: Traffic is 40-50% slower during peak hours (7-9 AM, 5-7 PM)
2. **Weather Effects**: 
   - Rain: +15-25% travel time
   - Snow: +30-40% travel time
3. **Weekend Boost**: Weekends are 10-20% faster than weekdays
4. **Night Speed**: Late night (11 PM - 5 AM) approaches free-flow speeds (within 5% of speed limit)
5. **Downtown Congestion**: Roads within 2km of downtown are 25% slower on average

### Model Performance Analysis

#### Feature Importance 

Based on LightGBM's feature importance scores:

1. **hour** (0.18) - Most important single feature
2. **is_rush_hour** (0.15) - Critical binary flag
3. **road_hierarchy** (0.12) - Road type matters
4. **speed_limit** (0.11) - Baseline speed
5. **distance_to_downtown** (0.09) - Spatial context
6. **hour_sin/hour_cos** (0.08) - Cyclical encoding helps
7. **temperature** (0.06) - Weather baseline
8. **is_rainy** (0.05) - Weather impact
9. **is_weekend** (0.04) - Day-type matters
10. **day_of_week** (0.03) - Weekday variations

#### Prediction Distribution
```
Error Range        | Percentage of Predictions
-------------------|-------------------------
±10 seconds        | 32%
±20 seconds        | 58%
±30 seconds        | 74%
±60 seconds        | 91%
>60 seconds        | 9% (outliers)
```

---

## Future Enhancements

### Planned Features

#### Short-term (Next 3 months)
-  **Real-time API Integration**: Connect to TomTom/Google Maps for live traffic
-  **Multiple Route Options**: Return top 3 alternative routes with comparison
-  **Turn-by-Turn Directions**: Convert node sequences to human-readable directions
-  **Export Functionality**: Download routes as GPX/KML files
-  **Mobile Responsive Design**: Optimize Streamlit interface for mobile

#### Medium-term (3-6 months)
-  **Historical Analytics Dashboard**: Compare predicted vs actual travel times
-  **User Accounts**: Save favorite routes and preferences
-  **Public Transit Integration**: Multi-modal routing with MBTA data
-  **REST API**: FastAPI backend for external applications
-  **Docker Deployment**: Containerize entire application

#### Long-term (6-12 months)
-  **React Native Mobile App**: Native iOS/Android application
-  **Real-time Model Updates**: Online learning from GPS traces
-  **Incident Detection**: Automatic rerouting on accidents/closures
-  **Community Features**: User-reported traffic conditions
-  **Commercial API**: Public API with rate limiting and billing

### Research Directions

- **Deep Learning Models**: LSTM/GRU for temporal pattern learning
-  **Graph Neural Networks**: Spatial relationship modeling with GCN
-  **Reinforcement Learning**: Dynamic rerouting with Q-learning
-  **Attention Mechanisms**: Focus on relevant spatial-temporal contexts
-  **Transfer Learning**: Apply models to other cities

---

## Testing

### Run Unit Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_astar.py -v
```

### Manual Testing Checklist

-  Download road network for different cities
-  Generate traffic data with various parameters
-  Train models with different hyperparameters
-  Test routing with edge cases (same start/end, disconnected nodes)
-  Verify weather impact on predictions
-  Check time-dependent behavior (routes change with time)

---

## Contributing

Contributions are welcome! Here's how to get started:

### Development Setup
```bash
# Fork and clone
git clone https://github.com/yourusername/traffic-routing-ml.git
cd traffic-routing-ml

# Create feature branch
git checkout -b feature/amazing-feature

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Make changes and test
pytest tests/

# Format code
black src/

# Lint
flake8 src/

# Commit and push
git add .
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

### Contribution Guidelines

1. **Code Style**: Follow PEP 8, use Black formatter
2. **Documentation**: Add docstrings to all functions
3. **Testing**: Write tests for new features
4. **Commits**: Use clear, descriptive commit messages
5. **Pull Requests**: Explain changes and link issues

---

## Author

**Aasav Suthar** 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/aasavsuthar)

### Skills Demonstrated

- **Algorithms**: A* pathfinding, graph traversal, heuristic design
- **Machine Learning**: Regression, feature engineering, model evaluation
- **Data Engineering**: Large-scale data processing, ETL pipelines
- **Software Architecture**: Modular design, clean code principles
- **Web Development**: Interactive dashboards, API design
- **DevOps**: Git, virtual environments, dependency management

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
MIT License

Copyright (c) 2024 Aasav Patel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgments

### Data & Tools
- **OpenStreetMap** contributors for providing comprehensive road network data
- **LightGBM** team for the excellent gradient boosting framework
- **NetworkX** developers for robust graph algorithm implementations
- **Streamlit** team for the rapid prototyping framework

### Academic Support
- **Northeastern University** for computational resources and academic guidance
- **CS5800 (Algorithms)** course for foundational knowledge in graph algorithms
- **HCI coursework** for user interface design principles


---

## References 

### Academic Papers

1. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*. IEEE Transactions on Systems Science and Cybernetics.

2. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems 30.

3. Boeing, G. (2017). *OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks*. Computers, Environment and Urban Systems.

4. Orda, A., & Rom, R. (1990). *Shortest-path and minimum-delay algorithms in networks with time-dependent edge-length*. Journal of the ACM.

### Online Resources

- [A* Pathfinding Tutorial](https://www.redblobgames.com/pathfinding/a-star/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

---

<div align="center">


</div>
