# Last-Mile Delivery Optimizer

A decision-support tool for urban grocery delivery that uses a predict-then-optimize approach to generate robust daily delivery routes.

## Project Overview

This project addresses the challenge of last-mile delivery in urban environments where:
- **Customer demands** are uncertain until arrival
- **Delivery time windows** are soft (violations incur penalties)
- **Travel times** vary based on traffic and other factors

The system uses a two-stage pipeline:
1. **Predict**: Machine learning models forecast travel times and demand ranges from historical data
2. **Optimize**: OR-Tools solver generates cost-optimal routes considering soft time windows

## Features

- 🔮 **Demand Prediction**: Random Forest models predict delivery sizes with uncertainty bounds
- ⏱️ **Travel Time Forecasting**: ML models estimate travel times based on distance, time of day, and day of week
- 🚚 **Route Optimization**: Vehicle Routing Problem (VRP) solver with capacity and time window constraints
- 📊 **Comprehensive Analysis**: Compare predicted vs actual costs, visualize routes, analyze penalties
- 📈 **Robust Planning**: Uses prediction intervals to handle uncertainty

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or navigate to the project directory:
```bash
cd last_mile_delivery
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python main.py
```

This will:
1. Generate synthetic historical data
2. Train prediction models
3. Generate test scenarios
4. Optimize routes using predictions
5. Evaluate performance
6. Generate visualizations

### Running Individual Components

**Generate historical data only:**
```bash
python data_generator.py
```

**Train prediction models:**
```bash
python predictor.py
```

**Run optimization on a scenario:**
```bash
python optimizer.py
```

**Visualize results:**
```bash
python visualize.py
```

## Project Structure

```
last_mile_delivery/
│
├── config.py              # Configuration parameters
├── data_generator.py      # Historical data generation
├── predictor.py           # ML prediction models
├── optimizer.py           # OR-Tools route optimization
├── pipeline.py            # Predict-then-optimize integration
├── visualize.py           # Visualization tools
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
└── README.md             # This file

data/                      # Generated data (auto-created)
├── historical_data.csv
├── test_scenarios.csv
└── models/               # Trained ML models

results/                   # Optimization results (auto-created)
├── routes.json
├── performance_metrics.json
└── visualizations/
```

## Methodology

### 1. Data Generation
- Simulates 100 days of historical delivery data
- Generates realistic customer locations, demands, and time windows
- Adds noise to simulate real-world uncertainty

### 2. Prediction Phase
- **Demand Prediction**: Random Forest Regressor with quantile estimation
  - Features: customer location, day of week, historical average
  - Outputs: predicted demand with 80% confidence intervals
  
- **Travel Time Prediction**: Random Forest Regressor
  - Features: distance, time of day, day of week, weather proxy
  - Outputs: estimated travel time

### 3. Optimization Phase
- Formulation: Capacitated VRP with soft time windows
- Objective: Minimize total cost = travel cost + early/late penalties + vehicle costs
- Constraints: Vehicle capacity, route duration, single visit per customer
- Solver: Google OR-Tools CP-SAT solver

### 4. Evaluation
- Compare predicted costs vs actual realized costs
- Analyze prediction accuracy (MAE, RMSE)
- Measure time window violation rates
- Compare against baselines (no prediction, average values)

## Key Parameters

Edit [config.py](config.py) to adjust:

- `NUM_VEHICLES`: Number of available delivery vehicles (default: 3)
- `VEHICLE_CAPACITY`: Maximum load per vehicle in kg (default: 100)
- `LATE_PENALTY`: Cost per hour of late delivery (default: 20 euros)
- `EARLY_PENALTY`: Cost per hour of early delivery (default: 10 euros)
- `NUM_CUSTOMERS_MIN/MAX`: Customer range per day (default: 15-25)

## Results and Analysis

The system generates:

1. **Route Visualizations**: Maps showing optimized delivery routes
2. **Performance Metrics**: 
   - Total cost (travel + penalties)
   - Prediction accuracy (demand and travel time)
   - Time window compliance rates
   - Vehicle utilization
3. **Comparative Analysis**: 
   - Predicted vs actual performance
   - Baseline comparisons
   - Sensitivity analysis

## Limitations and Future Work

### Current Limitations
- Assumes homogeneous vehicle fleet
- Static depot location
- Does not consider real-time traffic updates
- Simplified demand uncertainty model

### Potential Improvements
- Multi-depot support
- Dynamic re-routing based on real-time updates
- Deep learning for demand prediction
- Stochastic optimization approaches
- Integration with real traffic APIs

## Dependencies

Core libraries:
- **OR-Tools**: Constraint programming and routing optimization
- **scikit-learn**: Machine learning models
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization

## License

This project is for academic purposes as part of the Combinatorial Optimization course.

## Authors

ENSI - Combinatorial Optimization Course

## References

1. Toth, P., & Vigo, D. (2014). Vehicle routing: problems, methods, and applications.
2. Elmachtoub, A. N., & Grigas, P. (2022). Smart "predict, then optimize". Management Science.
3. Google OR-Tools Documentation: https://developers.google.com/optimization

---

For questions or issues, please refer to the project documentation or course materials.
