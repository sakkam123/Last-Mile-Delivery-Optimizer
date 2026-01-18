"""
Configuration file for Last-Mile Delivery Optimizer
"""

# Depot configuration
DEPOT_LOCATION = (0, 0)

# Vehicle configuration
NUM_VEHICLES = 3
VEHICLE_CAPACITY = 250  # kg - increased for robust demand bounds
VEHICLE_SPEED = 50  # km/h

# Cost parameters
COST_PER_KM = 0.5  # euros
EARLY_PENALTY = 20   # euros per hour early - Balanced to enforce time windows
LATE_PENALTY = 40   # euros per hour late - Balanced to enforce time windows
FIXED_VEHICLE_COST = 50  # euros per vehicle used

# Time window parameters
SERVICE_TIME = 0.083  # hours (5 minutes per delivery - realistic)
WORK_DAY_START = 8   # 8:00 AM
WORK_DAY_END = 18    # 6:00 PM
MAX_ROUTE_DURATION = 8  # hours - tighter to encourage efficient routes

# Grid size for customer locations
GRID_SIZE = 15  # km x km (compact for realistic urban delivery)

# Data generation parameters
NUM_HISTORICAL_DAYS = 100
NUM_CUSTOMERS_MIN = 15
NUM_CUSTOMERS_MAX = 25

# Random seed for reproducibility
RANDOM_SEED = 42

# Prediction model parameters
TRAIN_TEST_SPLIT = 0.2
CV_FOLDS = 5
