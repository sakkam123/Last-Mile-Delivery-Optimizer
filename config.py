"""
Configuration file for Last-Mile Delivery Optimizer
"""

# Depot configuration
DEPOT_LOCATION = (0, 0)

# Vehicle configuration
NUM_VEHICLES = 3
VEHICLE_CAPACITY = 200  # kg → augmenter pour pouvoir livrer la demande robuste
VEHICLE_SPEED = 50  # km/h → plus rapide pour respecter fenêtres horaires

# Cost parameters
COST_PER_KM = 0.5  # euros
EARLY_PENALTY = 5   # euros per hour early → réduire pour que solver accepte
LATE_PENALTY = 10   # euros per hour late → réduire
FIXED_VEHICLE_COST = 50  # euros per vehicle used

# Time window parameters
SERVICE_TIME = 0.05  # heures (≈3 minutes par livraison) → réduit pour gagner du temps
WORK_DAY_START = 8   # 8:00 AM
WORK_DAY_END = 18    # 6:00 PM
MAX_ROUTE_DURATION = 12  # heures → allonger pour éviter contraintes strictes

# Grid size for customer locations
GRID_SIZE = 15  # km x km → plus compact pour réduire distances

# Data generation parameters
NUM_HISTORICAL_DAYS = 100
NUM_CUSTOMERS_MIN = 15
NUM_CUSTOMERS_MAX = 25

# Random seed for reproducibility
RANDOM_SEED = 42

# Prediction model parameters
TRAIN_TEST_SPLIT = 0.2
CV_FOLDS = 5
