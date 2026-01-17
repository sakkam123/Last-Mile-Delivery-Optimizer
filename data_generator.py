"""
Data Generator for Last-Mile Delivery Optimizer
Generates synthetic historical delivery data for training prediction models
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import config


class DeliveryDataGenerator:
    """Generate realistic historical delivery data"""
    
    def __init__(self, random_seed=None):
        self.random_seed = random_seed or config.RANDOM_SEED
        np.random.seed(self.random_seed)
        
    def generate_customer_location(self):
        """Generate random customer location within grid"""
        x = np.random.uniform(0, config.GRID_SIZE)
        y = np.random.uniform(0, config.GRID_SIZE)
        return (x, y)
    
    def calculate_distance(self, loc1, loc2):
        """Calculate Euclidean distance between two locations"""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def generate_travel_time(self, distance, hour, day_of_week, add_noise=True):
        """
        Generate travel time based on distance and time factors
        
        Args:
            distance: Distance in km
            hour: Hour of day (0-23)
            day_of_week: 0=Monday, 6=Sunday
            add_noise: Whether to add random noise
        """
        # Base travel time from distance and speed
        base_time = distance / config.VEHICLE_SPEED
        
        # Traffic multiplier based on hour (rush hours slower)
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            traffic_multiplier = 1.4  # Rush hour
        elif 12 <= hour <= 14:
            traffic_multiplier = 1.2  # Lunch time
        else:
            traffic_multiplier = 1.0
            
        # Weekend traffic is lighter
        if day_of_week >= 5:
            traffic_multiplier *= 0.85
            
        travel_time = base_time * traffic_multiplier
        
        # Add noise to simulate uncertainty
        if add_noise:
            noise = np.random.normal(0, 0.1 * travel_time)
            travel_time = max(0.05, travel_time + noise)  # Minimum 3 minutes
            
        return travel_time
    
    def generate_demand(self, customer_type, day_of_week, add_noise=True):
        """
        Generate customer demand (order size in kg)
        
        Args:
            customer_type: Type of customer (affects base demand)
            day_of_week: 0=Monday, 6=Sunday
            add_noise: Whether to add random noise
        """
        # Base demand by customer type
        base_demands = {
            'small': 5,
            'medium': 15,
            'large': 30
        }
        
        base_demand = base_demands[customer_type]
        
        # Weekend shopping is higher
        weekend_multiplier = 1.3 if day_of_week >= 5 else 1.0
        
        demand = base_demand * weekend_multiplier
        
        # Add noise
        if add_noise:
            noise = np.random.normal(0, 0.2 * demand)
            demand = max(1, demand + noise)  # Minimum 1 kg
            
        return demand
    
    def generate_time_window(self, preferred_hour):
        """
        Generate time window around preferred delivery hour
        
        Args:
            preferred_hour: Customer's preferred delivery hour
        """
        # Window size (typically 2-4 hours)
        window_size = np.random.uniform(2, 4)
        
        # Center window around preferred hour with some randomness
        center = preferred_hour + np.random.uniform(-0.5, 0.5)
        
        early = max(config.WORK_DAY_START, center - window_size/2)
        late = min(config.WORK_DAY_END, center + window_size/2)
        
        return (early, late)
    
    def generate_historical_data(self, num_days=None):
        """
        Generate complete historical dataset
        
        Returns:
            DataFrame with historical delivery data
        """
        num_days = num_days or config.NUM_HISTORICAL_DAYS
        
        data = []
        customer_profiles = {}  # Store customer locations and types
        
        for day in range(num_days):
            # Determine date
            start_date = datetime(2025, 1, 1)
            current_date = start_date + timedelta(days=day)
            day_of_week = current_date.weekday()
            
            # Random number of customers for this day
            num_customers = np.random.randint(
                config.NUM_CUSTOMERS_MIN, 
                config.NUM_CUSTOMERS_MAX + 1
            )
            
            for customer_idx in range(num_customers):
                customer_id = f"C{customer_idx:03d}"
                
                # Assign consistent location and type to each customer
                if customer_id not in customer_profiles:
                    location = self.generate_customer_location()
                    customer_type = np.random.choice(
                        ['small', 'medium', 'large'],
                        p=[0.4, 0.4, 0.2]
                    )
                    customer_profiles[customer_id] = {
                        'location': location,
                        'type': customer_type
                    }
                
                profile = customer_profiles[customer_id]
                location = profile['location']
                customer_type = profile['type']
                
                # Generate demand
                demand = self.generate_demand(customer_type, day_of_week)
                
                # Generate preferred delivery time
                preferred_hour = np.random.uniform(
                    config.WORK_DAY_START + 1,
                    config.WORK_DAY_END - 1
                )
                
                # Generate time window
                time_window_early, time_window_late = self.generate_time_window(preferred_hour)
                
                # Calculate distance from depot
                distance = self.calculate_distance(config.DEPOT_LOCATION, location)
                
                # Generate travel time (we'll use this as "actual" travel time)
                travel_time = self.generate_travel_time(
                    distance, 
                    int(preferred_hour), 
                    day_of_week
                )
                
                data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'day_of_week': day_of_week,
                    'customer_id': customer_id,
                    'customer_type': customer_type,
                    'location_x': location[0],
                    'location_y': location[1],
                    'distance_from_depot': distance,
                    'demand': demand,
                    'time_window_early': time_window_early,
                    'time_window_late': time_window_late,
                    'preferred_hour': preferred_hour,
                    'travel_time': travel_time
                })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_test_scenarios(self, num_scenarios=10):
        """
        Generate test scenarios for evaluation
        
        Returns:
            DataFrame with test scenarios
        """
        scenarios = []
        
        for scenario_idx in range(num_scenarios):
            # Random date in the future
            test_date = datetime(2025, 6, 1) + timedelta(days=scenario_idx)
            day_of_week = test_date.weekday()
            
            # Random number of customers
            num_customers = np.random.randint(
                config.NUM_CUSTOMERS_MIN,
                config.NUM_CUSTOMERS_MAX + 1
            )
            
            for customer_idx in range(num_customers):
                customer_id = f"TEST_C{scenario_idx:02d}_{customer_idx:02d}"
                
                # Generate location
                location = self.generate_customer_location()
                
                # Generate customer type
                customer_type = np.random.choice(
                    ['small', 'medium', 'large'],
                    p=[0.4, 0.4, 0.2]
                )
                
                # Generate actual demand (unknown to optimizer)
                actual_demand = self.generate_demand(customer_type, day_of_week)
                
                # Preferred delivery time
                preferred_hour = np.random.uniform(
                    config.WORK_DAY_START + 1,
                    config.WORK_DAY_END - 1
                )
                
                # Time window
                time_window_early, time_window_late = self.generate_time_window(preferred_hour)
                
                # Distance
                distance = self.calculate_distance(config.DEPOT_LOCATION, location)
                
                # Actual travel time (unknown to optimizer)
                actual_travel_time = self.generate_travel_time(
                    distance,
                    int(preferred_hour),
                    day_of_week
                )
                
                scenarios.append({
                    'scenario_id': scenario_idx,
                    'date': test_date.strftime('%Y-%m-%d'),
                    'day_of_week': day_of_week,
                    'customer_id': customer_id,
                    'customer_type': customer_type,
                    'location_x': location[0],
                    'location_y': location[1],
                    'distance_from_depot': distance,
                    'actual_demand': actual_demand,  # Ground truth
                    'time_window_early': time_window_early,
                    'time_window_late': time_window_late,
                    'preferred_hour': preferred_hour,
                    'actual_travel_time': actual_travel_time  # Ground truth
                })
        
        df = pd.DataFrame(scenarios)
        return df


def main():
    """Generate and save data"""
    print("Generating historical delivery data...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    generator = DeliveryDataGenerator()
    
    # Historical data
    print(f"Generating {config.NUM_HISTORICAL_DAYS} days of historical data...")
    historical_data = generator.generate_historical_data()
    historical_data.to_csv('data/historical_data.csv', index=False)
    print(f"  Generated {len(historical_data)} historical deliveries")
    print(f"  Saved to data/historical_data.csv")
    
    # Test scenarios
    print("\nGenerating test scenarios...")
    test_scenarios = generator.generate_test_scenarios(num_scenarios=10)
    test_scenarios.to_csv('data/test_scenarios.csv', index=False)
    print(f"  Generated {len(test_scenarios)} test deliveries across 10 scenarios")
    print(f"  Saved to data/test_scenarios.csv")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("HISTORICAL DATA SUMMARY")
    print("="*60)
    print(historical_data.describe())
    
    print("\n" + "="*60)
    print("TEST SCENARIOS SUMMARY")
    print("="*60)
    print(test_scenarios.describe())


if __name__ == "__main__":
    main()
