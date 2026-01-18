"""
Route Optimizer for Last-Mile Delivery
Implements Vehicle Routing Problem with Soft Time Windows using OR-Tools
"""

import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import json
import os
import config


def solve_vrp_with_soft_time_windows(manager, routing, data, time_windows, vehicle_capacities):
    """
    Solve VRP with soft time windows and capacity constraints.
    
    Args:
        manager: Routing index manager
        routing: Routing model
        data: Dictionary containing distance_matrix, time_matrix, demands, service_times, etc.
        time_windows: List of (early, late) tuples for each node
        vehicle_capacities: List of capacity for each vehicle
        
    Returns:
        Solution object if found, None otherwise
    """
    # ========================================================================
    # DISTANCE CALLBACK
    # ========================================================================
    def distance_callback(from_index, to_index):
        """Calculate distance cost between two nodes"""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['distance_matrix'][from_node][to_node] * 100)  # scale for integer
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # ========================================================================
    # CAPACITY DIMENSION
    # ========================================================================
    def demand_callback(from_index):
        """Return demand at node"""
        from_node = manager.IndexToNode(from_index)
        return int(data['demands'][from_node])
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        vehicle_capacities,
        True,  # start cumul to zero
        'Capacity'
    )
    
    # ========================================================================
    # TIME DIMENSION
    # ========================================================================
    def time_callback(from_index, to_index):
        """Calculate travel time plus service time"""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = data['time_matrix'][from_node][to_node]
        service_time = data['service_times'][from_node]
        return int((travel_time + service_time) * 60)  # convert hours -> minutes
    
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        60 * 2,  # allow slack (max 2h) - REDUCED for tighter scheduling
        int(data['max_route_duration'] * 60),  # max route duration in minutes
        False,  # don't force start cumul to zero
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')
    # Increase priority on time dimension to respect time windows better
    time_dimension.SetGlobalSpanCostCoefficient(10)
    
    # ========================================================================
    # APPLY SOFT TIME WINDOWS
    # ========================================================================
    PENALTY_EARLY = int(data['early_penalty'] * 60)  # scale to minutes
    PENALTY_LATE = int(data['late_penalty'] * 60)    # scale to minutes
    
    for node_idx, (early, late) in enumerate(time_windows):
        index = manager.NodeToIndex(node_idx)
        early_min = int(early * 60)
        late_max = int(late * 60)
        
        # Soft lower bound (penalty for arriving early)
        time_dimension.SetCumulVarSoftLowerBound(index, early_min, PENALTY_EARLY)
        # Soft upper bound (penalty for arriving late)
        time_dimension.SetCumulVarSoftUpperBound(index, late_max, PENALTY_LATE)
    
    # ========================================================================
    # SOLVER PARAMETERS
    # ========================================================================
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 60  # 60s timeout
    search_parameters.solution_limit = 1000  # Explore more solutions
    search_parameters.log_search = False  # Disable verbose logging
    
    # ========================================================================
    # SOLVE
    # ========================================================================
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        return solution
    else:
        print("✗ No solution found!")
        return None


class DeliveryOptimizer:
    """Optimize delivery routes using OR-Tools"""
    
    def __init__(self, use_predictions=True):
        """
        Args:
            use_predictions: If True, use predicted values; if False, use actual values
        """
        self.use_predictions = use_predictions
        self.solution = None
        self.routes = None
        
    def create_data_model(self, scenario_df):
        """
        Create data model for OR-Tools
        
        Args:
            scenario_df: DataFrame with customer data (with predictions if available)
        
        Returns:
            Dictionary with all routing data
        """
        data = {}
        
        # Locations (depot + customers)
        locations = [(config.DEPOT_LOCATION[0], config.DEPOT_LOCATION[1])]
        for idx, row in scenario_df.iterrows():
            locations.append((row['location_x'], row['location_y']))
        
        data['locations'] = locations
        data['num_vehicles'] = config.NUM_VEHICLES
        data['depot'] = 0
        
        # Demands (use predictions or actual based on flag)
        if self.use_predictions and 'predicted_demand' in scenario_df.columns:
            # Use upper bound for robust planning
            demands = [0] + scenario_df['demand_upper_bound'].tolist()
        elif 'actual_demand' in scenario_df.columns:
            demands = [0] + scenario_df['actual_demand'].tolist()
        else:
            demands = [0] + scenario_df['demand'].tolist()
        
        data['demands'] = demands
        data['vehicle_capacities'] = [config.VEHICLE_CAPACITY] * config.NUM_VEHICLES
        
        # Time windows
        time_windows = [(config.WORK_DAY_START, config.WORK_DAY_END)]  # Depot
        for idx, row in scenario_df.iterrows():
            time_windows.append((row['time_window_early'], row['time_window_late']))
        
        data['time_windows'] = time_windows
        
        # Service times (depot has 0, customers have SERVICE_TIME)
        service_times = [0] + [config.SERVICE_TIME] * len(scenario_df)
        data['service_times'] = service_times
        data['service_time'] = config.SERVICE_TIME
        
        # Additional parameters for the solver
        data['max_route_duration'] = config.MAX_ROUTE_DURATION
        data['early_penalty'] = config.EARLY_PENALTY
        data['late_penalty'] = config.LATE_PENALTY
        
        # Store scenario data for later use
        data['scenario_df'] = scenario_df
        
        return data
    
    def calculate_distance_matrix(self, locations):
        """Calculate Euclidean distance matrix"""
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = locations[i][0] - locations[j][0]
                    dy = locations[i][1] - locations[j][1]
                    distance_matrix[i][j] = np.sqrt(dx**2 + dy**2)
        
        return distance_matrix
    
    def calculate_time_matrix(self, distance_matrix, scenario_df):
        """
        Calculate travel time matrix
        
        Uses predicted travel times if available, otherwise calculates from distance
        """
        n = len(distance_matrix)
        time_matrix = np.zeros((n, n))
        
        # From depot to customers and between customers
        for i in range(n):
            for j in range(n):
                if i == j:
                    time_matrix[i][j] = 0
                elif i == 0:  # From depot to customer j
                    if self.use_predictions and 'predicted_travel_time' in scenario_df.columns:
                        time_matrix[i][j] = scenario_df.iloc[j-1]['predicted_travel_time']
                    else:
                        # Use distance / speed
                        time_matrix[i][j] = distance_matrix[i][j] / config.VEHICLE_SPEED
                else:
                    # Between customers: use distance / speed
                    time_matrix[i][j] = distance_matrix[i][j] / config.VEHICLE_SPEED
        
        return time_matrix
        
    def solve(self, scenario_df, time_limit=60):
        """
        Solve the VRP with soft time windows using the integrated solver function
        
        Args:
            scenario_df: DataFrame with scenario data
            time_limit: Time limit for solver in seconds
        
        Returns:
            Dictionary with solution details
        """
        print("\n" + "="*60)
        print("SOLVING VEHICLE ROUTING PROBLEM")
        print("="*60)
        
        # Create data model
        data = self.create_data_model(scenario_df)
        
        # Calculate matrices
        distance_matrix = self.calculate_distance_matrix(data['locations'])
        time_matrix = self.calculate_time_matrix(distance_matrix, scenario_df)
        
        # Add matrices to data dictionary
        data['distance_matrix'] = distance_matrix
        data['time_matrix'] = time_matrix
        
        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(
            len(data['locations']),
            data['num_vehicles'],
            data['depot']
        )
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        # ========================================================================
        # USE INTEGRATED SOLVER FUNCTION
        # ========================================================================
        print("\nSolving with integrated VRP solver...")
        print(f"  Number of customers: {len(scenario_df)}")
        print(f"  Number of vehicles: {data['num_vehicles']}")
        print(f"  Time limit: {time_limit}s")
        
        # Call the integrated solver function
        solution = solve_vrp_with_soft_time_windows(
            manager=manager,
            routing=routing,
            data=data,
            time_windows=data['time_windows'],
            vehicle_capacities=data['vehicle_capacities']
        )
        
        if solution:
            print("  ✓ Solution found!")
            self.solution = solution
            self.routes = self.extract_routes(
                manager, routing, solution, data,
                distance_matrix, time_matrix
            )
            return self.routes
        else:
            print("  ✗ No solution found!")
            return None
    
    def extract_routes(self, manager, routing, solution, data, 
                       distance_matrix, time_matrix):
        """Extract routes from solution"""
        routes = []
        
        time_dimension = routing.GetDimensionOrDie('Time')
        capacity_dimension = routing.GetDimensionOrDie('Capacity')
        
        total_distance = 0
        total_time = 0
        total_load = 0
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route = {
                'vehicle_id': vehicle_id,
                'stops': [],
                'total_distance': 0,
                'total_time': 0,
                'total_load': 0
            }
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                load_var = capacity_dimension.CumulVar(index)
                
                stop_info = {
                    'node': node,
                    'arrival_time': solution.Min(time_var) / 60.0,  # convert minutes to hours
                    'load': solution.Min(load_var)
                }
                
                if node != 0:  # Not depot
                    customer_idx = node - 1
                    scenario_row = data['scenario_df'].iloc[customer_idx]
                    stop_info['customer_id'] = scenario_row['customer_id']
                    stop_info['location'] = (scenario_row['location_x'], 
                                            scenario_row['location_y'])
                    stop_info['time_window'] = (scenario_row['time_window_early'],
                                               scenario_row['time_window_late'])
                    stop_info['demand'] = data['demands'][node]
                
                route['stops'].append(stop_info)
                
                # Move to next node
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                
                if not routing.IsEnd(index):
                    from_node = manager.IndexToNode(previous_index)
                    to_node = manager.IndexToNode(index)
                    route['total_distance'] += distance_matrix[from_node][to_node]
                    route['total_time'] += time_matrix[from_node][to_node]
            
            # Add final depot
            node = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            route['stops'].append({
                'node': node,
                'arrival_time': solution.Min(time_var) / 60.0,  # convert minutes to hours
                'load': 0
            })
            
            # Update totals
            route['total_load'] = solution.Max(capacity_dimension.CumulVar(
                routing.End(vehicle_id)
            ))
            
            if len(route['stops']) > 2:  # Has actual customers
                routes.append(route)
                total_distance += route['total_distance']
                total_time += route['total_time']
                total_load += route['total_load']
        
        # Calculate costs
        summary = {
            'routes': routes,
            'num_routes': len(routes),
            'total_distance': total_distance,
            'total_time': total_time,
            'total_load': total_load,
            'travel_cost': total_distance * config.COST_PER_KM,
            'vehicle_cost': len(routes) * config.FIXED_VEHICLE_COST,
            'penalty_cost': 0  # Will be calculated during evaluation
        }
        
        return summary
    
    def calculate_actual_costs(self, routes, scenario_df):
        """
        Calculate actual costs when using real demand and travel times
        
        Args:
            routes: Route solution
            scenario_df: DataFrame with actual values
        
        Returns:
            Dictionary with cost breakdown
        """
        total_penalty = 0
        route_details = []
        
        for route in routes['routes']:
            route_penalty = 0
            
            for stop in route['stops']:
                if 'customer_id' in stop:
                    arrival_time = stop['arrival_time']
                    time_window = stop['time_window']
                    
                    # Calculate early/late penalty
                    if arrival_time < time_window[0]:
                        # Early
                        early_hours = time_window[0] - arrival_time
                        penalty = early_hours * config.EARLY_PENALTY
                        route_penalty += penalty
                    elif arrival_time > time_window[1]:
                        # Late
                        late_hours = arrival_time - time_window[1]
                        penalty = late_hours * config.LATE_PENALTY
                        route_penalty += penalty
            
            total_penalty += route_penalty
            route_details.append({
                'vehicle_id': route['vehicle_id'],
                'penalty': route_penalty
            })
        
        total_cost = (routes['travel_cost'] + 
                     routes['vehicle_cost'] + 
                     total_penalty)
        
        return {
            'travel_cost': routes['travel_cost'],
            'vehicle_cost': routes['vehicle_cost'],
            'penalty_cost': total_penalty,
            'total_cost': total_cost,
            'route_penalties': route_details
        }
    
    def save_solution(self, filepath='results/solution.json'):
        """Save solution to JSON file"""
        if self.routes is None:
            print("No solution to save!")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to JSON-serializable format
        solution_data = {
            'num_routes': self.routes['num_routes'],
            'total_distance': float(self.routes['total_distance']),
            'total_time': float(self.routes['total_time']),
            'total_load': float(self.routes['total_load']),
            'travel_cost': float(self.routes['travel_cost']),
            'vehicle_cost': float(self.routes['vehicle_cost']),
            'routes': []
        }
        
        for route in self.routes['routes']:
            route_data = {
                'vehicle_id': route['vehicle_id'],
                'total_distance': float(route['total_distance']),
                'total_time': float(route['total_time']),
                'total_load': float(route['total_load']),
                'stops': []
            }
            
            for stop in route['stops']:
                stop_data = {
                    'node': stop['node'],
                    'arrival_time': float(stop['arrival_time']),
                    'load': float(stop['load'])
                }
                if 'customer_id' in stop:
                    stop_data['customer_id'] = stop['customer_id']
                    stop_data['location'] = [float(x) for x in stop['location']]
                    stop_data['time_window'] = [float(x) for x in stop['time_window']]
                    stop_data['demand'] = float(stop['demand'])
                
                route_data['stops'].append(stop_data)
            
            solution_data['routes'].append(route_data)
        
        with open(filepath, 'w') as f:
            json.dump(solution_data, f, indent=2)
        
        print(f"\n  Solution saved to {filepath}")


def main():
    """Main execution for testing"""
    import sys
    
    # Load test scenario
    print("Loading test scenario...")
    test_scenarios = pd.read_csv('data/test_scenarios.csv')
    
    # Get scenario 0
    scenario_df = test_scenarios[test_scenarios['scenario_id'] == 0].copy()
    print(f"  Loaded scenario 0 with {len(scenario_df)} customers")
    
    # Add predictions
    from predictor import make_predictions_for_scenario
    try:
        scenario_with_predictions = make_predictions_for_scenario(test_scenarios, 0)
    except:
        print("  Warning: Could not load predictions, using actual values")
        scenario_with_predictions = scenario_df
    
    # Solve with predictions
    print("\n--- SOLVING WITH PREDICTIONS ---")
    optimizer_pred = DeliveryOptimizer(use_predictions=True)
    routes_pred = optimizer_pred.solve(scenario_with_predictions, time_limit=30)
    
    if routes_pred:
        costs_pred = optimizer_pred.calculate_actual_costs(routes_pred, scenario_with_predictions)
        
        print("\n" + "="*60)
        print("SOLUTION SUMMARY (WITH PREDICTIONS)")
        print("="*60)
        print(f"Number of routes: {routes_pred['num_routes']}")
        print(f"Total distance: {routes_pred['total_distance']:.2f} km")
        print(f"Total time: {routes_pred['total_time']:.2f} hours")
        print(f"Total load: {routes_pred['total_load']:.2f} kg")
        print(f"\nCost Breakdown:")
        print(f"  Travel cost: €{costs_pred['travel_cost']:.2f}")
        print(f"  Vehicle cost: €{costs_pred['vehicle_cost']:.2f}")
        print(f"  Penalty cost: €{costs_pred['penalty_cost']:.2f}")
        print(f"  TOTAL COST: €{costs_pred['total_cost']:.2f}")
        
        # Print routes
        print("\nRoutes:")
        for route in routes_pred['routes']:
            customer_ids = [stop.get('customer_id', 'DEPOT') 
                          for stop in route['stops'] if stop['node'] != 0]
            print(f"  Vehicle {route['vehicle_id']}: DEPOT → " + 
                  " → ".join(customer_ids) + " → DEPOT")
            print(f"    Distance: {route['total_distance']:.2f} km, " +
                  f"Load: {route['total_load']:.2f} kg")
        
        optimizer_pred.save_solution()


if __name__ == "__main__":
    main()