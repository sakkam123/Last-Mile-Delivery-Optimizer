"""
Predict-then-Optimize Pipeline
Integrates prediction models with route optimization
"""

import numpy as np
import pandas as pd
import json
import os
from predictor import DemandPredictor, TravelTimePredictor
from optimizer import DeliveryOptimizer
import config


class PredictOptimizePipeline:
    """Complete predict-then-optimize pipeline"""
    
    def __init__(self):
        self.demand_predictor = None
        self.travel_time_predictor = None
        self.results = []
        
    def load_models(self):
        """Load trained prediction models"""
        print("Loading prediction models...")
        
        self.demand_predictor = DemandPredictor()
        self.demand_predictor.load()
        
        self.travel_time_predictor = TravelTimePredictor()
        self.travel_time_predictor.load()
        
        print("  ✓ Models loaded successfully")
    
    def run_scenario(self, scenario_df, scenario_id, use_predictions=True):
        """
        Run complete pipeline for one scenario
        
        Args:
            scenario_df: Full DataFrame with all scenarios
            scenario_id: ID of scenario to process
            use_predictions: Whether to use ML predictions or actual values
        
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING SCENARIO {scenario_id}")
        print(f"{'='*60}")
        
        # Extract scenario data
        scenario_data = scenario_df[scenario_df['scenario_id'] == scenario_id].copy()
        
        print(f"\nScenario details:")
        print(f"  Date: {scenario_data.iloc[0]['date']}")
        print(f"  Number of customers: {len(scenario_data)}")
        print(f"  Total actual demand: {scenario_data['actual_demand'].sum():.2f} kg")
        
        # Step 1: Make predictions
        if use_predictions:
            print("\n--- STEP 1: PREDICTION ---")
            
            # Demand prediction
            demand_pred, demand_lower, demand_upper = \
                self.demand_predictor.predict_with_uncertainty(scenario_data)
            
            scenario_data['predicted_demand'] = demand_pred
            scenario_data['demand_lower_bound'] = demand_lower
            scenario_data['demand_upper_bound'] = demand_upper
            
            # Travel time prediction
            travel_time_pred = self.travel_time_predictor.predict(scenario_data)
            scenario_data['predicted_travel_time'] = travel_time_pred
            
            # Print prediction summary
            demand_mae = np.mean(np.abs(scenario_data['actual_demand'] - 
                                        scenario_data['predicted_demand']))
            travel_mae = np.mean(np.abs(scenario_data['actual_travel_time'] - 
                                        scenario_data['predicted_travel_time']))
            
            print(f"  Demand prediction MAE: {demand_mae:.2f} kg")
            print(f"  Travel time prediction MAE: {travel_mae:.3f} hours ({travel_mae*60:.1f} min)")
            print(f"  Predicted total demand: {demand_pred.sum():.2f} kg")
            print(f"  Robust total demand (upper bound): {demand_upper.sum():.2f} kg")
        
        # Step 2: Optimize routes
        print("\n--- STEP 2: OPTIMIZATION ---")
        optimizer = DeliveryOptimizer(use_predictions=use_predictions)
        routes = optimizer.solve(scenario_data, time_limit=30)
        
        if routes is None:
            print("  ✗ Optimization failed!")
            return None
        
        # Save solution for visualization
        optimizer.save_solution(f'results/solution_scenario_{scenario_id}.json')
        
        # Step 3: Evaluate with actual values
        print("\n--- STEP 3: EVALUATION ---")
        
        # Calculate costs with predicted values (what we planned for)
        planned_costs = optimizer.calculate_actual_costs(routes, scenario_data)
        
        # Calculate costs with actual values (what really happened)
        # Need to recalculate considering actual demands and travel times
        actual_costs = self.evaluate_actual_performance(routes, scenario_data)
        
        results = {
            'scenario_id': scenario_id,
            'num_customers': len(scenario_data),
            'use_predictions': use_predictions,
            'routes': routes,
            'planned_costs': planned_costs,
            'actual_costs': actual_costs,
            'scenario_data': scenario_data
        }
        
        # Print results
        self.print_results(results)
        
        return results
    
    def evaluate_actual_performance(self, routes, scenario_data):
        """
        Evaluate actual performance considering real demand and travel times
        
        This simulates what happens when the driver follows the planned route
        but encounters actual (different) demands and travel times
        """
        total_distance = routes['total_distance']  # Distance doesn't change
        total_penalty = 0
        capacity_violations = 0
        
        # Check each route
        for route in routes['routes']:
            route_load = 0
            current_time = config.WORK_DAY_START
            
            for i, stop in enumerate(route['stops']):
                if 'customer_id' not in stop:
                    continue  # Skip depot
                
                # Find actual customer data
                customer_id = stop['customer_id']
                customer_data = scenario_data[scenario_data['customer_id'] == customer_id].iloc[0]
                
                # Actual demand (might exceed predicted)
                actual_demand = customer_data['actual_demand']
                route_load += actual_demand
                
                # Check capacity violation
                if route_load > config.VEHICLE_CAPACITY:
                    capacity_violations += 1
                
                # Actual travel time to this customer
                if i > 0:  # Not first customer
                    # Use actual travel time
                    actual_travel_time = customer_data['actual_travel_time']
                    current_time += actual_travel_time
                
                # Service time
                current_time += config.SERVICE_TIME
                
                # Calculate time window penalty
                time_window = stop['time_window']
                if current_time < time_window[0]:
                    # Early
                    early_hours = time_window[0] - current_time
                    total_penalty += early_hours * config.EARLY_PENALTY
                elif current_time > time_window[1]:
                    # Late
                    late_hours = current_time - time_window[1]
                    total_penalty += late_hours * config.LATE_PENALTY
        
        # Calculate total cost
        travel_cost = total_distance * config.COST_PER_KM
        vehicle_cost = routes['num_routes'] * config.FIXED_VEHICLE_COST
        total_cost = travel_cost + vehicle_cost + total_penalty
        
        return {
            'travel_cost': travel_cost,
            'vehicle_cost': vehicle_cost,
            'penalty_cost': total_penalty,
            'total_cost': total_cost,
            'capacity_violations': capacity_violations
        }
    
    def print_results(self, results):
        """Print results summary"""
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nPlanned Costs (based on predictions):")
        print(f"  Travel: €{results['planned_costs']['travel_cost']:.2f}")
        print(f"  Vehicles: €{results['planned_costs']['vehicle_cost']:.2f}")
        print(f"  Penalties: €{results['planned_costs']['penalty_cost']:.2f}")
        print(f"  TOTAL: €{results['planned_costs']['total_cost']:.2f}")
        
        print(f"\nActual Costs (based on realized values):")
        print(f"  Travel: €{results['actual_costs']['travel_cost']:.2f}")
        print(f"  Vehicles: €{results['actual_costs']['vehicle_cost']:.2f}")
        print(f"  Penalties: €{results['actual_costs']['penalty_cost']:.2f}")
        print(f"  TOTAL: €{results['actual_costs']['total_cost']:.2f}")
        
        cost_difference = (results['actual_costs']['total_cost'] - 
                          results['planned_costs']['total_cost'])
        
        print(f"\nCost Difference: €{cost_difference:.2f} " +
              f"({cost_difference/results['planned_costs']['total_cost']*100:.1f}%)")
        
        if results['actual_costs']['capacity_violations'] > 0:
            print(f"\n⚠ Capacity violations: {results['actual_costs']['capacity_violations']}")
    
    def compare_approaches(self, scenario_df, scenario_id):
        """
        Compare different approaches:
        1. Predict-then-optimize (with ML predictions)
        2. Optimize with perfect information (oracle)
        3. Optimize with average values (baseline)
        """
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS")
        print("="*60)
        
        results_comparison = {}
        
        # Approach 1: Predict-then-optimize
        print("\n### APPROACH 1: PREDICT-THEN-OPTIMIZE ###")
        results_pred = self.run_scenario(scenario_df, scenario_id, use_predictions=True)
        if results_pred:
            results_comparison['predict_optimize'] = results_pred['actual_costs']['total_cost']
        
        # Approach 2: Oracle (perfect information)
        print("\n### APPROACH 2: ORACLE (PERFECT INFORMATION) ###")
        scenario_data = scenario_df[scenario_df['scenario_id'] == scenario_id].copy()
        
        # Use actual values as "predictions"
        scenario_data['predicted_demand'] = scenario_data['actual_demand']
        scenario_data['demand_upper_bound'] = scenario_data['actual_demand']
        scenario_data['predicted_travel_time'] = scenario_data['actual_travel_time']
        
        optimizer_oracle = DeliveryOptimizer(use_predictions=True)
        routes_oracle = optimizer_oracle.solve(scenario_data, time_limit=30)
        
        if routes_oracle:
            costs_oracle = optimizer_oracle.calculate_actual_costs(routes_oracle, scenario_data)
            results_comparison['oracle'] = costs_oracle['total_cost']
            print(f"\nOracle Total Cost: €{costs_oracle['total_cost']:.2f}")
        
        # Approach 3: Baseline (use historical averages)
        print("\n### APPROACH 3: BASELINE (HISTORICAL AVERAGES) ###")
        
        # Load historical data
        historical_data = pd.read_csv('data/historical_data.csv')
        avg_demand = historical_data['demand'].mean()
        
        scenario_baseline = scenario_df[scenario_df['scenario_id'] == scenario_id].copy()
        scenario_baseline['predicted_demand'] = avg_demand
        scenario_baseline['demand_upper_bound'] = avg_demand * 1.2  # Add buffer
        scenario_baseline['predicted_travel_time'] = \
            scenario_baseline['distance_from_depot'] / config.VEHICLE_SPEED
        
        optimizer_baseline = DeliveryOptimizer(use_predictions=True)
        routes_baseline = optimizer_baseline.solve(scenario_baseline, time_limit=30)
        
        if routes_baseline:
            # Evaluate with actual values
            actual_costs_baseline = self.evaluate_actual_performance(
                routes_baseline, scenario_data
            )
            results_comparison['baseline'] = actual_costs_baseline['total_cost']
            print(f"\nBaseline Total Cost: €{actual_costs_baseline['total_cost']:.2f}")
        
        # Print comparison
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        
        for approach, cost in results_comparison.items():
            print(f"{approach:20s}: €{cost:8.2f}")
        
        if 'oracle' in results_comparison:
            for approach in ['predict_optimize', 'baseline']:
                if approach in results_comparison:
                    gap = ((results_comparison[approach] - results_comparison['oracle']) / 
                           results_comparison['oracle'] * 100)
                    print(f"\n{approach} gap from oracle: {gap:.2f}%")
        
        return results_comparison
    
    def save_results(self, results, filepath='results/pipeline_results.json'):
        """Save pipeline results"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to serializable format
        save_data = {
            'scenario_id': results['scenario_id'],
            'num_customers': results['num_customers'],
            'use_predictions': results['use_predictions'],
            'planned_costs': results['planned_costs'],
            'actual_costs': results['actual_costs'],
            'num_routes': results['routes']['num_routes']
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n  Results saved to {filepath}")


def main():
    """Main execution"""
    print("="*60)
    print("PREDICT-THEN-OPTIMIZE PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = PredictOptimizePipeline()
    pipeline.load_models()
    
    # Load test scenarios
    print("\nLoading test scenarios...")
    test_scenarios = pd.read_csv('data/test_scenarios.csv')
    print(f"  Loaded {test_scenarios['scenario_id'].nunique()} scenarios")
    
    # Run comparative analysis on first scenario
    results = pipeline.compare_approaches(test_scenarios, scenario_id=0)
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
