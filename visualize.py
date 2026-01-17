"""
Visualization Tools for Last-Mile Delivery Optimizer
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
import config


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class DeliveryVisualizer:
    """Visualization tools for delivery routes and analysis"""
    
    def __init__(self, output_dir='results/visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_routes(self, routes, scenario_data, title="Delivery Routes", 
                    filename="routes.png"):
        """
        Plot delivery routes on a map
        
        Args:
            routes: Route solution from optimizer
            scenario_data: DataFrame with customer data
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define colors for routes
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        # Plot depot
        ax.scatter(config.DEPOT_LOCATION[0], config.DEPOT_LOCATION[1], 
                  c='red', s=400, marker='s', label='Depot', 
                  edgecolors='black', linewidths=2, zorder=5)
        
        # Plot each route
        for i, route in enumerate(routes['routes']):
            color = colors[i % len(colors)]
            
            # Extract route coordinates
            route_coords = [(config.DEPOT_LOCATION[0], config.DEPOT_LOCATION[1])]
            customer_ids = []
            
            for stop in route['stops']:
                if 'location' in stop:
                    route_coords.append(stop['location'])
                    customer_ids.append(stop['customer_id'])
            
            route_coords.append((config.DEPOT_LOCATION[0], config.DEPOT_LOCATION[1]))
            
            # Plot route line
            route_x = [coord[0] for coord in route_coords]
            route_y = [coord[1] for coord in route_coords]
            
            ax.plot(route_x, route_y, c=color, linewidth=2, alpha=0.7,
                   label=f"Vehicle {route['vehicle_id']} ({len(customer_ids)} stops)")
            
            # Plot customer points
            customer_x = route_x[1:-1]
            customer_y = route_y[1:-1]
            
            ax.scatter(customer_x, customer_y, c=[color], s=150, 
                      edgecolors='black', linewidths=1.5, zorder=3, alpha=0.8)
            
            # Add customer labels
            for j, (x, y, cid) in enumerate(zip(customer_x, customer_y, customer_ids)):
                ax.annotate(f"{j+1}", (x, y), fontsize=8, ha='center', va='center',
                           fontweight='bold', color='white')
        
        # Plot all customers (including those not in routes, if any)
        all_customers_x = scenario_data['location_x'].values
        all_customers_y = scenario_data['location_y'].values
        
        ax.set_xlim(-1, config.GRID_SIZE + 1)
        ax.set_ylim(-1, config.GRID_SIZE + 1)
        ax.set_xlabel('X Coordinate (km)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Coordinate (km)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add statistics box
        total_distance = routes['total_distance']
        total_load = routes['total_load']
        num_routes = routes['num_routes']
        
        stats_text = f"Total Distance: {total_distance:.1f} km\n"
        stats_text += f"Total Load: {total_load:.1f} kg\n"
        stats_text += f"Vehicles Used: {num_routes}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Route plot saved to {filepath}")
        plt.close()
    
    def plot_time_windows(self, routes, scenario_data, filename="time_windows.png"):
        """
        Plot arrival times vs time windows
        
        Args:
            routes: Route solution
            scenario_data: DataFrame with customer data
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Collect data for all stops
        stops_data = []
        
        for route in routes['routes']:
            for stop in route['stops']:
                if 'customer_id' in stop:
                    customer_id = stop['customer_id']
                    arrival_time = stop['arrival_time']
                    time_window = stop['time_window']
                    
                    stops_data.append({
                        'customer_id': customer_id,
                        'arrival_time': arrival_time,
                        'window_early': time_window[0],
                        'window_late': time_window[1],
                        'vehicle_id': route['vehicle_id']
                    })
        
        df_stops = pd.DataFrame(stops_data)
        df_stops = df_stops.sort_values('arrival_time')
        
        # Plot
        x_pos = np.arange(len(df_stops))
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        for i, row in df_stops.iterrows():
            vehicle_color = colors[row['vehicle_id'] % len(colors)]
            
            # Plot time window as a bar
            window_start = row['window_early']
            window_end = row['window_late']
            window_duration = window_end - window_start
            
            idx = list(df_stops.index).index(i)
            
            ax.barh(idx, window_duration, left=window_start, height=0.8,
                   color=vehicle_color, alpha=0.3, edgecolor='black', linewidth=0.5)
            
            # Plot arrival time as a marker
            arrival = row['arrival_time']
            
            if arrival < window_start:
                marker_color = 'blue'  # Early
                marker = '<'
            elif arrival > window_end:
                marker_color = 'red'  # Late
                marker = '>'
            else:
                marker_color = 'green'  # On time
                marker = 'o'
            
            ax.scatter(arrival, idx, c=marker_color, s=100, marker=marker,
                      edgecolors='black', linewidths=1, zorder=5)
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels([f"C{i+1}" for i in range(len(df_stops))], fontsize=8)
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Customer', fontsize=12, fontweight='bold')
        ax.set_title('Arrival Times vs Time Windows', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='On Time', markeredgecolor='black'),
            Line2D([0], [0], marker='<', color='w', markerfacecolor='blue',
                   markersize=10, label='Early', markeredgecolor='black'),
            Line2D([0], [0], marker='>', color='w', markerfacecolor='red',
                   markersize=10, label='Late', markeredgecolor='black'),
            Patch(facecolor='gray', alpha=0.3, label='Time Window')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Time window plot saved to {filepath}")
        plt.close()
    
    def plot_prediction_accuracy(self, scenario_data, filename="prediction_accuracy.png"):
        """
        Plot prediction accuracy for demand and travel time
        
        Args:
            scenario_data: DataFrame with predictions and actual values
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Demand prediction accuracy
        ax1 = axes[0]
        
        if 'predicted_demand' in scenario_data.columns:
            actual_demand = scenario_data['actual_demand']
            predicted_demand = scenario_data['predicted_demand']
            
            # Scatter plot
            ax1.scatter(actual_demand, predicted_demand, alpha=0.6, s=100,
                       edgecolors='black', linewidths=0.5)
            
            # Perfect prediction line
            max_val = max(actual_demand.max(), predicted_demand.max())
            ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            # Calculate metrics
            mae = np.mean(np.abs(actual_demand - predicted_demand))
            rmse = np.sqrt(np.mean((actual_demand - predicted_demand)**2))
            
            ax1.set_xlabel('Actual Demand (kg)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Predicted Demand (kg)', fontsize=12, fontweight='bold')
            ax1.set_title('Demand Prediction Accuracy', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Add metrics text
            metrics_text = f"MAE: {mae:.2f} kg\nRMSE: {rmse:.2f} kg"
            ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Travel time prediction accuracy
        ax2 = axes[1]
        
        if 'predicted_travel_time' in scenario_data.columns:
            actual_time = scenario_data['actual_travel_time']
            predicted_time = scenario_data['predicted_travel_time']
            
            # Scatter plot
            ax2.scatter(actual_time, predicted_time, alpha=0.6, s=100,
                       edgecolors='black', linewidths=0.5, color='orange')
            
            # Perfect prediction line
            max_val = max(actual_time.max(), predicted_time.max())
            ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            # Calculate metrics
            mae = np.mean(np.abs(actual_time - predicted_time))
            rmse = np.sqrt(np.mean((actual_time - predicted_time)**2))
            
            ax2.set_xlabel('Actual Travel Time (hours)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Predicted Travel Time (hours)', fontsize=12, fontweight='bold')
            ax2.set_title('Travel Time Prediction Accuracy', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Add metrics text
            metrics_text = f"MAE: {mae:.3f} hours\nRMSE: {rmse:.3f} hours"
            ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Prediction accuracy plot saved to {filepath}")
        plt.close()
    
    def plot_cost_comparison(self, comparison_results, filename="cost_comparison.png"):
        """
        Plot cost comparison between different approaches
        
        Args:
            comparison_results: Dictionary with approach names and costs
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        approaches = list(comparison_results.keys())
        costs = list(comparison_results.values())
        
        # Create bar plot
        bars = ax.bar(approaches, costs, color=['#3498db', '#2ecc71', '#e74c3c'],
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'€{height:.2f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Total Cost (€)', fontsize=12, fontweight='bold')
        ax.set_title('Cost Comparison: Different Approaches', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Set x-tick labels
        ax.set_xticklabels(['Predict-Then-\nOptimize', 'Oracle\n(Perfect Info)', 
                           'Baseline\n(Averages)'], fontsize=11)
        
        # Calculate and display improvement
        if 'oracle' in comparison_results and 'predict_optimize' in comparison_results:
            gap = ((comparison_results['predict_optimize'] - comparison_results['oracle']) / 
                   comparison_results['oracle'] * 100)
            
            gap_text = f"Predict-Optimize Gap from Oracle: {gap:.1f}%"
            ax.text(0.5, 0.95, gap_text, transform=ax.transAxes,
                   fontsize=11, ha='center', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Cost comparison plot saved to {filepath}")
        plt.close()
    
    def plot_vehicle_utilization(self, routes, filename="vehicle_utilization.png"):
        """
        Plot vehicle capacity utilization
        
        Args:
            routes: Route solution
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Capacity utilization
        ax1 = axes[0]
        
        vehicle_ids = [route['vehicle_id'] for route in routes['routes']]
        loads = [route['total_load'] for route in routes['routes']]
        capacities = [config.VEHICLE_CAPACITY] * len(loads)
        
        x_pos = np.arange(len(vehicle_ids))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, loads, width, label='Load', 
                       color='#3498db', edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x_pos + width/2, capacities, width, label='Capacity',
                       color='#95a5a6', edgecolor='black', linewidth=1, alpha=0.5)
        
        # Add utilization percentage on bars
        for i, (load, capacity) in enumerate(zip(loads, capacities)):
            utilization = (load / capacity) * 100
            ax1.text(i, load + 2, f'{utilization:.1f}%', ha='center', 
                    va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_ylabel('Load (kg)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Vehicle', fontsize=12, fontweight='bold')
        ax1.set_title('Vehicle Capacity Utilization', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'V{vid}' for vid in vehicle_ids])
        ax1.legend(fontsize=10)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Route duration
        ax2 = axes[1]
        
        durations = [route['total_time'] for route in routes['routes']]
        max_duration = [config.MAX_ROUTE_DURATION] * len(durations)
        
        bars1 = ax2.bar(x_pos - width/2, durations, width, label='Duration',
                       color='#e74c3c', edgecolor='black', linewidth=1)
        bars2 = ax2.bar(x_pos + width/2, max_duration, width, label='Max Duration',
                       color='#95a5a6', edgecolor='black', linewidth=1, alpha=0.5)
        
        # Add percentage on bars
        for i, (duration, max_dur) in enumerate(zip(durations, max_duration)):
            utilization = (duration / max_dur) * 100
            ax2.text(i, duration + 0.2, f'{utilization:.1f}%', ha='center',
                    va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_ylabel('Time (hours)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Vehicle', fontsize=12, fontweight='bold')
        ax2.set_title('Vehicle Time Utilization', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'V{vid}' for vid in vehicle_ids])
        ax2.legend(fontsize=10)
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Vehicle utilization plot saved to {filepath}")
        plt.close()


def main():
    """Test visualization functions"""
    print("="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Load solution
    print("\nLoading solution data...")
    
    try:
        with open('results/solution.json', 'r') as f:
            routes = json.load(f)
        print("  ✓ Solution loaded")
    except FileNotFoundError:
        print("  ✗ Solution file not found. Run optimizer.py first.")
        return
    
    # Load scenario data
    test_scenarios = pd.read_csv('data/test_scenarios.csv')
    scenario_data = test_scenarios[test_scenarios['scenario_id'] == 0].copy()
    
    # Add predictions
    try:
        from predictor import make_predictions_for_scenario
        scenario_data = make_predictions_for_scenario(test_scenarios, 0)
        print("  ✓ Predictions loaded")
    except:
        print("  ⚠ Predictions not available")
    
    # Create visualizer
    visualizer = DeliveryVisualizer()
    
    # Generate visualizations
    print("\nGenerating plots...")
    
    visualizer.plot_routes(routes, scenario_data, 
                          title="Optimized Delivery Routes - Scenario 0")
    
    visualizer.plot_time_windows(routes, scenario_data)
    
    if 'predicted_demand' in scenario_data.columns:
        visualizer.plot_prediction_accuracy(scenario_data)
    
    visualizer.plot_vehicle_utilization(routes)
    
    # Example cost comparison
    comparison_results = {
        'predict_optimize': 450.0,
        'oracle': 420.0,
        'baseline': 480.0
    }
    
    visualizer.plot_cost_comparison(comparison_results)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nVisualizations saved to {visualizer.output_dir}/")


if __name__ == "__main__":
    main()
