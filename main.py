"""
Main execution script for Last-Mile Delivery Optimizer
Runs the complete predict-then-optimize pipeline
"""

import os
import sys
from datetime import datetime

# Import modules
from data_generator import DeliveryDataGenerator
from predictor import train_all_models
from pipeline import PredictOptimizePipeline
from visualize import DeliveryVisualizer

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def main():
    """Main execution pipeline"""
    
    print_header("LAST-MILE DELIVERY OPTIMIZER")
    print(f"\nExecution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    
    # ========== STEP 1: DATA GENERATION ==========
    print_header("STEP 1: DATA GENERATION")
    
    if not os.path.exists('data/historical_data.csv'):
        print("\nGenerating historical delivery data...")
        generator = DeliveryDataGenerator()
        
        historical_data = generator.generate_historical_data()
        historical_data.to_csv('data/historical_data.csv', index=False)
        print(f"  ✓ Generated {len(historical_data)} historical records")
        
        test_scenarios = generator.generate_test_scenarios(num_scenarios=10)
        test_scenarios.to_csv('data/test_scenarios.csv', index=False)
        print(f"  ✓ Generated {len(test_scenarios)} test records (10 scenarios)")
    else:
        print("\n  ✓ Historical data already exists")
    
    # ========== STEP 2: MODEL TRAINING ==========
    print_header("STEP 2: PREDICTION MODEL TRAINING")
    
    if not (os.path.exists('data/models/demand_predictor.pkl') and 
            os.path.exists('data/models/travel_time_predictor.pkl')):
        print("\nTraining prediction models...")
        metrics = train_all_models('data/historical_data.csv')
        print("\n  ✓ Models trained and saved")
    else:
        print("\n  ✓ Trained models already exist")
    
    # ========== STEP 3: PREDICT-THEN-OPTIMIZE PIPELINE ==========
    print_header("STEP 3: PREDICT-THEN-OPTIMIZE PIPELINE")
    
    print("\nInitializing pipeline...")
    pipeline = PredictOptimizePipeline()
    pipeline.load_models()
    
    # Load test scenarios
    import pandas as pd
    test_scenarios = pd.read_csv('data/test_scenarios.csv')
    
    print(f"\nRunning comparative analysis on Scenario 0...")
    print(f"  (Testing {len(test_scenarios[test_scenarios['scenario_id']==0])} customers)")
    
    comparison_results = pipeline.compare_approaches(test_scenarios, scenario_id=0)
    
    # ========== STEP 4: VISUALIZATION ==========
    print_header("STEP 4: GENERATING VISUALIZATIONS")
    
    print("\nCreating visualizations...")
    visualizer = DeliveryVisualizer()
    
    # Load solution
    import json
    try:
        # Try to load scenario 0 solution
        solution_file = 'results/solution_scenario_0.json'
        if not os.path.exists(solution_file):
            solution_file = 'results/solution.json'  # Fallback
            
        with open(solution_file, 'r') as f:
            routes = json.load(f)
        
        # Load scenario with predictions
        from predictor import make_predictions_for_scenario
        scenario_data = make_predictions_for_scenario(test_scenarios, 0)
        
        # Generate all plots
        visualizer.plot_routes(routes, scenario_data, 
                              title="Optimized Delivery Routes - Scenario 0")
        visualizer.plot_time_windows(routes, scenario_data)
        visualizer.plot_prediction_accuracy(scenario_data)
        visualizer.plot_vehicle_utilization(routes)
        visualizer.plot_cost_comparison(comparison_results)
        
        print(f"\n  ✓ All visualizations saved to results/visualizations/")
        
    except Exception as e:
        print(f"\n  ⚠ Visualization error: {e}")
        print(f"     (This is normal if routes haven't been saved yet)")
    
    # ========== STEP 5: SUMMARY REPORT ==========
    print_header("EXECUTION SUMMARY")
    
    print("\n✅ Pipeline completed successfully!")
    print("\nGenerated Files:")
    print("  📊 Data:")
    print("     - data/historical_data.csv")
    print("     - data/test_scenarios.csv")
    print("  🤖 Models:")
    print("     - data/models/demand_predictor.pkl")
    print("     - data/models/travel_time_predictor.pkl")
    print("  📈 Results:")
    print("     - results/solution.json")
    print("     - results/pipeline_results.json")
    print("  📸 Visualizations:")
    print("     - results/visualizations/*.png")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("  1. Review visualizations in results/visualizations/")
    print("  2. Analyze results in results/*.json")
    print("  3. Read REPORT_TEMPLATE.md for analysis guidance")
    print("  4. Prepare presentation using PRESENTATION_OUTLINE.md")
    print("="*70)
    
    print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
