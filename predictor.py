"""
Prediction Models for Last-Mile Delivery Optimizer
Implements ML models to predict travel times and demand with uncertainty
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import config


class DemandPredictor:
    """Predict customer demand with uncertainty bounds"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
        self.trained = False
        self.global_avg_demand = None
        self.customer_avg_demand = {}
        
    def prepare_features(self, df, is_training=False):
        """
        Prepare features for demand prediction
        
        Features:
        - Customer location (x, y)
        - Day of week
        - Historical average demand for customer
        - Distance from depot
        """
        features = []
        
        # For training data, calculate and store historical averages
        if is_training and 'demand' in df.columns:
            self.customer_avg_demand = df.groupby('customer_id')['demand'].mean().to_dict()
            self.global_avg_demand = df['demand'].mean()
        
        # Use stored averages or default
        default_avg = self.global_avg_demand if self.global_avg_demand is not None else 15.0
        
        for idx, row in df.iterrows():
            customer_id = row['customer_id']
            avg_demand = self.customer_avg_demand.get(customer_id, default_avg)
            
            feature_vector = [
                row['location_x'],
                row['location_y'],
                row['day_of_week'],
                row['distance_from_depot'],
                avg_demand
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def train(self, df):
        """Train demand prediction model"""
        print("\nTraining Demand Prediction Model...")
        
        X = self.prepare_features(df, is_training=True)
        y = df['demand'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TRAIN_TEST_SPLIT, random_state=config.RANDOM_SEED
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"  MAE: {mae:.2f} kg")
        print(f"  RMSE: {rmse:.2f} kg")
        print(f"  R²: {r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=config.CV_FOLDS, 
            scoring='neg_mean_absolute_error'
        )
        print(f"  Cross-val MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} kg")
        
        self.trained = True
        return {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def predict_with_uncertainty(self, df, quantile=0.8):
        """
        Predict demand with uncertainty bounds using quantile regression
        
        Returns:
            predictions: Point estimates
            lower_bounds: Lower confidence bounds
            upper_bounds: Upper confidence bounds
        """
        if not self.trained:
            raise ValueError("Model not trained yet!")
        
        X = self.prepare_features(df)
        
        # Point prediction
        predictions = self.model.predict(X)
        
        # Estimate uncertainty using tree predictions
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Calculate quantiles
        lower_quantile = (1 - quantile) / 2
        upper_quantile = 1 - lower_quantile
        
        lower_bounds = np.quantile(tree_predictions, lower_quantile, axis=0)
        upper_bounds = np.quantile(tree_predictions, upper_quantile, axis=0)
        
        return predictions, lower_bounds, upper_bounds
    
    def save(self, filepath='data/models/demand_predictor.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save model and historical averages
        model_data = {
            'model': self.model,
            'global_avg_demand': self.global_avg_demand,
            'customer_avg_demand': self.customer_avg_demand
        }
        joblib.dump(model_data, filepath)
        print(f"  Demand model saved to {filepath}")
    
    def load(self, filepath='data/models/demand_predictor.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.global_avg_demand = model_data.get('global_avg_demand', 15.0)
            self.customer_avg_demand = model_data.get('customer_avg_demand', {})
        else:
            # Backward compatibility
            self.model = model_data
            self.global_avg_demand = 15.0
            self.customer_avg_demand = {}
        self.trained = True
        print(f"  Demand model loaded from {filepath}")


class TravelTimePredictor:
    """Predict travel time between locations"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
        self.trained = False
    
    def prepare_features(self, df):
        """
        Prepare features for travel time prediction
        
        Features:
        - Distance
        - Hour of day (from preferred_hour)
        - Day of week
        - Is rush hour (binary)
        - Is weekend (binary)
        """
        features = []
        
        for idx, row in df.iterrows():
            hour = int(row['preferred_hour'])
            day_of_week = row['day_of_week']
            
            # Rush hour indicator
            is_rush_hour = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
            
            # Weekend indicator
            is_weekend = 1 if day_of_week >= 5 else 0
            
            feature_vector = [
                row['distance_from_depot'],
                hour,
                day_of_week,
                is_rush_hour,
                is_weekend
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def train(self, df):
        """Train travel time prediction model"""
        print("\nTraining Travel Time Prediction Model...")
        
        X = self.prepare_features(df)
        y = df['travel_time'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TRAIN_TEST_SPLIT, random_state=config.RANDOM_SEED
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"  MAE: {mae:.3f} hours ({mae*60:.1f} minutes)")
        print(f"  RMSE: {rmse:.3f} hours ({rmse*60:.1f} minutes)")
        print(f"  R²: {r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=config.CV_FOLDS,
            scoring='neg_mean_absolute_error'
        )
        print(f"  Cross-val MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f} hours")
        
        self.trained = True
        return {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def predict(self, df):
        """Predict travel times"""
        if not self.trained:
            raise ValueError("Model not trained yet!")
        
        X = self.prepare_features(df)
        predictions = self.model.predict(X)
        
        return predictions
    
    def save(self, filepath='data/models/travel_time_predictor.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"  Travel time model saved to {filepath}")
    
    def load(self, filepath='data/models/travel_time_predictor.pkl'):
        """Load trained model"""
        self.model = joblib.load(filepath)
        self.trained = True
        print(f"  Travel time model loaded from {filepath}")


def train_all_models(historical_data_path='data/historical_data.csv'):
    """Train all prediction models"""
    print("="*60)
    print("TRAINING PREDICTION MODELS")
    print("="*60)
    
    # Load data
    print("\nLoading historical data...")
    df = pd.read_csv(historical_data_path)
    print(f"  Loaded {len(df)} historical records")
    
    # Train demand predictor
    demand_predictor = DemandPredictor()
    demand_metrics = demand_predictor.train(df)
    demand_predictor.save()
    
    # Train travel time predictor
    travel_time_predictor = TravelTimePredictor()
    travel_metrics = travel_time_predictor.train(df)
    travel_time_predictor.save()
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    
    return {
        'demand_metrics': demand_metrics,
        'travel_time_metrics': travel_metrics
    }


def make_predictions_for_scenario(scenario_df, scenario_id=0):
    """
    Make predictions for a test scenario
    
    Args:
        scenario_df: DataFrame with test scenario data
        scenario_id: ID of scenario to predict
    
    Returns:
        DataFrame with predictions added
    """
    # Load models
    demand_predictor = DemandPredictor()
    demand_predictor.load()
    
    travel_time_predictor = TravelTimePredictor()
    travel_time_predictor.load()
    
    # Filter scenario
    scenario_data = scenario_df[scenario_df['scenario_id'] == scenario_id].copy()
    
    # Predict demand with uncertainty
    demand_pred, demand_lower, demand_upper = demand_predictor.predict_with_uncertainty(
        scenario_data, quantile=0.8
    )
    
    scenario_data['predicted_demand'] = demand_pred
    scenario_data['demand_lower_bound'] = demand_lower
    scenario_data['demand_upper_bound'] = demand_upper
    
    # Predict travel time
    travel_time_pred = travel_time_predictor.predict(scenario_data)
    scenario_data['predicted_travel_time'] = travel_time_pred
    
    return scenario_data


def main():
    """Main execution"""
    # Train models
    metrics = train_all_models()
    
    # Test predictions on first scenario
    print("\n" + "="*60)
    print("TESTING PREDICTIONS ON SCENARIO 0")
    print("="*60)
    
    test_scenarios = pd.read_csv('data/test_scenarios.csv')
    predictions = make_predictions_for_scenario(test_scenarios, scenario_id=0)
    
    print("\nPrediction vs Actual Comparison:")
    print("-" * 60)
    
    comparison = predictions[['customer_id', 'actual_demand', 'predicted_demand', 
                              'demand_lower_bound', 'demand_upper_bound',
                              'actual_travel_time', 'predicted_travel_time']]
    
    print(comparison.to_string(index=False))
    
    # Calculate prediction errors
    demand_error = mean_absolute_error(
        predictions['actual_demand'], 
        predictions['predicted_demand']
    )
    travel_error = mean_absolute_error(
        predictions['actual_travel_time'],
        predictions['predicted_travel_time']
    )
    
    print(f"\nDemand Prediction MAE: {demand_error:.2f} kg")
    print(f"Travel Time Prediction MAE: {travel_error:.3f} hours ({travel_error*60:.1f} min)")


if __name__ == "__main__":
    main()
