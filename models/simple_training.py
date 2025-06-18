#!/usr/bin/env python3
"""
Simple Model Training Script - Creates a basic trained model for demonstration
This script creates a mock trained model to test the system without requiring full ML libraries.
"""

import os
import sys
import json
import pickle
from datetime import datetime

def create_mock_trained_model():
    """
    Create a mock trained model for testing the system.
    This simulates what a real trained model would look like.
    """
    print("🚀 Creating mock trained model for system testing...")
    
    # Mock model performance scores (realistic values)
    model_scores = {
        'rf': {'mae': 1.8542, 'rmse': 2.9871},
        'gb': {'mae': 1.7234, 'rmse': 2.8456},  # Best model
        'lr': {'mae': 2.1567, 'rmse': 3.2891}
    }
    
    # Mock feature importance (realistic feature names and values)
    feature_importance = [
        {'feature': 'sales_lag_1', 'importance': 0.2847},
        {'feature': 'sales_lag_7', 'importance': 0.1923},
        {'feature': 'sales_rolling_mean_7', 'importance': 0.1234},
        {'feature': 'sell_price', 'importance': 0.0876},
        {'feature': 'sales_lag_14', 'importance': 0.0645},
        {'feature': 'day_of_week', 'importance': 0.0543},
        {'feature': 'sales_rolling_std_7', 'importance': 0.0432},
        {'feature': 'month', 'importance': 0.0398},
        {'feature': 'price_change', 'importance': 0.0354},
        {'feature': 'is_weekend', 'importance': 0.0298},
        {'feature': 'has_event', 'importance': 0.0234},
        {'feature': 'sales_lag_28', 'importance': 0.0212},
        {'feature': 'snap_CA', 'importance': 0.0198},
        {'feature': 'year', 'importance': 0.0156},
        {'feature': 'item_id', 'importance': 0.0144}
    ]
    
    # Mock label encoders (for categorical variables)
    mock_label_encoders = {
        'item_id': {'HOBBIES_1_001': 0, 'HOBBIES_1_002': 1, 'HOUSEHOLD_1_118': 2, 'FOODS_3_555': 3},
        'dept_id': {'HOBBIES_1': 0, 'HOBBIES_2': 1, 'HOUSEHOLD_1': 2, 'HOUSEHOLD_2': 3, 'FOODS_1': 4, 'FOODS_2': 5, 'FOODS_3': 6},
        'cat_id': {'HOBBIES': 0, 'HOUSEHOLD': 1, 'FOODS': 2},
        'store_id': {'CA_1': 0, 'CA_2': 1, 'CA_3': 2, 'CA_4': 3, 'TX_1': 4, 'TX_2': 5, 'TX_3': 6, 'WI_1': 7, 'WI_2': 8, 'WI_3': 9},
        'state_id': {'CA': 0, 'TX': 1, 'WI': 2}
    }
    
    # Feature columns used in training
    feature_cols = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'wm_yr_wk', 'weekday', 'month', 'year',
        'day_of_week', 'day_of_month', 'week_of_year', 'is_weekend',
        'sell_price', 'price_change', 'price_change_pct',
        'has_event', 'is_sporting_event', 'is_cultural_event', 
        'is_national_event', 'is_religious_event',
        'total_snap', 'snap_CA', 'snap_TX', 'snap_WI',
        'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
        'sales_rolling_mean_7', 'sales_rolling_mean_14', 'sales_rolling_mean_28',
        'sales_rolling_std_7', 'sales_rolling_std_14', 'sales_rolling_std_28',
        'price_lag_1'
    ]
    
    # Create mock model data structure
    model_data = {
        'models': {
            'rf': MockModel('RandomForest'),
            'gb': MockModel('GradientBoosting'),  # This will be the best model
            'lr': MockModel('LinearRegression')
        },
        'best_model_name': 'gb',
        'label_encoders': mock_label_encoders,
        'feature_cols': feature_cols,
        'model_scores': model_scores,
        'feature_importance': feature_importance,
        'training_date': datetime.now().isoformat(),
        'is_mock_model': True  # Flag to indicate this is a mock model
    }
    
    return model_data

class MockModel:
    """Mock model class that simulates a trained ML model"""
    
    def __init__(self, model_type):
        self.model_type = model_type
        self.is_fitted = True
    
    def predict(self, X):
        """
        Mock prediction function that returns realistic demand values
        based on input features
        """
        import random
        
        # Simulate realistic demand prediction based on inputs
        n_samples = len(X) if hasattr(X, '__len__') else 1
        predictions = []
        
        for i in range(n_samples):
            # Base prediction between 0-20 units
            base_prediction = random.uniform(0.5, 15.0)
            
            # Adjust based on price (higher price = lower demand)
            if hasattr(X, 'iloc') and 'sell_price' in X.columns:
                price = X.iloc[i]['sell_price'] if i < len(X) else 10.0
                price_factor = max(0.3, 2.0 - (price / 20.0))  # Price elasticity
                base_prediction *= price_factor
            
            # Add some department-based variation
            if hasattr(X, 'iloc') and 'dept_id' in X.columns:
                dept = X.iloc[i]['dept_id'] if i < len(X) else 0
                if dept in [4, 5, 6]:  # FOODS departments tend to have higher demand
                    base_prediction *= 1.3
                elif dept in [2, 3]:  # HOUSEHOLD departments moderate demand
                    base_prediction *= 1.1
            
            # Add weekend effect
            if hasattr(X, 'iloc') and 'is_weekend' in X.columns:
                is_weekend = X.iloc[i]['is_weekend'] if i < len(X) else 0
                if is_weekend:
                    base_prediction *= 1.2  # Higher demand on weekends
            
            # Add event effect
            if hasattr(X, 'iloc') and 'has_event' in X.columns:
                has_event = X.iloc[i]['has_event'] if i < len(X) else 0
                if has_event:
                    base_prediction *= 1.4  # Higher demand during events
            
            # Ensure non-negative and realistic range
            prediction = max(0.1, min(50.0, base_prediction))
            predictions.append(prediction)
        
        return predictions if n_samples > 1 else [predictions[0]]

def save_mock_model(model_data, filepath):
    """Save the mock model to a pickle file"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Mock model saved to: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

def create_training_report(model_data):
    """Create a training report for the mock model"""
    report_content = f"""# 🤖 AI Demand Forecasting - Training Report

**Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model Type**: Mock Model for System Testing
**Training Duration**: Simulated (Instant)

## 🏆 Model Performance

**Best Model**: {model_data['best_model_name'].upper()}

| Model | MAE | RMSE |
|-------|-----|------|
| RF | {model_data['model_scores']['rf']['mae']:.4f} | {model_data['model_scores']['rf']['rmse']:.4f} |
| GB 🏆 | {model_data['model_scores']['gb']['mae']:.4f} | {model_data['model_scores']['gb']['rmse']:.4f} |
| LR | {model_data['model_scores']['lr']['mae']:.4f} | {model_data['model_scores']['lr']['rmse']:.4f} |

## 🎯 Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|"""

    for idx, feature in enumerate(model_data['feature_importance'][:10], 1):
        report_content += f"\n| {idx} | {feature['feature']} | {feature['importance']:.4f} |"

    report_content += """

## 📊 Performance Analysis

🟡 **Good Performance**: MAE < 2.0 units (Target achieved)

## ⚠️ Important Note

This is a **mock model** created for system testing. For production use:

1. Install required packages: `pip install pandas numpy scikit-learn joblib`
2. Run actual training: `python training.py`
3. Replace this mock model with real trained model

## ✅ Status

Mock model created successfully for system testing. Ready for frontend integration testing.
"""

    try:
        with open('training_report.md', 'w') as f:
            f.write(report_content)
        print("📄 Training report saved: training_report.md")
        return True
    except Exception as e:
        print(f"❌ Error creating report: {e}")
        return False

def main():
    """Main function to create mock trained model"""
    print("🤖 AI DEMAND FORECASTING - MOCK MODEL CREATION")
    print("=" * 60)
    print("⚠️  This creates a MOCK model for system testing.")
    print("   For production, install ML libraries and run training.py")
    print("=" * 60)
    
    # Create mock model
    model_data = create_mock_trained_model()
    
    # Save model
    model_path = "demand_forecasting_model.pkl"
    if save_mock_model(model_data, model_path):
        print(f"✅ Mock model created successfully!")
        
        # Create training report
        if create_training_report(model_data):
            print(f"✅ Training report generated!")
        
        print("\n" + "=" * 60)
        print("🎉 MOCK MODEL READY FOR TESTING")
        print("=" * 60)
        print("✅ You can now:")
        print("   1. Start the web application: python main.py")
        print("   2. Test predictions through the web interface")
        print("   3. See model performance metrics")
        print("   4. Test the complete system workflow")
        print("\n🔄 To use a real model:")
        print("   1. pip install pandas numpy scikit-learn joblib")
        print("   2. python training.py")
        print("   3. Replace the mock model with real trained model")
        
        return True
    else:
        print("❌ Failed to create mock model")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)