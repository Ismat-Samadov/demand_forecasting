#!/usr/bin/env python3
"""
AI Demand Forecasting Model Training Script

This script trains multiple ML models on the retail sales data and saves the best performing model.
The training process includes comprehensive data preprocessing, feature engineering, and model evaluation.
"""

from demand_forecasting_model import DemandForecastingModel
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"🤖 {title}")
    print("="*80)

def print_data_info(sales_df, calendar_df, prices_df):
    """Print comprehensive data information"""
    print_header("DATA OVERVIEW")
    
    print(f"📊 Sales Data: {sales_df.shape[0]:,} rows × {sales_df.shape[1]} columns")
    print(f"📅 Calendar Data: {calendar_df.shape[0]:,} rows × {calendar_df.shape[1]} columns") 
    print(f"💰 Prices Data: {prices_df.shape[0]:,} rows × {prices_df.shape[1]} columns")
    
    print(f"\n📦 Products: {sales_df['item_id'].nunique():,} unique items")
    print(f"🏪 Stores: {sales_df['store_id'].nunique()} stores")
    print(f"🗓️ Time Range: {calendar_df['date'].min()} to {calendar_df['date'].max()}")
    print(f"💵 Price Range: ${prices_df['sell_price'].min():.2f} - ${prices_df['sell_price'].max():.2f}")
    
    # Sample data preview
    print(f"\n📋 Sales Sample:")
    print(sales_df.head(3).to_string(index=False))

def validate_data_quality(sales_df, calendar_df, prices_df):
    """Validate data quality and report issues"""
    print_header("DATA QUALITY VALIDATION")
    
    issues = []
    
    # Check for missing values
    sales_missing = sales_df.isnull().sum().sum()
    calendar_missing = calendar_df.isnull().sum().sum()
    prices_missing = prices_df.isnull().sum().sum()
    
    print(f"❓ Missing Values:")
    print(f"  Sales: {sales_missing:,} missing values")
    print(f"  Calendar: {calendar_missing:,} missing values") 
    print(f"  Prices: {prices_missing:,} missing values")
    
    if sales_missing > 0:
        issues.append("Sales data has missing values")
    if calendar_missing > 0:
        issues.append("Calendar data has missing values")
    if prices_missing > 0:
        issues.append("Price data has missing values")
    
    # Check data ranges
    sales_cols = [col for col in sales_df.columns if col.startswith('d_')]
    negative_sales = (sales_df[sales_cols] < 0).sum().sum()
    
    if negative_sales > 0:
        issues.append(f"Found {negative_sales} negative sales values")
        print(f"⚠️ Found {negative_sales} negative sales values")
    
    if len(issues) == 0:
        print("✅ Data quality validation passed!")
    else:
        print(f"⚠️ Data quality issues found: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
    
    return len(issues) == 0

def main():
    """Main training pipeline"""
    print_header("AI DEMAND FORECASTING TRAINING PIPELINE")
    print(f"🕒 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize model
    print("\n🔧 Initializing DemandForecastingModel...")
    model = DemandForecastingModel()
    
    # Data paths
    data_dir = "../data"
    sales_path = os.path.join(data_dir, "sales_train_evaluation.csv")
    calendar_path = os.path.join(data_dir, "calendar.csv")
    prices_path = os.path.join(data_dir, "sell_prices.csv")
    
    # Validate file existence
    required_files = [sales_path, calendar_path, prices_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ ERROR: Required file not found: {file_path}")
            return False
    
    print("✅ All data files found!")
    
    try:
        # Load and validate data
        print("\n📥 Loading data files...")
        sales_df, calendar_df, prices_df = model.load_data(sales_path, calendar_path, prices_path)
        
        # Print data information
        print_data_info(sales_df, calendar_df, prices_df)
        
        # Validate data quality
        if not validate_data_quality(sales_df, calendar_df, prices_df):
            print("⚠️ Warning: Data quality issues detected. Training will continue...")
        
        # Train the model
        print_header("MODEL TRAINING")
        print("🚀 Starting model training pipeline...")
        print("   - Data preprocessing and feature engineering")
        print("   - Training Random Forest, Gradient Boosting, and Linear Regression")
        print("   - Model evaluation and selection")
        print("\n⏳ This may take several minutes...")
        
        start_time = datetime.now()
        scores = model.train(sales_path, calendar_path, prices_path, validation_split=0.2)
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Print results
        print_header("TRAINING RESULTS")
        print(f"⏱️ Training completed in: {training_duration}")
        print(f"🎯 Best Model: {model.best_model_name.upper()}")
        
        print(f"\n📊 Model Performance Comparison:")
        print("-" * 60)
        print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'Status'}")
        print("-" * 60)
        
        best_mae = min(metrics['mae'] for metrics in scores.values())
        
        for model_name, metrics in scores.items():
            status = "🏆 BEST" if metrics['mae'] == best_mae else ""
            print(f"{model_name.upper():<20} {metrics['mae']:<12.4f} {metrics['rmse']:<12.4f} {status}")
        
        # Performance interpretation
        print(f"\n🎯 Performance Analysis:")
        best_score = scores[model.best_model_name]
        if best_score['mae'] < 1.0:
            print("🟢 Excellent performance: MAE < 1.0 unit")
        elif best_score['mae'] < 2.0:
            print("🟡 Good performance: MAE < 2.0 units")
        elif best_score['mae'] < 3.0:
            print("🟠 Acceptable performance: MAE < 3.0 units")
        else:
            print("🔴 Performance needs improvement: MAE > 3.0 units")
        
        # Save the model
        model_path = "demand_forecasting_model.pkl"
        print(f"\n💾 Saving trained model to: {model_path}")
        model.save_model(model_path)
        
        # Show feature importance
        importance = model.get_feature_importance()
        if importance is not None:
            print_header("FEATURE IMPORTANCE ANALYSIS")
            print("🎯 Top 15 Most Important Features for Demand Prediction:")
            print("-" * 60)
            print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12} {'Impact'}")
            print("-" * 60)
            
            for idx, (_, row) in enumerate(importance.head(15).iterrows(), 1):
                if row['importance'] > 0.1:
                    impact = "🔥 High"
                elif row['importance'] > 0.05:
                    impact = "🔸 Medium"
                else:
                    impact = "🔹 Low"
                
                print(f"{idx:<6} {row['feature']:<30} {row['importance']:<12.4f} {impact}")
            
            # Feature categories analysis
            print(f"\n📊 Feature Category Analysis:")
            lag_importance = importance[importance['feature'].str.contains('lag_')]['importance'].sum()
            rolling_importance = importance[importance['feature'].str.contains('rolling_')]['importance'].sum()
            price_importance = importance[importance['feature'].str.contains('price')]['importance'].sum()
            time_importance = importance[importance['feature'].str.contains('day|week|month|year')]['importance'].sum()
            
            print(f"  📈 Historical Sales (lag features): {lag_importance:.2%}")
            print(f"  📊 Rolling Statistics: {rolling_importance:.2%}")
            print(f"  💰 Price Features: {price_importance:.2%}")
            print(f"  📅 Time Features: {time_importance:.2%}")
        
        # Generate training summary
        print_header("TRAINING SUMMARY")
        print("✅ Training completed successfully!")
        print(f"📦 Model saved: {model_path}")
        print(f"🏆 Best algorithm: {model.best_model_name.upper()}")
        print(f"📊 Final MAE: {scores[model.best_model_name]['mae']:.4f}")
        print(f"📊 Final RMSE: {scores[model.best_model_name]['rmse']:.4f}")
        print(f"⏱️ Training time: {training_duration}")
        print(f"🎯 Model ready for production deployment!")
        
        # Create training report
        create_training_report(model, scores, importance, training_duration)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Training failed with exception:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_training_report(model, scores, importance, training_duration):
    """Create a detailed training report"""
    report_path = "training_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# 🤖 AI Demand Forecasting - Training Report\n\n")
        f.write(f"**Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Training Duration**: {training_duration}\n\n")
        
        f.write("## 🏆 Model Performance\n\n")
        f.write(f"**Best Model**: {model.best_model_name.upper()}\n\n")
        
        f.write("| Model | MAE | RMSE |\n")
        f.write("|-------|-----|------|\n")
        for model_name, metrics in scores.items():
            marker = " 🏆" if model_name == model.best_model_name else ""
            f.write(f"| {model_name.upper()}{marker} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} |\n")
        
        f.write("\n## 🎯 Feature Importance\n\n")
        if importance is not None:
            f.write("| Rank | Feature | Importance |\n")
            f.write("|------|---------|------------|\n")
            for idx, (_, row) in enumerate(importance.head(10).iterrows(), 1):
                f.write(f"| {idx} | {row['feature']} | {row['importance']:.4f} |\n")
        
        f.write(f"\n## ✅ Status\n\n")
        f.write("Model training completed successfully and is ready for production deployment.\n")
    
    print(f"📄 Training report saved: {report_path}")

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 Training pipeline completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Training pipeline failed!")
        sys.exit(1)