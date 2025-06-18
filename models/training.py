from demand_forecasting_model import DemandForecastingModel
import os

def main():
    # Initialize model
    model = DemandForecastingModel()
    
    # Data paths
    data_dir = "../data"
    sales_path = os.path.join(data_dir, "sales_train_evaluation.csv")
    calendar_path = os.path.join(data_dir, "calendar.csv")
    prices_path = os.path.join(data_dir, "sell_prices.csv")
    
    # Train the model
    print("Starting demand forecasting model training...")
    scores = model.train(sales_path, calendar_path, prices_path)
    
    # Print results
    print("\n=== Model Performance ===")
    for model_name, metrics in scores.items():
        print(f"{model_name.upper()}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    # Save the model
    model_path = "demand_forecasting_model.pkl"
    model.save_model(model_path)
    
    # Show feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        print("\n=== Top 15 Feature Importance ===")
        print(importance.head(15))
    
if __name__ == "__main__":
    main()