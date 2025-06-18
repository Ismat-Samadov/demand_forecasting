# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import uvicorn

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from demand_forecasting_model import DemandForecastingModel

app = FastAPI(
    title="AI Demand Forecasting System", 
    description="Improving accuracy and reducing supply chain costs with machine learning",
    version="1.0.0"
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global model instance
model = None
model_loaded = False

class PredictionRequest(BaseModel):
    item_id: str
    store_id: str
    dept_id: str
    sell_price: float
    prediction_date: str
    has_event: int = 0

class PredictionResponse(BaseModel):
    success: bool
    prediction: float = None
    error: str = None

def load_model():
    """Load the trained demand forecasting model"""
    global model, model_loaded
    
    try:
        model = DemandForecastingModel()
        model_path = os.path.join("models", "demand_forecasting_model.pkl")
        
        if os.path.exists(model_path):
            model.load_model(model_path)
            model_loaded = True
            print("Model loaded successfully!")
        else:
            print("No pre-trained model found. Please train the model first.")
            model_loaded = False
            
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False

def get_data_stats():
    """Get basic statistics about the data"""
    try:
        # Load a sample of the sales data for stats
        sales_path = os.path.join("data", "sales_train_evaluation.csv")
        if os.path.exists(sales_path):
            # Read just a sample for performance
            df_sample = pd.read_csv(sales_path, nrows=1000)
            
            stats = {
                'total_products': len(df_sample['item_id'].unique()),
                'total_stores': len(df_sample['store_id'].unique()),
                'date_range': '1913 days',  # Known from the data
                'avg_daily_sales': '2.5 units'  # Estimated
            }
            return stats
    except Exception as e:
        print(f"Error getting data stats: {e}")
    
    return None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    
    # Get model performance data if available
    model_scores = None
    best_model = None
    feature_importance = None
    
    if model_loaded and model:
        model_scores = getattr(model, 'model_scores', None)
        best_model = getattr(model, 'best_model_name', None)
        
        # Get feature importance
        try:
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                feature_importance = feature_importance.head(15)
        except:
            feature_importance = None
    
    # Get data statistics
    data_stats = get_data_stats()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_scores": model_scores,
        "best_model": best_model,
        "feature_importance": feature_importance,
        "data_stats": data_stats
    })

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest):
    """Make demand prediction"""
    
    if not model_loaded or not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Parse the prediction date
        pred_date = datetime.strptime(request.prediction_date, "%Y-%m-%d")
        
        # Extract date features
        day_of_week = pred_date.weekday()
        day_of_month = pred_date.day
        week_of_year = pred_date.isocalendar()[1]
        month = pred_date.month
        year = pred_date.year
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Create a mock prediction input (simplified for demo)
        # In a real scenario, you'd need more sophisticated feature engineering
        prediction_data = {
            'item_id': [request.item_id],
            'dept_id': [request.dept_id],
            'cat_id': [request.dept_id.split('_')[0]],  # Extract category from dept
            'store_id': [request.store_id],
            'state_id': [request.store_id.split('_')[0]],  # Extract state from store
            'wm_yr_wk': [year * 100 + week_of_year],  # Approximate week format
            'weekday': [day_of_week + 1],  # Convert to 1-7 format
            'month': [month],
            'year': [year],
            'day_of_week': [day_of_week],
            'day_of_month': [day_of_month],
            'week_of_year': [week_of_year],
            'is_weekend': [is_weekend],
            'sell_price': [request.sell_price],
            'price_change': [0],  # Default values for demo
            'price_change_pct': [0],
            'has_event': [request.has_event],
            'is_sporting_event': [0],
            'is_cultural_event': [0],
            'is_national_event': [0],
            'is_religious_event': [0],
            'total_snap': [1],  # Default assumption
            'snap_CA': [1 if 'CA' in request.store_id else 0],
            'snap_TX': [1 if 'TX' in request.store_id else 0],
            'snap_WI': [1 if 'WI' in request.store_id else 0],
        }
        
        # Add lag features with default values (for demo)
        for lag in [1, 7, 14, 28]:
            prediction_data[f'sales_lag_{lag}'] = [2.0]  # Default historical sales
        
        # Add rolling features with default values
        for window in [7, 14, 28]:
            prediction_data[f'sales_rolling_mean_{window}'] = [2.0]
            prediction_data[f'sales_rolling_std_{window}'] = [0.5]
        
        # Add price lag
        prediction_data['price_lag_1'] = [request.sell_price * 0.95]  # Slight variation
        
        # Create DataFrame
        df = pd.DataFrame(prediction_data)
        
        # Encode categorical variables using the model's label encoders
        categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        
        for col in categorical_cols:
            if col in model.label_encoders:
                try:
                    df[col] = model.label_encoders[col].transform(df[col])
                except ValueError:
                    # If the value is not in the encoder, use a default value
                    df[col] = 0
            else:
                df[col] = 0
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Ensure prediction is non-negative
        prediction = max(0, float(prediction))
        
        return PredictionResponse(success=True, prediction=prediction)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return PredictionResponse(success=False, error=str(e))

@app.post("/api/train")
async def train_model():
    """Train the demand forecasting model"""
    global model, model_loaded
    
    try:
        # Check if data files exist
        data_dir = "data"
        required_files = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]
        
        for file in required_files:
            if not os.path.exists(os.path.join(data_dir, file)):
                raise HTTPException(status_code=404, detail=f"Data file {file} not found")
        
        # Initialize and train model
        model = DemandForecastingModel()
        
        sales_path = os.path.join(data_dir, "sales_train_evaluation.csv")
        calendar_path = os.path.join(data_dir, "calendar.csv")
        prices_path = os.path.join(data_dir, "sell_prices.csv")
        
        # Train model (this will take some time)
        scores = model.train(sales_path, calendar_path, prices_path)
        
        # Save the trained model
        model_path = os.path.join("models", "demand_forecasting_model.pkl")
        model.save_model(model_path)
        
        model_loaded = True
        
        return {
            "success": True,
            "message": "Model trained successfully!",
            "scores": scores,
            "best_model": model.best_model_name
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/model/status")
async def model_status():
    """Get model status"""
    return {
        "loaded": model_loaded,
        "model_name": getattr(model, 'best_model_name', None) if model_loaded else None,
        "scores": getattr(model, 'model_scores', None) if model_loaded else None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model_loaded}

if __name__ == "__main__":
    print("Starting AI Demand Forecasting System...")
    print("Dashboard will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )