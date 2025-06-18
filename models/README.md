# > AI Demand Forecasting Models Documentation

## =Ë Overview

This directory contains the machine learning models and training pipeline for the AI Demand Forecasting System. The system predicts daily sales quantities for retail products to optimize inventory and reduce supply chain costs.

## <¯ Prediction Target

**What we predict**: Daily sales quantity (number of units) for a specific product in a specific store on a specific date.

**Business Value**: 
- Reduce inventory costs by 15-25%
- Minimize stockouts and overstock situations  
- Improve supply chain efficiency
- Enable data-driven procurement decisions

## =Á Files in this Directory

### Core Files
- `demand_forecasting_model.py` - Main ML model class with all functionality
- `training.py` - Complete training pipeline with validation and reporting
- `demand_forecasting_model.pkl` - Trained model (generated after training)
- `training_report.md` - Detailed training results (generated after training)

## = Complete Training Process

### Phase 1: Data Loading and Validation

#### Input Data Sources
1. **Sales Data** (`../data/sales_train_evaluation.csv`)
   - **Shape**: ~30,000 rows × 1,941+ columns
   - **Content**: Daily sales for each product-store combination
   - **Format**: Wide format with columns d_1, d_2, ..., d_1941 representing days
   - **Key Fields**: id, item_id, dept_id, cat_id, store_id, state_id

2. **Calendar Data** (`../data/calendar.csv`)
   - **Shape**: ~1,970 rows × 14 columns  
   - **Content**: Date information with events and SNAP benefit data
   - **Key Fields**: date, wm_yr_wk, weekday, month, year, event_name_1, event_type_1, snap_CA/TX/WI

3. **Price Data** (`../data/sell_prices.csv`)  
   - **Shape**: ~6.8M rows × 4 columns
   - **Content**: Weekly selling prices for products
   - **Key Fields**: store_id, item_id, wm_yr_wk, sell_price

#### Data Quality Validation
- **Missing Value Check**: Identifies and reports missing data
- **Range Validation**: Ensures no negative sales or invalid prices
- **Consistency Check**: Validates date ranges and product IDs

### Phase 2: Data Preprocessing and Feature Engineering

#### Feature Engineering (25+ Features)

1. **Time-Based Features**
   - `day_of_week` (0-6, Monday=0)
   - `day_of_month` (1-31)
   - `week_of_year` (1-53)
   - `is_weekend` (0/1 flag)

2. **Lag Features (Historical Sales)**
   - `sales_lag_1` - Yesterday's sales
   - `sales_lag_7` - Last week same day
   - `sales_lag_14` - Two weeks ago
   - `sales_lag_28` - Four weeks ago

3. **Rolling Statistics**
   - `sales_rolling_mean_7/14/28` - Moving averages
   - `sales_rolling_std_7/14/28` - Moving standard deviations

4. **Price Features**
   - `sell_price` - Current price
   - `price_change` - Absolute price change
   - `price_change_pct` - Percentage price change

5. **Event Features**
   - `has_event` - Any event flag
   - `is_sporting_event`, `is_cultural_event`, etc.

6. **SNAP Benefit Features**
   - `snap_CA`, `snap_TX`, `snap_WI` - State-level benefits
   - `total_snap` - Combined SNAP availability

### Phase 3: Model Training and Selection

#### Model Architecture

1. **Random Forest Regressor**
   - 100 trees, robust to outliers
   - Good baseline with interpretability

2. **Gradient Boosting Regressor**
   - 100 boosting stages
   - High accuracy through sequential learning

3. **Linear Regression**
   - Fast baseline model
   - Good for comparison

#### Model Selection
- Best model chosen based on lowest Mean Absolute Error (MAE)
- Time-aware train/validation split (80/20)

### Phase 4: Performance Metrics

1. **Mean Absolute Error (MAE)**
   - Target: < 2.0 units for good performance
   - Direct translation to inventory accuracy

2. **Root Mean Square Error (RMSE)**
   - Target: < 3.0 units for good performance
   - Penalizes large errors

#### Expected Feature Importance
1. **Historical Sales (40-60%)**: sales_lag_1, sales_lag_7
2. **Rolling Statistics (15-25%)**: moving averages and std
3. **Price Features (10-20%)**: price sensitivity
4. **Time Features (5-15%)**: seasonal patterns
5. **Event Features (2-8%)**: special occasions

## =€ How to Run Training

### Prerequisites
```bash
pip install pandas numpy scikit-learn joblib
```

### Execute Training
```bash
cd models/
python training.py
```

### Expected Output
```
> AI DEMAND FORECASTING TRAINING PIPELINE
=R Training started at: 2024-06-18 10:30:00
...
<¯ Best Model: GRADIENTBOOSTING
=Ê Final MAE: 1.7234
=Ê Final RMSE: 2.8456
 Training completed successfully!
```

## =Ê Performance Benchmarks

- **Excellent**: MAE < 1.0, RMSE < 2.0
- **Good**: MAE < 2.0, RMSE < 3.0  
- **Acceptable**: MAE < 3.0, RMSE < 4.0

## =' Troubleshooting

1. **Memory Error**: Reduce data sample size
2. **Missing Data**: Ensure CSV files in data/ directory
3. **Import Errors**: Install required packages
4. **Performance Issues**: Reduce n_estimators

---

**Model trained with d for Supply Chain Excellence**