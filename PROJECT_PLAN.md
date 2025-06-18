# 🎯 AI Demand Forecasting System - Detailed Project Plan

## 📋 Project Overview

**Goal**: Build an AI-powered demand forecasting system that predicts daily sales for retail products to optimize supply chain and reduce costs.

## 🎯 What We're Predicting

### Primary Target
- **Daily Sales Quantity** for specific products in specific stores on specific dates
- **Output**: Number of units expected to be sold (integer/float)

### Business Impact
- **Inventory Optimization**: Reduce overstock and stockouts
- **Cost Reduction**: Minimize carrying costs and lost sales
- **Supply Chain Efficiency**: Better planning and procurement

## 📊 Input Features (What User Provides)

### 1. **Product Information**
- **Item ID**: Specific product identifier (e.g., "HOBBIES_1_001")
- **Department**: Product category (HOBBIES_1, HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2, FOODS_1, FOODS_2, FOODS_3)
- **Category**: Auto-derived from department (HOBBIES, HOUSEHOLD, FOODS)

### 2. **Store Information**  
- **Store ID**: Specific store location (CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3)
- **State**: Auto-derived from store (CA, TX, WI)

### 3. **Time Information**
- **Prediction Date**: Date for which to predict sales (YYYY-MM-DD)
- **Auto-generated time features**: weekday, month, year, day_of_week, is_weekend

### 4. **Price Information**
- **Selling Price**: Current price of the product ($0.01 - $1000.00)

### 5. **Event Information**
- **Special Event**: Whether it's a special event day (checkbox: Yes/No)
- **Event Type**: Type of event (Sporting, Cultural, National, Religious) - optional

### 6. **SNAP Benefits** (Auto-calculated based on store state)
- **SNAP CA/TX/WI**: Whether SNAP benefits are available in that state

## 🧠 ML Model Features (Auto-Generated)

### Historical Sales Features (From Training Data)
- **Lag Features**: sales_lag_1, sales_lag_7, sales_lag_14, sales_lag_28
- **Rolling Statistics**: 
  - rolling_mean_7, rolling_mean_14, rolling_mean_28
  - rolling_std_7, rolling_std_14, rolling_std_28

### Price Features
- **Price Changes**: price_change, price_change_pct
- **Price History**: price_lag_1

### Categorical Encoding
- All categorical variables encoded using LabelEncoder
- Handles unseen categories with default values

## 🔄 Complete Workflow

### Phase 1: Data Preparation & Training
1. **Load Raw Data**: sales, calendar, prices CSV files
2. **Data Preprocessing**: merge datasets, handle missing values
3. **Feature Engineering**: create lag features, rolling statistics, time features
4. **Model Training**: train multiple algorithms (RF, GB, Linear)
5. **Model Selection**: choose best performing model based on MAE
6. **Model Saving**: save trained model with preprocessing pipeline

### Phase 2: Frontend Development
1. **User Input Form**: 
   - Product selection dropdown
   - Store selection dropdown  
   - Date picker (future dates only)
   - Price input field
   - Event checkbox
2. **Result Display**: 
   - Predicted sales quantity
   - Confidence indicators
   - Historical context
3. **Dashboard Features**:
   - Model performance metrics
   - Feature importance visualization
   - Data insights

### Phase 3: Backend API
1. **Model Loading**: load pre-trained model on startup
2. **Input Validation**: validate user inputs
3. **Feature Engineering**: apply same preprocessing as training
4. **Prediction**: use trained model to predict
5. **Response**: return formatted prediction result

### Phase 4: Integration & Testing
1. **End-to-end Testing**: full pipeline validation
2. **Performance Testing**: response time optimization
3. **Error Handling**: robust error management
4. **Documentation**: complete system documentation

## 📈 Expected Model Performance

### Target Metrics
- **MAE (Mean Absolute Error)**: < 2.0 units
- **RMSE (Root Mean Square Error)**: < 3.0 units
- **MAPE (Mean Absolute Percentage Error)**: < 20%

### Model Comparison
- **Random Forest**: Baseline robust model
- **Gradient Boosting**: High accuracy model
- **Linear Regression**: Simple interpretable model

## 🌐 Frontend User Experience

### Main Dashboard Sections
1. **Prediction Panel**: 
   - Clean, intuitive form
   - Real-time validation
   - Clear result display

2. **Model Insights**:
   - Performance metrics
   - Feature importance chart
   - Model confidence indicators

3. **Data Overview**:
   - Dataset statistics
   - Historical trends
   - Business insights

### User Input Flow
```
1. User selects product (dropdown with search)
2. User selects store location (dropdown)
3. User picks prediction date (date picker, future only)
4. User enters current price (number input with validation)
5. User indicates if special event (checkbox)
6. User clicks "Predict Demand" button
7. System shows loading animation
8. System displays prediction result with context
```

## 🔧 Technical Architecture

### Backend Stack
- **FastAPI**: REST API framework
- **scikit-learn**: ML algorithms
- **pandas/numpy**: Data processing
- **joblib**: Model serialization

### Frontend Stack  
- **HTML5/CSS3**: Structure and styling
- **JavaScript**: Interactive functionality
- **Chart.js**: Data visualization
- **Jinja2**: Template rendering

### Data Flow
```
User Input → Frontend Validation → API Request → 
Feature Engineering → Model Prediction → 
API Response → Result Display
```

## 📝 Deliverables

### 1. Trained Model
- `demand_forecasting_model.pkl` - Best performing trained model
- Model performance report with metrics
- Feature importance analysis

### 2. Documentation
- `models/README.md` - Complete training process documentation
- API documentation with examples
- Frontend user guide

### 3. Application
- Working web application with prediction interface
- REST API with all endpoints
- Comprehensive testing suite

### 4. Results Analysis
- Model performance comparison
- Feature importance insights
- Business impact projections

## 🚀 Success Criteria

1. **Model Accuracy**: Achieves target performance metrics
2. **User Experience**: Intuitive, fast, and reliable interface
3. **System Reliability**: Handles edge cases and errors gracefully
4. **Documentation**: Complete and clear documentation
5. **Business Value**: Demonstrates clear ROI potential

This plan ensures we build a complete, production-ready demand forecasting system that delivers real business value!