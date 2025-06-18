# 🤖 AI Demand Forecasting System

An intelligent demand forecasting system that improves accuracy and reduces supply chain costs using machine learning.

## ✨ Features

- **Multi-Algorithm ML Models**: Random Forest, Gradient Boosting, and Linear Regression
- **Advanced Feature Engineering**: 25+ engineered features including lag features, rolling statistics, and price changes
- **Interactive Web Dashboard**: Modern Jinja2 template with real-time predictions
- **RESTful API**: FastAPI-powered endpoints for predictions and model management
- **Real-time Predictions**: Live demand forecasting through web interface

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd models
python training.py
```

### 3. Start the Server

```bash
python main.py
```

### 4. Access the Dashboard

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📊 Data Structure

The system expects the following data files in the `data/` directory:

- `sales_train_evaluation.csv` - Historical sales data
- `calendar.csv` - Calendar information with events
- `sell_prices.csv` - Product pricing data

## 🎯 API Endpoints

### Main Endpoints

- `GET /` - Interactive dashboard
- `POST /api/predict` - Make demand predictions
- `POST /api/train` - Train/retrain the model
- `GET /api/model/status` - Get model status
- `GET /health` - Health check

### Prediction Example

```bash
curl -X POST "http://localhost:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "item_id": "HOBBIES_1_001",
       "store_id": "CA_1",
       "dept_id": "HOBBIES_1",
       "sell_price": 9.99,
       "prediction_date": "2024-06-20",
       "has_event": 0
     }'
```

## 🧠 Model Features

### Input Features (25+)
- **Product Info**: item_id, dept_id, cat_id, store_id, state_id
- **Time Features**: weekday, month, year, day_of_week, is_weekend
- **Price Features**: sell_price, price_change, price_change_pct
- **Event Features**: has_event, event_type flags
- **Lag Features**: sales_lag_1, sales_lag_7, sales_lag_14, sales_lag_28
- **Rolling Features**: rolling_mean and rolling_std for 7, 14, 28 days
- **SNAP Benefits**: snap_CA, snap_TX, snap_WI

### Algorithms
- **Random Forest**: Ensemble method for robust predictions
- **Gradient Boosting**: Sequential learning for high accuracy
- **Linear Regression**: Baseline model for comparison

## 🎨 Dashboard Features

- **Model Performance**: Live metrics (MAE, RMSE) for all algorithms
- **Prediction Form**: Interactive form for real-time forecasting
- **Feature Importance**: Visualization of most important features
- **Data Insights**: Key statistics about the dataset

## 🔧 Development

### Project Structure

```
demand_forecasting/
├── main.py                    # FastAPI application
├── requirements.txt           # Dependencies
├── test_system.py            # System validation
├── models/
│   ├── demand_forecasting_model.py  # ML model class
│   └── training.py                  # Training script
├── templates/
│   └── index.html            # Jinja2 dashboard template
├── static/
│   ├── styles.css           # Dashboard styling
│   └── app.js               # Frontend JavaScript
└── data/
    ├── sales_train_evaluation.csv
    ├── calendar.csv
    └── sell_prices.csv
```

### Testing

Run the comprehensive test suite:

```bash
python test_system.py
```

## 📈 Performance

The system achieves:
- **High Accuracy**: Multi-algorithm approach with automatic model selection
- **Fast Predictions**: Sub-second response times
- **Scalable**: Handles 30K+ products across multiple stores
- **Supply Chain Optimization**: Reduces costs through accurate demand forecasting

## 🛠️ Technical Stack

- **Backend**: FastAPI, Python 3.8+
- **ML Libraries**: scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Template Engine**: Jinja2
- **Model Persistence**: joblib

## 📝 License

This project is created for AI-powered supply chain optimization.

---

**Built with ❤️ for Supply Chain Excellence**