# Artificial Intelligence in Demand Forecasting
**Improving Accuracy and Reducing Supply Chain Costs**

## 📘 Project Overview

This project explores how Artificial Intelligence (AI), particularly machine learning and deep learning models, can be leveraged to improve demand forecasting accuracy and reduce supply chain costs. Demand forecasting plays a critical role in modern supply chain management by enabling businesses to better plan for inventory, logistics, and customer satisfaction.

Traditional methods often fall short in environments with high variability, promotions, seasonality, or limited historical data. This project uses real-world datasets and state-of-the-art AI techniques to demonstrate practical forecasting solutions.

---

## 🔍 Objectives

- Evaluate and compare traditional and AI-based demand forecasting models.
- Quantify the impact of AI forecasting on inventory and logistics cost.
- Apply machine learning and deep learning algorithms to real-world retail datasets.
- Explore real-time and hierarchical forecasting techniques using AI.

---

## 📦 Datasets

Below is a curated list of publicly available datasets suitable for demand forecasting research and development:

### 1. [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
- **Source**: Walmart
- **Granularity**: Daily sales per SKU/store/state
- **Includes**: Prices, calendar events, product hierarchy
- **Use Case**: LSTM, Prophet, XGBoost, hierarchical forecasting

### 2. [Walmart Sales Forecasting](https://www.kaggle.com/datasets/yasserh/walmart-dataset)
- **Frequency**: Weekly sales
- **Features**: Store metadata, fuel prices, markdowns, holidays
- **Use Case**: Time series regression, ML models (Random Forest, XGBoost)

### 3. [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data)
- **Region**: Germany
- **Includes**: Daily sales, promotions, store types, holidays
- **Use Case**: Deep learning and time-aware attention models

### 4. [Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Region**: Ecuador
- **Size**: Over 50,000 products across 54 stores
- **Use Case**: Multi-variate forecasting with sparse data

### 5. [Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data)
- **Type**: Recurrent purchase data (orders by users)
- **Focus**: Basket-level demand and product reordering
- **Use Case**: Recommender systems and sequence modeling

### 6. [Open Food Facts](https://world.openfoodfacts.org/data)
- **Type**: Product-level metadata
- **Use Case**: Supplement demand datasets with nutrition/brand/category info

### 7. [UCI Online Retail II Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- **Transactions**: Invoices and item quantities
- **Timeframe**: 2009–2011
- **Use Case**: Time-series + RFM analysis + customer-level demand

### 8. [SAP Sample Supply Chain Datasets](https://www.sap.com/about/free-training/data-sets.html)
- **Industry**: Manufacturing
- **Use Case**: Predictive modeling of KPI-level demand and operations

### 9. [IBM Watson Retail Datasets](https://www.ibm.com/communities/analytics/watson-analytics-blog/retail-data-analytics-sample-datasets/)
- **Includes**: Product sales, marketing expenditures, promotions
- **Use Case**: Demand elasticity and price modeling

### 10. **Synthetic Dataset Generation (Custom)**
- Tools: `scikit-learn`, `Faker`, `TimeSynth`, `NumPy`
- **Use Case**: Tailored simulations for testing in specific domains (e.g., healthcare, aerospace, logistics)

---

## 🧠 Models & Techniques (To Be Implemented)

| Model Type        | Algorithms                          | Libraries                       |
|------------------|--------------------------------------|---------------------------------|
| Classical         | ARIMA, Holt-Winters                  | `statsmodels`, `pmdarima`       |
| Machine Learning  | XGBoost, Random Forest, SVR          | `scikit-learn`, `xgboost`       |
| Deep Learning     | LSTM, GRU, N-BEATS, Transformer      | `TensorFlow`, `PyTorch`, `Darts`|
| Hybrid Models     | Prophet + Regressors                 | `fbprophet`                     |

---

## 📈 Evaluation Metrics

- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Squared Error  
- **MAPE**: Mean Absolute Percentage Error  
- **SMAPE**: Symmetric MAPE  
- **WAPE**: Weighted Absolute Percentage Error  

---

## 🧰 Tools & Libraries

- Python (Jupyter Notebooks)
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `lightgbm`
- `tensorflow`, `keras`, `pytorch`
- `prophet`, `darts`, `gluonts`
- Visualization: `Tableau`, `Power BI` (optional)

---

## 📊 Project Phases

1. **Literature Review** – Study state-of-the-art in AI forecasting.
2. **Data Preprocessing** – Clean, impute, and transform features.
3. **Feature Engineering** – Include lags, seasonality, pricing, promotions.
4. **Model Development** – Train/test models, tune hyperparameters.
5. **Forecast Evaluation** – Compare AI models vs. traditional baselines.
6. **Business Impact Analysis** – Assess cost savings & forecasting improvements.

---

## 💸 Impact on Supply Chain

AI-based forecasting helps companies to:
- Reduce inventory holding costs
- Decrease stockouts and backorders
- Improve logistics and warehouse operations
- Respond dynamically to external factors (weather, promotions, trends)

---

## 🚧 Challenges

- Data sparsity and missing values
- Demand volatility during crisis periods (e.g., COVID-19)
- Integration with ERP/SCM systems
- Model interpretability and business adoption

---

## ✅ Expected Deliverables

- Forecasting model implementation with code
- Evaluation reports and comparative performance
- Visual dashboards or charts (forecast curves, error metrics)
- Business value assessment of AI-based forecasting

---

## 📚 References

- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). *The M4 Competition: Results, Findings, Conclusion and Way Forward.*
- Choi, T.-M., Wallace, S. W., & Wang, Y. (2018). *Big Data Analytics in Operations Management.*
- McKinsey (2023). *The AI Revolution in Supply Chain Forecasting.*
- Walmart, Kaggle Competitions, UCI Repository, IBM Watson Data Portal

---

## 📮 Contact

If you would like help setting up the model, building a dashboard, or writing a formal report based on this README, feel free to reach out or open an issue in this repository.
