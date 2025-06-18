import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class DemandForecastingModel:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression()
        }
        self.label_encoders = {}
        self.is_trained = False
        
    def load_data(self, sales_path, calendar_path, prices_path):
        """Load and merge all datasets"""
        print("Loading datasets...")
        
        # Load sales data
        sales_df = pd.read_csv(sales_path)
        calendar_df = pd.read_csv(calendar_path)
        prices_df = pd.read_csv(prices_path)
        
        return sales_df, calendar_df, prices_df
    
    def prepare_features(self, sales_df, calendar_df, prices_df):
        """Transform data into ML-ready format"""
        print("Preparing features...")
        
        # Melt sales data to long format
        id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        sales_melted = pd.melt(sales_df, id_vars=id_cols, var_name='d', value_name='sales')
        
        # Merge with calendar
        sales_melted = sales_melted.merge(calendar_df, on='d', how='left')
        
        # Create store_item_id for price merge
        sales_melted['store_item_id'] = sales_melted['store_id'] + '_' + sales_melted['item_id']
        prices_df['store_item_id'] = prices_df['store_id'] + '_' + prices_df['item_id']
        
        # Merge with prices
        sales_melted = sales_melted.merge(
            prices_df[['store_item_id', 'wm_yr_wk', 'sell_price']], 
            on=['store_item_id', 'wm_yr_wk'], 
            how='left'
        )
        
        # Fill missing prices with median
        sales_melted['sell_price'] = sales_melted.groupby('item_id')['sell_price'].transform(
            lambda x: x.fillna(x.median())
        )
        
        return sales_melted
    
    def create_features(self, df):
        """Create additional features for better forecasting"""
        print("Creating features...")
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Lag features
        df = df.sort_values(['id', 'date'])
        for lag in [1, 7, 14, 28]:
            df[f'sales_lag_{lag}'] = df.groupby('id')['sales'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 28]:
            df[f'sales_rolling_mean_{window}'] = df.groupby('id')['sales'].rolling(window).mean().reset_index(level=0, drop=True)
            df[f'sales_rolling_std_{window}'] = df.groupby('id')['sales'].rolling(window).std().reset_index(level=0, drop=True)
        
        # Price features
        df['price_lag_1'] = df.groupby('id')['sell_price'].shift(1)
        df['price_change'] = df['sell_price'] - df['price_lag_1']
        df['price_change_pct'] = df['price_change'] / df['price_lag_1'] * 100
        
        # Event features
        df['has_event'] = (~df['event_name_1'].isna()).astype(int)
        df['is_sporting_event'] = (df['event_type_1'] == 'Sporting').astype(int)
        df['is_cultural_event'] = (df['event_type_1'] == 'Cultural').astype(int)
        df['is_national_event'] = (df['event_type_1'] == 'National').astype(int)
        df['is_religious_event'] = (df['event_type_1'] == 'Religious').astype(int)
        
        # SNAP benefits
        df['total_snap'] = df['snap_CA'] + df['snap_TX'] + df['snap_WI']
        
        return df
    
    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables"""
        print("Encoding categorical variables...")
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df_encoded
    
    def train(self, sales_path, calendar_path, prices_path, validation_split=0.2):
        """Train the demand forecasting models"""
        print("Starting model training...")
        
        # Load and prepare data
        sales_df, calendar_df, prices_df = self.load_data(sales_path, calendar_path, prices_path)
        merged_df = self.prepare_features(sales_df, calendar_df, prices_df)
        feature_df = self.create_features(merged_df)
        
        # Define categorical columns to encode
        categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1']
        
        # Encode categorical variables
        encoded_df = self.encode_categorical(feature_df, categorical_cols)
        
        # Select features for training
        feature_cols = [
            'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
            'wm_yr_wk', 'weekday', 'month', 'year',
            'day_of_week', 'day_of_month', 'week_of_year', 'is_weekend',
            'sell_price', 'price_change', 'price_change_pct',
            'has_event', 'is_sporting_event', 'is_cultural_event', 'is_national_event', 'is_religious_event',
            'total_snap', 'snap_CA', 'snap_TX', 'snap_WI'
        ]
        
        # Add lag and rolling features if they exist
        lag_cols = [col for col in encoded_df.columns if 'lag_' in col or 'rolling_' in col]
        feature_cols.extend(lag_cols)
        
        # Remove missing values for training
        train_df = encoded_df.dropna(subset=feature_cols + ['sales'])
        
        # Split features and target
        X = train_df[feature_cols]
        y = train_df['sales']
        
        # Train-validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        
        # Train models
        self.model_scores = {}
        
        for name, model in self.models.items():
            print(f"Training {name} model...")
            model.fit(X_train, y_train)
            
            # Validate
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            self.model_scores[name] = {'mae': mae, 'rmse': rmse}
            print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        # Select best model
        best_model_name = min(self.model_scores.keys(), key=lambda x: self.model_scores[x]['mae'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"Best model: {best_model_name}")
        
        # Store feature columns for prediction
        self.feature_cols = feature_cols
        self.is_trained = True
        
        return self.model_scores
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure input has all required features
        X = input_data[self.feature_cols]
        
        # Make prediction with best model
        predictions = self.best_model.predict(X)
        
        return predictions
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'models': self.models,
            'best_model_name': self.best_model_name,
            'label_encoders': self.label_encoders,
            'feature_cols': self.feature_cols,
            'model_scores': self.model_scores
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.best_model_name = model_data['best_model_name']
        self.best_model = self.models[self.best_model_name]
        self.label_encoders = model_data['label_encoders']
        self.feature_cols = model_data['feature_cols']
        self.model_scores = model_data['model_scores']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
        print(f"Best model: {self.best_model_name}")
    
    def get_feature_importance(self):
        """Get feature importance for tree-based models"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print("Feature importance not available for this model type")
            return None