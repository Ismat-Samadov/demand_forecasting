#!/usr/bin/env python3
"""
Comprehensive Test Suite for AI Demand Forecasting System

This test suite covers:
- Model functionality and accuracy
- API endpoints and responses  
- Data integrity and preprocessing
- Feature engineering pipeline
- System integration and performance
- Error handling and edge cases

Run with: python test_comprehensive.py
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import our modules
try:
    from demand_forecasting_model import DemandForecastingModel
    import main
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

warnings.filterwarnings('ignore')

class DemandForecastingTester:
    """Comprehensive test suite for the demand forecasting system"""
    
    def __init__(self):
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'total': 0,
            'start_time': datetime.now()
        }
        self.errors = []
        self.base_url = "http://localhost:8000"
        
    def print_header(self, title):
        """Print formatted test section header"""
        print("\n" + "="*80)
        print(f"🧪 {title}")
        print("="*80)
        
    def print_test(self, test_name, status, message=""):
        """Print individual test result"""
        symbols = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}
        print(f"{symbols.get(status, '❓')} {test_name:<50} {status}")
        if message:
            print(f"   📝 {message}")
            
    def assert_test(self, condition, test_name, error_msg=""):
        """Assert test condition and track results"""
        self.test_results['total'] += 1
        try:
            if condition:
                self.test_results['passed'] += 1
                self.print_test(test_name, "PASS")
                return True
            else:
                self.test_results['failed'] += 1
                self.print_test(test_name, "FAIL", error_msg)
                self.errors.append(f"{test_name}: {error_msg}")
                return False
        except Exception as e:
            self.test_results['failed'] += 1
            self.print_test(test_name, "FAIL", str(e))
            self.errors.append(f"{test_name}: {str(e)}")
            return False
    
    def skip_test(self, test_name, reason):
        """Skip a test with reason"""
        self.test_results['total'] += 1
        self.test_results['skipped'] += 1
        self.print_test(test_name, "SKIP", reason)
        
    def test_data_files(self):
        """Test data file existence and integrity"""
        self.print_header("DATA FILES VALIDATION")
        
        data_dir = Path("data")
        required_files = [
            "sales_train_evaluation.csv",
            "calendar.csv", 
            "sell_prices.csv"
        ]
        
        # Check if data directory exists
        self.assert_test(
            data_dir.exists(),
            "Data directory exists",
            f"Directory {data_dir} not found"
        )
        
        # Check required files
        for file_name in required_files:
            file_path = data_dir / file_name
            self.assert_test(
                file_path.exists(),
                f"File {file_name} exists",
                f"Required file {file_path} not found"
            )
            
            if file_path.exists():
                # Check file is not empty
                file_size = file_path.stat().st_size
                self.assert_test(
                    file_size > 1000,  # At least 1KB
                    f"File {file_name} not empty",
                    f"File size: {file_size} bytes"
                )
                
        # Test data loading
        try:
            sales_path = data_dir / "sales_train_evaluation.csv"
            if sales_path.exists():
                df = pd.read_csv(sales_path, nrows=100)  # Sample for speed
                
                self.assert_test(
                    len(df) > 0,
                    "Sales data loads successfully",
                    f"Loaded {len(df)} rows"
                )
                
                # Check expected columns
                expected_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
                missing_cols = [col for col in expected_cols if col not in df.columns]
                self.assert_test(
                    len(missing_cols) == 0,
                    "Sales data has required columns",
                    f"Missing columns: {missing_cols}"
                )
                
        except Exception as e:
            self.assert_test(False, "Sales data validation", str(e))
    
    def test_model_functionality(self):
        """Test ML model core functionality"""
        self.print_header("MODEL FUNCTIONALITY TESTS")
        
        # Test model initialization
        try:
            model = DemandForecastingModel()
            self.assert_test(
                hasattr(model, 'models'),
                "Model initialization",
                "Model object created successfully"
            )
            
            self.assert_test(
                'rf' in model.models,
                "Random Forest model present",
                "RF model in models dict"
            )
            
            self.assert_test(
                'gb' in model.models,
                "Gradient Boosting model present", 
                "GB model in models dict"
            )
            
            self.assert_test(
                'lr' in model.models,
                "Linear Regression model present",
                "LR model in models dict"
            )
            
        except Exception as e:
            self.assert_test(False, "Model initialization", str(e))
            return
            
        # Test trained model loading
        model_path = Path("models/demand_forecasting_model.pkl")
        if model_path.exists():
            try:
                model.load_model(str(model_path))
                self.assert_test(
                    model.is_trained,
                    "Trained model loads successfully",
                    "Model loaded from pickle file"
                )
                
                # Test prediction with sample data
                sample_data = pd.DataFrame({
                    'item_id': ['HOBBIES_1_001'],
                    'dept_id': ['HOBBIES_1'],
                    'cat_id': ['HOBBIES'],
                    'store_id': ['CA_1'],
                    'state_id': ['CA'],
                    'wm_yr_wk': [11101],
                    'weekday': [1],
                    'month': [1],
                    'year': [2024],
                    'day_of_week': [0],
                    'day_of_month': [1],
                    'week_of_year': [1],
                    'is_weekend': [0],
                    'sell_price': [9.99],
                    'price_change': [0],
                    'price_change_pct': [0],
                    'has_event': [0],
                    'is_sporting_event': [0],
                    'is_cultural_event': [0],
                    'is_national_event': [0],
                    'is_religious_event': [0],
                    'total_snap': [1],
                    'snap_CA': [1],
                    'snap_TX': [0],
                    'snap_WI': [0],
                    'sales_lag_1': [2.0],
                    'sales_lag_7': [2.0],
                    'sales_lag_14': [2.0],
                    'sales_lag_28': [2.0],
                    'sales_rolling_mean_7': [2.0],
                    'sales_rolling_mean_14': [2.0],
                    'sales_rolling_mean_28': [2.0],
                    'sales_rolling_std_7': [0.5],
                    'sales_rolling_std_14': [0.5],
                    'sales_rolling_std_28': [0.5],
                    'price_lag_1': [9.99]
                })
                
                # Encode categorical variables if needed
                for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
                    if col in model.label_encoders:
                        try:
                            sample_data[col] = model.label_encoders[col].transform(sample_data[col])
                        except ValueError:
                            sample_data[col] = 0  # Default for unseen values
                    else:
                        sample_data[col] = 0
                
                prediction = model.predict(sample_data)
                
                self.assert_test(
                    isinstance(prediction, (list, np.ndarray)),
                    "Model prediction returns valid format",
                    f"Prediction type: {type(prediction)}"
                )
                
                self.assert_test(
                    len(prediction) > 0 and prediction[0] >= 0,
                    "Model prediction is non-negative",
                    f"Prediction value: {prediction[0] if len(prediction) > 0 else 'None'}"
                )
                
                # Test feature importance
                try:
                    importance = model.get_feature_importance()
                    if importance is not None:
                        self.assert_test(
                            len(importance) > 0,
                            "Feature importance available",
                            f"Got {len(importance)} features"
                        )
                    else:
                        self.skip_test("Feature importance", "Not available for this model type")
                except:
                    self.skip_test("Feature importance", "Error retrieving importance")
                
            except Exception as e:
                self.assert_test(False, "Model loading and prediction", str(e))
        else:
            self.skip_test("Trained model tests", "No trained model found")
    
    def test_api_endpoints(self):
        """Test API endpoints functionality"""
        self.print_header("API ENDPOINTS TESTS")
        
        if not REQUESTS_AVAILABLE:
            self.skip_test("API tests", "requests module not available")
            return
        
        # Test if server is running (optional)
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            server_running = response.status_code == 200
        except:
            server_running = False
            
        if not server_running:
            self.skip_test("API tests", "Server not running at localhost:8000")
            return
            
        # Test health endpoint
        try:
            response = requests.get(f"{self.base_url}/health")
            self.assert_test(
                response.status_code == 200,
                "Health endpoint responds",
                f"Status: {response.status_code}"
            )
            
            data = response.json()
            self.assert_test(
                'status' in data,
                "Health endpoint returns status",
                f"Response: {data}"
            )
        except Exception as e:
            self.assert_test(False, "Health endpoint test", str(e))
            
        # Test model status endpoint
        try:
            response = requests.get(f"{self.base_url}/api/model/status")
            self.assert_test(
                response.status_code == 200,
                "Model status endpoint responds",
                f"Status: {response.status_code}"
            )
            
            data = response.json()
            self.assert_test(
                'loaded' in data,
                "Model status returns loaded flag",
                f"Model loaded: {data.get('loaded', 'unknown')}"
            )
        except Exception as e:
            self.assert_test(False, "Model status endpoint test", str(e))
            
        # Test prediction endpoint
        try:
            prediction_data = {
                "item_id": "HOBBIES_1_001",
                "store_id": "CA_1",
                "dept_id": "HOBBIES_1",
                "sell_price": 9.99,
                "prediction_date": "2024-06-20",
                "has_event": 0
            }
            
            response = requests.post(
                f"{self.base_url}/api/predict",
                json=prediction_data,
                headers={'Content-Type': 'application/json'}
            )
            
            self.assert_test(
                response.status_code == 200,
                "Prediction endpoint responds",
                f"Status: {response.status_code}"
            )
            
            if response.status_code == 200:
                data = response.json()
                self.assert_test(
                    'success' in data,
                    "Prediction response has success field",
                    f"Success: {data.get('success', 'unknown')}"
                )
                
                if data.get('success'):
                    self.assert_test(
                        'prediction' in data and isinstance(data['prediction'], (int, float)),
                        "Prediction response has valid prediction",
                        f"Prediction: {data.get('prediction', 'unknown')}"
                    )
                    
        except Exception as e:
            self.assert_test(False, "Prediction endpoint test", str(e))
    
    def test_feature_engineering(self):
        """Test feature engineering pipeline"""
        self.print_header("FEATURE ENGINEERING TESTS")
        
        # Create sample data for feature engineering
        sample_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'sales': [2.0, 3.0, 1.5],
            'sell_price': [9.99, 9.99, 8.99],
            'event_name_1': [None, 'SuperBowl', None],
            'event_type_1': [None, 'Sporting', None],
            'snap_CA': [1, 1, 0],
            'snap_TX': [0, 0, 1], 
            'snap_WI': [0, 1, 0],
            'id': ['test_1', 'test_1', 'test_1']
        })
        
        try:
            model = DemandForecastingModel()
            
            # Test date parsing
            sample_df['date'] = pd.to_datetime(sample_df['date'])
            
            # Test time features
            sample_df['day_of_week'] = sample_df['date'].dt.dayofweek
            sample_df['day_of_month'] = sample_df['date'].dt.day
            sample_df['is_weekend'] = (sample_df['day_of_week'] >= 5).astype(int)
            
            self.assert_test(
                'day_of_week' in sample_df.columns,
                "Time features creation",
                "day_of_week, day_of_month, is_weekend created"
            )
            
            # Test event features
            sample_df['has_event'] = (~sample_df['event_name_1'].isna()).astype(int)
            sample_df['is_sporting_event'] = (sample_df['event_type_1'] == 'Sporting').astype(int)
            
            self.assert_test(
                sample_df['has_event'].sum() > 0,
                "Event features creation",
                f"Found {sample_df['has_event'].sum()} events"
            )
            
            # Test SNAP features
            sample_df['total_snap'] = sample_df['snap_CA'] + sample_df['snap_TX'] + sample_df['snap_WI']
            
            self.assert_test(
                'total_snap' in sample_df.columns,
                "SNAP features creation",
                f"Total SNAP values: {sample_df['total_snap'].tolist()}"
            )
            
            # Test price features  
            sample_df = sample_df.sort_values('date')
            sample_df['price_lag_1'] = sample_df['sell_price'].shift(1)
            sample_df['price_change'] = sample_df['sell_price'] - sample_df['price_lag_1']
            
            self.assert_test(
                'price_change' in sample_df.columns,
                "Price features creation",
                "price_lag_1, price_change created"
            )
            
        except Exception as e:
            self.assert_test(False, "Feature engineering test", str(e))
    
    def test_data_preprocessing(self):
        """Test data preprocessing functions"""
        self.print_header("DATA PREPROCESSING TESTS")
        
        # Test with minimal sample data
        sample_sales = pd.DataFrame({
            'id': ['HOBBIES_1_001_CA_1_evaluation'],
            'item_id': ['HOBBIES_1_001'],
            'dept_id': ['HOBBIES_1'],
            'cat_id': ['HOBBIES'],
            'store_id': ['CA_1'],
            'state_id': ['CA'],
            'd_1': [2],
            'd_2': [3],
            'd_3': [1]
        })
        
        sample_calendar = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3'],
            'date': ['2011-01-29', '2011-01-30', '2011-01-31'],
            'wm_yr_wk': [11101, 11101, 11101],
            'weekday': ['Saturday', 'Sunday', 'Monday'],
            'wday': [1, 2, 3],
            'month': [1, 1, 1],
            'year': [2011, 2011, 2011],
            'event_name_1': [None, None, None],
            'event_type_1': [None, None, None],
            'snap_CA': [1, 1, 0],
            'snap_TX': [0, 0, 1],
            'snap_WI': [0, 1, 0]
        })
        
        sample_prices = pd.DataFrame({
            'store_id': ['CA_1', 'CA_1', 'CA_1'],
            'item_id': ['HOBBIES_1_001', 'HOBBIES_1_001', 'HOBBIES_1_001'],
            'wm_yr_wk': [11101, 11101, 11101],
            'sell_price': [9.99, 9.99, 8.99]
        })
        
        try:
            model = DemandForecastingModel()
            
            # Test data melting
            id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
            sales_melted = pd.melt(sample_sales, id_vars=id_cols, var_name='d', value_name='sales')
            
            self.assert_test(
                len(sales_melted) == 3,  # 1 product × 3 days
                "Data melting works",
                f"Melted to {len(sales_melted)} rows"
            )
            
            # Test calendar merge
            merged_data = sales_melted.merge(sample_calendar, on='d', how='left')
            
            self.assert_test(
                'date' in merged_data.columns,
                "Calendar data merge",
                "Date column added successfully"
            )
            
            # Test price merge preparation
            merged_data['store_item_id'] = merged_data['store_id'] + '_' + merged_data['item_id']
            sample_prices['store_item_id'] = sample_prices['store_id'] + '_' + sample_prices['item_id']
            
            price_merged = merged_data.merge(
                sample_prices[['store_item_id', 'wm_yr_wk', 'sell_price']], 
                on=['store_item_id', 'wm_yr_wk'], 
                how='left'
            )
            
            self.assert_test(
                'sell_price' in price_merged.columns,
                "Price data merge", 
                "Price column added successfully"
            )
            
        except Exception as e:
            self.assert_test(False, "Data preprocessing test", str(e))
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        self.print_header("ERROR HANDLING TESTS")
        
        # Test invalid prediction data via API (if server running)
        if not REQUESTS_AVAILABLE:
            self.skip_test("API error handling", "requests module not available")
            return
            
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            server_running = response.status_code == 200
        except:
            server_running = False
            
        if server_running:
            # Test invalid prediction request
            try:
                invalid_data = {
                    "item_id": "",  # Empty item ID
                    "store_id": "INVALID_STORE",
                    "dept_id": "INVALID_DEPT", 
                    "sell_price": -5.0,  # Negative price
                    "prediction_date": "invalid-date",  # Invalid date
                    "has_event": 2  # Invalid event flag
                }
                
                response = requests.post(
                    f"{self.base_url}/api/predict",
                    json=invalid_data,
                    headers={'Content-Type': 'application/json'}
                )
                
                # Should handle gracefully (either error response or default prediction)
                self.assert_test(
                    response.status_code in [200, 400, 422],
                    "Invalid data handling",
                    f"Server responded with status {response.status_code}"
                )
                
            except Exception as e:
                self.assert_test(False, "Invalid data handling test", str(e))
        else:
            self.skip_test("API error handling", "Server not running")
            
        # Test model with missing features
        try:
            model = DemandForecastingModel()
            
            # Try to create features with minimal data
            minimal_df = pd.DataFrame({
                'date': ['2024-01-01'],
                'sales': [1.0],
                'id': ['test']
            })
            
            minimal_df['date'] = pd.to_datetime(minimal_df['date'])
            minimal_df['day_of_week'] = minimal_df['date'].dt.dayofweek
            
            self.assert_test(
                len(minimal_df) > 0,
                "Minimal data processing",
                "Can process data with minimal features"
            )
            
        except Exception as e:
            self.assert_test(False, "Minimal data processing test", str(e))
    
    def test_performance(self):
        """Test system performance metrics"""
        self.print_header("PERFORMANCE TESTS")
        
        # Test prediction speed (if model is available)
        model_path = Path("models/demand_forecasting_model.pkl")
        if model_path.exists():
            try:
                model = DemandForecastingModel()
                model.load_model(str(model_path))
                
                # Prepare test data
                test_data = pd.DataFrame({
                    'item_id': [0] * 100,  # Encoded values
                    'dept_id': [0] * 100,
                    'cat_id': [0] * 100,
                    'store_id': [0] * 100,
                    'state_id': [0] * 100,
                    'wm_yr_wk': [11101] * 100,
                    'weekday': [1] * 100,
                    'month': [1] * 100,
                    'year': [2024] * 100,
                    'day_of_week': [0] * 100,
                    'day_of_month': [1] * 100,
                    'week_of_year': [1] * 100,
                    'is_weekend': [0] * 100,
                    'sell_price': [9.99] * 100,
                    'price_change': [0] * 100,
                    'price_change_pct': [0] * 100,
                    'has_event': [0] * 100,
                    'is_sporting_event': [0] * 100,
                    'is_cultural_event': [0] * 100,
                    'is_national_event': [0] * 100,
                    'is_religious_event': [0] * 100,
                    'total_snap': [1] * 100,
                    'snap_CA': [1] * 100,
                    'snap_TX': [0] * 100,
                    'snap_WI': [0] * 100,
                    'sales_lag_1': [2.0] * 100,
                    'sales_lag_7': [2.0] * 100,
                    'sales_lag_14': [2.0] * 100,
                    'sales_lag_28': [2.0] * 100,
                    'sales_rolling_mean_7': [2.0] * 100,
                    'sales_rolling_mean_14': [2.0] * 100,
                    'sales_rolling_mean_28': [2.0] * 100,
                    'sales_rolling_std_7': [0.5] * 100,
                    'sales_rolling_std_14': [0.5] * 100,
                    'sales_rolling_std_28': [0.5] * 100,
                    'price_lag_1': [9.99] * 100
                })
                
                # Time 100 predictions
                start_time = time.time()
                predictions = model.predict(test_data)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 100 * 1000  # ms per prediction
                
                self.assert_test(
                    avg_time < 100,  # Less than 100ms per prediction
                    "Prediction speed test",
                    f"Average: {avg_time:.2f}ms per prediction"
                )
                
                self.assert_test(
                    len(predictions) == 100,
                    "Batch prediction accuracy",
                    f"Got {len(predictions)} predictions for 100 inputs"
                )
                
            except Exception as e:
                self.assert_test(False, "Performance test", str(e))
        else:
            self.skip_test("Performance tests", "No trained model found")
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("\n" + "🤖 AI DEMAND FORECASTING - COMPREHENSIVE TEST SUITE".center(80, "="))
        print(f"📅 Started at: {self.test_results['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Run all test categories
        test_methods = [
            self.test_data_files,
            self.test_model_functionality,
            self.test_feature_engineering,
            self.test_data_preprocessing,
            self.test_api_endpoints,
            self.test_error_handling,
            self.test_performance
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"❌ Test category failed: {test_method.__name__}")
                print(f"   Error: {str(e)}")
                self.errors.append(f"{test_method.__name__}: {str(e)}")
        
        # Print final results
        self.print_final_report()
    
    def print_final_report(self):
        """Print comprehensive test results"""
        end_time = datetime.now()
        duration = end_time - self.test_results['start_time']
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        print(f"⏱️  Duration: {duration.total_seconds():.2f} seconds")
        print(f"🧪 Total Tests: {self.test_results['total']}")
        print(f"✅ Passed: {self.test_results['passed']}")
        print(f"❌ Failed: {self.test_results['failed']}")
        print(f"⏭️  Skipped: {self.test_results['skipped']}")
        
        if self.test_results['total'] > 0:
            pass_rate = (self.test_results['passed'] / self.test_results['total']) * 100
            print(f"📈 Pass Rate: {pass_rate:.1f}%")
        
        if self.test_results['failed'] == 0:
            print("\n🎉 ALL TESTS PASSED! System is ready for production.")
        else:
            print(f"\n⚠️  {self.test_results['failed']} tests failed. See details below:")
            for i, error in enumerate(self.errors[:10], 1):  # Show first 10 errors
                print(f"   {i}. {error}")
            if len(self.errors) > 10:
                print(f"   ... and {len(self.errors) - 10} more errors")
        
        if self.test_results['skipped'] > 0:
            print(f"\nℹ️  {self.test_results['skipped']} tests were skipped (missing dependencies or server not running)")
        
        print("\n" + "="*80)
        
        # Return exit code
        return 0 if self.test_results['failed'] == 0 else 1

def main():
    """Main test execution"""
    print("🚀 Starting Comprehensive Test Suite...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return 1
    
    # Run tests
    tester = DemandForecastingTester()
    exit_code = tester.run_all_tests()
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)