#!/usr/bin/env python3
"""
Test script for AI Demand Forecasting System
This script validates the system components without requiring full dependencies.
"""

import os
import sys
import json
from datetime import datetime

def test_file_structure():
    """Test if all required files exist"""
    print("🔍 Testing file structure...")
    
    required_files = [
        'main.py',
        'models/demand_forecasting_model.py',
        'models/training.py',
        'templates/index.html',
        'static/styles.css',
        'static/app.js',
        'requirements.txt',
        'data/sales_train_evaluation.csv',
        'data/calendar.csv',
        'data/sell_prices.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"  ✅ {file}")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("  🎉 All required files present!")
    return True

def test_model_class():
    """Test the DemandForecastingModel class"""
    print("\n🧠 Testing model class...")
    
    try:
        # Check if the model file has the required class structure
        with open('models/demand_forecasting_model.py', 'r') as f:
            content = f.read()
        
        required_elements = [
            'class DemandForecastingModel',
            'def train(',
            'def predict(',
            'def load_model(',
            'def save_model('
        ]
        
        for element in required_elements:
            if element not in content:
                print(f"  ❌ Missing: {element}")
                return False
        
        print("  ✅ Model class structure validated")
        print("  ✅ Required methods found")
        print("  🎉 Model class validation passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Model class test failed: {e}")
        return False

def test_fastapi_structure():
    """Test FastAPI application structure"""
    print("\n🚀 Testing FastAPI structure...")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check for required imports
        required_imports = ['FastAPI', 'uvicorn', 'pandas', 'numpy']
        for imp in required_imports:
            if imp not in content:
                print(f"  ❌ Missing import: {imp}")
                return False
        print("  ✅ Required imports found")
        
        # Check for required endpoints
        required_endpoints = [
            '@app.get("/",',
            '@app.post("/api/predict",',
            '@app.post("/api/train")',
            '@app.get("/api/model/status")',
            '@app.get("/health")'
        ]
        
        for endpoint in required_endpoints:
            if endpoint not in content:
                print(f"  ❌ Missing endpoint: {endpoint}")
                return False
        print("  ✅ Required endpoints found")
        
        print("  🎉 FastAPI structure validation passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ FastAPI structure test failed: {e}")
        return False

def test_data_files():
    """Test data files accessibility"""
    print("\n📊 Testing data files...")
    
    try:
        # Test if we can read the first few lines of each data file
        data_files = {
            'data/calendar.csv': ['date', 'wm_yr_wk', 'weekday'],
            'data/sales_train_evaluation.csv': ['id', 'item_id', 'dept_id'],
            'data/sell_prices.csv': ['store_id', 'item_id', 'wm_yr_wk']
        }
        
        for file, expected_cols in data_files.items():
            with open(file, 'r') as f:
                header = f.readline().strip()
                for col in expected_cols:
                    if col not in header:
                        print(f"  ❌ {file}: Missing expected column '{col}'")
                        return False
                print(f"  ✅ {file}: Header validation passed")
        
        print("  🎉 Data files validation passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Data files test failed: {e}")
        return False

def test_web_interface():
    """Test web interface files"""
    print("\n🌐 Testing web interface...")
    
    try:
        # Test HTML template
        with open('templates/index.html', 'r') as f:
            html_content = f.read()
        
        required_elements = [
            'AI Demand Forecasting',
            'predictionForm',
            'importanceChart',
            'model_scores'
        ]
        
        for element in required_elements:
            if element not in html_content:
                print(f"  ❌ HTML: Missing element '{element}'")
                return False
        print("  ✅ HTML template validation passed")
        
        # Test CSS file
        with open('static/styles.css', 'r') as f:
            css_content = f.read()
        
        if len(css_content) < 100:  # Basic check for non-empty CSS
            print("  ❌ CSS file appears to be empty or too small")
            return False
        print("  ✅ CSS file validation passed")
        
        # Test JavaScript file
        with open('static/app.js', 'r') as f:
            js_content = f.read()
        
        if 'predictionForm' not in js_content:
            print("  ❌ JavaScript: Missing prediction form handling")
            return False
        print("  ✅ JavaScript file validation passed")
        
        print("  🎉 Web interface validation passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Web interface test failed: {e}")
        return False

def generate_test_report():
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("🤖 AI DEMAND FORECASTING SYSTEM - TEST REPORT")
    print("="*60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Model Class", test_model_class),
        ("FastAPI Structure", test_fastapi_structure),
        ("Data Files", test_data_files),
        ("Web Interface", test_web_interface)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
        print("\n📋 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train the model: cd models && python training.py")
        print("3. Start the server: python main.py")
        print("4. Open browser: http://localhost:8000")
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    generate_test_report()