#!/usr/bin/env python3
"""
API Test Script - Tests the FastAPI endpoints
"""

import requests
import json
import time

def test_api():
    """Test the prediction API"""
    base_url = "http://localhost:8000"
    
    # Test data
    prediction_request = {
        "item_id": "HOBBIES_1_001",
        "store_id": "CA_1", 
        "dept_id": "HOBBIES_1",
        "sell_price": 9.99,
        "prediction_date": "2025-06-20",
        "has_event": 1
    }
    
    try:
        # Test health endpoint
        print("🏥 Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
            return False
        
        # Test prediction endpoint
        print("🔮 Testing prediction endpoint...")
        response = requests.post(
            f"{base_url}/api/predict", 
            json=prediction_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                print(f"✅ Prediction successful: {result['prediction']:.2f} units")
                return True
            else:
                print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - Make sure the server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 API TEST SUITE")
    print("=" * 40)
    print("Note: Start the server first with 'python3 main.py'")
    print("=" * 40)
    
    success = test_api()
    if success:
        print("\n🎉 ALL API TESTS PASSED!")
    else:
        print("\n❌ API tests failed")