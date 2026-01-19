#!/usr/bin/env python3
"""
Run the Stock Prediction Web Application
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    from app import app
    print("🚀 Starting Stock Prediction Web App...")
    print("📊 Visit http://localhost:5000 to view the dashboard")
    print("📈 Using yfinance for real-time stock data")
    
    # Check if yfinance is available
    try:
        import yfinance as yf
        print("✓ yfinance available - real-time data enabled")
    except ImportError:
        print("⚠ yfinance not installed - install with: pip install yfinance")
        print("  Falling back to local data files")
    
    app.run(debug=True, host='0.0.0.0', port=5000)