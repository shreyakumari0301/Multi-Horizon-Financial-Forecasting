# Stock Prediction Web Dashboard

A modern web interface for visualizing stock data, news, and AI predictions.

## Features

- 📈 **Interactive Stock Charts**: View historical price data with Chart.js
- 📰 **Real-time News Feed**: Display recent news with sentiment analysis
- 🤖 **AI Predictions**: Show model predictions with confidence scores
- 🎨 **Modern UI**: Clean, responsive design with Bootstrap
- 🔄 **Real-time Updates**: Dynamic data loading via REST API

## Quick Start

### 1. Install Dependencies

```bash
cd web
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python run.py
```

Or directly:
```bash
python app.py
```

### 3. Open Browser

Visit: http://localhost:5000

## API Endpoints

### GET `/api/stock/<symbol>`
Get historical stock data for a symbol (e.g., AAPL, GOOGL)

**Response:**
```json
{
  "success": true,
  "data": {
    "dates": ["2024-01-01", "2024-01-02", ...],
    "prices": [150.25, 152.10, ...],
    "volumes": [1000000, 1200000, ...],
    "returns": [0.0, 0.012, ...]
  }
}
```

### GET `/api/news/<symbol>`
Get recent news headlines for a symbol

**Response:**
```json
{
  "success": true,
  "news": [
    {
      "date": "2024-01-15",
      "headline": "Apple announces quarterly earnings...",
      "sentiment": "positive"
    }
  ]
}
```

### GET `/api/predict/<symbol>`
Get AI prediction for a symbol

**Response:**
```json
{
  "success": true,
  "prediction": {
    "direction": "LONG",
    "confidence": 0.72,
    "price": 150.50,
    "prediction": 152.30,
    "change_pct": 1.20,
    "timestamp": "2024-01-15T10:30:00"
  }
}
```

## Architecture

```
web/
├── app.py              # Flask backend API
├── run.py              # Application runner
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Main dashboard (HTML/CSS/JS)
└── README.md          # This file
```

## Integration with ML Models

The web app integrates with your trained models via the `ProductionPredictor` class:

1. **Model Loading**: Loads trained hybrid ensemble from `data/models/`
2. **Feature Processing**: Handles technical + news features (38 total)
3. **Real-time Prediction**: Generates predictions on demand

## Customization

### Add New Stock Symbols

Update the dropdown in `templates/index.html`:

```javascript
<option value="YOUR_SYMBOL">YOUR_SYMBOL - Company Name</option>
```

### Modify Chart Styling

Edit the Chart.js configuration in `templates/index.html` for custom visualizations.

### Add More Features

Extend the Flask API in `app.py` to add:
- Multiple timeframes
- Technical indicators
- Portfolio analysis
- Risk metrics

## Development

### Frontend Development

The frontend uses:
- **Chart.js**: Interactive charts
- **Bootstrap 5**: Responsive design
- **Vanilla JavaScript**: API integration

### Backend Development

Built with:
- **Flask**: Lightweight web framework
- **Flask-CORS**: Cross-origin support
- **Pandas/NumPy**: Data processing

## Deployment

For production deployment:

1. **Disable debug mode** in `app.py`
2. **Add authentication** if needed
3. **Configure web server** (nginx + gunicorn)
4. **Add SSL certificate** for HTTPS
5. **Set up monitoring** and logging

Example with gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Troubleshooting

### Common Issues

1. **"Predictor not initialized"**
   - Ensure trained models exist in `data/models/`
   - Check that `ProductionPredictor` can load the models

2. **Empty charts**
   - Verify stock data files exist in `data/raw/`
   - Check CSV format and column names

3. **CORS errors**
   - Flask-CORS is enabled, but check browser console
   - May need additional CORS configuration for production

4. **Slow loading**
   - Optimize Chart.js for large datasets
   - Consider pagination for news data

## Contributing

To extend the dashboard:

1. **API Endpoints**: Add new routes in `app.py`
2. **Frontend Components**: Modify `templates/index.html`
3. **Styling**: Update CSS in `<style>` tags
4. **Data Sources**: Integrate real news APIs and market data feeds

The modular design makes it easy to add features while maintaining clean separation between frontend and backend.