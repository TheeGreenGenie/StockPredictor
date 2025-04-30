from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data.data_fetcher import DataFetcher
from app.predict_service import PredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stock Market Prediction API",
    description="API for stock price prediction and trading recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
data_fetcher = DataFetcher()
prediction_service = PredictionService()

# Pydantic models for API
class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    prediction_horizon: int = Field(1, description="Number of weeks to predict ahead")
    include_history: bool = Field(False, description="Whether to include historical data in response")
    

class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    prediction: Dict[str, Any]
    recommendation: Dict[str, Any]
    history: Optional[List[Dict[str, Any]]] = None

# API routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Stock Market Prediction API"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/stocks/available")
async def available_stocks():
    """Get list of available stocks for prediction"""
    # This would typically come from a database or configuration
    # For demo purposes, just return a fixed list of popular stocks
    return {
        "stocks": [
            {"ticker": "AAPL", "name": "Apple Inc."},
            {"ticker": "MSFT", "name": "Microsoft Corporation"},
            {"ticker": "GOOGL", "name": "Alphabet Inc."},
            {"ticker": "AMZN", "name": "Amazon.com Inc."},
            {"ticker": "META", "name": "Meta Platforms Inc."},
            {"ticker": "TSLA", "name": "Tesla Inc."},
            {"ticker": "NVDA", "name": "NVIDIA Corporation"},
            {"ticker": "JPM", "name": "JPMorgan Chase & Co."},
            {"ticker": "V", "name": "Visa Inc."},
            {"ticker": "JNJ", "name": "Johnson & Johnson"}
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """
    Get stock prediction and trading recommendation
    
    This endpoint provides:
    - Price forecast for specified weeks ahead
    - Trend prediction (Up/Down/Stable)
    - Trading recommendation (Buy/Sell/Hold)
    - Historical data (optional)
    """
    ticker = request.ticker.upper()
    
    try:
        # Fetch stock data
        df = data_fetcher.fetch_data(ticker, period="1y")
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
        
        # Get current price
        current_price = df['Close'].iloc[-1]
        
        # Get prediction
        prediction_result = prediction_service.predict_price(
            df, 
            horizon_weeks=request.prediction_horizon
        )
        
        # Get trading recommendation
        recommendation = prediction_service.get_recommendation(df)
        
        # Prepare response
        response = {
            "ticker": ticker,
            "current_price": float(current_price),
            "prediction": prediction_result,
            "recommendation": recommendation
        }
        
        # Add historical data if requested
        if request.include_history:
            history = []
            for i, row in df.tail(30).iterrows():  # Last 30 days
                history.append({
                    "date": i.strftime('%Y-%m-%d') if isinstance(i, pd.Timestamp) else str(i),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume'])
                })
            response["history"] = history
            
        return response
        
    except Exception as e:
        logger.error(f"Error processing prediction for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fundamentals/{ticker}")
async def get_fundamentals(ticker: str):
    """Get fundamental data for a stock"""
    ticker = ticker.upper()
    
    try:
        fundamentals = data_fetcher.get_fundamentals(ticker)
        
        if not fundamentals:
            raise HTTPException(status_code=404, detail=f"No fundamental data found for {ticker}")
            
        return {
            "ticker": ticker,
            "fundamentals": fundamentals
        }
        
    except Exception as e:
        logger.error(f"Error getting fundamentals for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/{ticker}")
async def backtest(
    ticker: str, 
    start_date: str = Query(None, description="Start date for backtest (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date for backtest (YYYY-MM-DD)")
):
    """Backtest trading strategy on historical data"""
    ticker = ticker.upper()
    
    try:
        # Default to last 2 years if dates not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Get data for backtest period
        df = data_fetcher.fetch_data(ticker, start=start_date, end=end_date)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker} in specified date range")
            
        # Run backtest
        backtest_results = prediction_service.backtest_strategy(df)
        
        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "initial_balance": backtest_results["initial_balance"],
            "final_balance": backtest_results["final_portfolio"],
            "profit": backtest_results["profit"],
            "profit_percentage": backtest_results["profit_pct"],
            "sharpe_ratio": backtest_results["sharpe_ratio"],
            "max_drawdown": backtest_results["max_drawdown"],
            "trade_count": backtest_results["buy_actions"] + backtest_results["sell_actions"],
            "metrics": {
                "buy_actions": backtest_results["buy_actions"],
                "sell_actions": backtest_results["sell_actions"],
                "hold_actions": backtest_results["hold_actions"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error running backtest for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn server
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)