from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from app.predict_service import PredictionService
from data.data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["Stock Predictions"])

# Initialize services
data_fetcher = DataFetcher()
prediction_service = PredictionService()


# Pydantic models
class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    prediction_horizon: int = Field(1, description="Number of weeks to predict ahead")
    include_history: bool = Field(False, description="Whether to include historical data in response")


class BacktestRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: Optional[str] = Field(None, description="Start date for backtest (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date for backtest (YYYY-MM-DD)")
    initial_investment: float = Field(10000.0, description="Initial investment amount")
