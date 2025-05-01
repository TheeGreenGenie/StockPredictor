from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from app.predict_service import PredictionService
from data.data_fetcher import DataFetcher

