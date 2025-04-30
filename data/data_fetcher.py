import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#Fetches and prepares stock market data
class DataFetcher:

    def __init__(self, cache_dir: str = 'data/cache/'):
        self.cache_dir = cache_dir

    def fetch_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        # Fetch historical data for given ticker
        # Args: ticker (str), period (str), interval (str)
        # Returns : pd.Dataframe w/ OHLCV data
        try:
            logger.info(f"Fetching {ticker} data for {period} with {interval} interval")
            df = yf.download(ticker, period=period, interval=interval, progress=False)

            if df.empty:
                logger.warning(f"No data retrieved for {ticker}")
                return pd.DataFrame()
            
            logger.info(f"Successfully fetched {len(df)} records for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise
            
    def fetch_multiple_tickers(self, tickers: list, period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        #Fetch tickers data
        #Args: tickers (list), period (str), interval (str)
        #Returns: Dict[str, pd.DataFrame]: Dictionary of DataFrames by ticker (str)

        result = {}
        for ticker in tickers:
            result[ticker] = self.fetch_data(ticker, period, interval)
        return result
    
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        #Get fundamental data for ticker
        #Args: ticker (str)
        #Returns: dict with fundamental data

        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info


            fundamentals = {
                'name': info.get('shortName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'beta': info.get('beta', 'N/A'),
                'avg_volume': info.get('averageVolume', 'N/A')
            }