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
                'avg_volume': info.get('averageVolume', 'N/A'),
            }

            return fundamentals
        
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {str(e)}")
            return {}
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        #Preprocess stock data b4 model
        #Args: df (pd.DataFrame)
        #Returns: Tuple[np.ndarray, np.ndarray]

        if df.empty:
            return np.array([]), np.array([])
        
        df = df.copy()
        df.dropna(inplace=True)

        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

        weekly = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'MA_5': 'last',
            'MA_20': 'last',
            'RSI': 'last',
            'Volatility': 'last'
        })

        weekly.dropna(inplace=True)

        if weekly.empty:
            return np.array([]), np.array([])
        
        feature_columns = ['Close', 'MA_5', 'MA_20', 'RSI', 'Volatility', 'Volume']