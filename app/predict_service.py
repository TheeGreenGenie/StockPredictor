import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.price_model import PriceModel
from models.r1_trading_agent import RLTradingAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionService:
    """Service for loading models and generating predictions"""
    
    def __init__(self, model_dir: str = 'models/saved_models/'):
        """
        Initialize the prediction service
        
        Args:
            model_dir (str): Directory with saved models
        """
        self.model_dir = model_dir
        self.price_model = PriceModel(model_dir)
        self.trading_agent = RLTradingAgent(model_dir)
        
        # Load models if available
        self._load_models()
    
    def _load_models(self) -> bool:
        """
        Load saved models if available
        
        Returns:
            bool: Success flag
        """
        success = True
        
        # Try to load price model
        price_model_path = os.path.join(self.model_dir, 'price_model_dense.h5')
        if os.path.exists(price_model_path):
            price_model_loaded = self.price_model.load_model(price_model_path)
            if not price_model_loaded:
                logger.warning(f"Failed to load price model from {price_model_path}")
                success = False
        else:
            logger.warning(f"Price model not found at {price_model_path}")
            success = False
            
        # Try to load trading agent
        trading_agent_path = os.path.join(self.model_dir, 'trading_agent_ppo.pkl')
        if os.path.exists(trading_agent_path):
            agent_loaded = self.trading_agent.load_model(trading_agent_path, model_type='ppo')
            if not agent_loaded:
                logger.warning(f"Failed to load trading agent from {trading_agent_path}")
                success = False
        else:
            logger.warning(f"Trading agent model not found at {trading_agent_path}")
            success = False
            
        return success
    
    def predict_price(self, df: pd.DataFrame, horizon_weeks: int = 1) -> Dict[str, Any]:
        """
        Generate price prediction for future weeks
        
        Args:
            df (pd.DataFrame): Historical price data
            horizon_weeks (int): Number of weeks to predict ahead
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        try:
            # Check if we have enough data
            if len(df) < 30:  # Need at least 30 days of data
                logger.warning("Not enough data for prediction")
                return {
                    "success": False,
                    "error": "Not enough historical data for prediction"
                }
                
            # Process features
            features, _ = self._preprocess_data(df)
            
            if len(features) == 0:
                return {
                    "success": False,
                    "error": "Could not extract features from data"
                }
            
            # If price model is loaded, use it; otherwise use a simple model
            if self.price_model.model is not None:
                # Get the latest features for prediction
                latest_features = features[-1].reshape(1, -1)
                
                # Generate predictions for each week in horizon
                predictions = []
                feature_set = latest_features.copy()
                
                for week in range(horizon_weeks):
                    # Predict next week
                    pred = float(self.price_model.predict(feature_set)[0])
                    
                    # Create date for prediction
                    if isinstance(df.index, pd.DatetimeIndex):
                        last_date = df.index[-1]
                        pred_date = last_date + timedelta(days=7 * (week + 1))
                        pred_date_str = pred_date.strftime('%Y-%m-%d')
                    else:
                        pred_date_str = f"Week {week + 1}"
                    
                    # Store prediction
                    predictions.append({
                        "date": pred_date_str,
                        "predicted_price": pred,
                        "week": week + 1
                    })
                    
                    # Update feature set for next prediction (simple approach)
                    # In a real system, you would create a proper feature vector
                    feature_set[0][0] = pred  # Update price feature
                
                # Calculate overall trend
                current_price = df['Close'].iloc[-1]
                final_prediction = predictions[-1]['predicted_price']
                
                if final_prediction > current_price * 1.03:
                    trend = "Up"
                elif final_prediction < current_price * 0.97:
                    trend = "Down"
                else:
                    trend = "Stable"
                
                # Calculate confidence level (simplified approach)
                # In a real system, this would come from model uncertainty
                prediction_std = df['Close'].pct_change().std() * 100  # Volatility
                if prediction_std < 1.0:
                    confidence = "High"
                elif prediction_std < 2.0:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                
                return {
                    "success": True,
                    "forecast_prices": predictions,
                    "trend": trend,
                    "confidence": confidence,
                    "horizon_weeks": horizon_weeks,
                    "volatility": float(prediction_std)
                }
            else:
                # If model not loaded, use a simple moving average forecast
                logger.warning("Price model not loaded, using simple moving average forecast")
                
                # Calculate a simple moving average based forecast
                ma_period = min(20, len(df) // 2)  # Use half the data or 20 days, whichever is smaller
                last_ma = df['Close'].rolling(window=ma_period).mean().iloc[-1]
                
                # Calculate average weekly change over the last N weeks
                weekly_data = df.resample('W').last() if isinstance(df.index, pd.DatetimeIndex) else df
                weekly_returns = weekly_data['Close'].pct_change().dropna()
                avg_weekly_change = weekly_returns.mean()
                
                # Generate predictions
                predictions = []
                for week in range(horizon_weeks):
                    # Simple projection based on average weekly change
                    if week == 0:
                        pred_price = last_ma * (1 + avg_weekly_change)
                    else:
                        pred_price = predictions[week-1]['predicted_price'] * (1 + avg_weekly_change)
                        
                    # Create date for prediction
                    if isinstance(df.index, pd.DatetimeIndex):
                        last_date = df.index[-1]
                        pred_date = last_date + timedelta(days=7 * (week + 1))
                        pred_date_str = pred_date.strftime('%Y-%m-%d')
                    else:
                        pred_date_str = f"Week {week + 1}"
                        
                    predictions.append({
                        "date": pred_date_str,
                        "predicted_price": float(pred_price),
                        "week": week + 1
                    })
                
                # Determine trend
                current_price = df['Close'].iloc[-1]
                final_prediction = predictions[-1]['predicted_price']
                
                if final_prediction > current_price * 1.03:
                    trend = "Up"
                elif final_prediction < current_price * 0.97:
                    trend = "Down"
                else:
                    trend = "Stable"
                    
                # Lower confidence for fallback method
                return {
                    "success": True,
                    "forecast_prices": predictions,
                    "trend": trend,
                    "confidence": "Low",
                    "horizon_weeks": horizon_weeks,
                    "note": "Using fallback prediction method due to unavailable model"
                }
                
        except Exception as e:
            logger.error(f"Error in price prediction: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_recommendation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading recommendation based on current market data
        
        Args:
            df (pd.DataFrame): Historical price data
            
        Returns:
            Dict[str, Any]: Trading recommendation
        """
        try:
            # Check if we have enough data
            if len(df) < 30:  # Need at least 30 days of data
                logger.warning("Not enough data for recommendation")
                return {
                    "success": False,
                    "error": "Not enough historical data for recommendation"
                }
            
            # If trading agent is loaded, use it for prediction
            if self.trading_agent.model is not None:
                # Get prediction from trading agent
                recommendation = self.trading_agent.predict(df)
                
                # Return formatted recommendation
                return {
                    "success": True,
                    "action": recommendation['action'],
                    "confidence": float(recommendation['confidence']),
                    "reasoning": self._generate_reasoning(df, recommendation['action'])
                }
            else:
                # If model not loaded, use a simple rule-based approach
                logger.warning("Trading agent not loaded, using simple rule-based recommendation")
                
                # Calculate simple technical indicators
                df = df.copy()
                df['MA_10'] = df['Close'].rolling(window=10).mean()
                df['MA_30'] = df['Close'].rolling(window=30).mean()
                
                # Get latest values
                current_price = df['Close'].iloc[-1]
                ma_10 = df['MA_10'].iloc[-1]
                ma_30 = df['MA_30'].iloc[-1]
                
                # Simple moving average crossover strategy
                if ma_10 > ma_30 and ma_10 > current_price:
                    action = "buy"
                    confidence = 0.6  # Medium confidence
                elif ma_10 < ma_30 and ma_10 < current_price:
                    action = "sell"
                    confidence = 0.6
                else:
                    action = "hold"
                    confidence = 0.5
                    
                return {
                    "success": True,
                    "action": action,
                    "confidence": confidence,
                    "reasoning": self._generate_reasoning(df, action),
                    "note": "Using fallback recommendation method due to unavailable model"
                }
                
        except Exception as e:
            logger.error(f"Error in trading recommendation: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def backtest_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Backtest the trading strategy on historical data
        
        Args:
            df (pd.DataFrame): Historical price data
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        try:
            # Check if trading agent is loaded
            if self.trading_agent.model is not None:
                # Run backtest using the trading agent
                backtest_results = self.trading_agent.backtest(df)
                return backtest_results
            else:
                # If model not loaded, use a simple rule-based backtest
                logger.warning("Trading agent not loaded, using simple rule-based backtest")
                
                # Calculate simple technical indicators
                df = df.copy()
                df['MA_10'] = df['Close'].rolling(window=10).mean()
                df['MA_30'] = df['Close'].rolling(window=30).mean()
                df.dropna(inplace=True)
                
                # Initialize backtest variables
                initial_balance = 10000.0
                balance = initial_balance
                position = 0  # Number of shares held
                trades = []
                portfolio_values = [initial_balance]
                
                # Run backtest
                for i in range(1, len(df)):
                    # Get current and previous day data
                    current = df.iloc[i]
                    prev = df.iloc[i-1]
                    
                    # Moving average crossover strategy
                    prev_signal = 1 if prev['MA_10'] > prev['MA_30'] else -1
                    current_signal = 1 if current['MA_10'] > current['MA_30'] else -1
                    
                    # Trading logic
                    if current_signal == 1 and prev_signal == -1:  # Buy signal
                        if balance > 0:
                            # Calculate shares to buy (all available balance)
                            shares = balance / current['Close']
                            cost = shares * current['Close']
                            balance = 0
                            position += shares
                            
                            # Record trade
                            trades.append({
                                'date': df.index[i],
                                'action': 'buy',
                                'price': current['Close'],
                                'shares': shares,
                                'cost': cost
                            })
                            
                    elif current_signal == -1 and prev_signal == 1:  # Sell signal
                        if position > 0:
                            # Sell all positions
                            proceeds = position * current['Close']
                            balance += proceeds
                            
                            # Record trade
                            trades.append({
                                'date': df.index[i],
                                'action': 'sell',
                                'price': current['Close'],
                                'shares': position,
                                'proceeds': proceeds
                            })
                            
                            position = 0
                    
                    # Calculate portfolio value
                    portfolio_value = balance + (position * current['Close'])
                    portfolio_values.append(portfolio_value)
                
                # Calculate performance metrics
                final_portfolio = portfolio_values[-1]
                profit = final_portfolio - initial_balance
                profit_pct = profit / initial_balance * 100
                
                # Calculate Sharpe ratio
                daily_returns = []
                for i in range(1, len(portfolio_values)):
                    daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                    daily_returns.append(daily_return)
                
                daily_returns = np.array(daily_returns)
                sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * np.sqrt(252)  # Annualized
                
                # Calculate max drawdown
                max_drawdown = self._calculate_max_drawdown(portfolio_values)
                
                # Count trades by type
                buy_actions = len([t for t in trades if t['action'] == 'buy'])
                sell_actions = len([t for t in trades if t['action'] == 'sell'])
                
                return {
                    "initial_balance": initial_balance,
                    "final_portfolio": final_portfolio,
                    "profit": profit,
                    "profit_pct": profit_pct,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "buy_actions": buy_actions,
                    "sell_actions": sell_actions,
                    "hold_actions": len(df) - buy_actions - sell_actions,
                    "note": "Using fallback backtest method due to unavailable model"
                }
                
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Process stock data for model input"""
        if df.empty:
            return np.array([]), np.array([])
            
        # Clean missing data
        df = df.copy()
        df.dropna(inplace=True)
        
        # Create technical indicators
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # Create weekly resampled features
        if isinstance(df.index, pd.DatetimeIndex):
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
        else:
            # If not a datetime index, create pseudo-weekly data by grouping every 5 rows
            df['week_group'] = df.index // 5
            weekly = df.groupby('week_group').agg({
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
        
        # Drop rows with NaN values
        weekly.dropna(inplace=True)
        
        if weekly.empty:
            return np.array([]), np.array([])
        
        # Define features and target
        feature_columns = ['Close', 'MA_5', 'MA_20', 'RSI', 'Volatility', 'Volume']
        features = weekly[feature_columns].values
        
        # Target is next week's closing price
        target = weekly['Close'].shift(-1).dropna().values
        
        # Align features with target
        features = features[:-1]  # Remove the last row as we don't have a target for it
        
        return features, target
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        
        # Make two series: one for gains and one for losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS
        rs = avg_gain / (avg_loss + 1e-9)  # Add small epsilon to avoid division by zero
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values"""
        max_so_far = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > max_so_far:
                max_so_far = value
            
            drawdown = (max_so_far - value) / max_so_far
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown
    
    def _generate_reasoning(self, df: pd.DataFrame, action: str) -> str:
        """Generate human-readable reasoning for a trading recommendation"""
        # Get latest market data
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = (current_price - prev_price) / prev_price * 100
        
        # Calculate simple indicators
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        ma_10 = df['MA_10'].iloc[-1]
        ma_30 = df['MA_30'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        
        # Generate reasoning based on action and indicators
        if action == 'buy':
            if rsi < 30:
                reasoning = f"The stock appears oversold with an RSI of {rsi:.1f}, suggesting a potential buying opportunity."
            elif ma_10 > ma_30:
                reasoning = f"Short-term momentum is positive with the 10-day moving average (${ma_10:.2f}) above the 30-day moving average (${ma_30:.2f})."
            else:
                reasoning = f"Technical indicators suggest upward price movement in the near term."
                
        elif action == 'sell':
            if rsi > 70:
                reasoning = f"The stock appears overbought with an RSI of {rsi:.1f}, suggesting it may be time to take profits."
            elif ma_10 < ma_30:
                reasoning = f"Short-term momentum is negative with the 10-day moving average (${ma_10:.2f}) below the 30-day moving average (${ma_30:.2f})."
            else:
                reasoning = f"Technical indicators suggest downward price movement in the near term."
                
        else:  # hold
            reasoning = f"Current market conditions don't strongly favor either buying or selling. The RSI is at a neutral {rsi:.1f} and price is moving within normal ranges."
            
        return reasoning
    
if __name__ == "__main__":
    
    # Fetch some test data
    df = yf.download("AAPL", period="1y")
    
    # Initialize service
    service = PredictionService()
    
    # Get price prediction
    prediction = service.predict_price(df, horizon_weeks=4)
    print("Price Prediction:")
    print(prediction)
    
    # Get trading recommendation
    recommendation = service.get_recommendation(df)
    print("\nTrading Recommendation:")
    print(recommendation)
                