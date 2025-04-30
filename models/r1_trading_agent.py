import numpy as np
import pandas as pd
import gym
from gym import spaces
import os
import pickle
from typing import Dict, List, Tuple, Any, Union
import logging
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockTradingEnv(gym.Env):
    #Environment for stock trading with gym
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0,
                 max_steps: int = None, window_size: int = 10,
                 commision: float = 0.001):
        #Args: df (pd.DataFrame), initial_balance (float), max_steps (int), window_size (int), commission (float)
        super(StockTradingEnv, self).__init__()

        self.df = df.copy()
        self.window_size = window_size
        self.prices = self.df['Close'].values

        self.initial_balance = initial_balance
        self.commission = commision

        self.max_steps = max_steps if max_steps else len(self.df) - window_size - 1

        self.action_space = spaces.Discrete(3)

        num_features = 10
        self.observation_space = spaces.Box(
            low=np.inf,
            high=np.inf,
            shape=(window_size * num_features + 2,),
            dtype=np.float32
        )

        self.reset()

    def _next_observation(self) -> np.ndarray:
        #Returns: np.ndarray (current observation)

        frame = self.df.iloc[self.current_step:self.current_step + self.window_size].copy()

        obs = []
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            col_data = frame[col].values
            normalized = (col_data - np.mean(col_data) / (np.std(col_data) + 1e-10))
            obs.append(normalized)

        for col in ['MA_5', 'MA_20', 'RSI', 'Volatility', 'Daily_Return']:
            if col in frame.columns:
                col_data = frame[col].values
                normalized = (col_data - np.mean(col_data)) / (np.std(col_data) + 1e-10)
                obs.append(normalized)
            else:
                obs.append(np.zeros(self.window_size))

        obs = np.array(obs).flatten()


        balance_normalized = self.balance / self.initial_balance
        holdings_normalized = self.holdings * self.current_price / self.initial_balance

        obs = np.append(obs, [balance_normalized, holdings_normalized])

        return obs
    
    def reset(self) -> np.ndarray:
        #Reset the enviornment
        #Returns: np.ndarray
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        self.trades = []
        self.total_reward = 0
        self.current_price = self.prices[self.current_step + self.window_size - 1]

        return self._next_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        #Args: action (int)
        #Returns: Tuple[np.ndarray, float, bool, Dict]
        self.current_price

        reward = 0
        done = False
        info = {}

        if action == 1:
            if self.balance > 0:
                max_shares = self.balance / (self.current_price * (1 + self.commission))
                shares_to_buy = max_shares

                cost = shares_to_buy * self.current_price * (1 + self.commission)
                self.balance -= cost
                self.holdings += shares_to_buy

                self.trades.append({
                    'step': self.current_step,
                    'price': self.current_price,
                    'type': 'buy',
                    'shares': shares_to_buy,
                    'cost': cost
                })

                reward -= 0.1
            
        elif action == 2:
            if self.holdings > 0:
                proceeds = self.holdings * self.current_price * (1 - self.commission)
                self.balance += proceeds

                self.trades.append({
                    'step': self.current_step,
                    'price': self.current_price,
                    'type': 'sell',
                    'shares': self.holdings,
                    'proceeds': proceeds
                })

                self.holdings = 0

                reward -= 0.1

        portfolio_value = self.balance + (self.holdings * self.current_price)

        if self.current_step > 0:
            prev_price = self.prices[self.current_step + self.window_size - 1]
            price_change_pct = (self.current_price - prev_price) / prev_price

            if self.holdings > 0:
                reward += price_change_pct * 100

            elif price_change_pct > 0.01:
                reward -= 0.1

        self.current_step +=1

        if self.current_step >= self.max_steps or self.current_step >= len(self.df) - self.window_size - 1:
            done = True

            info['final_portfolio_value'] = portfolio_value

            profit_pct = (portfolio_value - self.initial_balance) / self.initial_balance
            reward += profit_pct * 100

        self.total_reward += reward

        info.update({
            'current_step': self.current_step,
            'current_price': self.current_price,
            'balance': self.balance,
            'holdings': self.holdings,
            'portfolio_value': portfolio_value,
            'trade_count': len(self.trades),
            'total_reward': self.total_reward
        })

        return self._next_observation(), reward, done, info
    
    def render(self, mode='human', close=False):
        #Render the enviornment
        if mode =='human':
            portfolio_value = self.balance + (self.holdings * self.current_price)
            profit_pct = (portfolio_value - self.initial_balance) / self.initial_balance * 100

            print(f"Step: {self.current_step}")
            print(f"Price: ${self.current_price:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Holdings: {self.holdings:.4f} shares")
            print(f"Portfolio Value: ${portfolio_value:.2f} ({profit_pct:.2f}%)")
            print(f"Total Reward: {self.total_reward:.2f}")
            print("=" * 50)
            
            return
            
        return
    
class RLTradingAgent:
    def __init__(self, model_dir: str = 'saved_models/'):
        self.model_dir = model_dir
        self.model = None

        os.makedirs(model_dir, exist_ok=True)

    def train(self, df: pd.DataFrame, model_type: str = 'ppo',
              total_timesteps: int = 100000, window_size: int = 10) -> Dict[str, Any]:
        #Args: df (pd.dataframe), model_type (str), total_timesteps (int), window_size (int)
        #Returns: DIct[str, Any]
        if df.empty:
            logger.error("Cannot train with empty DataFrame")
            return {}
        
        required_cols  = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"DataFrame missing required columns: {required_cols}")
            return {}
        
        df = self._add_technical_indicators(df)
        env = StockTradingEnv(df, window_size=window_size)
        env = DummyVecEnv([lambda: env])

        if model_type.lower() == 'ppo':
            self.model = PPO(
                'MlpPolicy',
                env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=1
            )
        elif model_type.lower() == 'a2c':
            self.model = A2C(
                'MlpPolicy',
                env,
                learning_rate=0.0007,
                verbose=1
            )
        elif model_type.lower() == 'dqn':
            self.model = DQN(
                'MlpPolicy',
                env,
                learning_rate=0.0001,
                buffer_size=50000,
                exploration_fraction=0.1,
                exploration_final_eps=0.02,
                verbose=1
            )
        else:
            logger.error(f"Unkown model type: {model_type}")
            return {}
        
        logger.info(f"Trainning {model_type} agent for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)

        model_path = os.path.join(self.model_dir, f"trading_agent_{model_type}.pk1")
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=10)

        return {
            'model_type': model_type,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'total_timesteps': total_timesteps,
            'model_path': model_path
        }
    
    def load_model(self, model_path: str, model_type: str = 'ppo') -> bool:
        #Args: model_path (str): Path to saved model, model_type (str): Type of RL algorithm
        # Returns: bool: Success flag
        try:
            if model_type.lower() == 'ppo':
                self.model = PPO.load(model_path)
            elif model_type.lower() == 'a2c':
                self.model = A2C.load(model_path)
            elif model_type.lower() == 'dqn':
                self.model = DQN.load(model_path)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
                
            logger.info(f"Loaded {model_type} model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, df: pd.DataFrame, window_size: int = 10) -> Dict[str, Any]:
        #Makes decision based on data
        #Args: df (pd.DataFrame): Recent market data, window_size (int): 
        #Returns: Dict[str, Any]: Prediction results with action and confidence
        
        if self.model is None:
            logger.error("Model not trained or loaded")
            return {'action': 'hold', 'confidence': 0.0}
            
        if len(df) < window_size:
            logger.error(f"Not enough data, need at least {window_size} rows")
            return {'action': 'hold', 'confidence': 0.0}
            
        # Add technical indicators if not present
        df = self._add_technical_indicators(df)
        
        # Create environment for prediction
        env = StockTradingEnv(df, window_size=window_size)
        
        # Get initial observation
        observation = env.reset()
        
        # Get model prediction
        action, _states = self.model.predict(observation, deterministic=True)
        
        # Map action to string
        action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
        action_str = action_map.get(action, 'hold')
        
        # Get action probabilities if available (for PPO/A2C)
        confidence = 0.7  # Default medium-high confidence
        
        try:
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'get_distribution'):
                dist = self.model.policy.get_distribution(observation)
                probs = dist.distribution.probs.detach().numpy()
                confidence = float(probs[0][action])
        except Exception as e:
            logger.warning(f"Could not get action probabilities: {str(e)}")
        
        return {
            'action': action_str,
            'action_code': int(action),
            'confidence': confidence,
            'current_price': df['Close'].iloc[-1],
            'timestamp': df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None
        }
    
    def backtest(self, df: pd.DataFrame, window_size: int = 10, 
                 initial_balance: float = 10000.0) -> Dict[str, Any]:
        #Backtest the trained agent on historical data
        #Args: df (pd.DataFrame): Historical price data, window_size (int), initial_balance (float): Initial account balance
        #Returns: Dict[str, Any]: Backtest results
        
        if self.model is None:
            logger.error("Model not trained or loaded")
            return {}
            
        if len(df) < window_size + 10:
            logger.error(f"Not enough data for backtest, need at least {window_size + 10} rows")
            return {}
            
        # Add technical indicators if not present
        df = self._add_technical_indicators(df)
        
        # Create environment for backtesting
        env = StockTradingEnv(df, window_size=window_size, initial_balance=initial_balance)
        
        # Track results
        observations = []
        actions = []
        rewards = []
        dones = []
        infos = []
        
        # Initialize
        observation = env.reset()
        done = False
        total_reward = 0
        
        # Run through the entire dataset
        while not done:
            # Get action from model
            action, _states = self.model.predict(observation, deterministic=True)
            
            # Take step in environment
            new_observation, reward, done, info = env.step(action)
            
            # Record data
            observations.append(observation.copy())
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info.copy())
            
            # Update for next iteration
            observation = new_observation
            total_reward += reward
            
        # Calculate performance metrics
        final_portfolio = infos[-1]['portfolio_value']
        profit = final_portfolio - initial_balance
        profit_pct = profit / initial_balance * 100
        
        # Calculate Sharpe ratio
        daily_returns = []
        portfolio_values = [initial_balance]
        
        for info in infos:
            portfolio_values.append(info['portfolio_value'])
            
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(daily_return)
            
        daily_returns = np.array(daily_returns)
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * np.sqrt(252)  # Annualized
        
        # Count trades
        buy_actions = actions.count(1)
        sell_actions = actions.count(2)
        
        # Calculate additional metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        return {
            'initial_balance': initial_balance,
            'final_portfolio': final_portfolio,
            'profit': profit,
            'profit_pct': profit_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_reward': total_reward,
            'num_steps': len(actions),
            'buy_actions': buy_actions,
            'sell_actions': sell_actions,
            'hold_actions': actions.count(0),
            'portfolio_values': portfolio_values,
            'actions': actions,
            'rewards': rewards
        }
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe if not present"""
        df = df.copy()
        
        # Check and add MA_5
        if 'MA_5' not in df.columns:
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            
        # Check and add MA_20
        if 'MA_20' not in df.columns:
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            
        # Check and add RSI
        if 'RSI' not in df.columns:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            df['RSI'] = 100 - (100 / (1 + rs))
            
        # Check and add Volatility
        if 'Volatility' not in df.columns:
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            
        # Check and add Daily_Return
        if 'Daily_Return' not in df.columns:
            df['Daily_Return'] = df['Close'].pct_change()
            
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        
        return df
    
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


# For direct script execution
if __name__ == "__main__":
    # Create synthetic data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    
    # Create a trending price series with some randomness
    price = 100.0
    prices = [price]
    
    for _ in range(499):
        rnd = np.random.randn() * 2.0  # Random component
        trend = 0.1  # Upward trend
        price = max(0.1, price * (1 + trend/100 + rnd/100))  # Ensure price is positive
        prices.append(price)
    
    # Create DataFrame
    data = {
        'Open': prices,
        'High': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
        'Low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
        'Close': [p * (1 + np.random.randn() * 0.005) for p in prices],
        'Volume': [int(1000000 * (1 + np.random.randn() * 0.3)) for _ in prices]
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Train an agent
    agent = RLTradingAgent()
    training_results = agent.train(df, model_type='ppo', total_timesteps=10000)
    
    # Backtest
    backtest_results = agent.backtest(df)
    
    # Display results
    print(f"Final portfolio value: ${backtest_results['final_portfolio']:.2f}")
    print(f"Profit: ${backtest_results['profit']:.2f} ({backtest_results['profit_pct']:.2f}%)")
    print(f"Sharpe ratio: {backtest_results['sharpe_ratio']:.4f}")
    print(f"Max drawdown: {backtest_results['max_drawdown'] * 100:.2f}%")