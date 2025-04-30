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