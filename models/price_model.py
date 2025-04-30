import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import logging
from typing import Tuple, Dict, Any, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PriceModel:
    #Builds trains and evaluates stock prediction models

    def __init__(self, model_dir: str = 'models/saved_models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()

        os.makedirs(model_dir, exist_ok=True)

    def build_lstm_models(self, input_shape: Tuple[int, int], units: int = 64) -> keras.Model:
        #Build an LSTM-based deep learning model
        #Args: input_shape (tuple), units (int)
        #Return: keras.Model
        model = keras.Sequential([
            layers.LSTM(units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0,2),
            layers.LSTM(units // 2),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        logger.info(f"Built LSTM model with shape {input_shape}")
        return model