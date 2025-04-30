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
    
    def build_dense_model(self, input_shape: int) -> keras.Model:
        #Simple feed-forward neural network
        #Args: input_shape (int)
        #Returns: keras.Model
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        logger.info(f"Built Dense model with {input_shape} input features")
        return model
    
    def prepare_sequence_data(self, features: np.ndarray, target: np.ndarray,
                              sequence_length: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        #Sequential Data for model
        #Args: features (np.ndarray), target (np.ndarray), sequencce_length (int)
        #Returns: Tuple[np.ndarray, np.ndarray]
        X, y = [], []

        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])

        return np.array(X), np.array(y)
    
    def train(self, features: np.ndarray, target: np.ndarray, model_type: str = 'dense',
              sequence_length: int = 4, epochs: int= 50, batch_size: int = 32,
              validation_split: float = 0.2) -> Dict[str, Any]:
        #Train price prediction model
        #Args: features (np.ndarray), tagret (np.ndarray), model_type (str), sequence_legnth (int), epochs (int), batch_size (int), validation_split (float)
        #Returns: Dict[str, Any]
        if len(features == 0 or len(target) == 0):
            logger.error("Cannot train with empty features or target")
            return {}
        
        scaled_features = self.scaler.fit_transform(features)

        X_train, X_val, y_train, y_val = train_test_split(
            scaled_features, target, test_size=validation_split, shuffle=False
        )

        if model_type == 'lstm':
            X_train_seq, y_train_seq = self.prepare_sequence_data(X_train, y_train, sequence_length)
            X_val_seq, y_val_seq = self.prepare_sequence_data(X_val, y_val, sequence_length)

            self.model = self.build_lstm_models(
                input_shape=(sequence_length, features.shape[1])
            )

            logger.info("Training LSTM model...")
            history = self.model.fit(
                X_train_seq, y_train_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_seq, y_val_seq),
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
                ],
                verbose=1
            )

            evaluation = self.model.evaluate(X_val_seq, y_val_seq)

        else:
            #Building dense model
            self.model = self.build_dense_model(features.shape([1]))

            logger.info("Training Dense model...")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
                ],
                verbose=1
            )

            evaluation = self.model.evaluate(X_val, y_val)

        model_path = os.path.join(self.model_dir, f"price_model_{model_type}.h5")
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        return {
            'history': history.history,
            'evaluation': {
                'loss': evaluation[0],
                'mae': evaluation[1]
            }
        }