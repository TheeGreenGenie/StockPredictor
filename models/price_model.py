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
from joblib import dump
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PriceModel:
    #Builds trains and evaluates stock prediction models

    def __init__(self, model_dir: str = 'saved_models'):
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
        if len(features) == 0 or len(target) == 0:
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
            self.model = self.build_dense_model(features.shape[1])

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
        scaler_path = os.path.join(self.model_dir, f"price_model_scaler_{model_type}.pk1")
        joblib.dump(self.scaler, scaler_path)

        logger.info(f"Model saved to {model_path}")

        return {
            'history': history.history,
            'evaluation': {
                'loss': evaluation[0],
                'mae': evaluation[1]
            }
        }
    
    def load_model(self, model_path: str) -> bool:
        #Load a saved model
        #Args: model_path (str)
        #Returns: bool
        try:
            self.model = keras.models.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
        
    def predict(self, features: np.ndarray, model_type: str = 'dense',
                sequence_length: int = 4) -> np.ndarray:
        #Generate predictions
        #Args: features (np.ndarray), model_type (str), sequence_length (int)
        #Returns: np.ndarray
        if self.model is None:
            logger.error("Model not trained or loaded")
            return np.array([])
        
        scaled_features = self.scaler.transform(features)

        if model_type == 'lstm':
            if len(scaled_features) < sequence_length:
                logger.error(f"Not enough data for sequence length {sequence_length}")
                return np.ndarray([])
            
            if len(scaled_features) == 1:
                scaled_features = scaled_features.reshape(1, -1)

            if len(scaled_features) == sequence_length:
                X = scaled_features.reshape(1, sequence_length, -1)
            else:
                X = np.array([scaled_features[-sequence_length:]])

            predictions = self.model.predict(X)
        else:
            if len(scaled_features.shape) == 1:
                scaled_features = scaled_features.reshape(1, -1)

            predictions = self.model.predict(scaled_features)

        return predictions.flatten()
    
    def plot_history(self, history: Dict[str, Any], save_path: str = None) -> None:
        #Plot training history
        #Args: history (dict), save_path (str)
        figs, axes = plt.subplots(1,2, figsize=(15,5))

        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].plot(history['mae'], label='Training MAE')
        axes[1].plot(history['val_mae'], label='Validation MAE')
        axes[1].set_title('Mean absolute Error')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legenc()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved training history plot to {save_path}")

        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 200

    features = np.random.randn(n_samples, 5)

    target = 2 * features[:, 0] + 0.5 * features[:, 1] - features[:, 2] + 0.1 * np.random.randn(n_samples)

    model = PriceModel()
    results = model.train(features, target, epochs=20)
    
    new_data = np.random.randn(1, 5)
    prediction = model.predict(new_data)
    print(f"Prediction for new data: {prediction[0]:.2f}")