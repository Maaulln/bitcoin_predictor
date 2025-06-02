"""
Bitcoin price prediction model with enhanced accuracy and visualization
"""
import csv
import math
import random
import statistics
import os
import json
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

class BitcoinPricePredictor:
    def __init__(self, window_size=14, train_ratio=0.8):
        """Initialize with larger window size for better accuracy"""
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.data = []
        self.train_data = []
        self.test_data = []
        self.model = None
        self.normalized_data = None
        self.scaler = None
        self.normalized_prices = None
        
    def load_data(self, file_path):
        """Load and validate Bitcoin price data"""
        print(f"Loading data from {file_path}...")
        try:
            self.data = []
            with open(file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        date = row.get('Date', row.get('date', None))
                        price = float(row.get('Close', row.get('close', 0)))
                        self.data.append({
                            'date': date,
                            'price': price
                        })
                    except (ValueError, KeyError) as e:
                        print(f"Skipping row due to error: {e}")
                        continue
            
            self.data.reverse()
            print(f"Loaded {len(self.data)} data points.")
            return len(self.data) > 0
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def normalize_data(self):
        """Normalize data using robust scaling"""
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        prices = np.array([entry['price'] for entry in self.data]).reshape(-1, 1)
        self.normalized_prices = self.scaler.fit_transform(prices)
        
        self.normalized_data = []
        for i, entry in enumerate(self.data):
            self.normalized_data.append({
                'date': entry['date'],
                'price': self.normalized_prices[i][0]
            })

    def calculate_technical_indicators(self, prices):
        """Calculate technical indicators for price data"""
        df = pd.DataFrame(prices, columns=['price'])
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['price'].ewm(span=12, adjust=False).mean()
        exp2 = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['sma'] = df['price'].rolling(window=20).mean()
        df['std'] = df['price'].rolling(window=20).std()
        df['upper_band'] = df['sma'] + (df['std'] * 2)
        df['lower_band'] = df['sma'] - (df['std'] * 2)
        
        return df.fillna(0).values

    def prepare_lstm_data(self):
        """Prepare data for LSTM model with technical indicators"""
        self.normalize_data()
        prices = [entry['price'] for entry in self.data]
        technical_data = self.calculate_technical_indicators(prices)
        
        X, y = [], []
        for i in range(len(technical_data) - self.window_size):
            X.append(technical_data[i:i+self.window_size])
            y.append(technical_data[i+self.window_size, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        split_idx = int(len(X) * self.train_ratio)
        self.train_X, self.train_y = X[:split_idx], y[:split_idx]
        self.test_X, self.test_y = X[split_idx:], y[split_idx:]
        
        print(f"Training data: {len(self.train_X)} samples")
        print(f"Testing data: {len(self.test_X)} samples")

    def create_lstm_model(self):
        """Create improved LSTM model with better regularization"""
        model = Sequential([
            # Input layer dengan normalisasi batch
            BatchNormalization(),
            
            # First LSTM layer
            LSTM(64, activation='tanh', return_sequences=True, 
                 input_shape=(self.window_size, 7),
                 kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(32, activation='tanh'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layers
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        # Gunakan optimizer dengan learning rate yang lebih kecil
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber')
        return model

    def train_lstm(self):
        """Train LSTM model with improved validation"""
        print("Training LSTM model...")
        
        if not hasattr(self, 'train_X') or len(self.train_X) == 0:
            raise ValueError("No training data available. Call prepare_lstm_data() first.")
        
        # Tambahkan callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Tambahkan model checkpoint
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        self.model = self.create_lstm_model()
        
        # Training dengan validasi
        history = self.model.fit(
            self.train_X,
            self.train_y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        # Evaluasi model
        val_loss = self.model.evaluate(self.test_X, self.test_y, verbose=0)
        print(f"\nValidation Loss: {val_loss:.4f}")
        
        return history

    def predict(self, input_data):
        """Make predictions with validation"""
        if not self.model:
            raise ValueError("Model not trained. Call train_lstm() first.")
        
        predictions = self.model.predict(input_data)
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        
        # Validate predictions
        actual = self.scaler.inverse_transform(self.test_y.reshape(-1, 1))
        predictions = self.validate_predictions(predictions, actual)
        
        return predictions

    def validate_predictions(self, predictions, actual):
        """Validate and clean predictions"""
        cleaned_predictions = []
        for i, pred in enumerate(predictions):
            # Batasi perubahan maksimum
            if i > 0:
                max_change = 0.1  # 10% perubahan maksimum
                prev_price = actual[i-1]
                max_price = prev_price * (1 + max_change)
                min_price = prev_price * (1 - max_change)
                pred = np.clip(pred, min_price, max_price)
            
            cleaned_predictions.append(pred)
        
        return np.array(cleaned_predictions)

    def evaluate_model(self):
        """Evaluate model performance"""
        predictions = self.predict(self.test_X)
        actual = self.scaler.inverse_transform(self.test_y.reshape(-1, 1))
        
        mse = np.mean((predictions - actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual))
        mape = mean_absolute_percentage_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        
        print("\nModel Performance Metrics:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2%}")
        print(f"R² Score: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }