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
from sklearn.metrics import r2_score, mean_absolute_percentage_error

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
        self.min_price = None
        self.max_price = None
        
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
        """Normalize with improved scaling"""
        prices = [entry['price'] for entry in self.data]
        self.min_price = min(prices)
        self.max_price = max(prices)
        price_range = self.max_price - self.min_price
        
        self.normalized_data = []
        for entry in self.data:
            normalized_price = (entry['price'] - self.min_price) / price_range
            self.normalized_data.append({
                'date': entry['date'],
                'price': normalized_price
            })

    def prepare_data(self):
        """Prepare data with enhanced feature engineering"""
        self.normalize_data()
        
        X, y = [], []
        for i in range(len(self.normalized_data) - self.window_size):
            window = [self.normalized_data[i+j]['price'] for j in range(self.window_size)]
            
            # Add technical indicators
            sma = sum(window) / len(window)
            momentum = window[-1] - window[0]
            volatility = statistics.stdev(window)
            
            features = window + [sma, momentum, volatility]
            X.append(features)
            y.append(self.normalized_data[i+self.window_size]['price'])
        
        split_idx = int(len(X) * self.train_ratio)
        self.train_X, self.train_y = X[:split_idx], y[:split_idx]
        self.test_X, self.test_y = X[split_idx:], y[split_idx:]
        
        print(f"Training data: {len(self.train_X)} samples")
        print(f"Testing data: {len(self.test_X)} samples")

    def train_linear_regression(self):
        """Train with enhanced gradient descent"""
        print("Training enhanced linear regression model...")
        
        if not hasattr(self, 'train_X') or len(self.train_X) == 0:
            raise ValueError("No training data available. Call prepare_data() first.")

        feature_count = len(self.train_X[0])
        self.weights = [0] * feature_count
        self.bias = 0
        
        learning_rate = 0.01
        epochs = 2000  # Increased epochs
        best_weights = None
        best_bias = None
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(self.train_X)):
                prediction = self.bias + sum(w * x for w, x in zip(self.weights, self.train_X[i]))
                error = prediction - self.train_y[i]
                total_loss += error ** 2
                
                # Update weights with momentum
                self.bias -= learning_rate * error
                for j in range(feature_count):
                    self.weights[j] -= learning_rate * error * self.train_X[i][j]
            
            avg_loss = total_loss / len(self.train_X)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_weights = self.weights.copy()
                best_bias = self.bias
            
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.weights = best_weights
        self.bias = best_bias
        self.model = {
            'weights': self.weights,
            'bias': self.bias,
            'window_size': self.window_size
        }
        
        # Calculate and display accuracy metrics
        train_predictions = [self.predict(x) for x in self.train_X]
        train_actuals = [self.denormalize_price(y) for y in self.train_y]
        train_r2 = r2_score(train_actuals, train_predictions)
        train_mape = mean_absolute_percentage_error(train_actuals, train_predictions) * 100
        
        print(f"\nTraining Accuracy Metrics:")
        print(f"R² Score: {train_r2:.4f}")
        print(f"Accuracy: {100 - train_mape:.2f}%")
        
        return self.model

    def predict(self, input_data):
        """Make prediction with enhanced model"""
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
            
        if isinstance(input_data[0], (int, float)):
            normalized_input = [(x - self.min_price) / (self.max_price - self.min_price) for x in input_data]
            
            # Calculate technical indicators
            sma = sum(normalized_input) / len(normalized_input)
            momentum = normalized_input[-1] - normalized_input[0]
            volatility = statistics.stdev(normalized_input)
            
            features = normalized_input + [sma, momentum, volatility]
        else:
            features = input_data
        
        prediction = self.model['bias'] + sum(w * x for w, x in zip(self.model['weights'], features))
        return self.denormalize_price(prediction)

    def plot_predictions(self, predictions, actuals, title="Bitcoin Price Predictions"):
        """Plot predictions vs actual prices"""
        plt.figure(figsize=(12, 6))
        plt.plot(actuals, label='Actual Prices', color='blue')
        plt.plot(predictions, label='Predicted Prices', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig('prediction_plot.png')
        plt.close()

    def evaluate(self):
        """Evaluate with enhanced metrics and visualization"""
        if not hasattr(self, 'test_X') or not self.model:
            raise ValueError("Model not trained or no test data available.")
            
        predictions = []
        for x in self.test_X:
            pred = self.predict(x)
            predictions.append(pred)
        
        actuals = [self.denormalize_price(y) for y in self.test_y]
        
        # Calculate enhanced metrics
        r2 = r2_score(actuals, predictions)
        mape = mean_absolute_percentage_error(actuals, predictions) * 100
        accuracy = 100 - mape
        
        print(f"\nModel Evaluation:")
        print(f"R² Score: {r2:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Plot results
        self.plot_predictions(predictions, actuals, "Bitcoin Price Prediction Results")
        print("\nPrediction plot saved as 'prediction_plot.png'")
        
        return {
            'r2_score': r2,
            'accuracy': accuracy,
            'predictions': predictions[-10:],
            'actuals': actuals[-10:]
        }

    def denormalize_price(self, normalized_price):
        """Convert normalized price back to USD"""
        return normalized_price * (self.max_price - self.min_price) + self.min_price

    def save_model(self, file_path="bitcoin_model.json"):
        """Save the trained model"""
        if not self.model:
            raise ValueError("No trained model to save.")
            
        model_data = {
            'model': self.model,
            'min_price': self.min_price,
            'max_price': self.max_price,
            'window_size': self.window_size,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {file_path}")

    def load_model(self, file_path="bitcoin_model.json"):
        """Load a trained model"""
        try:
            with open(file_path, 'r') as f:
                model_data = json.load(f)
                
            self.model = model_data['model']
            self.min_price = model_data['min_price']
            self.max_price = model_data['max_price']
            self.window_size = model_data['window_size']
            
            print(f"Model loaded from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False