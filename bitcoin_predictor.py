import csv
import math
import random
import statistics
import os
import json
from datetime import datetime, timedelta
import time

class BitcoinPricePredictor:
    def __init__(self, window_size=7, train_ratio=0.8):
        """Initialize the Bitcoin price predictor.
        
        Args:
            window_size (int): Number of previous days to use for prediction
            train_ratio (float): Ratio of data to use for training
        """
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
        """Load Bitcoin price data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file with Bitcoin price data
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        print(f"Loading data from {file_path}...")
        try:
            self.data = []
            with open(file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Assuming CSV has 'Date' and 'Close' columns
                    # Adapt to your specific Kaggle dataset format
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
            
            # Sort data by date (newest first typically)
            self.data.reverse()
            
            print(f"Loaded {len(self.data)} data points.")
            return len(self.data) > 0
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def normalize_data(self):
        """Normalize the price data to the range [0, 1]."""
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

    def denormalize_price(self, normalized_price):
        """Convert a normalized price back to the original scale."""
        return normalized_price * (self.max_price - self.min_price) + self.min_price

    def prepare_data(self):
        """Prepare data for training and testing."""
        self.normalize_data()
        
        # Create input-output pairs (X, y)
        X, y = [], []
        for i in range(len(self.normalized_data) - self.window_size):
            X.append([self.normalized_data[i+j]['price'] for j in range(self.window_size)])
            y.append(self.normalized_data[i+self.window_size]['price'])
        
        # Split into training and testing sets
        split_idx = int(len(X) * self.train_ratio)
        self.train_X, self.train_y = X[:split_idx], y[:split_idx]
        self.test_X, self.test_y = X[split_idx:], y[split_idx:]
        
        print(f"Training data: {len(self.train_X)} samples")
        print(f"Testing data: {len(self.test_X)} samples")
    def train_linear_regression(self):
        """Train a simple linear regression model."""
        print("Training linear regression model...")
        
        if not hasattr(self, 'train_X') or len(self.train_X) == 0:
            raise ValueError("No training data available. Call prepare_data() first.")

        # Calculate weights using basic linear regression
        # For each feature (day in the window), calculate its weight
        self.weights = [0] * self.window_size
        self.bias = 0
        
        # Simple gradient descent
        learning_rate = 0.01
        epochs = 1000
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(self.train_X)):
                # Forward pass
                prediction = self.bias
                for j in range(self.window_size):
                    prediction += self.weights[j] * self.train_X[i][j]
                
                # Compute loss
                error = prediction - self.train_y[i]
                total_loss += error ** 2
                
                # Backpropagation
                self.bias -= learning_rate * error
                for j in range(self.window_size):
                    self.weights[j] -= learning_rate * error * self.train_X[i][j]
            
            # Print progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / len(self.train_X)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.model = {
            'weights': self.weights,
            'bias': self.bias,
            'window_size': self.window_size
        }
        print("Model training completed.")
        return self.model

    