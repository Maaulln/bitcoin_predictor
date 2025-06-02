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
