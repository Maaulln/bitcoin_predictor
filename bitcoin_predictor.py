import random
import statistics
import os
import json
from datetime import datetime, timedelta
import time

class BitcoinPredictor:
  def __init__(self, window_size=7, train_ratio= 0.8):
    """Initializes the BitcoinPredictor price predictor.

    Args:
        windowsize (int):
        train_rasio (float):    """
        self.window_size = window_size
        self.train_rasio = train_ratio
        self.data = []
        self.train_data = []
        self.test_data = []
        self.model = None
        self.normalized_data = []
        self.min_price = none
        self.max_price = none