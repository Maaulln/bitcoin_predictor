import csv
import os
from datetime import datetime

def detect_csv_format(file_path):
    """
    Detect the format of the CSV file by examining its headers.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary with format information
    """
    try:
        with open(file_path, 'r') as f:
            # Read the first line to get headers
            headers = next(csv.reader(f))
            
            # Convert headers to lowercase for case-insensitive comparison
            headers_lower = [h.lower() for h in headers]
            
            format_info = {
                'date_column': None,
                'price_column': None,
                'has_headers': True
            }
            
            # Look for date column
            date_candidates = ['date', 'time', 'timestamp', 'datetime']
            for candidate in date_candidates:
                if candidate in headers_lower:
                    format_info['date_column'] = headers[headers_lower.index(candidate)]
                    break
            
            # Look for price column
            price_candidates = ['close', 'price', 'closing_price', 'last', 'value']
            for candidate in price_candidates:
                if candidate in headers_lower:
                    format_info['price_column'] = headers[headers_lower.index(candidate)]
                    break
            
            return format_info
    except Exception as e:
        print(f"Error detecting CSV format: {e}")
        return {'has_headers': False}

def parse_date(date_str):
    """
    Try to parse a date string in various formats.
    
    Args:
        date_str (str): Date string to parse
        
    Returns:
        datetime or None: Parsed datetime object or None if parsing failed
    """
    formats = [
        "%Y-%m-%d",           # 2023-01-31
        "%d/%m/%Y",           # 31/01/2023
        "%m/%d/%Y",           # 01/31/2023
        "%Y-%m-%d %H:%M:%S",  # 2023-01-31 12:30:45
        "%Y-%m-%dT%H:%M:%S",  # 2023-01-31T12:30:45
        "%Y-%m-%d %H:%M",     # 2023-01-31 12:30
        "%d-%m-%Y",           # 31-01-2023
        "%m-%d-%Y",           # 01-31-2023
        "%b %d, %Y",          # Jan 31, 2023
        "%B %d, %Y",          # January 31, 2023
        "%d %b %Y",           # 31 Jan 2023
        "%d %B %Y",           # 31 January 2023
        "%Y%m%d"              # 20230131
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None

def validate_data(file_path):
    """
    Validate that the data file exists and contains valid Bitcoin price data.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    # Check if file is empty
    if os.path.getsize(file_path) == 0:
        print(f"Error: File is empty: {file_path}")
        return False
    
    # Check if file is a CSV
    if not file_path.lower().endswith('.csv'):
        print(f"Warning: File does not have .csv extension: {file_path}")
    
    # Check if file has valid content
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            first_row = next(reader, None)
            
            if header is None or first_row is None:
                print(f"Error: File does not contain enough data: {file_path}")
                return False
            
            # Check if there's a price-like column
            has_numeric = False
            for value in first_row:
                try:
                    float(value)
                    has_numeric = True
                    break
                except ValueError:
                    continue
            
            if not has_numeric:
                print(f"Warning: No numeric values found in the first row of data")
    
    except Exception as e:
        print(f"Error validating data file: {e}")
        return False
    
    return True

def print_sample_data(file_path, num_rows=5):
    """
    Print a sample of data from the file.
    
    Args:
        file_path (str): Path to the CSV file
        num_rows (int): Number of rows to print
    """
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            
            print("\nSample data:")
            if headers:
                print(f"Headers: {', '.join(headers)}")
            
            print("\nFirst few rows:")
            for i, row in enumerate(reader):
                if i >= num_rows:
                    break
                print(f"Row {i+1}: {row}")
    
    except Exception as e:
        print(f"Error printing sample data: {e}")