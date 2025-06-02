import os
import sys
import csv
from datetime import datetime, timedelta
from bitcoin_predictor import BitcoinPricePredictor
from data_utils import validate_data, detect_csv_format, print_sample_data
from visualize import print_line_graph, print_comparison, animate_loading, display_forecast

def print_header():
    """Print the application header."""
    header = """
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║   🪙  BITCOIN PRICE PREDICTION MODEL  🪙             ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(header)

def print_menu():
    """Print the main menu options."""
    menu = """
    Select an option:
    
    1. Load data file
    2. Train prediction model
    3. Evaluate model performance
    4. Make future price predictions
    5. Save trained model
    6. Load saved model
    7. View sample data
    8. Exit
    
    Enter your choice (1-8): """
    
    return input(menu)

def get_data_file():
    """Get the data file path from the user."""
    print("\nEnter the path to your Bitcoin price CSV file:")
    print("(Example: bitcoin_prices.csv or /path/to/your/data.csv)")
    
    file_path = input("\nFile path: ").strip()
    
    if not file_path:
        print("No file path provided.")
        return None
    
    if not validate_data(file_path):
        print("Invalid data file. Please provide a valid CSV file with Bitcoin price data.")
        return None
    
    return file_path

def main():
    """Main application function."""
    print_header()
    print("Welcome to the Bitcoin Price Prediction Model!")
    print("\nThis application will help you analyze and predict Bitcoin prices using machine learning.")
    print("To get started, you'll need a CSV file with historical Bitcoin price data from Kaggle or another source.")
    
    # Initialize the predictor with default settings
    predictor = BitcoinPricePredictor(window_size=7, train_ratio=0.8)
    data_loaded = False
    model_trained = False
    
    while True:
        choice = print_menu()
        
        if choice == '1':  # Load data file
            file_path = get_data_file()
            if file_path:
                success = predictor.load_data(file_path)
                if success:
                    data_loaded = True
                    print("\n✅ Data loaded successfully!")
                    print_sample_data(file_path, num_rows=3)
        
        elif choice == '2':  # Train model
            if not data_loaded:
                print("\n⚠️ Please load data first (option 1).")
                continue
            
            print("\nPreparing data for training...")
            predictor.prepare_data()
            
            print("\nSelect the model type:")
            print("1. Linear Regression")
            print("2. Moving Average")
            model_type = input("Enter your choice (1-2): ")
            
            if model_type == '1':
                animate_loading(seconds=3, message="Training linear regression model")
                predictor.train_linear_regression()
                model_trained = True
            elif model_type == '2':
                animate_loading(seconds=1, message="Setting up moving average model")
                predictor.train_moving_average()
                model_trained = True
            else:
                print("Invalid choice. Please select 1 or 2.")
        
        elif choice == '3':  # Evaluate model
            if not model_trained:
                print("\n⚠️ Please train a model first (option 2).")
                continue
            
            print("\nEvaluating model performance...")
            animate_loading(seconds=2, message="Calculating metrics")
            
            eval_results = predictor.evaluate()
            
            # Visualize the comparison between predicted and actual values
            if eval_results['predictions'] and eval_results['actuals']:
                print("\nLast 10 predictions vs actual values:")
                print_comparison(
                    eval_results['predictions'], 
                    eval_results['actuals'],
                    labels=[f"Day {i+1}" for i in range(len(eval_results['predictions']))]
                )
                
                # Plot the values
                print("\nVisualization of predictions vs actuals:")
                combined = []
                for i in range(len(eval_results['predictions'])):
                    combined.append(eval_results['actuals'][i])
                    combined.append(eval_results['predictions'][i])
                
                labels = ["Actual", "Predicted"] * len(eval_results['predictions'])
                print_line_graph(combined, labels, width=40, height=10, title="Prediction Accuracy")
        
        elif choice == '4':  # Make predictions
            if not model_trained:
                print("\n⚠️ Please train a model first (option 2).")
                continue
            
            try:
                days = int(input("\nHow many days ahead would you like to predict? (1-30): "))
                if days < 1 or days > 30:
                    print("Please enter a number between 1 and 30.")
                    continue
                
                animate_loading(seconds=2, message=f"Generating {days}-day forecast")
                predictions = predictor.predict_next_days(days)
                
                # Generate dates for the predictions
                today = datetime.now()
                dates = [(today + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(days)]
                
                # Display the forecast
                display_forecast(dates, predictions, title=f"Bitcoin Price Forecast - Next {days} Days")
                
                # Visualize the prediction trend
                print("\nPrediction trend:")
                print_line_graph(
                    predictions, 
                    [dates[0], dates[-1]], 
                    width=50, 
                    height=10, 
                    title="Bitcoin Price Forecast Trend"
                )
            
            except ValueError:
                print("Please enter a valid number.")
        
        elif choice == '5':  # Save model
            if not model_trained:
                print("\n⚠️ Please train a model first (option 2).")
                continue
            
            file_name = input("\nEnter file name to save the model (default: bitcoin_model.json): ").strip()
            if not file_name:
                file_name = "bitcoin_model.json"
            
            predictor.save_model(file_name)
            print(f"\n✅ Model saved to {file_name}")
        
        elif choice == '6':  # Load model
            file_name = input("\nEnter the file name of the saved model (default: bitcoin_model.json): ").strip()
            if not file_name:
                file_name = "bitcoin_model.json"
            
            if predictor.load_model(file_name):
                print(f"\n✅ Model loaded from {file_name}")
                model_trained = True
            else:
                print(f"\n❌ Failed to load model from {file_name}")
        
        elif choice == '7':  # View sample data
            if not data_loaded:
                print("\n⚠️ Please load data first (option 1).")
                continue
            
            print("\nSample of loaded Bitcoin price data:")
            # Print the first and last 5 data points
            print("\nFirst 5 data points:")
            for i in range(min(5, len(predictor.data))):
                print(f"{predictor.data[i]['date']}: ${predictor.data[i]['price']:.2f}")
            
            print("\nLast 5 data points:")
            for i in range(max(0, len(predictor.data)-5), len(predictor.data)):
                print(f"{predictor.data[i]['date']}: ${predictor.data[i]['price']:.2f}")
            
            # Print basic statistics
            prices = [entry['price'] for entry in predictor.data]
            print(f"\nStatistics:")
            print(f"Number of records: {len(prices)}")
            print(f"Min price: ${min(prices):.2f}")
            print(f"Max price: ${max(prices):.2f}")
            print(f"Average price: ${sum(prices)/len(prices):.2f}")
        
        elif choice == '8':  # Exit
            print("\nThank you for using the Bitcoin Price Prediction Model. Goodbye!")
            sys.exit(0)
        
        else:
            print("\nInvalid choice. Please enter a number between 1 and 8.")
        
        print("\nPress Enter to continue...", end="")
        input()

if __name__ == "__main__":
    main()