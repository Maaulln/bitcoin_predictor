import os
import sys
from bitcoin_predictor import BitcoinPricePredictor
from visualize import print_line_graph, print_comparison
import matplotlib.pyplot as plt

def train_and_evaluate(predictor, model_type="linear"):
    """Train and evaluate the specified model type"""
    if model_type == "linear":
        print("\nTraining Linear Regression model...")
        predictor.prepare_data()
        predictor.train_linear_regression()
    else:
        print("\nPreparing data for LSTM model...")
        predictor.prepare_lstm_data()
        history = predictor.train_lstm()
        
        # Visualisasi hasil training LSTM
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_type} Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{model_type}_training_history.png')
        plt.close()
    
    # Evaluasi model
    print("\nEvaluating model performance...")
    metrics = predictor.evaluate_model()
    
    if model_type == "lstm":
        predictions = predictor.predict(predictor.test_X)
        actual = predictor.scaler.inverse_transform(predictor.test_y.reshape(-1, 1))
    else:
        predictions = predictor.predict(predictor.test_X)
        actual = predictor.test_y
    
    # Visualisasi prediksi vs aktual
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title(f'Bitcoin Price Prediction ({model_type})')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig(f'{model_type}_prediction_results.png')
    plt.close()
    
    # Print perbandingan prediksi
    print(f"\nPrediction Comparison (Last 10 points) - {model_type}:")
    last_predictions = predictions[-10:].flatten()
    last_actual = actual[-10:].flatten()
    dates = [f"Point {i+1}" for i in range(10)]
    
    print_comparison(last_predictions, last_actual, dates)
    
    # Print grafik ASCII untuk trend
    print(f"\nPrice Trend (ASCII Graph) - {model_type}:")
    print_line_graph(
        values=last_actual,
        labels=dates,
        title=f"Bitcoin Price Trend ({model_type})"
    )
    
    print(f"\nModel Performance Summary ({model_type}):")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']:.2%}")
    print(f"R² Score: {metrics['r2']:.4f}")
    
    return metrics

def main():
    # Inisialisasi predictor
    predictor = BitcoinPricePredictor(window_size=14)
    
    # Cek file data
    data_file = 'BTC-2017min.csv'
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        sys.exit(1)
    
    # Load data
    if not predictor.load_data(data_file):
        print("Failed to load data!")
        sys.exit(1)
    
    # Train dan evaluasi kedua model
    linear_metrics = train_and_evaluate(predictor, "linear")
    lstm_metrics = train_and_evaluate(predictor, "lstm")
    
    # Bandingkan performa kedua model
    print("\nModel Comparison:")
    print("Metric\t\tLinear\t\tLSTM")
    print("-" * 40)
    print(f"MSE\t\t{linear_metrics['mse']:.2f}\t\t{lstm_metrics['mse']:.2f}")
    print(f"RMSE\t\t{linear_metrics['rmse']:.2f}\t\t{lstm_metrics['rmse']:.2f}")
    print(f"MAE\t\t{linear_metrics['mae']:.2f}\t\t{lstm_metrics['mae']:.2f}")
    print(f"MAPE\t\t{linear_metrics['mape']:.2%}\t\t{lstm_metrics['mape']:.2%}")
    print(f"R² Score\t{linear_metrics['r2']:.4f}\t\t{lstm_metrics['r2']:.4f}")
    
    print("\nVisualisasi hasil telah disimpan dalam:")
    print("1. linear_prediction_results.png - Grafik prediksi Linear Regression")
    print("2. lstm_prediction_results.png - Grafik prediksi LSTM")
    print("3. lstm_training_history.png - Grafik proses training LSTM")

if __name__ == "__main__":
    main()