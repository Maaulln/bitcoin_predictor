import time

def print_line_graph(values, labels=None, width=60, height=15, title=None):
    """
    Print a simple ASCII line graph.
    
    Args:
        values (list): List of numeric values to plot
        labels (list, optional): List of labels for each value
        width (int): Width of the graph in characters
        height (int): Height of the graph in characters
        title (str, optional): Title for the graph
    """
    if not values:
        print("No data to visualize")
        return
    
    # Find min and max values for scaling
    min_val = min(values)
    max_val = max(values)
    
    # Ensure there's a range to plot
    if max_val == min_val:
        max_val = min_val + 1
    
    # Calculate y-axis scale
    value_range = max_val - min_val
    
    # Print title
    if title:
        print(f"\n{title}")
    
    # Print the y-axis scale
    print(f"${max_val:.2f} ┐")
    
    # Generate the graph rows
    for i in range(height):
        y_level = max_val - (i * value_range / (height - 1))
        row = f"${y_level:.2f} │"
        
        for j in range(len(values)):
            x_pos = int((j / (len(values) - 1 if len(values) > 1 else 1)) * (width - 1))
            normalized_val = (values[j] - min_val) / value_range
            y_pos = int((1 - normalized_val) * (height - 1))
            
            if y_pos == i:
                row += "*"
            else:
                row += " "
        
        print(row)
    
    # Print x-axis
    x_axis = "      └" + "─" * width
    print(x_axis)
    
    # Print x-axis labels if provided
    if labels:
        # Only show first and last labels to avoid clutter
        if len(labels) > 1:
            label_line = " " * 7 + labels[0] + " " * (width - len(labels[0]) - len(labels[-1])) + labels[-1]
            print(label_line)

def print_comparison(predicted, actual, labels=None):
    """
    Print a comparison table of predicted vs actual values.
    
    Args:
        predicted (list): List of predicted values
        actual (list): List of actual values
        labels (list, optional): List of labels for each comparison
    """
    if len(predicted) != len(actual):
        print("Error: Predicted and actual lists must have the same length")
        return
    
    if not labels:
        labels = [f"Point {i+1}" for i in range(len(predicted))]
    
    # Calculate errors
    abs_errors = [abs(predicted[i] - actual[i]) for i in range(len(predicted))]
    pct_errors = [abs(predicted[i] - actual[i]) / max(abs(actual[i]), 0.0001) * 100 for i in range(len(predicted))]
    
    # Print header
    print("\nPrediction Comparison:")
    print(f"{'Label':<10} {'Predicted':>12} {'Actual':>12} {'Error':>10} {'Error %':>10}")
    print("-" * 60)
    
    # Print rows
    for i in range(len(predicted)):
        print(f"{labels[i]:<10} ${predicted[i]:>11.2f} ${actual[i]:>11.2f} ${abs_errors[i]:>9.2f} {pct_errors[i]:>9.2f}%")
    
    # Print summary
    mae = sum(abs_errors) / len(abs_errors)
    mape = sum(pct_errors) / len(pct_errors)
    print("-" * 60)
    print(f"{'Average':<10} {'':<12} {'':<12} ${mae:>9.2f} {mape:>9.2f}%")

def animate_loading(seconds=3, message="Processing"):
    """
    Display an animated loading message.
    
    Args:
        seconds (int): Duration of the animation in seconds
        message (str): Message to display during loading
    """
    animations = ["|", "/", "-", "\\"]
    end_time = time.time() + seconds
    
    i = 0
    while time.time() < end_time:
        print(f"\r{message} {animations[i % len(animations)]}", end="", flush=True)
        time.sleep(0.1)
        i += 1
    
    print("\r" + " " * (len(message) + 2), end="\r")  # Clear the line

