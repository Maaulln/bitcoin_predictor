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

