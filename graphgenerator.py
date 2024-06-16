import matplotlib.pyplot as plt
import numpy as np
import os

def generate_random_function():
    # List of mathematical functions
    functions = [np.sin, np.cos, np.tan, np.exp, np.sqrt, np.log]
    # Choose a random function
    selected_function = np.random.choice(functions)
    return selected_function

def generate_random_domain(selected_func):
    # Define domain generation rules based on the function
    if selected_func == np.sqrt:
        start_point = np.random.uniform(0, 1000)
    elif selected_func == np.log:
        start_point = np.random.uniform(1, 1000)
    else:
        start_point = np.random.uniform(-2, 2)
    
    end_point = np.random.uniform(start_point, 2 if selected_func in [np.sin, np.cos, np.tan, np.exp] else 1000)
    
    # Generate the domain
    domain = np.linspace(start_point, end_point, 1000)
    return domain

def generate_and_plot_random_graph(n):
    try:
        # Generate random function and domain
        func = generate_random_function()
        domain = generate_random_domain(func)
        
        # Calculate y values
        y = func(domain)
        
        # Handle potential numerical issues
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print(f"Skipping plot {n} due to numerical issues.")
            return
        
        # Plotting the function
        plt.figure(figsize=(3, 2), dpi=100)
        plt.title(f"Random Function: {func.__name__}")
        plt.plot(domain, y, label=func.__name__, color='blue')
        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')
        plt.gca().spines['bottom'].set_color('none')
        plt.gca().spines['left'].set_color('none')
        
        # Save the figure
        random_filename = f'graph{n}.png'
        plt.savefig(os.path.join('output', random_filename), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating plot {n}: {e}")

# Ensure the 'output' directory exists for saving figures
os.makedirs('output', exist_ok=True)

# Generate and plot 500 random graphs
for i in range(501):
    generate_and_plot_random_graph(i)
