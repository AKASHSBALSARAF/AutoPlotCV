import matplotlib.pyplot as plt
import numpy as np
import os
import random
import string

def generate_random_function():
    # Define a list of mathematical functions
    functions = [np.sin, np.cos, np.tan, np.exp, np.sqrt, np.log]

    # Choose a random function from the list
    selected_function = np.random.choice(functions)

    return selected_function


def generate_random_domain(selected_func):

    # Randomly generate the starting and ending points of the domain
    if(selected_func==np.sqrt):
        start_point = np.random.uniform(0, 1000)
        end_point = np.random.uniform(start_point, 1000)
    elif(selected_func==np.log):
        start_point = np.random.uniform(1, 1000)
        end_point = np.random.uniform(start_point, 1000)
    elif(selected_func==np.sin or selected_func==np.cos or selected_func==np.tan):
        start_point = np.random.uniform(-2, 2)
        end_point = np.random.uniform(start_point, 2)
    else:
        start_point = np.random.uniform(-1000, 1000)
        end_point = np.random.uniform(start_point, 1000)

    # Generate the domain
    domain = np.linspace(start_point, end_point, 1000)

    return domain

def generate_and_plot_random_graph(n):
    # Generate random function, grid, and domain
    func = generate_random_function()
    domain = generate_random_domain(func)

    # Calculate y values using the randomly chosen function
    y = func(domain)

    # Create a new figure with a random grid
    plt.figure(figsize=(8, 8))
    plt.grid(random.choice([True, False]))
    plt.title(f"Random Function: {func.__name__}")

    # Plot the function
    plt.plot(domain, y, label=func.__name__)
    plt.legend()

    # Save the figure with a random name
    random_filename = 'graph'+str(n)+'.png'
    plt.savefig(os.path.join('output', random_filename))

# Ensure the 'output' directory exists for saving figures
os.makedirs('output', exist_ok=True)

# Generate and plot a random graph
for i in range(0,100):
    generate_and_plot_random_graph(i)