import torch
import matplotlib.pyplot as plt
import numpy as np


# Function to draw the fractal tree using recursion
def draw_tree(ax, x, y, angle, length, depth):
    if depth == 0:
        return

    # Calculate the end point of the branch
    x_end = x + length * np.cos(angle)
    y_end = y + length * np.sin(angle)

    # Draw the branch
    ax.plot([x, x_end], [y, y_end], color="green")

    # Recursively draw the two sub-branches
    draw_tree(ax, x_end, y_end, angle - np.pi / 6, length * 0.7, depth - 1)
    draw_tree(ax, x_end, y_end, angle + np.pi / 6, length * 0.7, depth - 1)


def plot_fractal_tree():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')

    # Initialize parameters
    x0, y0 = 0, 0  # Starting point
    angle = np.pi / 2  # Vertical upwards
    length = 60  # Initial length of the trunk
    depth = 12  # Depth of recursion

    # Set plot limits
    ax.set_xlim(-150, 150)
    ax.set_ylim(0, 250)
    ax.axis('off')  # Turn off the axis

    # Draw the tree
    draw_tree(ax, x0, y0, angle, length, depth)

    plt.show()


plot_fractal_tree()
