# com
import torch
import numpy as np
print("PyTorch Version:", torch.__version__)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
# transfer to the GPU device
x = x.to(device)
y = y.to(device)
z1 = torch.sin(2*x+y)
# Compute sin
z2 = torch.exp(-(x**2+y**2)/2.0)
# Compute Gaussian
z = z1 * z2
#plot
import matplotlib.pyplot as plt
plt.imshow(z.cpu().numpy())#Updated!
plt.tight_layout()
plt.show()


import torch
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
Y, X = np.mgrid[-0.8:0.8:0.005, -0.8:0.8:0.005]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
c = torch.complex(torch.tensor(-0.835),torch.tensor(-0.2123))
z = torch.complex(x, y) #important!
zs = z.clone() #Updated!
ns = torch.zeros_like(z)

# transfer to the GPU device
z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)
c = c.to(device)
#Mandelbrot Set

for i in range(200):
    #Compute the new values of z: z^2 + x
    zs_ = zs*zs + c
    #Have we diverged with this new value?
    not_diverged = torch.abs(zs_) < 4.0
    #Update variables to compute
    ns += not_diverged
    zs = zs_
#plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,10))
def processFractal(a):
    """Display an array of iteration counts as a colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a
plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()


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
