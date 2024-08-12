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
