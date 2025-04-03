# Create a plot of the softplus function
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

x = torch.linspace(-5, 5, 100).view(-1, 1)  # Create a tensor of shape (100, 1)
y = torch.nn.functional.softplus(x, beta=50, threshold=20)  # Apply softplus function

# Plot
plt.plot(x.numpy(), y.numpy())
plt.title('Softplus Function')
plt.xlabel('x')
plt.ylabel('softplus(x)')
plt.grid()
plt.show()