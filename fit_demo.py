import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, 10),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.linear(x)

fig, ax = plt.subplots()

real, = ax.plot([], [], lw=3)
estimate, = ax.plot([], [], lw=3)
ax.set_xlim(-10, 10)
ax.set_ylim(-3, 60)
net = BasicNN()
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()
input_range = torch.linspace(-10, 10, 10000).view(-1, 1)
mean, std = input_range.mean(), input_range.std()

def normalize(x):
    return (x - mean) / std

def update(frame):
    x = np.linspace(-10, 10, 10000).astype(np.float32)
    y = x**2
    real.set_data(x, y)

    optimizer.zero_grad()
    inputs = torch.linspace(-10, 10, 10000).view(-1, 1)
    labels = function(inputs)
    inputs = normalize(inputs)
    preds = net(inputs)
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()
    if frame == 1:
        print(f'Loss: {loss.item()}')
    if frame % 10 == 0:
        x_tensor = torch.from_numpy(x).view(-1, 1)
        x_tensor = normalize(x_tensor)
        y_estimate = net(x_tensor).detach().numpy().flatten()
        estimate.set_data(x, y_estimate)

    return real, estimate


def function(x):
    return torch.pow(x, 2)

if __name__ == "__main__":
    ani = FuncAnimation(fig, update, frames=range(100), blit=True, interval=1)
    plt.show()