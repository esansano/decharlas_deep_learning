import torch
from torch.autograd import Variable
torch.manual_seed(0)

# Neural network dimensions
n, in_dim, hid_dim, out_dim = 50, 100, 500, 10

# Training data
x = Variable(torch.randn(n, in_dim), requires_grad=False)
y = Variable(torch.randn(n, out_dim), requires_grad=False)

# Model definition
model = torch.nn.Sequential(
    torch.nn.Linear(in_dim, hid_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hid_dim, out_dim)
)

# Loss function
criterion = torch.nn.MSELoss()

# Optimization function
learning_rate = 1e-6
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

for epoch in range(500):

    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()
