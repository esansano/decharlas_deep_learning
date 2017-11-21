import torch
from torch.autograd import Variable
torch.manual_seed(0)

# Neural network dimensions
n, in_dim, hid_dim, out_dim = 50, 100, 500, 10

# Training data
x = Variable(torch.randn(n, in_dim), requires_grad=False)
y = Variable(torch.randn(n, out_dim), requires_grad=False)

# Weight inizialization
wl1 = Variable(torch.randn(in_dim, hid_dim), requires_grad=True)
wl2 = Variable(torch.randn(hid_dim, out_dim), requires_grad=True)

learning_rate = 1e-6
for epoch in range(500):

    # Computational graph
    h = x.mm(wl1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(wl2)
    loss = (y_pred - y).pow(2).sum()

    # Remove previous gradients
    if wl1.grad is not None: wl1.grad.data.zero_()
    if wl2.grad is not None: wl2.grad.data.zero_()

    # Gradient computation
    loss.backward()

    # Update weights
    wl1.data -= learning_rate * wl1.grad.data
    wl2.data -= learning_rate * wl2.grad.data