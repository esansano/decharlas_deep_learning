import torch
torch.manual_seed(0)

# Neural network dimensions
n, in_dim, hid_dim, out_dim = 50, 100, 500, 10

# Training data
x = torch.randn(n, in_dim).type(torch.FloatTensor)
y = torch.randn(n, out_dim).type(torch.FloatTensor)

# Weight inizialization
wl1 = torch.randn(in_dim, hid_dim).type(torch.FloatTensor)
wl2 = torch.randn(hid_dim, out_dim).type(torch.FloatTensor)

learning_rate = 1e-6
for epoch in range(500):
    h = x.mm(wl1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(wl2)
    loss = (y_pred - y).pow(2).sum()

    print(loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_wl2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(wl2.t())
    grad_h = grad_h_relu.clone()
    grad_h[grad_h < 0] = 0
    grad_wl1 = x.t().mm(grad_h)

    wl1 -= learning_rate * grad_wl1
    wl2 -= learning_rate * grad_wl2