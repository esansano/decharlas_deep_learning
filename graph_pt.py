import torch
from torch.autograd import Variable
torch.manual_seed(0)

# Tensor dimensions
d1, d2 = 4, 5

# Computational graph definition with Variables
t1 = Variable(torch.randn(d1, d2), requires_grad=True)
t2 = Variable(torch.randn(d1, d2), requires_grad=True)
t3 = Variable(torch.randn(d1, d2), requires_grad=True)
a = t1 + t2
b = a * t3
c = torch.sum(b)

# Gradient computation
c.backward()

grad_t1 = t1.grad.data
grad_t2 = t2.grad.data
grad_t3 = t3.grad.data

