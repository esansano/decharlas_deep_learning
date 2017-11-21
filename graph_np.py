import numpy as np
np.random.seed(0)

# Tensor dimensions
d1, d2 = 4, 5

# Initialize three tensors
t1 = np.random.rand(d1, d2)
t2 = np.random.rand(d1, d2)
t3 = np.random.rand(d1, d2)

# Computational graph
a = t1 + t2
b = a * t3
c = np.sum(b) # 8.80812307798

grad_c = 1.0
grad_b = grad_c * np.ones((d1, d2))
grad_a = grad_b * t3
grad_t3 = grad_b * a
grad_t1 = grad_a.copy()
grad_t2 = grad_a.copy()
