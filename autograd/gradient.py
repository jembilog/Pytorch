import torch

x = torch.tensor(4.0, requires_grad=True)

# define a function
y = 3*x**2 + 2*x + 1

# backpropagation
y.backward()

print(x.grad)

x.grad.zero_()

y = 3*x**2 + 2*x + 1

y.backward()

print(x.grad)

