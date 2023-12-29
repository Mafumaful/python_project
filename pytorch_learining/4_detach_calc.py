import torch

x = torch.arange(4.0, requires_grad=True)
y = x*x
u = y.detach()

z = u*x

print(z)
z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2*x)