import torch

x = torch.arange(4.0)
print(x)

x.requires_grad_(True)
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)
# y.backward()
# print("backward two times:", x.grad)

print(x.grad == 4 * x)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)
