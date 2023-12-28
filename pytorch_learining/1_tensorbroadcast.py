import torch

a = torch.arange(4).reshape(2,2)
print("a=",a)
b = torch.arange(12).reshape(3,2,2)
print("b=",b)

print("a+b=",a+b)