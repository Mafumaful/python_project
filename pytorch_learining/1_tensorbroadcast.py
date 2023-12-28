import torch

a = torch.arange(4).reshape(2,2)
print("a=\n",a)
b = torch.arange(12).reshape(3,2,2)
print("b=\n",b)

print("a+b=\n",a+b)