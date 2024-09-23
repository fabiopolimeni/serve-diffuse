import torch
x = torch.rand(5, 3)
print(x)

print(f"CUDA is available? {torch.cuda.is_available()}")