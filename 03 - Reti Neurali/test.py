import torch
tensor = torch.randn(2, 2)
try:
    tensor_cuda = tensor.to('cuda')
    print("CUDA is available and working!")
except Exception as e:
    print("CUDA error:", e)
