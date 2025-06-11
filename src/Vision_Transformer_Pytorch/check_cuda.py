# Download CUDA:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install tqdm matplotlib netron 

import torch

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())