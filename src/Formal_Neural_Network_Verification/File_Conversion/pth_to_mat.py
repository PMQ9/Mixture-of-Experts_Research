import torch
import scipy.io
import os
import sys

PRETRAINED_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'artifacts', 'nnv'))
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Vision_Transformer_Pytorch')
sys.path.append(SRC_DIR)

model = torch.load(
    os.path.join(PRETRAINED_MODEL_DIR, "meta_moe_convnext_tiny_best_og_1.pth"),
    map_location='cpu', 
    weights_only=False)
model.eval()

weights_dict = {}
for name, param in model.named_parameters():
    weights_dict[name] = param.detach().cpu().numpy()

scipy.io.savemat('converted_model_pth_2_mat.mat', weights_dict)
print("Model successfully converted to converted_model_pth_2_mat.mat")