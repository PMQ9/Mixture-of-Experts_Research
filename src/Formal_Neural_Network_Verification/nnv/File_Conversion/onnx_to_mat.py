import onnx
import numpy as np
import scipy.io
import os
import sys

PRETRAINED_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'artifacts', 'nnv'))
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Vision_Transformer_Pytorch')
sys.path.append(SRC_DIR)

onnx_model = onnx.load(os.path.join(PRETRAINED_MODEL_DIR, "metamoe_convnext_tiny_1.onnx"))

weights_dict = {}
for initializer in onnx_model.graph.initializer:
    weights_dict[initializer.name] = np.array(onnx.numpy_helper.to_array(initializer))

scipy.io.savemat('converted_model_onnx_2_mat.mat', weights_dict)
print("Model successfully converted to converted_model.mat")