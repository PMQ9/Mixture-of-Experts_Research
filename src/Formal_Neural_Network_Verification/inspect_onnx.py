"""Inspect ONNX model structure"""
import onnx

model_path = 'artifacts/nnv_models/gtsrb_ultra_verifiable_cnn.onnx'
model = onnx.load(model_path)

print(f'\nONNX Model: {model_path}')
print('=' * 60)
print(f'Inputs: {len(model.graph.input)}')
print(f'Outputs: {len(model.graph.output)}')
print(f'Initializers (weights/biases): {len(model.graph.initializer)}')
print(f'Total operators: {len(model.graph.node)}')

print('\nModel Structure:')
print('-' * 60)
for i, node in enumerate(model.graph.node):
    print(f'  {i+1}. {node.op_type:20s} {node.name}')

print('\nOperator Summary:')
print('-' * 60)
from collections import Counter
op_counts = Counter(node.op_type for node in model.graph.node)
for op, count in sorted(op_counts.items()):
    print(f'  {op:20s}: {count}')

print('\nKey Features:')
print('-' * 60)
print(f"  ✓ BatchNorm layers: {op_counts.get('BatchNormalization', 0)}")
print(f"  ✓ Conv layers: {op_counts.get('Conv', 0)}")
print(f"  ✓ Pooling type: {'AveragePool' if 'AveragePool' in op_counts else 'MaxPool' if 'MaxPool' in op_counts else 'None'}")
print(f"  ✓ Flatten: {'Yes' if 'Flatten' in op_counts else 'No (uses Shape+Reshape)'}")
print(f"  ✓ Fully connected (Gemm): {op_counts.get('Gemm', 0)}")
