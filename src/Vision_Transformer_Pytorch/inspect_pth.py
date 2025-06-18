import torch
import os
from train_moe import (PRETRAINED_MODEL_DIR)

DEVICE = torch.device("cpu")  # Use CPU for inspection to avoid GPU issues

def inspect_pth_file(filename):
    filepath = os.path.join(PRETRAINED_MODEL_DIR, filename)
    print(f"\nInspecting file: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        return
    
    try:
        # Load with weights_only=False to allow full models
        obj = torch.load(filepath, map_location=DEVICE, weights_only=False)
        print(f"Type of loaded object (weights_only=False): {type(obj)}")
        
        # If it's a model, print its class and check if it has a 'to' method
        if hasattr(obj, 'to'):
            print(f"Object is a model with 'to' method: {obj.__class__.__name__}")
        elif isinstance(obj, dict):
            print("Object is a state dictionary (OrderedDict or dict).")
            # Print first few keys to confirm it's a state dict
            print(f"First few keys: {list(obj.keys())[:5]}")
        else:
            print(f"Unexpected object type: {type(obj)}")
        
        # Try loading with weights_only=True to confirm state dictionary
        try:
            state_dict = torch.load(filepath, map_location=DEVICE, weights_only=True)
            print(f"Loaded with weights_only=True: Successfully loaded as state dictionary.")
            print(f"First few keys: {list(state_dict.keys())[:5]}")
        except Exception as e:
            print(f"Failed to load with weights_only=True: {str(e)}")
            
    except Exception as e:
        print(f"Error loading with weights_only=False: {str(e)}")

# Inspect both files
inspect_pth_file("vit_gtsrb_best.pth")
inspect_pth_file("vit_ptsd_best.pth")