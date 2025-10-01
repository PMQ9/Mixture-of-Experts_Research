import timm
import torch
import torch.nn as nn
from torchvision import models
from vision_transformer_moe import VisionTransformer, VisionTransformerConfig, LabelSmoothingCrossEntropy, TrafficSignTrainDataset, TrafficSignTestDataset

class ModelWrapper(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, x):
        output = self.model(x)
        return output, [torch.tensor(0.0, device=x.device)]

class LogitsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]
        return output
    
def create_model(model_arch, config):
    if model_arch == 'vit_moe':
        return VisionTransformer(config)
    elif model_arch == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, config.num_class)
        return ModelWrapper(model, config.num_class)
    elif model_arch == 'resnet101':
        model = models.resnet101(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, config.num_class)
        return ModelWrapper(model, config.num_class)
    elif model_arch == 'convnext_tiny':
        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=config.num_class)
        return ModelWrapper(model, config.num_class)
    elif model_arch == 'convnextv2_tiny':
        model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=config.num_class)
        return ModelWrapper(model, config.num_class)
    elif model_arch == 'convnext_small':
        model = timm.create_model('convnext_small', pretrained=True, num_classes=config.num_class)
        return ModelWrapper(model, config.num_class)
    elif model_arch == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=config.num_class)
        return ModelWrapper(model, config.num_class)
    elif model_arch == 'vit_base':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=config.num_class, img_size=config.img_size)
        return ModelWrapper(model, config.num_class)
    elif model_arch == 'convnext_large':
        model = timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=True, num_classes=config.num_class)
        return ModelWrapper(model, config.num_class)
    elif model_arch == 'vit_large':
        model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=config.num_class, img_size=config.img_size)
        return ModelWrapper(model, config.num_class)
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")
    
def get_num_classes(model):
    if hasattr(model, 'total_classes'):
        return model.total_classes
    elif isinstance(model, VisionTransformer):
        return model.config.num_class
    elif isinstance(model, ModelWrapper):
        return model.num_classes
    else:
        raise ValueError("Cannot determine the number of classes from the model")