import os
import csv
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision import models
import logging
from torchvision.models.resnet import ResNet18_Weights
import timm
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisionTransformerConfig:
    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    num_class: int = 10
    embed_dim: int = 192
    depth: int = 9
    num_heads: int = 12
    mlp_ratio: float = 2.0
    qkv_bias: bool = True
    drop_rate: float = 0.15
    attn_drop_rate: float = 0.1
    num_experts: int = 7 
    top_k: int = 3
    balance_loss_weight: float = 1.0
    drop_path_rate: float = 0.1
    router_weight_reg: float = 0.03

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
    
class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.n_patch = (self.img_size // self.patch_size) ** 2
        self.proj = nn.Conv2d(config.in_chans, config.embed_dim, config.patch_size, stride=config.patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        return x

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.embed_dim // config.num_heads
        self.scale = head_dim ** 0.5
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim*3, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attn_drop_rate)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj_drop = nn.Dropout(config.drop_rate)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionRouter(nn.Module):
    def __init__(self, embed_dim, num_experts):
        super().__init__()
        self.expert_tokens = nn.Parameter(torch.empty(num_experts, embed_dim))
        nn.init.xavier_uniform_(self.expert_tokens)
        self.embed_dim = embed_dim

    def forward(self, x):
        router_logits = torch.einsum('bse,ne->bsn', x, self.expert_tokens) / (self.embed_dim ** 0.5)
        return router_logits

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.embed_dim, int(config.embed_dim*config.mlp_ratio))
        self.af = nn.GELU()
        self.l2 = nn.Linear(int(config.embed_dim*config.mlp_ratio), config.embed_dim)
        self.drop = nn.Dropout(config.drop_rate)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.af(x)
        x = self.l2(x)
        x = self.drop(x)
        return x

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        n_classes = input.size(-1)
        with torch.no_grad():
            smooth_target = torch.zeros_like(log_probs).fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = -torch.sum(smooth_target * log_probs, dim=-1).mean()
        return loss

class TrafficSignTrainDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found at {csv_file}")
        if not os.path.exists(root):
            raise FileNotFoundError(f"Training dataset directory not found at {root}")
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        self.meta_classes = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                img_path = os.path.join(root, row['Filename'])
                if not os.path.exists(img_path):
                    logger.warning(f"Image not found at {img_path}, skipping")
                    continue
                self.images.append(row['Filename'])
                self.labels.append(int(row['ClassId']))
                self.meta_classes.append(int(row['meta_class']))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        meta_class = self.meta_classes[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, meta_class

class TrafficSignTestDataset(Dataset):
    def __init__(self, root, csv_file, transform=None, default_meta_class=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found at {csv_file}")
        if not os.path.exists(root):
            raise FileNotFoundError(f"Test dataset directory not found at {root}")
        self.root = root
        self.transform = transform
        self.default_meta_class = default_meta_class
        self.images = []
        self.labels = []
        self.meta_classes = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                img_path = os.path.join(root, "Images", row['Filename'])
                if not os.path.exists(img_path):
                    logger.warning(f"Image not found at {img_path}, skipping")
                    continue
                self.images.append(row['Filename'])
                self.labels.append(int(row['ClassId']))
                meta_class = int(row.get('meta_class', self.default_meta_class))
                if self.default_meta_class is not None and 'meta_class' not in row:
                    logger.info(f"Using default meta_class {meta_class} for {csv_file}")
                self.meta_classes.append(meta_class)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.images[idx])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        meta_class = self.meta_classes[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, meta_class
        
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)    
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.router_weight_reg = config.router_weight_reg
        self.router = AttentionRouter(self.embed_dim, self.num_experts)
        self.experts = nn.ModuleList([Block(config) for _ in range(self.num_experts)])

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        router_logits = self.router(x)
        temperature = 1.0
        router_probs = F.softmax(router_logits / temperature, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)
        for k in range(self.top_k):
            indices = top_k_indices[:, :, k].flatten().long()
            expert_counts += torch.bincount(indices, minlength=self.num_experts)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        top_k_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim)
        selected_outputs = torch.gather(expert_outputs, 2, top_k_indices)
        combine_output = (selected_outputs * top_k_probs.unsqueeze(-1)).sum(dim=2)
        combine_output = self.drop_path(combine_output)
        x = x + combine_output 
        f_i = expert_counts.float() / (batch_size * seq_len * self.top_k)
        P_i = router_probs.mean(dim=[0, 1])
        balance_loss = self.num_experts * torch.sum(f_i * P_i)
        router_weight_norm = torch.norm(self.router.expert_tokens, p=2)
        balance_loss += self.router_weight_reg * router_weight_norm
        return x, balance_loss

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.n_patch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        self.blocks = nn.ModuleList([MoEBlock(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_class)
        
    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        balance_losses = []
        for block in self.blocks:
            x, block_balance_loss = block(x)
            balance_losses.append(block_balance_loss)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x, balance_losses

class MetaGatingNet(nn.Module):
    def __init__(self, num_experts=2, backbone='convnext_tiny', temperature=0.025):
        super().__init__()
        self.backbone = backbone
        self.temperature = temperature

        # Custom CNN backbones
        custom_backbones = {
            'small_cnn': ('SmallExpertCNN_Features', 512),
            'tiny_cnn': ('TinyExpertCNN_Features', 256),
            'micro_cnn': ('MicroExpertCNN_Features', 1024)
        }

        if backbone in custom_backbones:
            # Use custom CNN backbone
            from small_expert import SmallExpertCNN_Features, TinyExpertCNN_Features, MicroExpertCNN_Features

            if backbone == 'small_cnn':
                self.model = SmallExpertCNN_Features()
            elif backbone == 'tiny_cnn':
                self.model = TinyExpertCNN_Features()
            elif backbone == 'micro_cnn':
                self.model = MicroExpertCNN_Features()

            feature_dim = self.model.num_features
        else:
            # Use timm backbone
            self.model = timm.create_model(self.backbone, pretrained=True, num_classes=0)
            feature_dim = self.model.num_features

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.model(x)
        logits = self.fc(features) / self.temperature
        return logits

class MetaMoE(nn.Module):
    def __init__(self, experts, num_classes_list, meta_gating_net, meta_top_k=1):
        super(MetaMoE, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.num_classes_list = num_classes_list
        self.total_classes = sum(num_classes_list)
        self.meta_gating_net = meta_gating_net
        self.meta_top_k = min(max(meta_top_k, 1), self.num_experts)
        self.class_offsets = [0] + [sum(num_classes_list[:i+1]) for i in range(self.num_experts)]

        # Basic architecture check: Ensure all experts handle the same input shape
        # dummy_input = torch.randn(1, 3, 32, 32)  # Assuming 3x32x32 RGB
        # for i, expert in enumerate(self.experts):
        #     try:
        #         output = expert(dummy_input)
        #         if isinstance(output, tuple):
        #             output = output[0]
        #         if output.shape != torch.Size([1, num_classes_list[i]]):
        #             raise ValueError(f"Expert {i} output shape {output.shape} does not match expected [1, {num_classes_list[i]}]")
        #     except Exception as e:
        #         raise ValueError(f"Expert {i} failed input compatibility check: {str(e)}")
            
    def forward(self, x):
        gates = self.meta_gating_net(x)  # [batch_size, num_experts], probabilities
        batch_size = x.shape[0]
        final_output = torch.zeros(batch_size, self.total_classes, device=x.device, dtype=x.dtype)
        
        top_k_probs, top_k_indices = torch.topk(gates, k=self.meta_top_k, dim=1)  # [batch_size, top_k]
        sum_top_k = top_k_probs.sum(dim=1, keepdim=True)
        normalized_probs = top_k_probs / sum_top_k  # Renormalize probabilities
        
        for i in range(self.num_experts):
            sample_indices, k_positions = torch.where(top_k_indices == i)
            if len(sample_indices) > 0:
                scaling_factors = normalized_probs[sample_indices, k_positions]
                expert_input = x[sample_indices]
                expert_output = self.experts[i](expert_input)
                if isinstance(expert_output, tuple):
                    expert_output = expert_output[0]
                scaled_output = expert_output * scaling_factors.unsqueeze(1)
                start_idx = self.class_offsets[i]
                end_idx = self.class_offsets[i+1]
                final_output[sample_indices, start_idx:end_idx] = scaled_output.to(dtype=x.dtype)
        
        return final_output, gates

    def forward_with_timing(self, x):
        if torch.cuda.is_available():
            start_total = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)
            start_router = torch.cuda.Event(enable_timing=True)
            end_router = torch.cuda.Event(enable_timing=True)
            start_experts = torch.cuda.Event(enable_timing=True)
            end_experts = torch.cuda.Event(enable_timing=True)
            start_post = torch.cuda.Event(enable_timing=True)
            end_post = torch.cuda.Event(enable_timing=True)
            start_total.record()
            start_router.record()
            gates = self.meta_gating_net(x)
            end_router.record()
            start_experts.record()
            batch_size = x.shape[0]
            final_output = torch.zeros(batch_size, self.total_classes, device=x.device, dtype=x.dtype)
            top_k_probs, top_k_indices = torch.topk(gates, k=self.meta_top_k, dim=1)
            sum_top_k = top_k_probs.sum(dim=1, keepdim=True)
            normalized_probs = top_k_probs / sum_top_k
            for i in range(self.num_experts):
                sample_indices, k_positions = torch.where(top_k_indices == i)
                if len(sample_indices) > 0:
                    scaling_factors = normalized_probs[sample_indices, k_positions]
                    expert_input = x[sample_indices]
                    expert_output = self.experts[i](expert_input)
                    if isinstance(expert_output, tuple):
                        expert_output = expert_output[0]
                    scaled_output = expert_output * scaling_factors.unsqueeze(1)
                    start_idx = self.class_offsets[i]
                    end_idx = self.class_offsets[i+1]
                    final_output[sample_indices, start_idx:end_idx] = scaled_output.to(dtype=x.dtype)
            end_experts.record()
            start_post.record()
            end_post.record()
            end_total.record()
            torch.cuda.synchronize()
            router_time = start_router.elapsed_time(end_router) / 1000
            experts_time = start_experts.elapsed_time(end_experts) / 1000
            post_time = start_post.elapsed_time(end_post) / 1000
            total_time = start_total.elapsed_time(end_total) / 1000
        else:
            import time
            start_total = time.time()
            start_router = time.time()
            gates = self.meta_gating_net(x)
            end_router = time.time()
            start_experts = time.time()
            batch_size = x.shape[0]
            final_output = torch.zeros(batch_size, self.total_classes, device=x.device, dtype=x.dtype)
            top_k_probs, top_k_indices = torch.topk(gates, k=self.meta_top_k, dim=1)
            sum_top_k = top_k_probs.sum(dim=1, keepdim=True)
            normalized_probs = top_k_probs / sum_top_k
            for i in range(self.num_experts):
                sample_indices, k_positions = torch.where(top_k_indices == i)
                if len(sample_indices) > 0:
                    scaling_factors = normalized_probs[sample_indices, k_positions]
                    expert_input = x[sample_indices]
                    expert_output = self.experts[i](expert_input)
                    if isinstance(expert_output, tuple):
                        expert_output = expert_output[0]
                    scaled_output = expert_output * scaling_factors.unsqueeze(1)
                    start_idx = self.class_offsets[i]
                    end_idx = self.class_offsets[i+1]
                    final_output[sample_indices, start_idx:end_idx] = scaled_output.to(dtype=x.dtype)
            end_experts = time.time()
            start_post = time.time()
            end_post = time.time()
            end_total = time.time()
            router_time = end_router - start_router
            experts_time = end_experts - start_experts
            post_time = end_post - start_post
            total_time = end_total - start_total
        return final_output, gates, router_time, experts_time, post_time, total_time

class CombinedDataset(Dataset):
    def __init__(self, datasets, num_classes_list):
        self.datasets = datasets
        self.num_classes_list = num_classes_list
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = [0] + [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]
        self.class_offsets = [0] + [sum(num_classes_list[:i]) for i in range(1, len(num_classes_list))]

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        for i in range(len(self.datasets)):
            if idx < self.cumulative_lengths[i+1]:
                local_idx = idx - self.cumulative_lengths[i]
                image, label, meta_class = self.datasets[i][local_idx]
                label = label + self.class_offsets[i]
                return image, label, meta_class
        raise IndexError("Index out of range")