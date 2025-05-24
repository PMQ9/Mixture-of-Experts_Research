import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# dataclass
@dataclass
class VisionTransformerConfig:
    img_size: int = 32
    patch_size: int = 4
    in_chans : int =3
    num_class: int = 10     #number of classes for classification
    embed_dim: int = 192
    depth: int = 9
    num_heads: int = 12
    mlp_ratio: float = 2.0
    qkv_bias: bool = True
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.0
    num_experts: int = 4    # number or experts
    top_k: int = 2

    
class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.n_patch = (config.img_size // config.patch_size) ** 2
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
        qkv  = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2,-1))* self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
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

"""
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
"""

class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.embed_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        # Router
        self.router = nn.Linear(self.embed_dim, self.num_experts)

        # Experts
        self.experts = nn.ModuleList([Block(config) for _ in range (self.num_experts)])

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim = -1)
        
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim = -1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim = -1, keepdim = True)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim = 2)
        top_k_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim)
        selected_outputs = torch.gather(expert_outputs, 2, top_k_indices)
        combine_output = (selected_outputs * top_k_probs.unsqueeze(-1)).sum(dim = 2)

        return combine_output

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.n_patch
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(p = config.drop_rate)
        
        #self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.blocks = nn.ModuleList([MoEBlock(config) for _ in range(config.depth)])


        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_class)
        
    def forward(self, x):
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        x = torch.cat((cls_token, x), dim = 1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x
           