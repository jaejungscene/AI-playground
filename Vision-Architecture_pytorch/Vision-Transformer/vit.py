import torch
import torch.nn as nn
import numpy as np







class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    Drop Path and Drop Connect is same.
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob) # binary tensor
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)



class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(
        self, 
        img_size: int=224, 
        patch_size: int=16,
        in_ch: int=3,
        embed_dim: int=768,
        norm_layer=None,
        flatten=True,
        bias=True
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1,2) # B,numPatchH*numPatchW, C
        x = self.norm(x)
        return x



class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        dim : int, 
        num_heads : int,
        qkv_bias : bool = False,
        attn_drop : float = 0.,
        proj_drop : float = 0.
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # assert C % self.num_heads == 0, "D of x should be divisible by num_heads"
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn_probs = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_probs)

        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_probs



class Mlp(nn.Module):
    def __init__(
        self,
        in_features : int,
        hidden_features : int,
        out_features : int,
        act_layer = nn.GELU,
        bias : bool = True,
        drop : float = 0.,
    ) -> None:
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x



class Block(nn.Module):
    def __init__(
        self,
        dim : int,
        num_heads : int,
        mlp_ratio : int = 4,
        qkv_bias : bool = False,
        drop_rate : float = 0.,
        drop_path : float = 0.,
        attn_drop : float = 0.,
        act_layer = nn.GELU,
        norm_layer = nn.LayerNorm,
    ) -> None:
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, qkv_bias, attn_drop, drop_rate)
        self.drop_path1 = DropPath(drop_path) if drop_path>0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim*mlp_ratio), dim, act_layer, drop_rate)
        self.drop_path2 = DropPath(drop_path) if drop_path>0. else nn.Identity()
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    r"""
    Args:
        dim (int): Patch embedded dimension/
        num_head (int): Number of attention heads in different layers.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        attn_drop (float): Attention dropout rate.
    """
    def __init__(
        self,
        img_size:int=224,
        patch_size:int=16,
        in_ch:int=3,
        num_classes:int=1000,
        global_pool='token',
        embed_dim:int=768,
        depth:int=12,
        num_heads:int=12,
        mlp_ratio:float=4.,
        qkv_bias:bool=True,
        class_token:bool=True,
        drop_rate:float=0.,
        drop_path:float=0.,
        attn_drop:float=0.,
        weight_init:str='',
        norm_layer=None,
        act_layer=None,
    ) -> None:
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_ch=in_ch,
            embed_dim=embed_dim,
            bias=True,
        )
        num_patches = (img_size//patch_size)**2
        
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim)) if class_token else None
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim)*0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x for x in np.linspace(0, drop_path, depth)]
        self.blocks = nn.Squential(
            *[Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                act_layer=act_layer
            )   for i in range(depth)]
        )

        if num_classes > 0:
            self.norm = norm_layer(embed_dim)
            self.head = nn.Linear(self.embed_dim, num_classes)
        
        if weight_init != 'skip':
            self.init_weights(weight_init)
        
    # def init_weights(self, mode=''):
    #     assert 
        