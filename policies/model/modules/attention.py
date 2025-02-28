from torch import nn


class Attention(nn.Module):
    """
    A simplified version of attention from DSTFormer that also considers x tensor to be (B, N, C) instead of
    (B * N, C)
    """

    def __init__(self, dim_in, dim_out, num_heads=8, dim_proj=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.dim_proj = dim_proj or dim_out

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.qkv = nn.Linear(dim_in, self.dim_proj * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if dim_in // num_heads != 0:
            raise ValueError("dim_in must be divisible by num_heads")

    def forward(self, x):
        B, L, C = x.shape
        
        qkv = self.qkv(x) # (B, N, 3 * C)

        qkv = qkv.reshape(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, L, C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        

        x = self.proj_drop(x)
        return x


    def attention(self, q, k, v, mask=None):
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, 
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        return attn @ v  # (B, H, J, T, C)
