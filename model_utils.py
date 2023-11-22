import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Encoder(nn.Module):
    def __init__(self, input_dim, nChannel):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)

        self.conv2 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0, groups=nChannel)
        self.bn2 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(
        self, embedding_dim, projection_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
    
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)
    
    
class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)

    def forward(self, x, H, W):
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out
    
    
class CrossAttention(nn.Module):
    """
    args:
        in_channels:    (int) : Embedding Dimension.
        key_channels:   (int) : Key Embedding Dimension,   Best: (in_channels).
        value_channels: (int) : Value Embedding Dimension, Best: (in_channels or in_channels//2). 
    input:
        x : [B, D, H, W]
    output:
        Efficient Attention : [B, D, H, W]
    
    """
    
    def __init__(self, in_channels, key_channels, value_channels, height, width,):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.H = height
        self.W = width
        self.reprojection = nn.Conv2d(value_channels, in_channels*2, 1)       
        self.norm = nn.LayerNorm(2 * in_channels)
        
    def forward(self, x1, x2):
        B, N, D = x1.size()
        
        # Efficient Attention
        keys = F.softmax(x1.transpose(1, 2), dim=2)
        queries = F.softmax(x1.transpose(1, 2), dim=1)
        values = x2.transpose(1, 2)          
        context = keys @ values.transpose(1, 2) # dk*dv            
        attended_value = (context.transpose(1, 2) @ queries).reshape(B, self.value_channels, self.H, self.W) # n*dv
        
        eff_attention  = self.reprojection(attended_value).reshape(B, 2 * D, N).permute(0, 2, 1)   
        eff_attention = self.norm(eff_attention)

        return eff_attention

    
class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(self, in_channels, key_channels, value_channels, height, width):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.H = height
        self.W = width
        self.attn = CrossAttention(in_channels, key_channels, value_channels, height, width)
        self.norm2 = nn.LayerNorm((in_channels * 2))
        self.mlp = MixFFN_skip((in_channels * 2), int(in_channels * 4))

    def forward(self, x1, x2):
        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)
        
        attn = self.attn(norm_1, norm_2)
        residual = torch.cat([x1, x2], dim=-1)            
        tx = residual + attn
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W)
        mx = rearrange(mx, 'b (h w) c -> b c h w', h=self.H, w=self.W)
        
        return mx