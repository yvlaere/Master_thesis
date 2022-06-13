import torch
from torch import nn
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F

class WindowAttention(nn.Module):
    def __init__(self, heads, channels, window):
        super().__init__()
        assert window % 2 == 1, 'Window size should be an odd integer.'
        
        self.heads = heads
        self.cph = channels // heads
        
        self.softmax = nn.Softmax(dim = -2)
        self.w = int((window-1)/2)
        
        self.query = nn.Linear(channels, self.heads * self.cph)
        self.key = nn.Linear(channels, self.heads * self.cph)
        self.value = nn.Linear(channels, self.heads * self.cph)
    
    def forward(self, x, mask):
        #assert k.shape[1] == q.shape[1], 'q and k should have same input length.'
        batch_size = x.shape[0]
        
        #prepare query, key, value through linear transformation
        # (B, C, S) -trans-> (B, S, C) -proj-> (B, S, C) -split-> (B, S, Heads, Cph)
        q = self.query(x.transpose(1, 2)).view(batch_size, -1, self.heads, self.cph)     #Q: (batch x seq_len x heads x cph)
        k = self.key(x.transpose(1, 2)).view(batch_size, -1, self.heads, self.cph)       #K: (batch x seq_len x heads x cph)
        v = self.value(x.transpose(1, 2)).view(batch_size, -1, self.heads, self.cph)     #V: (batch x seq_len x heads x cph)
        
        b, s, nh, h = k.shape
        
        q = q * (h ** -.5)      #Q: (batch x seq_len x heads x cph)

        k = F.pad(k, (0,)*4 + (self.w,)*2).unfold(1, s, 1)      #(batch, window, heads, cph, seq_len)
        v = F.pad(v, (0,)*4 + (self.w,)*2).unfold(1, s, 1)

        A = einsum('b q n h, b k n h q -> b q k n', q, k)       #(batch, seq_len, window, heads)

        mask = F.pad(mask.to(device = k.device), (self.w,)*2, value = False).unfold(1, s, 1)
        mask = mask.permute(0,2,1).unsqueeze(-1)

        mask_value = -torch.finfo(A.dtype).max
        #print(mask_value)
        A.masked_fill_(~mask, mask_value)
        #print(A.shape)
        A = self.softmax(A)#, dim = -2)
        z = einsum('b q k n, b k n h q -> b q n h', A, v)
        #print("z")
        #print(z.shape)#(batch, seq_len, heads, cph)
        
        z = z.transpose(1, 2).reshape(batch_size, -1, self.heads * self.cph) #x: (batch x seq_len x heads * cph) = (B, S, C)
        return z


class TransformerBlock(nn.Module):
    def __init__(
        self,
        channels,
        heads = 8,
        layer_or_batchnorm = "layer",
        dropout = 0.0,
        window = 7,
    ):
        super().__init__()

        if layer_or_batchnorm == "layer":
            self.norm = LayerNorm(channels, data_format="channels_first")
        else:
            self.norm = nn.BatchNorm1d(channels)
            
        self.WHA = WindowAttention(heads, channels, window = window)
        
        self.linear = nn.Linear(channels, channels)

        self.pointwise_net = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * channels, channels),
        )        
      
    def forward(self, x, mask):
        
        #first normalization
        z1 = self.norm(x)
        
        
        #multi-headed attention
        z1 = self.WHA(z1, mask) #(B, S, C)
        #print(z1.shape)
        z1 = self.linear(z1)
        z1 = z1.permute(0, 2, 1)  #(B, C, S)
        
        #first residual connection
        z1 += x
        
        #second normalization
        z2 = self.norm(z1)
        
        #inverted bottleneck
        z2 = z2.permute(0, 2, 1)
        z2 = self.pointwise_net(z2)
        
        #second residual connection
        return z2.permute(0, 2, 1) + z1
    
# adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x
