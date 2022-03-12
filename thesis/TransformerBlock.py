import torch
from torch import nn

#bias = true for linear layer? sommige doen dat, maar niet allemaal

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, channels):#, dropout_prob = 0.1):
        super().__init__()
        
        self.heads = heads
        self.cph = channels // heads #afgeronde breuk, channels per head
        
        self.query = nn.Linear(channels, self.heads * self.cph)     # zo geschreven, nochtans **meestal**: features_in = features_out
        self.key = nn.Linear(channels, self.heads * self.cph)
        self.value = nn.Linear(channels, self.heads * self.cph)  
    
    def ScaledDotProductAttention(self, Q, K, V, mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.cph) # scores : [batch_size x heads x seq_len x seq_len] (seq_len x seq_len is attention matrix)
        #print(scores.shape)
        scores.masked_fill_(mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
    
    def forward(self, Q, K, V, mask):
        #shape of Q, K and V is (Batch, Channels, Seq_len)
        #dimensies github 
        #print(Q.shape)
        batch_size = Q.shape[0]

        #prepare query, key, value
        # (B, C, S) -trans-> (B, S, C) -proj-> (B, S, C) -split-> (B, S, Heads, Cph) -trans-> (B, H, S, Cph)
        Q = self.query(Q.transpose(1, 2)).view(batch_size, -1, self.heads, self.cph).transpose(1,2)  # Q: [batch_size x heads x seq_len x cph]
        K = self.key(K.transpose(1, 2)).view(batch_size, -1, self.heads, self.cph).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        V = self.value(V.transpose(1, 2)).view(batch_size, -1, self.heads, self.cph).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        #mask wordt gedupliceerd voor elke head
        #mask (batch_size x seq_len x seq_len) -> (batch_size x n_heads x seq_len x seq_len)
        #print("voor")
        #print(mask.shape)
        #print(mask)
        mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        #1 mask per head per batch, dus voor alle channels in die head zelfde mask
        #print("na")
        #print(mask.shape)
        #print(mask)

        #self attention
        x, _ = self.ScaledDotProductAttention(Q, K, V, mask)
        #print("shape of x")
        #print(x.shape) # zelfde als Q, K en V: (B, H, S, Cph)
        #print(x)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.cph) # context: [batch_size x len_q x n_heads * d_v] = (B, S, C)
        #print(x.shape)

        return x     #waarom contiguous? het is geen binary array?


class TransformerBlock(nn.Module):
    def __init__(
        self,
        channels,
        heads = 7,
        layer_or_batchnorm = "layer",
        dropout = 0.0,
    ):
        super().__init__()

        if layer_or_batchnorm == "layer":
            self.norm = LayerNorm(dim, data_format="channels_first")
        else:
            self.norm = nn.BatchNorm1d(dim)
            
        self.MHA = MultiHeadAttention(heads, channels)
        
        self.linear = nn.Linear(channels, channels)

        self.pointwise_net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        
    def create_mask(x):
        #print(x.size())
        x = x[:, 0, :]
        batch_size, seq_len = x.size()
        #print(batch_size, seq_len)
        # eq(zero) is PAD token
        mask = x.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking    eq vergelijkt
        return mask.expand(batch_size, seq_len, seq_len)  # batch_size x len_q x len_k       expand expands the singleton dimensie
        #creeert B (batch_size) masks van grootte seq_len bij seq_len
        #wat met channels?
        #met enkel 0 in?
        #dus namaakbaar, want onafhankelijk van Q, K en V (behalve grootte)

    def forward(self, x):
        #first normalization
        z1 = self.norm(x)
        
        #multi-headed attention
        #mask = create_mask(x)
        z1 = self.MHA(z1, z1, z1, mask) #(B, S, C)
        z1 = self.linear(z1)
        z1 = z1.permute(0, 2, 1)  #(B, C, S)
        
        #first residual connection
        z1 += x
        
        #second normalization
        z2 = self.norm(z1)
        
        #inverted bottleneck
        z2 = z2.permute(0, 2, 1)
        z2 = self.pointwise_net(z)
        
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
