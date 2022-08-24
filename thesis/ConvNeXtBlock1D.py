import torch
from torch import nn

# adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim, kernel_size = 7, layer_or_batchnorm = "layer", depthwise = True, dropout = 0.0):
        super().__init__()

        self.conv = nn.Conv1d(dim, dim,kernel_size = kernel_size, padding = kernel_size // 2, groups = (dim if depthwise else 1))

        if layer_or_batchnorm == "layer":
            self.norm = LayerNorm(dim, data_format="channels_first")
        else:
            self.norm = nn.BatchNorm1d(dim)

        self.pointwise_net = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, x):
        z = self.norm(self.conv(x))
        z = z.permute(0, 2, 1)
        z = self.pointwise_net(z)
        return z.permute(0, 2, 1) + x
    
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
