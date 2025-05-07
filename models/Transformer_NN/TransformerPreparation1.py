import torch
from torch import nn
import numpy as np

class TransformerPreparation(nn.Module):
    def __init__(self):
        super().__init__()
        pass
        
    #create a general mask that can be used by all transformer blocks
    def create_mask(self, x):
        x = x[:, 0, :]
        
        #get dimensions
        batch_size, seq_len = x.size()
        
        #create mask
        mask = x.data.eq(0)#.unsqueeze(1)                #mask: (batch_size x 1 x seq_len), True is masking
        return ~mask#.expand(batch_size, seq_len, seq_len)#mask: (batch_size x seq_len x seq_len), expands the singleton dimension, channel dimension is created in the transformer block
        
    # Code adapted from https://www.tensorflow.org/tutorials/text/transformer
    def get_angles(self, seq_len_vec, channels_vec, channels):
        angle_rates = 1 / np.power(10000, (2 * (channels_vec//2)) / np.float32(channels))
        return seq_len_vec * angle_rates

    def positional_encoding(self, x):
        #get dimensions
        batch, channels, seq_len = x.shape
        
        angle_rads = self.get_angles(np.arange(seq_len)[:, np.newaxis], np.arange(channels)[np.newaxis, :], channels)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        pos_encoding = torch.tensor(pos_encoding)
        pos_encoding = pos_encoding.repeat(batch, 1, 1)
        pos_encoding = pos_encoding.permute(0, 2, 1)
        #print(pos_encoding.shape)

        return pos_encoding
