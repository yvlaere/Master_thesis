#neural network
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
import torchmetrics

#bias = true for linear layer? sommige doen dat, maar niet allemaal

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, channels):#, dropout_prob = 0.1):
        super().__init__()
        
        self.heads = heads
        self.cph = channels // heads #afgeronde breuk, channels per head
        
        self.query = nn.Linear(channels, self.heads * self.cph)     # zo geschreven, nochtans **meestal**: features_in = features_out
        self.key = nn.Linear(channels, self.heads * self.cph)
        self.value = nn.Linear(channels, self.heads * self.cph)  
      
    def attention(self, Q, K, V, mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.cph) # scores : [batch_size x heads x seq_len x seq_len] (seq_len x seq_len is attention matrix)
        #print(scores.shape)
        scores.masked_fill_(mask, -1e9)#float('-inf')) # Fills elements of self tensor with value where mask is one.  #but mask is zero everywhere?
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        
        #############################
        del scores
        del attn
        torch.cuda.empty_cache()
        ###########################
        
        return context#, attn

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
        mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        #1 mask per head per batch, dus voor alle channels in die head zelfde mask
        
        #self attention
        x = self.attention(Q, K, V, mask)
         # shape van x zelfde als Q, K en V: (B, H, S, Cph)
        #print(x)
        """
        #####################################
        #werkt niet
        del Q
        del K
        del V
        #test
        del mask
        torch.cuda.empty_cache()
        ########################################
        """
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.cph) # context: [batch_size x len_q x n_heads * d_v] = (B, S, C)


        return x     #waarom contiguous? het is geen binary array?  .contiguous()
                            
        


        
def create_mask(x):
    #print(x.size())
    x = x[:, 0, :]
    batch_size, seq_len = x.size()
    #print(batch_size, seq_len)
    # eq(zero) is PAD token
    mask = x.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking    eq vergelijkt
    #print("masktest")
    #print(True in mask)
    return mask.expand(batch_size, seq_len, seq_len)  # batch_size x len_q x len_k       expand expands the singleton dimensie
    #creeert B (batch_size) masks van grootte seq_len bij seq_len
    #wat met channels?
    #met enkel 0 in?
    #dus namaakbaar, want onafhankelijk van Q, K en V (behalve grootte)
    
        
        
class Transformer_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
      nn.Conv1d(in_channels = 7, out_channels = 32, kernel_size = 7, padding = 3),
      nn.ReLU(),
      nn.Conv1d(in_channels = 32, out_channels = 9, kernel_size = 7, padding = 3),
      nn.ReLU())
        
        self.MHA = MultiHeadAttention(heads = 3, channels = 9)
        self.linear = nn.Linear(9, 1)
        self.sigm = nn.Sigmoid()
        
    def forward(self, x):
        x = self.encoder(x)

        mask = create_mask(x)
        #print("mask?")
        #print(mask.shape)

        y = self.MHA(x, x, x, mask)
        #print("problem?")
        #print(y.shape)
        y = self.linear(y)
        #print(y.shape)
        y = y.permute(0, 2, 1)
        #print(y.shape)
        y = self.sigm(y)
        y = y.view(y.size(0), -1)
        
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        
        y_lab = y[y != 2]
        y_hat = y_hat[y != 2]
        
        #evaluation metrics
        accuracy = torchmetrics.functional.accuracy(y_hat, y_lab.type(torch.cuda.IntTensor))
        self.log('train_acc', accuracy)
        aur = torchmetrics.functional.auroc(y_hat, y_lab.type(torch.cuda.IntTensor))
        self.log('train_auroc', aur)
        f1 = torchmetrics.functional.f1(y_hat, y_lab.type(torch.cuda.IntTensor))
        self.log('train_F1', f1)
        precision = torchmetrics.functional.precision(y_hat, y_lab.type(torch.cuda.IntTensor))
        self.log('train_precision', precision)
        recall = torchmetrics.functional.recall(y_hat, y_lab.type(torch.cuda.IntTensor))
        self.log('train_recall', recall)
        prauc = torchmetrics.functional.average_precision(y_hat, y_lab.type(torch.cuda.IntTensor))
        self.log('train_AP', prauc)
        loss = F.binary_cross_entropy(y_hat, y_lab)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        
        y_lab = y[y != 2]
        y_hat = y_hat[y != 2]
        
        #evaluation metrics
        accuracy = torchmetrics.functional.accuracy(y_hat, y_lab.type(torch.cuda.IntTensor))
        self.log('val_acc', accuracy)
        aur = torchmetrics.functional.auroc(y_hat, y_lab.type(torch.cuda.IntTensor))
        self.log('val_auroc', aur)
        f1 = torchmetrics.functional.f1(y_hat, y_lab.type(torch.cuda.IntTensor))
        self.log('val_F1', f1)
        precision = torchmetrics.functional.precision(y_hat, y_lab.type(torch.cuda.IntTensor))
        self.log('val_precision', precision)
        recall = torchmetrics.functional.recall(y_hat, y_lab.type(torch.cuda.IntTensor))
        self.log('val_recall', recall)
        prauc = torchmetrics.functional.average_precision(y_hat, y_lab.type(torch.cuda.IntTensor))
        self.log('val_AP', prauc)
        loss = F.binary_cross_entropy(y_hat, y_lab)
        self.log('val_loss', loss)
        return loss
        
    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [('total', total_params)]
        return params_per_layer


    
    
    #mask is niet het probleem
    #score is het probleem? heel grote matrix
    #size increases exponentially with seq length
    #loss niet altijd tss 0 en 1????????
    
    #mask is er voor de padding (in het begin is dat er), anders zou dat een invloed hebben op de attention matrix
