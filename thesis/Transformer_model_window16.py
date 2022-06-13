#add significant depth for windowed self-attention

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
import matplotlib.pyplot as plt

from TransformerBlock1 import TransformerBlock
from TransformerPreparation1 import TransformerPreparation

class Transformer_model(pl.LightningModule):
    def __init__(self, dims, dropout, heads, window):
        super().__init__()
        self.save_hyperparameters()
        
        self.tp = TransformerPreparation()
        
        self.connection_layers = nn.ModuleList()
        
        self.transformer_layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            #connection layer
            self.connection_layers.append(nn.Conv1d(in_channels = dims[i], out_channels = dims[i + 1], kernel_size = 1))
            #Transformer
            self.transformer_layers.append(TransformerBlock(channels = dims[i + 1], heads = heads, dropout = dropout, window = window))
            
        self.decoder = nn.Sequential(
      nn.Conv1d(in_channels = dims[-1], out_channels = 1, kernel_size = 1),
      nn.Sigmoid())  
        
    def forward(self, x):
        #create a mask
        mask = self.tp.create_mask(x).to(device = x.device)
    
        #create positional encoding
        pos_enc = self.tp.positional_encoding(x).to(device = x.device)
        
        x += pos_enc
        
        for i in range(len(self.connection_layers)):
            x = self.connection_layers[i](x)
            x = self.transformer_layers[i](x, mask)
            
        z = self.decoder(x)
        
        z = z.view(z.size(0), -1)
        
        return z
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
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
