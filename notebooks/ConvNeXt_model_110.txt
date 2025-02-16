#neural network
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
import torchmetrics

from ConvNeXtBlock1D import ConvNeXtBlock1D

class ConvNeXt_model(pl.LightningModule):
    def __init__(self, dims, dropout, kernel_size):
        super().__init__()
        self.save_hyperparameters()
            
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            #connection layer
            self.layers.append(nn.Conv1d(in_channels = dims[i], out_channels = dims[i + 1], kernel_size = 1))
            #ConvNeXtBlock
            self.layers.append(ConvNeXtBlock1D(dim = dims[i + 1], kernel_size = kernel_size, dropout = dropout))
            
        self.decoder = nn.Sequential(
      nn.Conv1d(in_channels = dims[-1], out_channels = 1, kernel_size = 1),
      nn.Sigmoid())

    def forward(self, x):
        #x = x[:, 0:3, :]
        
        for layer in self.layers:
            x = layer(x) 
            
        z = self.decoder(x)
        
        z = z.view(z.size(0), -1)
        
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 5e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        y_hat = y_hat.view(y_hat.size(0), -1)

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
        y_hat = y_hat.view(y_hat.size(0), -1)
        
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

