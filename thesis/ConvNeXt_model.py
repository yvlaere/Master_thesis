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
    def __init__(self):
        super().__init__()
        
        self.stem = nn.Conv1d(in_channels = 7, out_channels = 96, kernel_size = 1)
            
        self.res2 = ConvNeXtBlock1D(96, dropout = 0.5)
            
        self.res23 = nn.Conv1d(in_channels = 96, out_channels = 192, kernel_size = 1)
        
        self.res3 = ConvNeXtBlock1D(192, dropout = 0.5)
            
        self.res34 = nn.Conv1d(in_channels = 192, out_channels = 384, kernel_size = 1)
        
        self.res4 = ConvNeXtBlock1D(384, dropout = 0.5)
            
        self.res45 = nn.Conv1d(in_channels = 384, out_channels = 768, kernel_size = 1)
        
        self.res5 = ConvNeXtBlock1D(768, dropout = 0.5)
            
        self.decoder = nn.Sequential(
      nn.Conv1d(in_channels = 768, out_channels = 1, kernel_size = 1),
      nn.Sigmoid())

    def forward(self, x):
        y = self.stem(x)
        #for i in range(3):
        y = self.res2(y)
        y = self.res23(y)
        #for i in range(3):
        y = self.res3(y)
        y = self.res34(y)
        for i in range(3):
            y = self.res4(y)
        
        y = self.res45(y)
        #for i in range(3):
        y = self.res5(y)
        
        output = self.decoder(y)
        
        output = output.view(output.size(0), -1)
        
        return output 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 5e-4)
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

