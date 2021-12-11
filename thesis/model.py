import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
import torchmetrics

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
      nn.Conv1d(in_channels = 7, out_channels = 32, kernel_size = 7, padding = 3),
      nn.ReLU(),
      #nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 5, padding = 2),
      #nn.ReLU(),
      nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 7, padding = 3),
      nn.ReLU())
        self.decoder = nn.Sequential(
      nn.Conv1d(in_channels = 32, out_channels = 1, kernel_size = 7, padding = 3),
      nn.Sigmoid())

    def forward(self, x):
        embedding = self.encoder(x)
        output = self.decoder(embedding) #not originally here
        output = output.view(output.size(0), -1)
        
        return output #embedding

        
        #for layer in self.encoder:   
            #x = layer(x)
            #print(x.shape)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):      #batch_idx is index of batch
        x, y = train_batch
        #x = x.view(x.size(0), -1)
        #or use self.forward 
        z = self.encoder(x)
        y_hat = self.decoder(z)
        y_hat = y_hat.view(y_hat.size(0), -1)

        y_lab = y[y != 2]
        y_hat = y_hat[y != 2]
        #correct = y_lab == torch.round(y_hat)
        #print(y_hat)
        #print(y_lab)
        #print(y_hat > 0.5)
        #print('True' in y_hat > 0.5)
        #y_test = y_hat.detach().numpy()
        #y_test = np.around(y_test)
        #num = 1
        #if num in y_test:
            #print('succes')

        #print(y_lab)
        #print(y_hat)
        
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
        self.log('train_prauc', prauc)
        loss = F.binary_cross_entropy(y_hat, y_lab)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        #print(val_batch)
        x, y = val_batch
        #x = x.view(x.size(0), -1)
        z = self.encoder(x)
        y_hat = self.decoder(z)
        y_hat = y_hat.view(y_hat.size(0), -1)
        
        y_lab = y[y != 2]
        y_hat = y_hat[y != 2]
        #correct = y_lab == torch.round(y_hat)
        
        #acc = sum(correct)/len(correct)
        
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
        self.log('val_prauc', prauc)
        loss = F.binary_cross_entropy(y_hat, y_lab)
        self.log('val_loss', loss)
        return loss
        
    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [('total', total_params)]
        return params_per_layer

#dataset = torch.utils.data.TensorDataset(X, Y)
#dataset = list(zip(X, Y))
#print(dataset)

#directory = '/home/yarivl/DNA_data/hx1_ab231_2/BNP17L0004-0207-D1_GA30000/0/workspace/pass/0/'


#tensor dataset
#dataset class in dataloader
train_loader = DataLoader(train, batch_size = 8, num_workers = 16, collate_fn = custom_collate)
val_loader = DataLoader(val, batch_size = 8, num_workers = 16, collate_fn = custom_collate)

# model
model = LitAutoEncoder()

# training
#trainer = pl.Trainer(gpu = 4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer = pl.Trainer(gpus=1, max_epochs=15)
trainer.fit(model, train_loader, val_loader)
