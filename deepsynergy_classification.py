import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L



import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip

import matplotlib.pyplot as plt

hyperparameter_file = 'hyperparameters' # textfile which contains the hyperparameters of the model
data_file = '../data_test_fold0_tanh.p.gz' # pickle file which contains the data (produced with normalize.ipynb)

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

layers = [8182,4096,1] 
epochs = 1000 
act_func = 'relu'
dropout = 0.5 
input_dropout = 0.2
eta = 0.00001 
norm = 'tanh' 

file = gzip.open(data_file, 'rb')
X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = pickle.load(file)
file.close()

y_tr.shape

X_tr.shape

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(X_tr.shape[1], layers[0]), 
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[0], layers[1]),
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[1], layers[2]),
                                nn.Sigmoid()
                               )

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.apply(init_weights)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self.encoder(x)
        loss = F.binary_cross_entropy(z,y.view(x.size(0), -1).float())
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.encoder(x)
        val_loss = F.binary_cross_entropy(z,y.view(x.size(0), -1).float())
        return val_loss

        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.encoder(x)
        test_loss = F.binary_cross_entropy(z,y.view(x.size(0), -1).float())
        self.log("test_loss", test_loss)
        return test_loss
        
    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=eta, momentum=0.5)
        return optimizer

X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test =torch.FloatTensor(X_tr),torch.FloatTensor(X_val),torch.FloatTensor(X_train),torch.FloatTensor(X_test),torch.FloatTensor(y_tr), torch.FloatTensor(y_val), torch.FloatTensor(y_train), torch.FloatTensor(y_test)

X_tr.shape

X_test.shape

y_tr = (y_tr>30)*1
y_val = (y_val>30)*1
y_test = (y_test>30)*1

train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
valid_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

model = LitAutoEncoder(Encoder())

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64)
valid_loader = DataLoader(valid_dataset, batch_size=64)

# train with both splits
trainer = L.Trainer()
trainer.fit(model, train_loader, valid_loader)

trainer.test(model, dataloaders=DataLoader(test_dataset))

with torch.no_grad():
    y_pred = model.encoder(X_test).detach()
