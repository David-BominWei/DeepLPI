molembed_path = "/home/wbm001/deeplpi/data/davis/mol.csv"
seqembed_path = "/home/wbm001/deeplpi/data/davis/seq_6165.csv"
train_path = "/home/wbm001/deeplpi/data/davis/trainset.csv"
test_path = "/home/wbm001/deeplpi/data/davis/testset.csv"
tensorboard_path = "/home/wbm001/deeplpi/DeepLPI/output/tensorboard/"
data_path = "/home/wbm001/deeplpi/DeepLPI/output/"

RAMDOMSEED = 11
CLASSIFYBOUND = -2

import pandas as pd

seqembed = pd.read_csv(seqembed_path)
molembed = pd.read_csv(molembed_path)
train = pd.read_csv(train_path)

seqembed = seqembed.set_index("id").iloc[:,1:]
molembed = molembed.set_index("id").iloc[:,1:]

import torch
from torch import tensor
from torch.utils.data import DataLoader,TensorDataset,SequentialSampler,RandomSampler
import numpy as np
from sklearn.model_selection import train_test_split

train, val = train_test_split(train, test_size=1000, random_state=RAMDOMSEED)

# train
train_seq = tensor(np.array(seqembed.loc[train["seq"]])).to(torch.float32)
train_mol = tensor(np.array(molembed.loc[train["mol"]])).to(torch.float32)
train_classify = tensor(np.array(train["pKd (nM)"])).to(torch.float32)

trainDataset = TensorDataset(train_mol,train_seq,train_classify)
trainDataLoader = DataLoader(trainDataset, batch_size=256)

#val
val_seq = tensor(np.array(seqembed.loc[val["seq"]])).to(torch.float32)
val_mol = tensor(np.array(molembed.loc[val["mol"]])).to(torch.float32)
val_classify = tensor(np.array(val["pKd (nM)"])).to(torch.float32)

# valDataset = TensorDataset(val_mol,val_seq,val_classify)
# valDataLoader = DataLoader(valDataset, batch_size=256)

from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import torch

class resBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1=False, strides=1, dropout=0.3):
        super().__init__()
        
        self.process = nn.Sequential (
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        
        if use_conv1:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv1 = None
        
    def forward(self, x):
        left = self.process(x)
        right = x if self.conv1 is None else self.conv1(x)
        
        return F.relu(left + right)

class cnnModule(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=32, dropout=0.3):
        super().__init__()
        
        self.head = nn.Sequential (
            nn.Conv1d(in_channel, hidden_channel, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(2)
        )
        
        self.cnn = nn.Sequential (
            resBlock(hidden_channel, out_channel, use_conv1=True, strides=1),
            resBlock(out_channel, out_channel, strides=1),
            resBlock(out_channel, out_channel, strides=1),
        )
    
    def forward(self, x):
        x = self.head(x)
        x = self.cnn(x)
        
        return x

class DeepLPI(nn.Module):
    def __init__(self, molshape, seqshape, dropout=0.3):
        super().__init__()
        
        self.molshape = molshape
        self.seqshape = seqshape

        self.molcnn = cnnModule(1,16)
        self.seqcnn = cnnModule(1,16)
        
        self.pool = nn.AvgPool1d(5, stride = 3)
        self.lstm = nn.LSTM(16, 16, num_layers=2, batch_first=True, bidirectional=True)
        
        self.mlp = nn.Sequential (
            nn.Linear(round(((300+6165)/4-2)*2/3) * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(128, 1),
        )

    def forward(self, mol, seq):
        mol = self.molcnn(mol.reshape(-1,1,self.molshape))
        seq = self.seqcnn(seq.reshape(-1,1,self.seqshape))
        
        # put data into lstm        
        x = torch.cat((mol,seq),2)
        x = self.pool(x)
        # print(x.shape)
        x = x.reshape(-1,round(((self.molshape+self.seqshape)/4-2)/3),16)

        x,_ = self.lstm(x)
        # fully connect layer
        x = self.mlp(x.flatten(1))
        
        x = x.flatten()
        
        return x

def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

from matplotlib import pyplot as plt
import io

def train_loop(model, train_dataloader, lossfunc, optimizer, scheduler):
    model = model.to("cuda")
    model.train()
    loop_loss = 0
    
    for step, batch in enumerate(train_dataloader):
        step_mol, step_seq, step_label = batch
        step_mol, step_seq, step_label = step_mol.to("cuda"), step_seq.to("cuda"), step_label.to("cuda")

        optimizer.zero_grad()
        logits = model(step_mol, step_seq)
        loss = lossfunc(logits, step_label)
        loss.backward()
        optimizer.step()
        loop_loss += float(loss.to("cpu"))

        if step%20 == 0:
            print("step " + str(step) + " loss: " + str(float(loss.to("cpu"))))
        
    with torch.no_grad():
        return loop_loss/len(train_dataloader)

from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score


def test_loop(model, val_mol, val_seq, val_lab, writer, epoch):
    model.eval()
    model = model.to("cuda")
    with torch.no_grad():
        step_mol, step_seq = val_mol.to("cuda"), val_seq.to("cuda")
        logits = model(step_mol,step_seq)
    logits = logits.to("cpu")

    fig = plt.figure(figsize=(6, 6))
    plt.xlabel("true value")
    plt.ylabel("predict value")
    plt.scatter(logits, val_lab, alpha = 0.2, color='Black')
    plt.plot(range(-9,4), range(-9,4),color="r",linewidth=2)
    plt.xlim(-9,4)
    plt.ylim(-9,4)
    writer.add_figure(tag='test evaluate', figure=fig, global_step=epoch)

    return mean_squared_error(val_lab,logits), r2_score(val_lab,logits)

import torch.optim as optim

model = DeepLPI(300,6165)

model.apply(initialize_weights)
loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, min_lr=0.00001)

from torch.utils.tensorboard import SummaryWriter
import time

version = "v9b1reg"

writer = SummaryWriter(tensorboard_path)

for epoch in range(1000):
    print("--"*20)
    print("epoch: " + str(epoch))
    time0 = time.time()

    avgloss = train_loop(model, trainDataLoader, loss_fn, optimizer, scheduler)
    msescore, r2score = test_loop(model, val_mol, val_seq, val_classify, writer, epoch)

    writer.add_scalar("test time", time.time()-time0, epoch)
    writer.add_scalar('avgloss', avgloss , epoch)
    writer.add_scalar('mse', msescore , epoch)
    writer.add_scalar('r2', r2score , epoch)
    writer.add_scalar('current lr', optimizer.param_groups[0]['lr'], epoch)

    print()
    print("R2: " + str(r2score) + "\t MSE: " + str(msescore))
    print("use time: " + str(time.time() - time0))
    
    model.eval()
    if epoch % 50 == 0:
        torch.save({'state_dict': model.state_dict()}, data_path + 'model/' + str(version) + "e" + str(epoch) + '.pth.tar')
    else:
        torch.save({'state_dict': model.state_dict()}, data_path + "model/quicksave.pth.tar")
