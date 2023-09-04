#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from pytorchtools_Transformer_realistic import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
import scipy.stats as st
from matplotlib.offsetbox import AnchoredText

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, d_model, num_layers, p, neg_slope, n_head, pos_dropout):
        super(Transformer, self).__init__()

        self.dmodel = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = n_head, dim_feedforward = 2048, dropout = 0.25, batch_first = False, dtype = torch.float)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = self.encoder_layer, num_layers = num_layers)
        self.pos_encoder = PositionalEncoding(d_model = d_model, dropout = pos_dropout)
        
        self.linear = nn.Linear(3, d_model)
        
        self.fcff = nn.Sequential(
          
            nn.Linear(512, 1200, device = device),
            nn.BatchNorm1d(1200),
            nn.Dropout1d(p = p),
            nn.LeakyReLU(neg_slope),
            
            nn.Linear(1200, 450, device = device),
            nn.BatchNorm1d(450),
            nn.Dropout1d(p = p),
            nn.LeakyReLU(neg_slope),

            nn.Linear(450, 30, device = device),
            nn.BatchNorm1d(30),
            nn.Dropout1d(p = p),
            nn.LeakyReLU(neg_slope),

            nn.Linear(30, 2,  device = device),
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, 0, 0.1)
            if module.bias is not None:
                module.bias.data.fill_(0.08)
        
    
    def forward(self, x1, x2):

        x1 = x1.permute(1, 0, 2)
        x1 = self.linear(x1)
        x1 = self.pos_encoder(x1 * torch.sqrt(torch.tensor(self.dmodel)))
        x1 = self.transformer_encoder(x1)
        x1 = x1.permute(1, 0, 2)
        x1 = torch.flatten(x1, 1)
        x1 = self.fcff(x1)

        x2 = x2.permute(1, 0, 2)
        x2 = self.linear(x2)
        x2 = self.pos_encoder(x2 * torch.sqrt(torch.tensor(self.dmodel)))
        x2 = self.transformer_encoder(x2)
        x2 = x2.permute(1, 0, 2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fcff(x2)
        
        return torch.stack([x1, x2])

class MyLoss (nn.Sequential):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, W, E, ALPHA):
        
        xw, yw = torch.nonzero(W, as_tuple = True)
        
        W = W[xw,yw]
        E = E[xw,yw]

        loss = torch.mean(ALPHA * (torch.abs(E-W)/W) + (1 - ALPHA) * (torch.abs(E-W)/(E + 2e-2)))

        return loss

def Dist(W, E):
    
    x, y = torch.nonzero(W, as_tuple = True)
    
    W = W[x,y]
    E = E[x,y]
    
    dist = E/W
    
    return dist

def Train(train, W_dist_train, device, n_batches_train, ALPHA):
    losses = []

    for batch_x in range (n_batches_train):
        for batch_y in range (n_batches_train):
            if torch.count_nonzero(W_dist_train[batch_x][batch_y]) != 0:
                
                sample_emb_train = model(train[batch_x], train[batch_y]).to(device)
                E_dist_train = torch.cdist(sample_emb_train[0], sample_emb_train[1], p = 2).to(device) 

                loss = criterion(W_dist_train[batch_x][batch_y], E_dist_train, ALPHA)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.requires_grad_()
                loss = loss.to(torch.float)
                loss.backward(retain_graph=True)

                optimizer.step()
    
    return losses

def Validation(validation, W_dist_validation, device, ALPHA):
    
    sample_emb_validation = model(validation, validation).to(device)
    E_dist_validation = torch.cdist(sample_emb_validation[0], sample_emb_validation[0], p = 2).to(device)
    valid_loss = criterion(W_dist_validation, E_dist_validation, ALPHA)
    
    return valid_loss

def Test(test, W_dist_test, ALPHA, device):
    
    sample_emb_test = model(test, test).to(device)
    E_dist_test = torch.cdist(sample_emb_test[0], sample_emb_test[1], p = 2).to(device)
    test_loss = criterion(W_dist_test, E_dist_test, ALPHA)
    print(f'Test error {test_loss}')
    test_dist = Dist(W_dist_test, E_dist_test)

    median_test = torch.median(test_dist)
    
    fig3, ax3_test = plt.subplots(1,1, figsize=(7,7))

    ax3_test.set_title('Test set distortion', fontsize = 15)
    ax3_test.set_xlim([-0.1,3])
    ax3_test.set_xlabel("Distortion")
    ax3_test.set_ylabel("Density")
    ax3_test.tick_params(axis='both', which='major')
    labels = ['Ideal']

    count_test, bins_test, p_test = ax3_test.hist(test_dist.cpu().detach().numpy(), color = 'b', bins = 'auto', density = True, edgecolor = 'b')
    ax3_test.axvline(x=1, color='r', linestyle='dashed')

    ax3_test.legend(labels, fontsize = 12, loc = 'upper right', edgecolor = 'black',  bbox_to_anchor=(1, 1), bbox_transform=ax3_test.transAxes)
    anchored_text_test = AnchoredText('Median:'+str('%.2f' % median_test.item()), loc='upper right', bbox_to_anchor = (0.99,0.93), bbox_transform=ax3_test.transAxes)
    ax3_test.add_artist(anchored_text_test)
    
    return sample_emb_test[0]

def DataLoader(type: str, device):
    if type == 'train':

        data, _ = torch.load('/data/gabrieleb/data/Jets/dataset_train_real', map_location=device)
        W_dist, _ = torch.load('/data/gabrieleb/data/Jets/Wasserstein_dist_train_real_s=12000', map_location=device)

        data = data.to(torch.float)

        return [data, W_dist]
    
    elif type == 'validation':

        data, _  = torch.load('/data/gabrieleb/data/Jets/dataset_validation_real', map_location=device)
        W_dist, _ = torch.load('/data/gabrieleb/data/Jets/Wasserstein_dist_validation_real_s=2400', map_location=device)

        data = data.to(torch.float)

        return [data, W_dist]

    elif type == 'test':
        
        device = 'cpu'
        data, _ = torch.load('/data/gabrieleb/data/Jets/dataset_test_real', map_location = device)
        W_dist, targets_test = torch.load('/data/gabrieleb/data/Jets/Wasserstein_dist_test_real_s=4000', map_location = device)

        data = data.to(torch.float)

        return [data, W_dist, targets_test]

def Batches_Maker(train, W_dist_train, batch_size: int):

    n_sample_train = train.size(0)
    n_batches_train = int(n_sample_train/batch_size)

    train = torch.stack(torch.chunk(train, n_batches_train, dim = 0), dim = 0)
    W_dist_train = torch.stack(torch.chunk(torch.stack(torch.chunk(W_dist_train, n_batches_train, dim = -1), dim = 0), n_batches_train, dim = 1), dim = 0)

    return [train, W_dist_train]

def Embedding_plot(sample_emb_test, targets_test):

    labels = ['1','2','3','4']
    colors = ['g', 'r','c','m']

    var, col = [], []

    fig, ax = plt.subplots(1,1, figsize = (7,7))

    fig.suptitle('Embedded space', fontsize = 15)
    ax.set_xlabel('Embedded x')
    ax.set_ylabel('Embedded y')

    for j in [1,2,3,4]:
        for i in torch.nonzero(targets_test == j)[:,0]:
            ax.scatter(sample_emb_test[i][0].cpu().detach().numpy(), sample_emb_test[i][1].cpu().detach().numpy(), color = colors[j-1], s=10, alpha = 0.5)

        var.append(labels[j-1])
        col.append(colors[j-1])
        
    ax.legend(var,loc = 'best', fontsize = 12)

    leg = ax.get_legend()    
    for i in range(len(var)):

        leg.legend_handles[i].set_color(col[i])
        
    var, col = [], []

def KDE_plot(sample_emb_test, targets_test):

    labels = ['1','2','3','4']
    colors = ['g', 'r','c','m']

    fig, ax = plt.subplots(1,1, figsize = (7,7))
    fig.suptitle('Kernel density plot', fontsize = 15)
    ax.set_xlabel('Embedded x')
    ax.set_ylabel('Embedded y')

    lab, h = [], []

    for k in [1,2,3,4]:
        ind = torch.count_nonzero(targets_test == k)
        X = torch.zeros(1,1).expand(ind, 2).clone()

        l=0
        for i in torch.nonzero(targets_test == k)[:,0]:
            X[l] = sample_emb_test[i]
            l = l+1

        # Extract x and y
        x = X[:, 0]
        y = X[:, 1]

        # Define the borders
        deltaX = (torch.max(x) - torch.min(x))/5
        deltaY = (torch.max(y) - torch.min(y))/5
        xmin = torch.min(x).item() - deltaX.item()
        xmax = torch.max(x).item() + deltaX.item()
        ymin = torch.min(y).item() - deltaY.item()
        ymax = torch.max(y).item() + deltaX.item()

        # Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]

        x = x.detach().numpy()
        y = y.detach().numpy()

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])

        kernel = st.gaussian_kde(values)

        f = np.reshape(kernel(positions).T, xx.shape)

        cset = ax.contour(xx, yy, f, levels = [10], colors = colors[k-1])

        h1, l1 = cset.legend_elements()
        h.append(h1[0])
        lab.append(labels[k-1])

    ax.legend(h, lab, loc = 'best', fontsize = 12)
    lab, h = [], []

def Euclidean_plotter(sample_emb_test, targets):

    E = torch.triu(torch.cdist(sample_emb_test, sample_emb_test, p = 2))
    fig2, ax2 = plt.subplots(1,4,figsize = (28,7))

    xtitle = 'Euclidean Distance'
    ytitle = 'Density'

    fig2.suptitle('Euclidean distance', fontsize = 20)

    ax2[0].set_xlabel(xtitle)
    ax2[0].set_ylabel(ytitle)
    ax2[0].tick_params(axis='both', which='major')
    
    ax2[1].set_xlabel(xtitle)
    ax2[1].set_ylabel(ytitle)
    ax2[1].tick_params(axis='both', which='major')

    ax2[2].set_xlabel(xtitle)
    ax2[2].set_ylabel(ytitle)
    ax2[2].tick_params(axis='both', which='major')
    
    ax2[3].set_xlabel(xtitle)
    ax2[3].set_ylabel(ytitle)
    ax2[3].tick_params(axis='both', which='major')
    
    labels = [['1-1','1-2','1-3', '1-4'],['2-1','2-2','2-3', '2-4'],['3-1','3-2','3-3', '3-4'], ['4-1','4-2','4-3', '4-4']]#jets

    color_1 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    color_2 = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']
    color_3 = ['#2ca02c', '#d62728', '#1f77b4','#ff7f0e']
    color_4 = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']

    v = []
    l = 0

    ind = torch.nonzero(targets == 1)[:,0]
    for val in [1,2,3,4]:
        ind_target = torch.nonzero(targets == val)[:,0]
        for i in ind:
            for j in filter(lambda h: h>i, ind_target):
                v.append(E[i][j].item())
        ax2[0].hist(v, bins = 'auto', linewidth = 0.8, alpha = 0.5, density = True, color = color_1[l-1])
        ax2[0].legend(labels[0])
        l = l+1

        v = []

    l = 0
    ind = torch.nonzero(targets == 2)[:,0]
    for val in [1,2,3,4]:

        ind_target = torch.nonzero(targets == val)[:,0]
        for i in ind:
            for j in filter(lambda h: h>i, ind_target):
                v.append(E[i][j].item())
        ax2[1].hist(v, bins = 'auto', linewidth = 0.8, alpha = 0.5, density = True, color = color_2[l-1])
        ax2[1].legend(labels[1])
        l = l+1

        v =[]

    l = 0
    ind = torch.nonzero(targets == 3)[:,0]
    for val in [1,2,3,4]:
        ind_target = torch.nonzero(targets == val)[:,0]
        for i in ind:
            for j in filter(lambda h: h>i, ind_target):
                v.append(E[i][j].item())
        ax2[2].hist(v, bins = 'auto', linewidth = 0.8, alpha = 0.5, density = True, color = color_3[l-1])
        ax2[2].legend(labels[2])
        l = l+1

        v = []
        
    l = 0
    ind = torch.nonzero(targets == 4)[:,0]
    for val in [1,2,3,4]:
        ind_target = torch.nonzero(targets == val)[:,0]
        for i in ind:
            for j in filter(lambda h: h>i, ind_target):
                v.append(E[i][j].item())
        ax2[3].hist(v, bins = 'auto', linewidth = 0.8, alpha = 0.5, density = True, color = color_4[l-1])
        ax2[3].legend(labels[3])
        l = l+1

        v = []
    del l

#Device setting
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#Data loading
train, W_dist_train = DataLoader('train', device)
validation, W_dist_validation  = DataLoader('validation', device)

batch_size = 300
train, W_dist_train = Batches_Maker(train, W_dist_train, batch_size)

#Transformer parameters
d_model = 32
num_layers = 2
n_head = 4
pos_dropout = 0.1
neg_slope = 1e-3
p = 0.1

model = Transformer(d_model, num_layers, p, neg_slope, n_head, pos_dropout).to(device)

#Optimaizer parameters
learning_rate = 1e-3

optimizer = optim.Adam(model.parameters(),lr = learning_rate, weight_decay = 0)

#Loss
criterion = MyLoss()
ALPHA = 0.2

#Early stopping 
patience = 30

early_stopping = EarlyStopping(patience = patience, verbose = True)

#Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.5)

#Train
iteration = 1000

fig = plt.figure(figsize = (14, 7), constrained_layout=False)
gs = GridSpec(1, 2, figure = fig)

ax3 = fig.add_subplot(gs[0, 0])
ax3.set_title('Training loss average', fontsize = 15)
ax3.set_ylabel('Training loss')
ax3.set_ybound(lower = 0, upper = None)

ax4 = fig.add_subplot(gs[0, 1])
ax4.set_title('Validation loss', fontsize = 15)
ax4.set_ylabel('Validation loss')
ax4.set_ybound(lower = 0, upper = None)

for ite in range (iteration):
    #Train
    model.train()
    train_losses = Train(train, W_dist_train, device, train.size(0), ALPHA)
    train_loss = np.average(train_losses)
    ax3.scatter(ite, train_loss, color = 'b', s = 5)

    #Validation
    model.eval()
    valid_loss = Validation(validation, W_dist_validation, device, ALPHA)
    ax4.scatter(ite, valid_loss.cpu().detach().numpy(), color = 'b', s = 5)

    early_stopping(valid_loss, model)
    scheduler.step()


    if early_stopping.early_stop:
            print("Early stopping")
            break

#Test
test, W_dist_test, targets_test = DataLoader('test', 'cpu')
model.load_state_dict(torch.load('/home/gabrieleb/GitHub/Checkpoint/checkpoint_transformer_simple.pth', map_location = 'cpu'))

model.eval()
model.to('cpu')
sample_emb_test = Test(test, W_dist_test, ALPHA, device = 'cpu')

#Embedding space plot
Embedding_plot(sample_emb_test, targets_test)

#Kernel density plot
KDE_plot(sample_emb_test, targets_test)

#Euclidean distance plot
Euclidean_plotter(sample_emb_test, targets_test)