#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import ot

#GPU
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def wasserstein_dist(data_source, data_target):#1-Wasserstein
    
    source = data_source[:,0]/torch.sum(data_source[:,0], dtype = torch.double)
    target = data_target[:,0]/torch.sum(data_target[:,0], dtype = torch.double)
    
    source = source.to(device)
    target = target.to(device)
    
    M = ot.dist(torch.stack([data_source[:,1], data_source[:,2]], dim = -1).to(torch.float), torch.stack([data_target[:,1], data_target[:,2]], dim = -1).to(torch.float), metric = 'euclidean').to(device)
    
    T = ot.emd(source, target, M).to(device)
    W = torch.sum(T*M).to(device)
    
    return W


# In[4]:


jets, targets = torch.load('data/Jets/dataset_train_real', map_location = device)
jets = jets.to(torch.double)


# In[5]:


#Wasserstein distance 

size = jets.size(0)

Wasserstein_dist=torch.zeros(size, size, dtype = torch.double).to(device)

for i in range (size):
    for j in filter(lambda h: h>i, range (size)):
        Wasserstein_dist[i][j] = wasserstein_dist(jets[i], jets[j]).to(device)
    print('riga n:',(i/size)*100)
#Wasserstein_dist = Wasserstein_dist/torch.max(Wasserstein_dist)


# In[38]:


torch.save([Wasserstein_dist, targets],'data/Jets/Wasserstein_dist_train_real_s='+str(size))

