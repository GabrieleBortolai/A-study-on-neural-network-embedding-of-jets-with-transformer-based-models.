#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import ot
from multiprocessing import Pool
from torchvision import datasets, transforms

def set_device(Device):

    if Device == 'cpu':
        device = 'cpu'
        print(f"Using {device} device")
    
    elif Device == 'cuda':
        device = 'cuda'
        print(f"Using {device} device")
    
    return device

def wasserstein_dist(a,b,metric, d, device):
    
    source = a[torch.nonzero(a, as_tuple = False)[:,0], torch.nonzero(a, as_tuple = False)[:,1]].view(torch.nonzero(a, as_tuple = False).size(0))
    target = b[torch.nonzero(b, as_tuple = False)[:,0], torch.nonzero(b, as_tuple = False)[:,1]].view(torch.nonzero(b, as_tuple = False).size(0))
    
    source = source.to(device)
    target = target.to(device)
    
    source = source/torch.sum(source, dtype = torch.double)
    target = target/torch.sum(target, dtype = torch.double)
    
    M = ot.dist(torch.nonzero(a, as_tuple = False).to(torch.float), torch.nonzero(b, as_tuple = False).to(torch.float), metric = metric).to(device)
    M = M/d
    
    T = ot.emd(source, target, M).to(device)
    W = torch.sum(T*M).to(device).to(device)
    
    W = torch.sqrt(W)
    
    return W

def fun(data_source, data_target, metric, d, ind1, ind2, device):
    
    W = torch.zeros(data_source.size(0), data_target.size(0))
    
    if ind1 == ind2:
        for i in range(data_source.size(0)):
            for j in filter(lambda h: h>i, range(data_target.size(0))):

                W[i][j] = wasserstein_dist(data_source[i], data_target[j], metric, d, device)
                
    elif ind2 > ind1:
        
        for i in range(data_source.size(0)):
            for j in range(data_target.size(0)):
                
                W[i][j] = wasserstein_dist(data_source[i], data_target[j], metric, d, device)
            
    return W
            
def iterable(data, processes, metric, d, device):
    
    data = torch.stack(torch.chunk(data, processes, dim = 0), dim = 0)
    
    for i in range(data.size(0)):
        for j in range(data.size(0)):
            
            yield([data[i], data[j], metric, d, i, j, device])

def shuffle(dataset, targets):
    
    idx = torch.randperm(targets.size(0))
    
    dataset = dataset[idx]
    targets = targets[idx]
    
    
    return [dataset, targets]

def MNIST_DataLoader(type):

    if type == 'Train':

        data, targets = torch.zeros(60000, 28, 28, dtype = torch.float), torch.zeros(60000, dtype = torch.int)

        train_dataset = datasets.MNIST(root = './data/', train = True, download = True, transform = transforms.ToTensor())

        for i in range(60000):
            data[i] = train_dataset[i][0]
            targets[i] = train_dataset[i][1]

        data, targets = shuffle(data, targets)

    elif type == 'Test':

        data, targets = torch.zeros(60000, 28, 28, dtype = torch.float), torch.zeros(60000, dtype = torch.int)

        test_dataset = datasets.MNIST(root = './data/', train = False, transform = transforms.ToTensor()) 

        for i in range(60000):
            data[i] = test_dataset[i][0]
            targets[i] = test_dataset[i][1]

        data, targets = shuffle(data, targets)

    return [data, targets]

def Processing(data, size, type_process, processes, save):

    W_dist = torch.zeros(size, size, dtype = torch.float)
    metric = 'sqeuclidean'
    d = 2*28*28

    if type_process == 'multi':
        device = set_device('cpu')
        data = data.to(device)
        if data.size(0)%processes == 0:
            ite = iterable(data, processes, metric, d, device)

            with Pool(processes = processes) as p:
                W_dist = torch.cat(torch.chunk(torch.cat(p.starmap(fun, ite), dim = 1), processes, dim = 1), dim = 0)

        else: print('The size of the dataset should be divisible by the number of processes')

    elif type_process == 'GPU':

        device = set_device('cuda')
        data = data.to(device)

        for i in range(size):
            for j in filter(lambda h, h>i, size):
                
                W_dist[i][j] = wasserstein_dist(data[i], data[j], metric, d, device)

            if i%5 == 0:
                print(f'Raws processed {i/size}%')
    
    elif save == True:
        torch.save(W_dist, './data')

    return W_dist

data, targest = MNIST_DataLoader('Train')
type_process = ['multi', 'GPU']
processes = 10
save = False

W_dist = Processing(data, 50, type_process[0], processes, save)
