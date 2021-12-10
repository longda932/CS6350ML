import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.nn import Module, Parameter
import pandas as pd
import numpy as np
import sys
import torch
import numpy as np

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import torch.optim as optim
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.nn import Module, Parameter
import pandas as pd
import torch.utils.data as Data
import sys
import numpy as np


class DNN(nn.Module):
    def __init__(self,depth,width):
        super().__init__()
        self.moduleList=nn.ModuleList()
        self.moduleList.append(nn.Linear(4, width,bias=True))
        self.moduleList.append(nn.Tanh())
        for i in range(depth-2):
            self.moduleList.append(nn.Linear(width, width,bias=True))
            self.moduleList.append(nn.Tanh()) 
        self.moduleList.append(nn.Linear(width, 1,bias=True))
        
    def forward(self,input):
        for m in self.moduleList:
            
            input=m(input)

        return input
        
        
        


def load_data3(file1=None,file2=None):
    fileName1=file1
    rawdata2 = np.loadtxt(fileName1,delimiter=",")
    dataTorch2=torch.from_numpy(rawdata2).float()
    train_loader = Data.DataLoader(
    dataset=dataTorch2,      
    batch_size=1,     
    shuffle=True,             
    num_workers=0,
    )
    
    fileName1=file1
    rawdata2 = np.loadtxt(fileName1,delimiter=",")
    dataTorch2=torch.from_numpy(rawdata2).float()
    train_loader2 = Data.DataLoader(
    dataset=dataTorch2,      
    batch_size=872,     
    shuffle=False,             
    num_workers=0,
    )


    fileName2=file2
    rawdata2 = np.loadtxt(fileName2,delimiter=",")
    dataTorch2=torch.from_numpy(rawdata2).float()
    test_loader = Data.DataLoader(
    dataset=dataTorch2,      
    batch_size=500,     
    shuffle=False,             
    num_workers=0,
    )        
    
    return train_loader,train_loader2,test_loader

file1="/home/zenus/Da/Neural Networks/bank-note/train.csv"
file2="/home/zenus/Da/Neural Networks/bank-note/test.csv"
train_loader,train_loader2,test_loader=load_data3(file1,file2)

def cal_error(test_loader,model):
    for time,(data) in enumerate(test_loader):
        
        batch_x=data[:,:-1].float()
        batch_y=data[:,-1].view(-1)
        out=model(batch_x).view(-1)
        r=0
        w=0
        break
   
        
    for i in range(batch_y.shape[0]):
        if(out[i]<=0.5 and batch_y[i]==0):
            r=r+1
        elif(out[i]>0.5 and batch_y[i]==1):
            r=r+1
        else:
            w=w+1
    return w/(r+w)

def initialize_weights_xavier(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data,gain=1)
def initialize_weights_he(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)

def training(depth,width,epoch,init):
    model=DNN(depth,width)
    if(init==0):
        initialize_weights_xavier(model)
    else:
        initialize_weights_he(model)
        
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    for i in range(epoch):
        loss=0
    
        for time,(data) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x=data[:,:-1].view(-1,1).float()
 

            batch_y=data[:,-1].view(-1)
            out=model(batch_x.t())
            L=0.5*(out-batch_y)**2
            L.backward()
            optimizer.step()
            
            
        
    if(init==0):
        inits="xavier"
    else:
        inits="he"
    print("The setting is: depth is: ",depth," width is: ",width," init: ",inits)
    print("Training error: ",cal_error(train_loader2,model))
    print("Test error: ",cal_error(test_loader,model))
    
    
depths=[3,5,9]
widths=[5,10,25,50,100]

for i in range(3):
    for j in range(5):
        for k in range(2):
            training(depths[i],widths[j],50,k)
    
            


