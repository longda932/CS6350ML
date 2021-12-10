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
import torch.utils.data as Data

training_rate=0.001
epoch=1000

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
sig=nn.Sigmoid()
def forward(X,W1,W2,W3):
    S1=torch.mm(W1,X).view(-1)
    Z1=sig(S1).view(-1)


    S2=torch.mm(W2,torch.cat((Z1.view(-1,1),torch.ones(1).view(1,1)),axis=0)).view(-1)



    Z2=sig(S2).view(-1)




    return Z1,Z2,torch.mm(W3,torch.cat((Z2.view(-1,1),torch.ones(1).view(1,1)),axis=0)).view(-1)

def cal_loss(X,Y):
    return (0.5*(X-Y)**2).sum()

def cal_grad(W2,W3,X,Y,num_nodes,num_in,YY,Z1,Z2):
    dW1=torch.zeros(num_nodes,num_in+1)
    dW2=torch.zeros(num_nodes,num_nodes+1)
    dW3=torch.zeros(1,num_nodes+1)
    
    temp1=YY-Y
    
    dW3[0,:-1]=(temp1*Z2)
    dW3[0,-1]=temp1*1
    # print("S2",S2)
    
    dW2[:,:-1]=torch.mm(((temp1)*W3[0,:-1]*Z2*(1-Z2)).view(-1,1),Z1.view(1,-1))
    dW2[:,-1]=(temp1)*W3[0,:-1]*Z2*(1-Z2)
    
    dW1[:,:-1]=torch.mm((((temp1*W3[0,:-1]*Z2*(1-Z2)).view(1,-1)*W2[:,:-1].t()).sum(axis=1).view(-1)*Z1*(1-Z1)).view(-1,1) ,X.view(1,-1))
    dW1[:,-1]=(((temp1*W3[0,:-1]*Z2*(1-Z2)).view(1,-1)*W2[:,:-1].t()).sum(axis=1).view(-1)*Z1*(1-Z1))
    return dW1,dW2,dW3
    
file1="/home/zenus/Da/Neural Networks/bank-note/train.csv"
file2="/home/zenus/Da/Neural Networks/bank-note/test.csv"
train_loader,train_loader2,test_loader=load_data3(file1,file2)

def cal_learning_rate(a,b,t):
    return a/(1+(a/b)*t)



num_in=4



def training(a,b,width,epoch):
    num_nodes=width
    W1=torch.randn(num_nodes,num_in+1,requires_grad=True)
    W2=torch.randn(num_nodes,num_nodes+1,requires_grad=True)
    W3=torch.randn(1,num_nodes+1,requires_grad=True)

    t=1

    for i in range(epoch):
        loss=0
    
        for time,(data) in enumerate(train_loader):

            batch_x2=data[:,:-1].view(-1,1).float()
            batch_x=torch.cat((data[:,:-1].view(-1,1).long(),torch.ones(1,1)),axis=0)

            batch_y=data[:,-1].view(-1)
            Z1,Z2,out=forward(batch_x,W1,W2,W3)
            lr=cal_learning_rate(a,b,t)
            dw1,dw2,dw3=cal_grad(W2,W3,batch_x2,batch_y,num_nodes,num_in,out,Z1,Z2)
            
    
            W1=W1-lr*dw1
            W2=W2-lr*dw2
            W3=W3-lr*dw3
            loss=loss+cal_loss(out,batch_y)
            
        
        
        print("LOSS: ",loss)
        t=t+1
    print("Training error: ",cal_error(train_loader2,W1,W2,W3))
    print("Test error: ",cal_error(test_loader,W1,W2,W3))
    return W1,W2,W3


def forward2(X,W1,W2,W3):
    S1=torch.mm(W1,X)
    Z1=sig(S1)


    S2=torch.mm(W2,torch.cat((Z1,torch.ones(1,Z1.shape[1])),axis=0))



    Z2=sig(S2)




    return torch.mm(W3,torch.cat((Z2,torch.ones(1,Z2.shape[1])),axis=0)).view(-1)
    


def training2(a,b,width,epoch):
    num_nodes=width
    W1=torch.zeros(num_nodes,num_in+1,requires_grad=True)
    W2=torch.zeros(num_nodes,num_nodes+1,requires_grad=True)
    W3=torch.zeros(1,num_nodes+1,requires_grad=True)

    t=1

    for i in range(epoch):
        loss=0
    
        for time,(data) in enumerate(train_loader):

            batch_x2=data[:,:-1].view(-1,1).float()
            batch_x=torch.cat((data[:,:-1].view(-1,1).long(),torch.ones(1,1)),axis=0)

            batch_y=data[:,-1].view(-1)
            Z1,Z2,out=forward(batch_x,W1,W2,W3)
            lr=cal_learning_rate(a,b,t)
            dw1,dw2,dw3=cal_grad(W2,W3,batch_x2,batch_y,num_nodes,num_in,out,Z1,Z2)
            
    
            W1=W1-lr*dw1
            W2=W2-lr*dw2
            W3=W3-lr*dw3
            loss=loss+cal_loss(out,batch_y)
            
        
    
        print("LOSS: ",loss)
        t=t+1
    print("Training error: ",cal_error(train_loader2,W1,W2,W3))
    print("Test error: ",cal_error(test_loader,W1,W2,W3))
    return W1,W2,W3
            
    
    
def cal_error(test_loader,W1,W2,W3):
    for time,(data) in enumerate(test_loader):
        
        batch_x=torch.cat((data[:,:-1].float(),torch.ones(data.shape[0],1)),axis=1)
        batch_y=data[:,-1].view(-1)
        out=forward2(batch_x.t(),W1,W2,W3)
        r=0
        w=0
        
    for i in range(batch_y.shape[0]):
        if(out[i]<=0.5 and batch_y[i]==0):
            r=r+1
        elif(out[i]>0.5 and batch_y[i]==1):
            r=r+1
        else:
            w=w+1
    return w/(r+w)
                
        
        
    
    
    
    


W1,W2,W3=training(1,30,5,250) #Training: Test:

# W1,W2,W3=training(1,30,10,200) #Training: Test:

# W1,W2,W3=training(1,30,25,200) #Training: Test:

# W1,W2,W3=training(1,30,50,200) #Training: Test:

# W1,W2,W3=training(1,30,100,200) #Training: Test:



# W1,W2,W3=training2(1,30,5,200) #Training: Test:

# W1,W2,W3=training2(1,30,10,200) #Training: Test:

# W1,W2,W3=training2(1,30,25,200) #Training: Test:

# W1,W2,W3=training2(1,30,50,200) #Training: Test:

# W1,W2,W3=training2(1,30,100,200) #Training: Test:





# training(1,10,5)

# training(1,10,5)

# training(1,10,5)

# training(1,10,5)
 
    
    
    