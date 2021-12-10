import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import os

dir=os.getcwd()

def load_data3(file1=None,file2=None):
    fileName1=file1
    rawdata2 = np.loadtxt(fileName1,delimiter=",")
    dataTorch2=torch.from_numpy(rawdata2).float()
    size=dataTorch2.shape[0]
    for i in range(size):
        if(dataTorch2[i,-1]==0):
            dataTorch2[i,-1]=-1
    train_loader = Data.DataLoader(
    dataset=dataTorch2,      
    batch_size=1,     
    shuffle=True,             
    num_workers=0,
    )
    
    fileName1=file1
    rawdata2 = np.loadtxt(fileName1,delimiter=",")
    dataTorch2=torch.from_numpy(rawdata2).float()
    size=dataTorch2.shape[0]
    for i in range(size):
        if(dataTorch2[i,-1]==0):
            dataTorch2[i,-1]=-1
            
    train_loader2 = Data.DataLoader(
    dataset=dataTorch2,      
    batch_size=872,     
    shuffle=False,             
    num_workers=0,
    )


    fileName2=file2
    rawdata2 = np.loadtxt(fileName2,delimiter=",")
    dataTorch2=torch.from_numpy(rawdata2).float()
    size=dataTorch2.shape[0]
    for i in range(size):
        if(dataTorch2[i,-1]==0):
            dataTorch2[i,-1]=-1
    test_loader = Data.DataLoader(
    dataset=dataTorch2,      
    batch_size=500,     
    shuffle=False,             
    num_workers=0,
    )        
    
    return train_loader,train_loader2,test_loader

def cal_learning_rate(a,b,t):
    return a/(1+(a/b)*t)

def cal_grad(X,Y,W,var):
    temp=(-Y*torch.mm(W.t(),X)).exp()
    out=1/(1+temp)*temp*(-Y*X)+1/var*W
    return out

def cal_grad2(X,Y,W):
    temp=(-Y*torch.mm(W.t(),X)).exp()
    out=1/(1+temp)*temp*(-Y*X)
    return out

def forward(W,batch_x,Y):
    
    temp=(-Y*torch.mm(W.t(),batch_x)).exp()

    return 1/(1+temp)

def cal_error(test_loader,W):
    for time,(data) in enumerate(test_loader):
        
        # batch_x=data[:,:-1].float()
        batch_x=torch.cat((data[:,:-1].long(),torch.ones(data.shape[0],1)),axis=1)
        batch_y=data[:,-1].view(-1)
        temp=(-torch.mm(W.t(),batch_x.t())).exp()
        out=(1/(1+temp)).view(-1)
        r=0
        w=0
        break
   
    # print(batch_y.shape)
    for i in range(batch_y.shape[0]):
        if(out[i]<=0.5 and batch_y[i]==-1):
            # print(out[i])
            r=r+1
        elif(out[i]>0.5 and batch_y[i]==1):
            # print(out[i])
            r=r+1
        else:
            w=w+1
    return w/(r+w)

# def cal_error(test_loader,W1,W2,W3):
#     for time,(data) in enumerate(test_loader):
        
#         batch_x=torch.cat((data[:,:-1].float(),torch.ones(data.shape[0],1)),axis=1)
#         batch_y=data[:,-1].view(-1)
#         out=forward2(batch_x.t(),W1,W2,W3)
#         r=0
#         w=0
        
#     for i in range(batch_y.shape[0]):
#         if(out[i]<=0.5 and batch_y[i]==0):
#             r=r+1
#         elif(out[i]>0.5 and batch_y[i]==1):
#             r=r+1
#         else:
#             w=w+1
#     return w/(r+w)
file1=dir+"/bank-note/train.csv"
file2=dir+"/bank-note/test.csv"
train_loader,train_loader2,test_loader=load_data3(file1,file2)
epoch=200
def training(var,a,b):
    W=torch.zeros(5,1)
    t=1
    for i in range(epoch):
        loss=0
        lr=cal_learning_rate(a,b,t)
        for time,(data) in enumerate(train_loader):

            # batch_x2=data[:,:-1].view(-1,1).float()
            batch_x=torch.cat((data[:,:-1].view(-1,1).long(),torch.ones(1,1)),axis=0).view(-1,1)

            batch_y=data[:,-1].view(-1)
            
            
            
          
            
            dW=cal_grad(batch_x,batch_y,W,var)
            # print(dW)
            W=W-0.01*dW
            
            out=forward(W,batch_x,batch_y)
            
            loss=loss+out
            
        
        # print("W",W)
        # print("LOSS: ",loss)
        t=t+1
    print("MAP Setting: ","var: ",var," Training error: ",cal_error(train_loader2,W)," Test error: ",cal_error(test_loader,W))
    # print("Training error: ",cal_error(train_loader2,W))
    # print("Test error: ",cal_error(test_loader,W))

var=[0.01,0.1,0.5,1,3,5,10,100]
for i in range(8):
    
    training(var[i],0.01,0.1)
    
def training2(a,b):
    W=torch.zeros(5,1)
    t=1
    for i in range(epoch):
        loss=0
        lr=cal_learning_rate(a,b,t)
        for time,(data) in enumerate(train_loader):

            # batch_x2=data[:,:-1].view(-1,1).float()
            batch_x=torch.cat((data[:,:-1].view(-1,1).long(),torch.ones(1,1)),axis=0).view(-1,1)

            batch_y=data[:,-1].view(-1)
            
            
            
            # out=forward(W,batch_x)
            
            dW=cal_grad2(batch_x,batch_y,W)
            # print(dW)
            W=W-lr*dW
            
            # out=forward(W,batch_x)
            
            # loss=loss+out
            
        
        # print("W",W)
        # print("LOSS: ",loss)
        t=t+1
    print("MLE Setting: ")
    print("Training error: ",cal_error(train_loader2,W))
    print("Test error: ",cal_error(test_loader,W))
    
training2(0.01,0.1)