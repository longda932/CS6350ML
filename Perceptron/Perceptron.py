import os
import numpy as np
import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.nn import Module, Parameter
import torch.distributions as dis
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.utils.data as Data
import datetime
import random
torch.manual_seed(4)


def load_data(shuffle):
    dir=os.getcwd()
    labelEncoder = LabelEncoder()
    file_loc1=dir+"/bank-note/train.csv"
    file_loc2=dir+"/bank-note/test.csv"
    rawdata1 = pd.read_csv(file_loc1,header=None)
    dataTorch1=torch.from_numpy(rawdata1.values).float()
    X1=dataTorch1[:,:-1]
    Y1=dataTorch1[:,-1]

    for i in range(X1.shape[0]):
        if(Y1[i]==0.0):
            Y1[i]=-1.0
    


    X1=torch.cat((X1,torch.ones(X1.shape[0],1)),1)
    data1=torch.cat((X1,Y1.view(-1,1)),1).float()


    rawdata2=pd.read_csv(file_loc2,header=None)
    dataTorch2=torch.from_numpy(rawdata2.values).float()
    X2=dataTorch2[:,:-1]
    Y2=dataTorch2[:,-1]
    for i in range(X2.shape[0]):
        if(Y2[i]==0.0):
            Y2[i]=-1.0
    X2=torch.cat((X2,torch.ones(X2.shape[0],1)),1)
    data2=torch.cat((X2,Y2.view(-1,1)),1).float()



    train_loader = Data.DataLoader(
    dataset=data1,      
    batch_size=data1.shape[0],     
    shuffle=shuffle,             
    num_workers=0,             
    )
    return data1,data2,train_loader




def standardPerceptron(T,r,train_loader):
    weights=torch.tensor([[0.0],[0.0],[0.0],[0.0],[0.0]])
    for i in range(T):
        for step,(data) in enumerate(train_loader):
            n_examples=data.shape[0]
            for j in range(n_examples):
                if(data[j,-1]*torch.mm(weights.t(),data[j,:-1].view(-1,1))<=0):
                    weights=weights+r*data[j,-1]*data[j,:-1].view(-1,1)
    return weights



def prediction_error(set,weights):
    n_examples=set.shape[0]
    right=0
    wrong=0
    for i in range(n_examples):
        if(torch.mm(weights.t(),set[i,:-1].view(-1,1))>0 and set[i,-1]==1.0):
            right=right+1
        elif(torch.mm(weights.t(),set[i,:-1].view(-1,1))<=0 and set[i,-1]==-1.0):
            right=right+1
        else:
            wrong=wrong+1
    return (wrong/(right+wrong))
        
            





def voted_perceptron(T,r,train_loader):
    weights=torch.tensor([[0.0],[0.0],[0.0],[0.0],[0.0]])
    m=0
    dict=torch.zeros(10000,5)
    dict[m,:]=weights.view(-1)
    c=torch.zeros(10000,1)
    c[m,0]=0
    for i in range(T):
        for step,(data) in enumerate(train_loader):
            n_examples=data.shape[0]
            for j in range(n_examples):
                if(data[j,-1]*torch.mm(weights.t(),data[j,:-1].view(-1,1))<=0):
                    weights=weights+r*data[j,-1]*data[j,:-1].view(-1,1)
                    m=m+1
                    # print(m)
                    dict[m,:]=weights.view(-1)
                    c[m,0]=1
                else:
                    c[m,0]=c[m,0]+1
    return c,dict,m
                    





def voted_prediction_error(set,t_c,t_dict,m):
    c=t_c[:m+1,0]
    dict=t_dict[:m+1,:]

    n_examples=set.shape[0]
    right=0
    wrong=0
    for i in range(n_examples):
        if(voted_pred_helper(c,dict,set[i,:-1].view(-1,1))>0 and set[i,-1]==1.0):
            # print("HERE")
            right=right+1
        elif(voted_pred_helper(c,dict,set[i,:-1].view(-1,1))<=0 and set[i,-1]==-1.0):
            # print("TERE")
            right=right+1
        else:
            wrong=wrong+1
    return (wrong/(right+wrong))

def voted_pred_helper(c,dict,point):
    # print(c)
    # print(c.sum())
    # print(dict.shape)
    temp=(c.view(-1)*(torch.mm(dict,point)).view(-1))
    # print((torch.mm(dict,point)).view(-1).shape)
    # print(temp.shape)
    # print(temp)
    temp1=temp.sum()

    if(temp1>0):
        return 1.0
    else:
        return -1.0




# train_data,test_data,train_loader=load_data(True)


def averaged_perceptron(T,r,train_loader):
    weights=torch.tensor([[0.0],[0.0],[0.0],[0.0],[0.0]])
    m=0
    dict=torch.zeros(10000,5)
    dict[m,:]=weights.view(-1)
    for i in range(T):
        for step,(data) in enumerate(train_loader):
            n_examples=data.shape[0]
            for j in range(n_examples):
                if(data[j,-1]*torch.mm(weights.t(),data[j,:-1].view(-1,1))<=0):
                    weights=weights+r*data[j,-1]*data[j,:-1].view(-1,1)
                    m=m+1
                    dict[m,:]=weights.view(-1)
    return dict,m

def averaged_prediction_error(set,t_dict,m):
    dict=t_dict[:m+1,:]

    n_examples=set.shape[0]
    right=0
    wrong=0
    for i in range(n_examples):
        if(averaged_pred_helper(dict,set[i,:-1].view(-1,1))>0 and set[i,-1]==1.0):
            right=right+1
        elif(averaged_pred_helper(dict,set[i,:-1].view(-1,1))<=0 and set[i,-1]==-1.0):
            right=right+1
        else:
            wrong=wrong+1
    return (wrong/(right+wrong))

def averaged_pred_helper(dict,point):
    
    temp=((torch.mm(dict,point)))
    temp1=temp.sum()
    

    if(temp1>0):
        return 1.0
    else:
        return -1.0







train_data,test_data,train_loader=load_data(True)


weights=standardPerceptron(10,1,train_loader)
print("Weight for standard perceptron with learning rate of 0.5:",weights)

error=prediction_error(test_data,weights)
print("Prediction error:",error)


c,dict,m=voted_perceptron(10,1,train_loader)
# print(m)

error1=voted_prediction_error(test_data,c,dict,m)
dict=dict[:1+m,:]
c=c[:1+m,:]
print(error1)
dict_np = dict.numpy()
df = pd.DataFrame(dict_np)
df.to_csv("weights",index=False)

c_np = c.numpy()
df = pd.DataFrame(c_np)
df.to_csv("counts",index=False)


dict,m=averaged_perceptron(10,1,train_loader)
error1=averaged_prediction_error(test_data,dict,m)
# print(m)
dict=dict[:1+m,:]

dict_np = dict.numpy()
df = pd.DataFrame(dict_np)
df.to_csv("weights2",index=False)

print(error1)



