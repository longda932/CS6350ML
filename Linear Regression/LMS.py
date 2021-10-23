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
import torch.nn.functional as F
import random as rand


labelEncoder = LabelEncoder()
rawdata1 = pd.read_csv(r'//Users/dalong/CS6350ML/Linear Regression/concrete/train.csv',header=None).values

x1 = rawdata1[:,:-1].astype(float) 
Y1 = rawdata1[:,-1].astype(float)

rawdata2 = pd.read_csv(r'/Users/dalong/CS6350ML/Linear Regression/concrete/test.csv',header=None).values

x2 = rawdata2[:,:-1].astype(float) 
Y2 = rawdata2[:,-1].astype(float)



learning_rate=0.1

x1=torch.tensor(x1)
Y1=torch.tensor(Y1).view(-1,1)
x2=torch.tensor(x2)
Y2=torch.tensor(Y2).view(-1,1)

X1=torch.cat((x1,torch.ones(x1.size()[0],1)),1)
X2=torch.cat((x2,torch.ones(x2.size()[0],1)),1)



def cost_func(W,X,Y):
    temp1=Y-torch.mm(X,W)
    temp2=0.5*torch.dot(temp1.view(-1),temp1.view(-1))
    return temp2

def grad_func(W,X,Y):
    # print(X.size())
    # print(torch.mm(X,W).size())
    # print((Y.view(-1,1)-torch.mm(X,W)).size())
    temp1=2*(Y-torch.mm(X,W))*(-X)
    temp2=0.5*torch.sum(temp1,dim=0)

    return temp2.view(-1,1)

iteration=200000
def training_model(X,Y):
    W=torch.zeros(X.size()[1],1,dtype=float)
    learning_rate=0.5

    

    rate_set=False
    costs=torch.zeros(iteration)
    for i in range(iteration):
        
        grad_v=grad_func(W,X,Y)
        new_W=W-learning_rate*grad_v
        if(not rate_set):
            temp1=W-new_W
            temp2=temp1*temp1
            temp3=1/W.size()[0]*temp2.sum()
            # print(temp3)
            if(temp3<10**-4):
                rate_set=True
                print("Rate found")
                print(learning_rate)
            else:
                learning_rate=learning_rate*0.8
        if(rate_set):
            W=new_W
        else:
            W=W
        costs[i]=cost_func(W,X,Y)
    return learning_rate,W,costs
        

learning_rate,W,costs=training_model(X1,Y1)
print(learning_rate)
print(W)

fig=plt.figure(1)
plt.plot(range(iteration),costs)
plt.show()


print(cost_func(W,X2,Y2))



iteration=1000000
def GSD_func(X,Y):
    W=torch.zeros(X.size()[1],1,dtype=float)
    learning_rate=0.1

    

    rate_set=False
    costs=torch.zeros(iteration)
    for i in range(iteration):
        rand_num=rand.randint(0,X.size()[0]-1)
        grad_v=grad_func(W,X[rand_num,:].view(1,-1),Y[rand_num,:].view(1,-1))
        new_W=W-learning_rate*grad_v
        if(not rate_set):
            temp1=W-new_W
            temp2=temp1*temp1
            temp3=1/W.size()[0]*temp2.sum()
            # print(temp3)
            if(temp3<10**-4 and learning_rate<0.008):
                rate_set=True
                print("Rate found")
                print(learning_rate)
            else:
                learning_rate=learning_rate*0.9
        if(rate_set):
            W=new_W
        else:
            W=W
        costs[i]=cost_func(W,X,Y)
        # print(W)
    return learning_rate,W,costs

learning_rate,W,costs=GSD_func(X1,Y1)
print(learning_rate)
print(W)

fig=plt.figure(2)
plt.plot(range(iteration),costs)
plt.show()


print(cost_func(W,X2,Y2))

X1=X1.t()

temp1=torch.mm(X1,X1.t())

temp2=torch.inverse(temp1)

temp3=torch.mm(temp2,X1)

result=torch.mm(temp3,Y1)

print(result)