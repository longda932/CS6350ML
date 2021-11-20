from operator import matmul
import os
import torch
import pandas as pd
import torch.utils.data as Data
import numpy as np

from scipy.optimize import minimize, rosen, rosen_der

torch.manual_seed(4)

dir=os.getcwd()
def load_data(shuffle):
    dir=os.getcwd()
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
    batch_size=1,     
    shuffle=shuffle,             
    num_workers=0,             
    )
    loader2 = Data.DataLoader(
    dataset=data1,      
    batch_size=data1.shape[0],     
    shuffle=False,             
    num_workers=0,             
    )
    return data1,data2,train_loader,loader2

def get_r(r0,a,t):
    return r0/(1+r0/a*t)

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

def cal_X(X,Y,k):
    l1=X.shape[0]
    l2=Y.shape[0]
    temp=np.zeros((l1,l2))
    for i in range(l1):
        for j in range(l2):
            t1=np.exp(-np.linalg.norm(X[i,:]-Y[j,:],ord=2)/k)
            temp[i,j]=t1
    # print("HERE")

    return temp


# def cal_X2(set):


def prediction_error2(a,Y,big_x,b,set):
    le=Y.shape[0]
    set=np.array(set)
    a=a.reshape(le,1)

    t1=np.multiply(a,Y)



    n_examples=big_x.shape[1]
    right=0
    wrong=0
    for i in range(n_examples):

        t2=(np.multiply(t1,big_x[:,i].reshape(le,1))).sum()
        if((t2+b)>0 and set[i,-1]==1.0):
            right=right+1
        elif((t2+b)<=0 and set[i,-1]==-1.0):
            right=right+1
        else:
            wrong=wrong+1
    return (wrong/(right+wrong))



def SGD_SVM(C,train_loader,n_examples,loader2,r0,a,type=1):
    
    d1=4

    w=torch.zeros(d1+1,1)
    

    T=100

    b=0
    
    for i in range(T):
        for step,(data) in enumerate(train_loader):
            
            batch_x=data[:,:-1]
            batch_y=data[:,-1]
            w_temp=torch.cat((w[:-1,0].view(-1,1),torch.zeros(1,1)),0)
            w0=w[:-1,0].view(-1,1)

            if(type==1):
                lr=get_r(r0,a,T+1)
            # else:
            #     print("HERE")
                lr=r0/(1+T+1)
            if(batch_y*torch.mm(w.t(),batch_x.view(-1,1))<=1):
                
                w=w-lr*w_temp+lr*C*n_examples*batch_y*batch_x.view(-1,1)

            else:
                w=w-lr*w_temp
        for step,(data) in enumerate(loader2):
            batch_x=data[:,:-1]
            batch_y=data[:,-1]
            temp3=(1-batch_y.view(-1,1)*(torch.mm(w.t(),batch_x.t())).view(-1,1))
       
            # print(temp3)
            temp3=torch.where(temp3<0,torch.tensor(0.0),temp3)
            w0=w[:-1,0].view(-1,1)
            
            temp=0.5*torch.mm(w0.t(),w0)+C*temp3.sum()
            # print(temp)

        #     print(temp)
        # if(temp>0):
        #         J=0.5*torch.mm(w0.t(),w0)+C*n_examples*temp
        #         # print(J)
        #     else:
        #         J=0.5*torch.mm(w0.t(),w0)
    return w


# def SGD_SVM2(C,train_loader,n_examples):
    
#     d1=3

#     w=torch.zeros(d1+1,1)
    

#     T=1

#     b=0

#     lr=[0.01,0.005,0.0025]
    
#     for i in range(T):
#         k=0
#         for step,(data) in enumerate(train_loader):
            
#             batch_x=data[:,:-1]
#             batch_y=data[:,-1]
#             w_temp=torch.cat((w[:-1,0].view(-1,1),torch.zeros(1,1)),0)
#             w0=w[:-1,0].view(-1,1)

           
#             if(batch_y*torch.mm(w.t(),batch_x.view(-1,1))<=1):
                
#                 w=w-lr[k]*w_temp+lr[k]*C*n_examples*batch_y*batch_x.view(-1,1)
#                 print(w_temp-C*n_examples*batch_y*batch_x.view(-1,1))
#                 print("HERE")
#                 print(w)

#             else:
#                 w=w-lr[k]*w_temp
#             k=k+1
#         # for step,(data) in enumerate(loader2):
#         #     batch_x=data[:,:-1]
#         #     batch_y=data[:,-1]
#         #     temp3=(1-batch_y.view(-1,1)*(torch.mm(w.t(),batch_x.t())).view(-1,1))
       
#         #     # print(temp3)
#         #     temp3=torch.where(temp3<0,torch.tensor(0.0),temp3)
#         #     w0=w[:-1,0].view(-1,1)
            
#         #     temp=0.5*torch.mm(w0.t(),w0)+C*temp3.sum()
#         #     print(temp)

#         #     print(temp)
#         # if(temp>0):
#         #         J=0.5*torch.mm(w0.t(),w0)+C*n_examples*temp
#         #         # print(J)
#         #     else:
#         #         J=0.5*torch.mm(w0.t(),w0)
#     return w

data1,data2,loader,loader2=load_data(True)
# data1=torch.tensor([[0.5,-1,0.3,1,1],[-1,-2,-2,1,-1],[1.5,0.2,-2.5,1,1]])
# train_loader = Data.DataLoader(
#     dataset=data1,      
#     batch_size=1,     
#     shuffle=False,             
#     num_workers=0,             
# )

n_examples=data1.shape[0]
# print(n_examples)

Cs=torch.tensor([100/873,500/873,700/873])
# print(Cs[1])



w=SGD_SVM(Cs[0],loader,n_examples,loader2,0.005,0.0005,type=1)
print(w)

k=prediction_error(data1,w)
print(k)
k=prediction_error(data2,w)
print(k)

w=SGD_SVM(Cs[1],loader,n_examples,loader2,0.001,0.0005,type=1)
print(w)

k=prediction_error(data1,w)
print(k)
k=prediction_error(data2,w)
print(k)

w=SGD_SVM(Cs[2],loader,n_examples,loader2,0.0003,0.0004,type=1)
print(w)

k=prediction_error(data1,w)
print(k)
k=prediction_error(data2,w)
print(k)

w=SGD_SVM(Cs[0],loader,n_examples,loader2,0.005,0.0005,type=0)
print(w)

k=prediction_error(data1,w)
print(k)
k=prediction_error(data2,w)
print(k)

w=SGD_SVM(Cs[1],loader,n_examples,loader2,0.001,0.0005,type=0)
print(w)

k=prediction_error(data1,w)
print(k)
k=prediction_error(data2,w)
print(k)

w=SGD_SVM(Cs[2],loader,n_examples,loader2,0.0005,0.0003,type=0)
print(w)

k=prediction_error(data1,w)
print(k)
k=prediction_error(data2,w)
print(k)







def helper_func(a,X,Y):
    a=a.reshape((len(a),1))
    t1=np.matmul(np.multiply(a,Y),Y.T)

    t3=np.matmul(X,X.T)
    t4=np.multiply(t1,t3)

    t2=np.matmul(t4,a)
    
    # print("HERE")
    return 0.5*(t2.sum())-a.sum()




def helper_func2(a,X,Y):
    a=a.reshape((len(a),1))
    t1=np.matmul(np.multiply(a,Y),Y.T)
    t3=X
   
    t4=np.multiply(t1,t3)

    t2=np.matmul(t4,a)
    
    # print("HERE")
    return 0.5*(t2.sum())-a.sum()

    
# def con(args):
    # Y=args
    
    # cons = ({'type':'eq','fun': lambda a:(np.dot(a,Y)).sum()}
    #         )
    
    # return cons

def get_w(a,X,Y,n_examples):
    a=a.reshape(n_examples,1)
    t1=np.multiply(a,Y)
    t2=np.multiply(t1,X)
    
    return np.sum(t2,axis=0)

# def get_w2(a,X,Y,n_examples):
#     a=a.reshape(n_examples,1)
#     t1=np.multiply(a,Y)
#     t2=np.multiply(t1,X)
    
#     return np.sum(t2,axis=0)

def get_b(a,X,Y,n_examples,w):
    a=a.reshape(n_examples,1)
    lw=len(w)-1
    w0=(w[:-1]).reshape(lw,1)
    x_t=X.T
    t1=Y-(np.matmul(w0.T,x_t[:-1,:])).reshape(n_examples,1)
    n_vectors=0
    t_s=0
    for i in range(n_examples):
        if(a[i,0]>0):
            t_s=t_s+t1[i,0]
            n_vectors=n_vectors+1
    t2=t_s/n_vectors
    # print(n_vectors)
    # print(t2)
    return t2

def get_b2(a,Y,n_examples,bigX):
    


    a=a.reshape(n_examples,1)


    




  

    
    n_vectors=0
    t_s=0

    for i in range(n_examples):
        if(a[i,0]>0):
            t_s=t_s+Y[i,0]- (np.multiply(np.multiply(a,Y),bigX[:,i].reshape(n_examples,1))).sum()


            
            n_vectors=n_vectors+1
    t2=t_s/n_vectors
    print("The number of support vectors in this setting is: ",n_vectors)
    # print(n_vectors)
    # print(t2)
    return t2

    


def dual_SVM(data1,n_examples,C):
    a1=np.array(data1[:,:-1])
    a2=np.array(data1[:,-1].reshape(n_examples,1))
    args=a2.reshape(n_examples)
    n=0
    for i in range(n_examples):
        if(a2[i]==1.0):
            n=n+1
    
    # print("HERE")
    # print(n)
        
    # cons=con(args)

    cons=({'type':'eq','fun': lambda a:(np.dot(a,args)).sum()})
    bounds=[]
    for i in range(n_examples):
        bounds.append((0,C))

    result=minimize(fun=helper_func,x0=0*np.ones(n_examples),args=(a1[:,:-1],a2),method='SLSQP',bounds=bounds,constraints=cons)
    a=result.x
    # print(result)
    # print(len(a))
    w=get_w(a,a1,a2,n_examples)
    # print(w)
    b=get_b(a,a1,a2,n_examples,w)
    w[-1]=b
    print(w)
    w=w.reshape(5,1)
    print("The setting: C is ",C)
    print("The training error: is ",prediction_error(data1,torch.from_numpy(w).float()))
    print("The testing error: is ",prediction_error(data2,torch.from_numpy(w).float()))
    
    
  




dual_SVM(data1,n_examples,Cs[0])
dual_SVM(data1,n_examples,Cs[1])
dual_SVM(data1,n_examples,Cs[2])





def kernel_SVM(data1,n_examples,C,k):
    a2=np.array(data1[:,-1].reshape(n_examples,1))
    args=a2.reshape(n_examples)



    cons=({'type':'eq','fun': lambda a:(np.dot(a,args)).sum()})
    bounds=[]
    for i in range(n_examples):
        bounds.append((0,C))


    atemp=cal_X(data1[:,:-2],data1[:,:-2],k)

    big_x1=cal_X(data1[:,:-2],data1[:,:-2],k)
    big_x2=cal_X(data1[:,:-2],data2[:,:-2],k)


    result=minimize(fun=helper_func2,x0=0.1*np.ones(n_examples),args=(atemp,a2),method='SLSQP',bounds=bounds,constraints=cons)
    # print(result)
    a=result.x

    b=get_b2(a,a2,n_examples,atemp)

    print("Training error: ",prediction_error2(a,a2,big_x1,b,data1))
    print("Testing error: ",prediction_error2(a,a2,big_x2,b,data2))
    return a


ks=[0.1,0.5,1,5,100]
vs=np.zeros((15,n_examples))
z=0
for j in range(3):
    for i in range(5):
        print("Setting: C is ",Cs[j],", and r is ",ks[i])
        vs[z,:]=kernel_SVM(data1,n_examples,Cs[j],ks[i])
        z=z+1
tempdir=dir+"/supVectors.txt"
np.savetxt(tempdir,vs)

indexs=[5,6,7,8,9]



for i in range(4):
    row1=vs[indexs[i],:]
    row2=vs[indexs[i+1],:]
    n=0
    for j in range(n_examples):
        if(row1[j]!=0 and row2[j]!=0):
            n=n+1
    
    print("The number of overlapped support vectors between ",ks[i]," and ",ks[i+1]," is: ",n)



train_loader = Data.DataLoader(
    dataset=data1,      
    batch_size=1,     
    shuffle=False,             
    num_workers=0,             
)


def pre_p(n,Y,C,big_x,b,set):
   


    Y=(data1[:,-1].view(-1,1)).numpy()
    n_examples=big_x.shape[1]
    right=0
    wrong=0
    for i in range(n_examples):

        

        t1=((np.multiply(np.multiply(C,Y),big_x[:,i].reshape(n,1))).sum()+b)
      
        if(t1>0 and set[i,-1]==1.0):
            right=right+1
        elif(t1<=0 and set[i,-1]==-1.0):
            right=right+1
        else:
            wrong=wrong+1
    return (wrong/(right+wrong))
def kernelP(n_examples,train_loader,k,data1,data2):
    T=1000
    C=np.ones((n_examples,1))
    atemp=cal_X(data1[:,:-2],data1[:,:-2],k)

    big_x1=cal_X(data1[:,:-2],data1[:,:-2],k)
    big_x2=cal_X(data1[:,:-2],data2[:,:-2],k)


    X=data1[:,:-2]
    # print(data1.shape)
    Y=(data1[:,-1].view(-1,1)).numpy()
    for i in range(T):
        p=0
        for step,(data) in enumerate(train_loader):
            
            # batch_x=data[:,:-2]
            batch_y=(data[:,-1]).numpy()


            tb=(np.multiply(C,Y)).sum()
            # print(tb.shape)
            # print(np.multiply(np.multiply(C,Y),big_x1[:,p].reshape(n_examples,1)).shape)
            t1=Y[p,0]*((np.multiply(np.multiply(C,Y),big_x1[:,p].reshape(n_examples,1))).sum()+tb)

            if(t1<=0):
                C[p,0]=C[p,0]+1
            

            p=p+1

    
    tb=(np.multiply(C,Y)).sum()
    print("The setting r is ",k)
    print("The training error is : ",pre_p(n_examples,data1[:,-1].view(-1,1),C,big_x1,tb,data1))
    print("The test error is : ",pre_p(n_examples,data1[:,-1].view(-1,1),C,big_x2,tb,data2))
    
  

kernelP(data1.shape[0],train_loader,0.1,data1,data2)
kernelP(data1.shape[0],train_loader,0.5,data1,data2)
kernelP(data1.shape[0],train_loader,1,data1,data2)
kernelP(data1.shape[0],train_loader,5,data1,data2)
kernelP(data1.shape[0],train_loader,100,data1,data2)

            

            









