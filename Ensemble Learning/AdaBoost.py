from os import error
import numpy as np
import math
from collections import Counter
import sys
import copy
from numpy.core.fromnumeric import transpose
from numpy.lib.shape_base import split
import matplotlib.pyplot as plt

train_set = list()
test_set = list()
labels=['yes','no']
T=500

def getColumn(list,k):
    n_row=len(list)
    newList=[]
    for i in range(n_row):
        newList.append(list[i][k])
    return newList

# import training and test data
with open('/Users/dalong/CS6350ML/DecisionTree/bank/train.csv', 'r') as f:
    for line in f :
        terms = line.strip( ).split(',')
        test_set.append(terms)

with open('/Users/dalong/CS6350ML/DecisionTree/bank/test.csv', 'r') as f:
    for line in f :
        terms = line.strip( ).split(',')
        train_set.append(terms)

train_labels=np.zeros(len(train_set))
test_lables=np.zeros(len(test_set))

for i in range(len(train_set)):
    if(train_set[i][-1]==labels[0]):
        train_labels[i]=1
    else:
        train_labels[i]=-1

for i in range(len(test_set)):
    if(test_set[i][-1]==labels[0]):
        test_lables[i]=1
    else:
        test_lables[i]=-1


age=np.median(np.array(getColumn(train_set,0)).astype(np.float))
balance=np.median(np.array(getColumn(train_set,5)).astype(np.float))
day=np.median(np.array(getColumn(train_set,9)).astype(np.float))
duration=np.median(np.array(getColumn(train_set,11)).astype(np.float))
campaign=np.median(np.array(getColumn(train_set,12)).astype(np.float))
pdays=np.median(np.array(getColumn(train_set,13)).astype(np.float))
previous=np.median(np.array(getColumn(train_set,14)).astype(np.float))

job=["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"]
marital=["married","divorced","single"]
education=["unknown","secondary","primary","tertiary"]
default=["yes","no"]
housing=["yes","no"]
loan=["yes","no"]
contact=["unknown","telephone","cellular"]
month=["jan", "feb", "mar", "apr","may","jun","jul","aug","sep","oct","nov", "dec"]
poutcome=["unknown","other","failure","success"]

values=[[age],job,marital,education,default,[balance],housing,loan,contact,[day],month,[duration],[campaign],[pdays],[previous],poutcome]
# print(values)
num_index=[0,5,9,11,12,13,14]
# print(poutcome)


features=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']


# Entropy function
def H(P):
    H = 0
    for i, N in enumerate(P):
        if(N!=0):
            H = H -N*math.log2(N)
    return H

# ME function
def ME(P):
    num=P.shape[0]
    ls = []
    for i in range(num):
        if(P[i]<=0.5):
            ls.append(P[i])
    return max(ls)

# GI function
def GI(P):
    result=0
    for i, N in enumerate(P):
        result=result+P[i]*P[i]
    return 1-result

# Entropy information gain
def Gain(p,k,P):
    # print("in Gain")
    # print(p)
    # print(k)
    # print(P)
    nr=P.shape[0]
    nc=P.shape[1]
    l=0
    for i in range(nr):
        l=l+k[i]*H(P[i,:])
    return H(p)-l

# ME gain
def Gain_ME(p,k,P):
    nr=P.shape[0]
    nc=P.shape[1]
    l=0
    for i in range(nr):
        l=l+k[i]*ME(P[i,:])
    return ME(p)-l

# GI gain
def Gain_GI(p,k,P):
    nr=P.shape[0]
    nc=P.shape[1]
    l=0
    for i in range(nr):
        l=l+k[i]*GI(P[i,:])
    return GI(p)-l

# 修这个,概率不一样了
# helper function used to calculate proportions passed to information gain functions
def proportions(feature_list,label_list,list,labels,num,weights): # num: 1 for numerical 0 for categorical

    n_labels=len(labels)
    if(num==1):
        n_values=2
    else:
        n_values=len(feature_list)

    n_examples=len(list)
    p = np.zeros([n_values,n_labels])
    p_labels=np.zeros(n_labels)
    
    if(num==0): #非数值
        for i in range(n_labels):
            # p_labels[i]=label_list.count(labels[i])/n_examples #这里数label的概率

            for i0 in range(n_examples):
                if(label_list[i0]==labels[i]):
                    p_labels[i]=p_labels[i]+weights[i0]
        for j in range(n_examples):
            for k in range(n_values): # values 
                if(list[j]==feature_list[k]):
                    # p_values[k]=p_values[k]+1
                    for z in range(n_labels):
                        if(label_list[j]==labels[z]):
                            # p[k,z]=p[k,z]+1 # 这里也要加
                            p[k,z]=p[k,z]+weights[j]
        p=p/(np.sum(p,axis=1).reshape(-1,1)+sys.float_info.epsilon)
    else:
        for i in range(n_labels):
            # p_labels[i]=label_list.count(labels[i])/n_examples #这里
            for i0 in range(n_examples):
                if(label_list[i0]==labels[i]):
                    p_labels[i]=p_labels[i]+weights[i0]
        for j in range(n_examples):
            # print(list[j])
            if(float(list[j])<=feature_list[0]):
                for z in range(n_labels):
                        if(label_list[j]==labels[z]):
                            # p[0,z]=p[0,z]+1 #这里
                            p[0,z]=p[0,z]+weights[j]
            else:
                for z in range(n_labels):
                        if(label_list[j]==labels[z]):
                            # p[1,z]=p[1,z]+1 #这里
                            p[1,z]=p[1,z]+weights[j]
        p=p/(np.sum(p,axis=1).reshape(-1,1)+sys.float_info.epsilon)
    # print("PPP")
    # print(p)
    # print("LABELS")
    # print(p_labels)

    return np.vstack((p_labels,p))
 
# 修改这个,概率不一样了
# helper function
def p_values(feature_list,list,num,weights): #不同feature比例
    if(num==0):
        n_values=len(feature_list)
    else:
        n_values=2
    p_values=np.zeros(n_values)
    n_examples=len(list)
    if(num==1):
        for j in range(n_examples):
            if(float(list[j])<=feature_list[0]):
                # p_values[0]=p_values[0]+1 #这里
                p_values[0]=p_values[0]+weights[j]
            else:
                #  p_values[1]=p_values[1]+1
                p_values[1]=p_values[1]+weights[j]
    else:
        for j in range(n_examples):
            for k in range(n_values):
                if(list[j]==feature_list[k]):
                    # p_values[k]=p_values[k]+1 #这里
                    p_values[k]=p_values[k]+weights[j]
    # p_values=p_values/(np.sum(p_values)+sys.float_info.epsilon)
    # print("P_VALUES")
    # print(p_values)
    return p_values

# get kth column of a 2-D list

# a class for tree node
class treeNode():
    def __init__(self,leaf=False,feature=None,value=None,children=None,label=None,level=0):
        self.leaf=leaf
        self.feature=feature
        self.children=children
        self.value=value
        self.label=label
        self.level=level
    def addChild(self,node):
        self.children.append(node)
    def addFeature(self,feature):
        self.feature=feature
    def addLabel(self,label):
        self.label=label
    def addLeaf(self):
        self.leaf=True
    def addLevel(self,level):
        self.level=level

# build tree function, f can be 'H' for Entropy gain, 'ME' for me gain and 'GI' for GI gain
def buildTree(examples,s_features,node,f,weights,level):
    # print(weights)
    # print("___________________________")
    s_labels=getColumn(examples,-1)
    if(len(set(s_labels))==1):
        node.addLabel(s_labels[0])
        node.addLeaf()
        return
    temp_fract=np.zeros(2)
    if(len(set(s_labels))!=1 and not s_features): 
        # print("HERE")
        for i in range(len(s_labels)):
            if(s_labels[i]==labels[0]):
                temp_fract[0]=temp_fract[0]+weights[i]
            else:
                temp_fract[1]=temp_fract[1]+weights[i]

        alabel = list(Counter(s_labels).most_common(1))[0][0]
        if(temp_fract[0]>=temp_fract[1]):
            alabel=labels[0]
        else:
            alabel=labels[1]
        node.addLabel(alabel)
        node.addLeaf()
        return
    
    n_features=len(s_features)
    v_Gains=np.zeros(n_features)
    for i in range(n_features): 
        index=features.index(s_features[i])
        x=getColumn(examples,index)
        y=getColumn(examples,-1)
        is_Num=0;
        if(index in num_index):
            # print(index)
            # print(values)
            is_Num=1
            # print(values[index])
            s_proportions=proportions(values[index],y,x,labels,1,weights)
        else:
            s_proportions=proportions(values[index],y,x,labels,0,weights)
        # print("S_PPPP")
        # print(p_values(values[index],x,is_Num,weights))
        if(f=='H'):
            v_Gains[i]=Gain(s_proportions[0,:],p_values(values[index],x,is_Num,weights),s_proportions[1:,:])
        if(f=='GI'):
            v_Gains[i]=Gain_GI(s_proportions[0,:],p_values(values[index],x,is_Num,weights),s_proportions[1:,:])
        if(f=='ME'):
            v_Gains[i]=Gain_ME(s_proportions[0,:],p_values(values[index],x,is_Num,weights),s_proportions[1:,:])
    temp=v_Gains.tolist()
    # print("GI__________________")
    # print(temp)
    max_index=temp.index(max(temp))
    # print(max(temp))
    # print(max_index)
    
    split_feature=s_features[max_index]
    # print(split_feature)
    f_index=features.index(split_feature)
    values_feature=values[f_index]
    n_examples=len(examples)
    n_feature=len(values_feature)
    # if(split_feature==None):
    #     print("SOS")
    node.addFeature(split_feature)

    dict={}
    if(f_index in num_index):
        is_Num=1
    else:
        is_Num=0
    # print(values_feature)
    weightDict={}

    if(is_Num==0):
        for j,N in enumerate(values_feature):
            dict[N]=list()
            weightDict[N]=list()
        for i in range(n_examples):
            for j in range(n_feature):
                if(values_feature[j]==examples[i][f_index]):
                    dict[values_feature[j]].append(examples[i])
                    weightDict[values_feature[j]].append(weights[i])
    else:
        dict[0]=list()
        dict[1]=list()
        weightDict[0]=list()
        weightDict[1]=list()
        for i in range(n_examples):
            for j in range(n_feature):
                if(float(examples[i][f_index])<=values_feature[0]):
                    dict[0].append(examples[i])
                    weightDict[0].append(weights[i])
                else:
                    dict[1].append(examples[i])
                    weightDict[1].append(weights[i])
    new_features=copy.copy(s_features)
    new_features.remove(split_feature)
    # print(split_feature)
    if(node.level==level):
        
        new_features=[]
    for key in dict:
        # print(key)
        if(len(dict[key])==0):
            # print("I am here")
            alabel = list(Counter(s_labels).most_common(1))[0][0]
            anode=treeNode(leaf=True,label=alabel,value=key)
            node.addChild(anode)
        else:         
            # print("LOL")
            anode=treeNode(value=key,label='not',children=list(),level=(node.level+1))
            node.addChild(anode)
            buildTree(dict[key],new_features,anode,f,weightDict[key],level)
            
# root=treeNode(children=list())
# buildTree(train_set,features,root,'GI')

# print tree
def printTree(node):
    if(node.children):
        print(node.feature)
        n_children=len(node.children)
        print(n_children)
        for i in range(n_children):
            printTree(node.children[i])
    else:
        if(node.leaf):
            print(node.label)

# predict function
def predict(x,root):
    cur=root
    while(not cur.leaf):
        # if(not cur.feature):
        #     print(cur.label)
        #     return cur.label
        f = cur.feature
        # print(f)
        index=features.index(f)
        if(index in num_index):
            v=x[index]
            if(float(v)<=values[index][0]):
                cur=cur.children[0]
            else:
                cur=cur.children[1]
        else:
            v=x[index]
            index2=values[index].index(v)
            cur=cur.children[index2]
    return cur.label

# training and output the prediction error
def training(set,f,level):
    n_train=len(set)
    root=treeNode(children=list())
    buildTree(train_set,features,root,f,level)
    right=0
    wrong=0

    for i in range(n_train):
        pre=predict(set[i],root)
        if(pre==set[i][-1]):
            right=right+1
        else:
            # print(pre)
            wrong=wrong+1
    
    return list([right,wrong,wrong/(right+wrong)])


def build_and_predict(weights,t):
    # print(t)
    n_train=len(train_set)
    root=treeNode(children=list())
    buildTree(train_set,features,root,'H',weights,2)
    predictions=np.zeros(n_train)
    right=0
    wrong=0

    for i in range(n_train):
        pre=predict(train_set[i],root)
        if(pre==train_set[i][-1]):
            right=right+weights[i]
            predictions[i]=1
        else:
            wrong=wrong+weights[i]
            predictions[i]=0
    
    return root,predictions, wrong/(right+wrong)


def predict_set(set,root,num_roots,fig_name):
    n_train=len(set)
    errors=np.zeros(num_roots)
    right=0
    wrong=0
    for i in range(num_roots):
        for j in range(n_train):
            pre=predict(set[j],root[i])
            if(pre==set[j][-1]):
                right=right+1
            else:
                wrong=wrong+1
        errors[i]=wrong/(wrong+right)
    fig=plt.figure(1)
    plt.scatter(range(num_roots),errors)
    temp_name="/Users/dalong/CS6350ML/Ensemble Learning/"+fig_name+".png"
    fig.savefig(temp_name)
    plt.show()



def generate_weights(error_t,T,num_examples,weights_t,predictions):
    if(T==1):
        return 1/num_examples * np.ones(num_examples)
    else:
        weights=np.zeros(num_examples)
        alpha_t=0.5*np.log((1-error_t)/(error_t+sys.float_info.epsilon))
        n_sum=0
        # print(alpha_t)
        for i in range(num_examples):
            if(predictions[i]==1):
                weights[i]=weights_t[i]*np.exp(-alpha_t)
                n_sum=n_sum+weights_t[i]*np.exp(-alpha_t)
            else:
                weights[i]=weights_t[i]*np.exp(alpha_t)
                n_sum=n_sum+weights_t[i]*np.exp(alpha_t)
        weights=weights/n_sum
        # print(weights.sum())
        return weights


n_examples=len(train_set)


# print(n_examples)
# print(weights)






# nodeList=list()
# votes_list=np.zeros(T)
# for t in range(T):
#     if(t==0):
#         weights,temp_a=generate_weights(0,1,n_examples,0,0)
#         votes_list[t]=temp_a
#     root,predictions_t,error_t=build_and_predict(weights,t)
#     nodeList.append(root)
#     weights,temp_a=generate_weights(error_t,t+1,n_examples,weights,predictions_t)
#     votes_list[t]=temp_a


def combined_tree_predict(nodeList,votes_list,sample,T):
    v=np.zeros((T))
    # v_2=np.zeros((T))
    for i in range(T):
        pre=predict(sample,nodeList[i])
        temp=0
        if(pre==labels[0]):
            temp=1
        else:
            temp=-1
        if(i==0):
            v[i]=votes_list[i]
        else:
            v[i]=v[i-1]+temp*votes_list[i]
        # if(i==0):
        #     v_1[i]=0
        #     v_2[i]=0
        # else:
        #     v_1[i]=v_1[i-1]
        #     v_2[i]=v_2[i-1]
        # c
        # if(pre==labels[0]):
        #     if(i==0):
        #         v_1[i]=votes_list[i]
        #     else:
        #         v_1[i]=v_1[i]+votes_list[i]
        # else:
        #     if(i==0):
        #         v_2[i]=votes_list[i]
        #     else:
        #         v_2[i]=v_2[i]+votes_list[i]
    return v


def learn_adanboost(T):
    nodeList=list()
    errors=np.zeros(T)
    votes_list=np.zeros(T)
    for t in range(T):
        print(t)
        if(t==0):
            weights=generate_weights(0,1,n_examples,0,0)
        root,predictions_t,error_t=build_and_predict(weights,t)
        nodeList.append(root)
        errors[t]=error_t
        weights=generate_weights(error_t,t+1,n_examples,weights,predictions_t)
        votes_list[t]=0.5*np.log((1-error_t)/(error_t+sys.float_info.epsilon))
    
 
    train_errors=np.zeros(T)
    test_errors=np.zeros(T)

    v=np.zeros((len(train_set),T))


    for i in range(len(train_set)):
        v[i,:]=combined_tree_predict(nodeList,votes_list,train_set[i],T)

    for t in range(T):
        r=0
        w=0
        for i in range(len(train_set)):
            pre=0
            if(v[i,t]>0):
                pre=labels[0]
            else:
                pre=labels[1]
            
            if(pre==train_set[i][-1]):
                r=r+1
            else:
                w=w+1
        train_errors[t]=w/(w+r)

    v=np.zeros((len(test_set),T))
    
    for i in range(len(test_set)):
        v[i,:]=combined_tree_predict(nodeList,votes_list,test_set[i],T)

    for t in range(T):
        r=0
        w=0
        for i in range(len(test_set)):
            pre=0
            if(v[i,t]>0):
                pre=labels[0]
            else:
                pre=labels[1]
            
            if(pre==train_set[i][-1]):
                r=r+1
            else:
                w=w+1
        test_errors[t]=w/(w+r)


    t_e=np.zeros(T)
    t_e2=np.zeros(T)
    for i in range(T):
        w=0
        r=0
        for j in range(len(train_set)):
            pre=predict(train_set[j],nodeList[i])
            if(pre==train_set[j][-1]):
                r=r+1
            else:
                w=w+1
        t_e[i]=w/(w+r)

    for i in range(T):
        w=0
        r=0
        for j in range(len(test_set)):
            pre=predict(test_set[j],nodeList[i])
            if(pre==test_set[j][-1]):
                r=r+1
            else:
                w=w+1
        t_e2[i]=w/(w+r)
    return train_errors,test_errors,errors,t_e,t_e2





train_errors,test_errors,errors,te,te2=learn_adanboost(T)

fig=plt.figure(1)
plt.scatter(range(T),train_errors,color='red',label="training errors")
plt.scatter(range(T),test_errors,color='green',label="test errors")
temp_name="/Users/dalong/CS6350ML/Ensemble Learning/figure1.png"
fig.savefig(temp_name)
plt.legend()
plt.show()

fig=plt.figure(2)
plt.scatter(range(T),te,color='red',label="training errors")
plt.scatter(range(T),te2,color='green',label="test errors")
temp_name="/Users/dalong/CS6350ML/Ensemble Learning/figure1.png"
fig.savefig(temp_name)
plt.legend()
plt.show()

fig=plt.figure(3)
plt.scatter(range(T),errors,color='red',label="weighted training errors")
temp_name="/Users/dalong/CS6350ML/Ensemble Learning/figure1.png"
fig.savefig(temp_name)
plt.legend()
plt.show()

# predict_set(train_set,nodeList,T,"train_set")
# predict_set(test_set,nodeList,T,"test_set")









    