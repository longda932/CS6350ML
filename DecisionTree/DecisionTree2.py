import numpy as np
import math
from collections import Counter
import sys
import copy
from numpy.lib.shape_base import split

train_set = list()
test_set = list()
labels=['yes','no']


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
        train_set.append(terms)

with open('/Users/dalong/CS6350ML/DecisionTree/bank/test.csv', 'r') as f:
    for line in f :
        terms = line.strip( ).split(',')
        test_set.append(terms)


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


# print(age)
# print(balance)
# print(day)
# print(duration)
# print(campaign)
# print(pday)
# print(previous)


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

# helper function used to calculate proportions passed to information gain functions
def proportions(feature_list,label_list,list,labels,num): # num: 1 for numerical 0 for categorical

    n_labels=len(labels)
    if(num==1):
        n_values=2
    else:
        n_values=len(feature_list)

    n_examples=len(list)
    p = np.zeros([n_values,n_labels])
    p_labels=np.zeros(n_labels)
    
    if(num==0):
        for i in range(n_labels):
            p_labels[i]=label_list.count(labels[i])/n_examples
        for j in range(n_examples):
            for k in range(n_values): # values 
                if(list[j]==feature_list[k]):
                    # p_values[k]=p_values[k]+1
                    for z in range(n_labels):
                        if(label_list[j]==labels[z]):
                            p[k,z]=p[k,z]+1
        p=p/(np.sum(p,axis=1).reshape(-1,1)+sys.float_info.epsilon)
    else:
        for i in range(n_labels):
            p_labels[i]=label_list.count(labels[i])/n_examples
        for j in range(n_examples):
            # print(list[j])
            if(float(list[j])<=feature_list[0]):
                for z in range(n_labels):
                        if(label_list[j]==labels[z]):
                            p[0,z]=p[0,z]+1
            else:
                for z in range(n_labels):
                        if(label_list[j]==labels[z]):
                            p[1,z]=p[1,z]+1
        p=p/(np.sum(p,axis=1).reshape(-1,1)+sys.float_info.epsilon)
    # print(p)

    return np.vstack((p_labels,p))
 
# helper function
def p_values(feature_list,list,num):
    if(num==0):
        n_values=len(feature_list)
    else:
        n_values=2
    p_values=np.zeros(n_values)
    n_examples=len(list)
    if(num==1):
        for j in range(n_examples):
            if(float(list[j])<=feature_list[0]):
                p_values[0]=p_values[0]+1
            else:
                 p_values[1]=p_values[1]+1
    else:
        for j in range(n_examples):
            for k in range(n_values):
                if(list[j]==feature_list[k]):
                    p_values[k]=p_values[k]+1
    p_values=p_values/(np.sum(p_values)+sys.float_info.epsilon)
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
def buildTree(examples,s_features,node,f,level):
    s_labels=getColumn(examples,-1)
    if(len(set(s_labels))==1):
        node.addLabel(s_labels[0])
        node.addLeaf()
        return
    if(len(set(s_labels))!=1 and not s_features): 
        alabel = list(Counter(s_labels).most_common(1))[0][0]
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
            s_proportions=proportions(values[index],y,x,labels,1)
        else:
            s_proportions=proportions(values[index],y,x,labels,0)
        if(f=='H'):
            v_Gains[i]=Gain(s_proportions[0,:],p_values(values[index],x,is_Num),s_proportions[1:,:])
        if(f=='GI'):
            v_Gains[i]=Gain_GI(s_proportions[0,:],p_values(values[index],x,is_Num),s_proportions[1:,:])
        if(f=='ME'):
            v_Gains[i]=Gain_ME(s_proportions[0,:],p_values(values[index],x,is_Num),s_proportions[1:,:])
    temp=v_Gains.tolist()
    max_index=temp.index(max(temp))
    
    split_feature=s_features[max_index]
    
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
    if(is_Num==0):
        for j,N in enumerate(values_feature):
            dict[N]=list()
        for i in range(n_examples):
            for j in range(n_feature):
                if(values_feature[j]==examples[i][f_index]):
                    dict[values_feature[j]].append(examples[i])
    else:
        dict[0]=list()
        dict[1]=list()
        for i in range(n_examples):
            for j in range(n_feature):
                if(float(examples[i][f_index])<=values_feature[0]):
                    dict[0].append(examples[i])
                else:
                    dict[1].append(examples[i])
    new_features=copy.copy(s_features)
    new_features.remove(split_feature)
    if(node.level==level):
        new_features=[]
    for key in dict:
        if(len(dict[key])==0):
            alabel = list(Counter(s_labels).most_common(1))[0][0]
            anode=treeNode(leaf=True,label=alabel,value=key)
            node.addChild(anode)
        else:         
            anode=treeNode(value=key,label='not',children=list(),level=(node.level+1))
            node.addChild(anode)
            buildTree(dict[key],new_features,anode,f,level)
            
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


print("For each row, from left to right: the number of correct predictions, the number of wrong predictions, prediction errors")
print("Entropy information gain for predicting training set:")
a1=0.0
a2=0.0
a3=0.0

b1=0.0
b2=0.0
b3=0.0
for i in range(16):
    r=training(train_set,'H',i)
    a1=a1+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",a1/16 )

print("---------------------------------")
print("EM for predicting training set:")
for i in range(16):
    r=training(train_set,'EM',i)
    a2=a2+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",a2/16 )

print("---------------------------------")
print("GI for predicting training set:")
for i in range(16):
    r=training(train_set,'GI',i)
    a3=a3+r[2]
    print("Tree level: ",(i+1),r)

print("The everage prediction error is: ",a3/16 )

print("---------------------------------")
print("Entropy information gain for predicting testing set:")
for i in range(16):
    r=training(test_set,'H',i)
    b1=b1+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",b1/16 )

print("---------------------------------")
print("EM for predicting testing set:")
for i in range(16):
    r=training(test_set,'EM',i)
    b2=b2+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",b2/16 )

print("---------------------------------")
print("GI for predicting testing set:")
for i in range(16):
    r=training(test_set,'GI',i)
    b3=b3+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",b3/16 )

print("---------------------------------")


job=["admin.","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"]
marital=["married","divorced","single"]
education=["secondary","primary","tertiary"]
default=["yes","no"]
housing=["yes","no"]
loan=["yes","no"]
contact=["telephone","cellular"]
month=["jan", "feb", "mar", "apr","may","jun","jul","aug","sep","oct","nov", "dec"]
poutcome=["other","failure","success"]

values=[[age],job,marital,education,default,[balance],housing,loan,contact,[day],month,[duration],[campaign],[pdays],[previous],poutcome]

n=len(train_set)
n1=len(test_set)


nlist=[]
for i in range(n):
    nlist.append(train_set[i][8])
print(list(Counter(nlist).most_common(2)))
for i in range(n):
    if(train_set[i][15]=='unknown'):
        train_set[i][15]='failure'
    if(train_set[i][1]=='unknown'):
        train_set[i][1]='blue-collar'
    if(train_set[i][3]=='unknown'):
        train_set[i][3]='secondary'
    if(train_set[i][8]=='unknown'):
        train_set[i][8]='cellular'


for i in range(n1):
    if(test_set[i][15]=='unknown'):
        test_set[i][15]='failure'
    if(test_set[i][1]=='unknown'):
        test_set[i][1]='blue-collar'
    if(test_set[i][3]=='unknown'):
        test_set[i][3]='secondary'
    if(test_set[i][8]=='unknown'):
        test_set[i][8]='cellular'
    



print("For each row, from left to right: the number of correct predictions, the number of wrong predictions, prediction errors")
print("Entropy information gain for predicting training set:")
a1=0.0
a2=0.0
a3=0.0

b1=0.0
b2=0.0
b3=0.0
for i in range(16):
    r=training(train_set,'H',i)
    a1=a1+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",a1/16 )

print("---------------------------------")
print("EM for predicting training set:")
for i in range(16):
    r=training(train_set,'EM',i)
    a2=a2+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",a2/16 )

print("---------------------------------")
print("GI for predicting training set:")
for i in range(16):
    r=training(train_set,'GI',i)
    a3=a3+r[2]
    print("Tree level: ",(i+1),r)

print("The everage prediction error is: ",a3/16 )

print("---------------------------------")
print("Entropy information gain for predicting testing set:")
for i in range(16):
    r=training(test_set,'H',i)
    b1=b1+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",b1/16 )

print("---------------------------------")
print("EM for predicting testing set:")
for i in range(16):
    r=training(test_set,'EM',i)
    b2=b2+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",b2/16 )

print("---------------------------------")
print("GI for predicting testing set:")
for i in range(16):
    r=training(test_set,'GI',i)
    b3=b3+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",b3/16 )

print("---------------------------------")
