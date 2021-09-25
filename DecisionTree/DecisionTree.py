import numpy as np
import math
from collections import Counter
import sys
import copy
from numpy.lib.shape_base import split

train_set = list()
test_set = list()
labels=['unacc','acc','good','vgood']
buying=['vhigh', 'high', 'med', 'low']
maint=['vhigh' ,'high', 'med', 'low']
doors=['2', '3', '4', '5more']
persons=['2', '4', 'more']
lug_boot=['small', 'med', 'big']
safety=['low', 'med', 'high']
values=[buying,maint,doors,persons,lug_boot,safety]
features=['buying','maint','doors','persons','lug_boot','safety']

# import training and test data
with open('/Users/dalong/CS6350ML/DecisionTree/Car/train.csv', 'r') as f:
    for line in f :
        terms = line.strip( ).split(',')
        train_set.append(terms)

with open('/Users/dalong/CS6350ML/DecisionTree/Car/test.csv', 'r') as f:
    for line in f :
        terms = line.strip( ).split(',')
        test_set.append(terms)

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
def proportions(feature_list,label_list,list,labels):
    n_labels=len(labels)
    n_values=len(feature_list)
    n_examples=len(list)
    p = np.zeros([n_values,n_labels])
    p_labels=np.zeros(n_labels)
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
    return np.vstack((p_labels,p))
 
# helper function
def p_values(feature_list,list):
    n_values=len(feature_list)
    p_values=np.zeros(n_values)
    n_examples=len(list)
    for j in range(n_examples):
        for k in range(n_values):
            if(list[j]==feature_list[k]):
                p_values[k]=p_values[k]+1
    p_values=p_values/(np.sum(p_values)+sys.float_info.epsilon)
    return p_values

# get kth column of a 2-D list
def getColumn(list,k):
    n_row=len(list)
    newList=[]
    for i in range(n_row):
        newList.append(list[i][k])
    return newList

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
        s_proportions=proportions(values[index],y,x,labels)
        if(f=='H'):
            v_Gains[i]=Gain(s_proportions[0,:],p_values(values[index],x),s_proportions[1:,:])
        if(f=='GI'):
            v_Gains[i]=Gain_GI(s_proportions[0,:],p_values(values[index],x),s_proportions[1:,:])
        if(f=='ME'):
            v_Gains[i]=Gain_ME(s_proportions[0,:],p_values(values[index],x),s_proportions[1:,:])
    temp=v_Gains.tolist()
    max_index=temp.index(max(temp))
    split_feature=s_features[max_index]
    f_index=features.index(split_feature)
    values_feature=values[f_index]
    n_examples=len(examples)
    n_feature=len(values_feature)
    node.addFeature(split_feature)
    dict={}
    for j,N in enumerate(values_feature):
        dict[N]=list()
    for i in range(n_examples):
        for j in range(n_feature):
            if(values_feature[j]==examples[i][f_index]):
                dict[values_feature[j]].append(examples[i])
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
            
# root=treeNode(children=list(),level=1)
# buildTree(train_set,features,root,'H',6)

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
        f = cur.feature
        index=features.index(f)
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
            wrong=wrong+1
    
    return list([right,wrong,wrong/(right+wrong)])

# print(training(train_set,'H',1))
# print(training(train_set,'GI'))
# print(training(train_set,'ME'))

# print(training(test_set,'H'))
# print(training(test_set,'GI'))
# print(training(test_set,'ME'))

print("For each row, from left to right: the number of correct predictions, the number of wrong predictions, prediction errors")
print("Entropy information gain for predicting training set:")
a1=0.0
a2=0.0
a3=0.0

b1=0.0
b2=0.0
b3=0.0
for i in range(6):
    r=training(train_set,'H',i)
    a1=a1+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",a1/6 )

print("---------------------------------")

print("EM for predicting training set:")
for i in range(6):
    r=training(train_set,'EM',i)
    a2=a2+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",a2/6 )

print("---------------------------------")

print("GI for predicting training set:")
for i in range(6):
    r=training(train_set,'GI',i)
    a3=a3+r[2]
    print("Tree level: ",(i+1),r)

print("The everage prediction error is: ",a3/6 )

print("---------------------------------")
print("Entropy information gain for predicting testing set:")
for i in range(6):
    r=training(test_set,'H',i)
    b1=b1+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",b1/6 )

print("---------------------------------")

print("EM for predicting testing set:")
for i in range(6):
    r=training(test_set,'EM',i)
    b2=b2+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",b2/6 )

print("---------------------------------")

print("GI for predicting testing set:")
for i in range(6):
    r=training(test_set,'GI',i)
    b3=b3+r[2]
    print("Tree level: ",(i+1),r)
print("The everage prediction error is: ",b3/6 )

print("---------------------------------")