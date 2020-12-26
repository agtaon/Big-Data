#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from mpl_toolkits import mplot3d
from sklearn import preprocessing



data=np.load('classification_hemtenta.npz')
x_train=data['X_train']
y_train=data['y_train']
x_valid=data['X_valid']
y_valid=data['y_valid']


# In[3]:


indices_ones=np.nonzero(y_train==1)
indices_zeros=np.nonzero(y_train==0)

train_data_1=x_train[indices_ones]
train_data_0=x_train[indices_zeros]

mean1=np.mean(train_data_1, 0).T
mean0=np.mean(train_data_0, 0).T


dims=np.linspace(0,799,800)

plt.figure(figsize=(10, 4))
plt.stem(dims,mean1,use_line_collection=True)
plt.xlabel('Dimension',fontsize=15)
plt.ylabel('Mean value along dimension', fontsize=15)
plt.title('Label=1')


plt.figure(figsize=(10, 4))
plt.stem(dims,mean0,use_line_collection=True)
plt.xlabel('Dimension',fontsize=15)
plt.ylabel('Mean value along dimension', fontsize=15)
plt.title('Label=0');


# In[4]:


c_values=np.exp(np.linspace(-10,-5,30))
scores_l1=[]
scores_l10=[]
scores_l11=[]

for c in c_values:
    lr1=LogisticRegression(penalty='l1',solver='saga',max_iter=10000,C=c)
    lr1.fit(x_train, y_train)
    
    score=lr1.score(x_valid,y_valid)
    scores_l1.append(score)
    
    score0=lr1.score(x_valid[y_valid==0],np.zeros(len(y_valid[y_valid==0])))
    scores_l10.append(score0)
    
    score1=lr1.score(x_valid[y_valid==1],np.ones(len(y_valid[y_valid==1])))
    scores_l11.append(score1)
    

plt.plot(np.log(c_values),scores_l1,label='Total score')
plt.plot(np.log(c_values),scores_l10,label='Score label 0')
plt.plot(np.log(c_values),scores_l11,label='Score label 1')

plt.title('Test scores plotted to hyperparameter C, Lasso regression');
plt.xlabel('log(C), inverse of regulatization strength')
plt.legend()
plt.ylabel('Accuracy on validation set');

index_max_score_c=np.argmax(scores_l1)
print('log(C) that gave the highest score: ', np.log(c_values[index_max_score_c]))
print('Score: ', scores_l1[index_max_score_c])

index_max_score_c0=np.argmax(scores_l10)
print('log(C) that gave the highest score on label 0: ', np.log(c_values[index_max_score_c0]))
print('Score: ', scores_l10[index_max_score_c0])

index_max_score_c1=np.argmax(scores_l11)
print('log(C) that gave the highest score on label 1: ', np.log(c_values[index_max_score_c1]))
print('Score: ', scores_l11[index_max_score_c1])


# In[5]:


lr1=LogisticRegression(penalty='l1',solver='saga',max_iter=10000,C=c_values[index_max_score_c])
lr1.fit(x_train, y_train)
print('Score at C=exp(', np.round(np.log(c_values[index_max_score_c]),2),') on validation set:',lr1.score(x_valid,y_valid))
print('Score at C=exp(', np.round(np.log(c_values[index_max_score_c]),2),') on validation set class 0:',lr1.score(x_valid[y_valid==0],np.zeros(len(y_valid[y_valid==0]))))
print('Score at C=exp(', np.round(np.log(c_values[index_max_score_c]),2),') on validation set class 1:',lr1.score(x_valid[y_valid==1],np.ones(len(y_valid[y_valid==1]))))
coefs_lr1=lr1.coef_

print('Remaining dimensions: ',np.nonzero(lr1.coef_[0])[0])
print(len(np.nonzero(lr1.coef_[0])[0]))
print(lr1.classes_)
plt.figure(figsize=(10, 4))
plt.stem(dims,lr1.coef_[0],use_line_collection=True)
plt.xlabel('Dimension',fontsize=15)
plt.ylabel('Coefficient value', fontsize=15)
plt.title('18 remaining coefficients in beta');


# In[13]:


valid_data_1=x_valid.T*y_valid*coefs_lr1.T
valid_data_0=x_valid.T*(y_valid-1)*(-1)*coefs_lr1.T

mean1=np.mean(valid_data_1, 1)
mean0=np.mean(valid_data_0, 1)

dims=np.linspace(0,799,800)
plt.figure(figsize=(10, 4))
plt.stem(dims,mean1,use_line_collection=True)
plt.xlabel('Dimension',fontsize=15)
plt.ylabel('Product', fontsize=15)
plt.title('Coefficients multiplied by means along respective dimension of label 1 in validation set, sum equals ' + str(round(np.sum(mean1),4)))

plt.figure(figsize=(10, 4))
plt.stem(dims,mean0,use_line_collection=True)
plt.xlabel('Dimension',fontsize=15)
plt.ylabel('Product', fontsize=15)
plt.title('Coefficients multiplied by means along respective dimension of label 0 in validation set, sum equals ' + str(round(np.sum(mean0),4)));


# In[14]:


shrink_ts=np.linspace(0,5,100) # Change 5 to 6 to get the plot in the exam. The 5 is to generate the optimal threshold<5
scores_nc=[]
scores_nc0=[]
scores_nc1=[]
for st in shrink_ts:
    shrink_c=NearestCentroid(shrink_threshold=st)
    shrink_c.fit(x_train, y_train)
    
    score_shrink_c=shrink_c.score(x_valid,y_valid)
    scores_nc.append(score_shrink_c)
    
    score_shrink_c0=shrink_c.score(x_valid[y_valid==0],np.zeros(len(y_valid[y_valid==0])))
    scores_nc0.append(score_shrink_c0)
    
    score_shrink_c1=shrink_c.score(x_valid[y_valid==1],np.ones(len(y_valid[y_valid==1])))
    scores_nc1.append(score_shrink_c1)
    
plt.plot(shrink_ts,scores_nc,label='Total score')
plt.plot(shrink_ts,scores_nc0,label='Score label 0')
plt.plot(shrink_ts,scores_nc1,label='Score labe 1')

plt.title('Test scores plotted to shrink threshold, Nearest shrunken centroids');
plt.xlabel('Shrink threshold')
plt.legend()
plt.ylabel('Accuracy');

index_max_score_st=np.argmax(scores_nc)
print('shrink_ts that gave the highest score on validation set: ', shrink_ts[index_max_score_st])
print('Total score: ', scores_nc[index_max_score_st])

index_max_score_st0=np.argmax(scores_nc0)
print('shrink_ts that gave the highest score on validation set label 0: ', shrink_ts[index_max_score_st0])
print('Score label 0: ', scores_nc0[index_max_score_st0])

index_max_score_st1=np.argmax(scores_nc1)
print('shrink_ts that gave the highest score on validation set label 1: ', shrink_ts[index_max_score_st1])
print('Score label 1: ', scores_nc1[index_max_score_st1])

print('Difference between score(shrink_ts=',shrink_ts[index_max_score_st0],') and score(shrink_ts=',shrink_ts[index_max_score_st1],') = ',scores_nc1[index_max_score_st1]-scores_nc1[index_max_score_st0])


# In[17]:


shrink_c=NearestCentroid(shrink_threshold=shrink_ts[index_max_score_st])
shrink_c.fit(x_train, y_train)
print('Total score: ', scores_nc[index_max_score_st])
print('Score at shrink threshold=',shrink_ts[index_max_score_st],' on validation set class 1:',shrink_c.score(x_valid[y_valid==1],np.ones(len(y_valid[y_valid==1]))))
print('Score at shrink threshold=',shrink_ts[index_max_score_st],' on validation set class 0:',shrink_c.score(x_valid[y_valid==0],np.zeros(len(y_valid[y_valid==0]))))
class_centroids=shrink_c.centroids_



print('Coefficients in common between the two centroids',len(set(class_centroids[0]).intersection(class_centroids[1])))
print('All coefficients in centroid 0 minus all coefficients in centroid 1 (to test equality)',sum(class_centroids[0]-class_centroids[1]))
print('"Important" features in common with l1 regularized: ',set(np.where(class_centroids[0]!=class_centroids[1])[0]).intersection(np.nonzero(lr1.coef_[0])[0]))
print('Features that had different coefficients in the two centroids: ') 
print(np.where(class_centroids[0]!=class_centroids[1])[0])

