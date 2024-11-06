#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[6]:


#Data collection And analysis


# In[9]:


customer_data = pd.read_csv(r'C:\Users\mkkor\Downloads\data\Mall_Customers.csv')


# In[13]:


customer_data.head()


# In[15]:


type(customer_data)


# In[17]:


customer_data.shape


# In[21]:


customer_data.info()


# In[22]:


customer_data.Gender


# In[24]:


customer_data.isnull().sum()


# In[27]:


X = customer_data.iloc[:,[3,4]].values


# In[32]:


X


# In[34]:


## Finidng WCSS values for diff number of clustors


# In[37]:


wcss=[]

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
   
    wcss.append(kmeans.inertia_)


# In[43]:


sns.set()
plt.plot(range(1,11),wcss)
plt.title('Elbow point Grph')
plt.label('No of clusters')
plt.label('wcss')
plt.show()


# In[44]:


#the optimum number of clustors are 5


# In[46]:


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)


# In[48]:


Y =kmeans.fit_predict(X)


# In[50]:


Y


# In[51]:


plt.figure(figsize=(9,9))
plt.scatter(X[Y==0,0],X[Y==0,1], s=50, c='green', label='clu 1')
plt.scatter(X[Y==1,0],X[Y==1,1], s=50, c='red', label='clu 2')
plt.scatter(X[Y==2,0],X[Y==2,1], s=50, c='blue', label='clu 3')
plt.scatter(X[Y==3,0],X[Y==3,1], s=50, c='orange', label='clu 4')
plt.scatter(X[Y==4,0],X[Y==4,1], s=50, c='olive', label='clu 5')


# In[54]:


plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], c='cyan', label='Centroid')


# In[56]:


plt.figure(figsize=(9,9))
plt.scatter(X[Y==0,0],X[Y==0,1], s=50, c='green', label='clu 1')
plt.scatter(X[Y==1,0],X[Y==1,1], s=50, c='red', label='clu 2')
plt.scatter(X[Y==2,0],X[Y==2,1], s=50, c='blue', label='clu 3')
plt.scatter(X[Y==3,0],X[Y==3,1], s=50, c='orange', label='clu 4')
plt.scatter(X[Y==4,0],X[Y==4,1], s=50, c='olive', label='clu 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], c='cyan', label='Centroid')

plt.title('Customer_groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending')

