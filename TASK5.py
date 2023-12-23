#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[31]:


df = pd.read_csv('Advertising.csv')


# In[32]:


df


# In[33]:


df.head()


# In[34]:


df.tail()


# In[35]:


df.shape


# In[36]:


df.info()


# In[37]:


df.describe()


# In[38]:


x=df.iloc[:,0:-1]
x


# In[39]:


y=df.iloc[:,-1]
y


# In[40]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)
x_train


# In[41]:


x_test


# In[42]:


y_train


# In[43]:


y_test


# In[44]:


x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
y_test=y_test.astype(int)


# In[45]:


from sklearn.preprocessing import StandardScaler


# In[46]:


sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.fit_transform(x_test)


# In[47]:


from sklearn.linear_model import LinearRegression


# In[48]:


lr=LinearRegression()


# In[49]:


lr.fit(x_train_scaled,y_train)


# In[50]:


y_pred=lr.predict(x_test_scaled)


# In[51]:


from sklearn.metrics import r2_score


# In[52]:


r2_score(y_test,y_pred)


# In[53]:


plt.scatter(y_test,y_pred,c='g')

