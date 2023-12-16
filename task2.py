#!/usr/bin/env python
# coding: utf-8

# In[144]:


import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


# In[145]:


df = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')
df


# In[146]:


df.head()


# In[147]:


df.tail()


# In[148]:


df.shape


# In[149]:


df.info()


# In[150]:


df.describe()


# In[151]:


x=df['Region']
x


# In[152]:


y=df[' Estimated Unemployment Rate (%)']
y


# In[153]:


df.columns


# In[154]:


df2=df.iloc[:,3]
df2


# In[156]:


fg=px.bar(df,x='Region',y=' Estimated Unemployment Rate (%)',color='Region',title='Unemployment Rate(State-Wise)by bar Graph',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()


# In[157]:


fg=px.bar(df,x='Region',y=' Estimated Unemployment Rate (%)',color='Region',title='Unemployment Rate(Region-Wise)by bar Graph',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()


# In[158]:


fg=px.box(df,x='Region',y=' Estimated Unemployment Rate (%)',color='Region',title='Unemployment Rate(Region-Wise)by bar Graph',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()


# In[159]:


fg=px.box(df,x='Region',y=' Estimated Unemployment Rate (%)',color='Region',title='Unemployment Rate(State-Wise)by bar Graph',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()


# In[160]:


fg=px.histogram(df,x='Region',y=' Estimated Unemployment Rate (%)',color='Region',title='Unemployment Rate(State-Wise)by bar Graph',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()


# In[161]:


fg=px.histogram(df,x='Region',y=' Estimated Unemployment Rate (%)',color='Region',title='Unemployment Rate(Region-Wise)by bar Graph',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()


# In[ ]:




