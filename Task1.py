#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
style.use('dark_background')


# In[2]:


iris=pd.read_csv('Iris.csv')


# In[3]:


iris.head()


# In[4]:


iris.tail()


# In[5]:


iris.describe()


# In[6]:


iris.size


# In[7]:


iris.columns


# In[10]:


iris.shape


# In[12]:


iris.groupby('Species').size()


# In[14]:


iris['Species'].unique().tolist()


# In[15]:


iris.isnull().sum()


# In[17]:


nameplot = iris['Species'].value_counts().plot.bar(title='Flower class distribution')
nameplot.set_xlabel('class',size=20)
nameplot.set_ylabel('count',size=20)


# In[18]:


sns.pairplot(iris,hue='Species')


# In[22]:


iris.hist()


# In[25]:


x = iris.drop("Species", axis=1)
y = iris["Species"]


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[27]:


print("X_train.shape:",x_train.shape)
print("X_test.shape:",x_test.shape)
print("Y_train.shape:",y_train.shape)
print("Y_test.shape:",y_train.shape)


# In[29]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
x_new=np.array([[151,5,2.9,1,0.2]])
prediction=knn.predict(x_new)
print("prediction: {}".format(prediction))


# In[31]:


model=KNeighborsClassifier()
model.fit(x_train,y_train)
model.score(x_train,y_train)


# In[32]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[39]:


def test_model(model):
    model.fit(x_train,y_train)
    predicitons=model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, predicitons))
    print("confusion matrix:")
    print(confusion_matrix(y_test, predicitons))
    print("classification report:")
    print(classification_report(y_test, predicitons))


# In[40]:


test_model(model)


# In[ ]:




