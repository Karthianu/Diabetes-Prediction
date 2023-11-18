#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd # read dataset
import numpy as np # numerical python
import matplotlib.pyplot as plt #to plot graph
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns #plot graph in graphical manner


# In[25]:


#read dataset
df=pd.read_csv('diabetes.csv')


# In[26]:


df.head() # Read Data in First Five


# In[27]:


#To Predict Weather diabetic or Not


# In[28]:


df.shape # How Many Rows and Columns


# In[29]:


df.info() # if Dtype is Object then u can convert it into Numerical Value


# In[30]:


df.isnull() # to check null values


# In[31]:


df.isnull().sum() #To chk null values in all column


# In[32]:


#df.dropna() #if it has any null remove this using dropna command


# Train and Test
# 

# In[33]:


from sklearn.model_selection import train_test_split #data #label


# In[34]:


x=df.iloc[:,df.columns!='Outcome'] #data
y=df.iloc[:,df.columns=='Outcome'] #label


# In[35]:


print(x)


# In[ ]:





# In[36]:


print(y)


# In[37]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)#20%test


# In[38]:


xtrain.head()


# In[39]:


ytrain.head()


# In[40]:


xtrain.shape


# Algoritham

# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


model=RandomForestClassifier()


# In[43]:


model.fit(xtrain,ytrain.values.ravel())#train the data
'''
history = model.fit(
      xtrain,
      steps_per_epoch=5,
      epochs=2,
      validation_data = ytrain,
      validation_steps = 50,
      verbose=1)


acc = history.history['acc']
plt.title('TRAINING ACCURACY')
plt.xlabel('Epoch')
plt.ylabel('Accuracy Value')
plt.show
'''


# In[44]:


predict_output=model.predict(xtest) #to test the alg
print(predict_output)


# In[45]:


from sklearn.metrics import accuracy_score


# In[46]:


acc=accuracy_score(predict_output,ytest)
print('The Accuracy Score for RF:',acc)


# In[47]:


#predict the data


# In[48]:


ip_data=(1,85,66,29,0,26.6,0.351,31)
ip_arr=np.asarray(ip_data)


# In[49]:


print(ip_arr)


# In[50]:


res_arr=ip_arr.reshape(1,-1)
res_arr


# In[51]:


pred=model.predict(res_arr)


# In[52]:


print(pred)


# In[53]:


if (pred==0):
    print("The Person is Not Diabetic")
else:
    print("The Person is Diabetic")

