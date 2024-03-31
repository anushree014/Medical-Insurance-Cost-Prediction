#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


# # Importing Dataset

# In[2]:


df=pd.read_csv("insurance.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# categorical value
# -sex
# -smoker
# -region
# 

# In[8]:


df.isnull().sum()


# # Data Analysis

# In[9]:


df.describe()


# In[10]:


df.describe(include='object')


# In[11]:


# distribustion of age value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(df['age'])
plt.title('Age Distribution')
plt.show()


# In[12]:


plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=df)
plt.title("sex distribution")
plt.show()


# In[13]:


df['sex'].value_counts()


# In[14]:


# distribustion of BMI value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(df['bmi'])
plt.title('BMI Distribution')
plt.show()


# normal bmi-------> 18.5 to 24.9

# In[15]:


plt.figure(figsize=(6,6))
sns.countplot(x='children',data=df)
plt.title("Children")
plt.show()


# In[16]:


df['children'].value_counts()


# In[17]:


plt.figure(figsize=(6,6))
sns.countplot(x='smoker',data=df)
plt.title("Smoker")
plt.show()


# In[18]:


plt.figure(figsize=(6,6))
sns.countplot(x='region',data=df)
plt.title("Region")
plt.show()


# In[19]:


df['region'].value_counts()


# In[20]:


sns.set()
plt.figure(figsize=(6,6))
sns.distplot(df['charges'])
plt.title('Charges Distribution')
plt.show()


# Data Pre-Processing

# In[21]:


# encoding sex column
df.replace({'sex':{'male':0,'female':1}},inplace=True)

# encoding smoker column
df.replace({'smoker':{'yes':0,"no":1}},inplace=True)

#encoding region column
df.replace({'region':{'southeast':0,"southwest":1,"northeast":2,'northwest':3}},inplace=True)



# In[22]:


df


# # Splitting the Dataset

# In[23]:


x=df.drop(columns='charges',axis=1)
y=df['charges']


# In[24]:


print(x)


# In[25]:


print(y)


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[27]:


print(x_test.shape)


# In[28]:


print(x_train.shape)


# # Model Training

# In[29]:


model=LinearRegression()


# In[30]:


model.fit(x_train,y_train)


# # Model Evaluation

# In[31]:


#prediction on training data
training_data_prediction=model.predict(x_train)


# In[32]:


# R-squared value
r2_train=metrics.r2_score(y_train,training_data_prediction)
print("R-squared value",r2_train)


# In[33]:


#prediction on testing data
testing_data_prediction=model.predict(x_test)


# In[34]:


# R-squared value
r2_test=metrics.r2_score(y_test,testing_data_prediction)
print("R-squared value",r2_test)


# # Building a predictive Model

# In[38]:


input_data=(31,1,25.74,0,1,0)

#changing to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the array
input_data_reshaped =input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)

print(prediction)
print("the insurance cost is USD",prediction[0])

