#!/usr/bin/env python
# coding: utf-8

# ### Employee Attrition Problem
# 
# Company X has an employee attrition problem and needs to know which type of employee are leave or prone to leave. 
# The data given consists of current employee and ex employee and canbe gotten from this link:
# 
# https://takenmind.com/takenmind-internship-project-statement/
# 
# The following attributes are available for every employee:
# - Satisfaction Level
# - Last evaluation
# - Number of projects 
# - Average monthly hours
# - Time spent at the company
# - Whether they have had a work accident
# - Whether they have had a promotion in the last 5 years
# - Departments (column sales)
# - Salary
# - Whether the employee has left
# 
# 
# 
# 

# ### Exploratory Analysis
# 
# This is the initial process of analysis, where a summary of the characteristics of the data sets is shown. Summary of patterns, trends, hypothesis testing can be done using descriptive statistics and data visualization.

# In[1]:


# Importing modules
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as  sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Loading the dataset

curr_employ_data = pd.read_excel('TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx',
                                 sheet_name= 'Existing employees')

ex_employ_data = pd.read_excel('TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx',
                              sheet_name= 'Employees who have left')


# In[3]:


curr_employ_data.head()


# In[4]:


curr_employ_data.tail()


# In[5]:


ex_employ_data.head()


# In[6]:


ex_employ_data.head()


# In[7]:


# To get info about the data sets

curr_employ_data.info()


# From the info above, the data has 11428 entries(rows) and 10 attribbutes (6 interger, 2 float, 2 objects).
# 
# No variable column has nullmissing values.

# In[8]:


ex_employ_data.info()


# From the info above, the data has 3570 entries(rows) and 10 attribbutes (6 interger, 2 float, 2 objects).
# 
# No variable column has nullmissing values

# In[9]:


# Combine the two data to help analyse easily.
# But before that Create a new column called 'left' to indicate employees that left
# and  existing employees. ex employ are denoted with '1' and 
# current employess are denoted with '0'

ex_employ_data['left'] = 1

curr_employ_data['left'] = 0


# In[10]:


df = pd.concat([ex_employ_data, curr_employ_data]) 


# In[11]:


left = df.groupby('left')
left.mean()


# From the above output we can see that the employess who left the company had low satisfaction level, high average monthly hours, low promotion rate.

# ## Visualizing 

# In[12]:


features = ['number_project','time_spend_company',
            'left','promotion_last_5years','dept','salary','average_montly_hours']

fig = plt.subplots(figsize=(10,15))

for i,j in enumerate(features):
    plt.subplot(4,2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data= df)
    plt.xticks(rotation=90)
    plt.title("No. of employee")
    


# From the above visualiztion:
# - Most of the employees do about 3-5 projects
# - Most of the employess have not spent more than 5 years at the company
# - Most of the employees have not been promoted in the last five years 
# - The sales department is having maximum no.of employee followed by technical and support
# - Most of the employees are getting salary either medium or low.

# In[13]:


# Comparing current employees and ex employees on the features
fig = plt.subplots(figsize=(10,15))

for i,j in enumerate(features):
    plt.subplot(4,2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data= df,hue = 'left')
    plt.xticks(rotation=90)
    plt.title("No. of employee")


# From the visulaization:
# - Employees who had less than 5 projects tend to leave the company
# - Most of the employees who left do not spend more than five years, this might seem to be bacause none of them had promotions in the last five years
# 

# ## In summary
# 
# From the above the analysis the following features are the most influencing to making a person leave the company:
# 
# - Promotions: Employees are most likely to leave if they haven't received a promotion in the last five years.
# 
# - Time with company: Most of the employees leave after 3 years and if not three years they leave after 6 years
# 
# - Salary: Most of the employee who quite have mid to low salary
# 
# - Number of projects: Enployees with 3-5 projects are less likely to leave the company. 

# ### Building the prediction model and fitting it.
# 
# Most machine learning algorithms require numerical data, therefore categorical columns need to be converted to numerical columns.
# 
# Recall that our attributes include:
# - Promotion in the last 5 years
# - satisfaction level 
# - number of proojects 
# - average monthly hours
# - salary 
# - dept 
# - time spent at the company 
# 
# From the attributes some columns are categorical so we need to label encode this columns. To do this, the sklearn label encoder will be used. 
# 
# So for example, the salary column's values the label encoder will represent them as low:0, medium:1, high:2
# 

# In[14]:


# import LabelEncoder

from sklearn.preprocessing import LabelEncoder

# Create an obeject of the class LabebEncoder

labelencoder_col = LabelEncoder()

# Convert the columns

df['salary'] = labelencoder_col.fit_transform(df['salary'])
df['dept'] = labelencoder_col.fit_transform(df['dept'])


# In[15]:


# After converting the columns,spliting the data set into training
# set and test set is a good strategy to help understand the performance of the 
# model

# selecting the features, target and splitting the data set

features = ['number_project','time_spend_company','Work_accident','time_spend_company',
            'promotion_last_5years','dept','salary','average_montly_hours','last_evaluation','satisfaction_level']

X = df[features]

y = df['left']

from sklearn.model_selection import train_test_split

# split the dataset into training set and test set. 30% for testing and validation

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.3, random_state=1)


# In[16]:


# model building using GradientBoostingClassifier()


# import Gradient Boosting Classifier 
from sklearn.ensemble import GradientBoostingClassifier

# create an object of the class GradientBoostingClassifier

gbc = GradientBoostingClassifier(learning_rate = 0.01,  max_leaf_nodes=500)

# train the model using the trainig set

gbc.fit(X_train,y_train)


# In[17]:


y_pred = gbc.predict(X_test)
print(y_pred[:5])


# ### Evaluating Model Performance
# 
# 

# In[18]:


# import scikit-learn metrics module for accuracy calculation

from sklearn import metrics 

# Model Accuracy, how often is the classifier correct?

model_acc = metrics.accuracy_score(y_test, y_pred)

print("Accuracy: ", model_acc)

# model precision

model_prec = metrics.precision_score(y_test,y_pred)

print("Precision: ", model_prec)


# In[19]:


X_test[:5]


# In[20]:


testpred = gbc.predict(X_test[:5])


# In[21]:


emp_Id = [8337,6937,5622,9516,506]


# In[22]:


test = pd.DataFrame(testpred, index=emp_Id)


# In[23]:


test.to_csv("sampleoutput.csv")


# In[ ]:




