#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv("train.csv")


# In[11]:


data.head(10)


# In[7]:


print("Shape:", data.shape)
print("\nHead:")
print(data.head())


print("\nInfo:")
print(data.info())


print("\nDescribe:")
print(data.describe(include='all'))


print("\nValue Counts for categorical columns:")
for col in data.select_dtypes(include='object').columns:
    print(f"\n{col}:\n", data[col].value_counts())


# In[8]:


print("\nMissing values per column:")
print(data.isnull().sum())

sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()


# In[9]:


data.select_dtypes(include=['int64', 'float64']).hist(bins=20, figsize=(15, 10))
plt.suptitle("Histograms of Numeric Features")
plt.show()

for col in data.select_dtypes(include=['int64', 'float64']).columns:
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# In[10]:


sns.pairplot(data.select_dtypes(include=['int64', 'float64']))
plt.suptitle("Pairplot - Relationships Between Numeric Features", y=1.02)
plt.show()


correlation_matrix = data.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[15]:


for col in data.select_dtypes(include=['int64', 'float64']).columns:
    sns.boxplot(x='Survived', y=col, data=data)  
    plt.title(f'{col} by target_column')
    plt.show()


# In[20]:


print("SUMMARY OF FINDINGS:")
print("""
- Numerical features are mostly normally distributed, except Fare and Age which are right skewed.
- Features Fare and Age have outliers (see boxplots).
- Strong correlations were found between Survived and Sex.
- Missing data exists in Age and Cabin; consider imputation or removal.
- Categorical feature Sex has dominant class imbalance.
- Feature Sex is highly predictive of target_column.
""")


# In[19]:


category_col = 'Survived'  
data[category_col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title(f'Count of {category_col}')
plt.xlabel(category_col)
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# In[ ]:




