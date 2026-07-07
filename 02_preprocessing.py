#!/usr/bin/env python
# coding: utf-8

# ## 02 - Data Preprocessing

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Data/combined_sold.csv')


# In[3]:


df.isnull().sum()


# In[4]:


# removing columns that have over 80% missing/null values

def get_nan_cols(df, thresh=0.8):
    threshold = len(df.index) * thresh
    return [c for c in df.columns if df[c].isnull().sum() >= threshold]

print(f"Columns With Over 80% Missing Values: {get_nan_cols(df)}")

df = df.dropna(axis=1, thresh=0.2*len(df))


# In[5]:


# drop these 2 columns since they have no more predictive power after filitering to single family homes
df = df.drop(columns=['PropertyType', 'PropertySubType'])


# drop columns that are solely used for identification and/or have no predictive power
df = df.drop(columns = ['ListAgentFullName', 'ListAgentFirstName', 'ListAgentLastName','ListAgentEmail', 
                        'CoListAgentFirstName', 'CoListAgentLastName', 'BuyerAgentFirstName', 'BuyerAgentLastName', 'BuyerAgentMlsId',
                        'BuyerAgentAOR', 'ListAgentAOR', 'BuyerOfficeAOR', 'ListOfficeName', 'BuyerOfficeName', 'CoListOfficeName',
                        'ListingKey', 'ListingId', 'ListingKeyNumeric', 'AssociationFeeFrequency'])

df = df.drop(columns = [])


# In[6]:


df.columns


# In[7]:


df.isnull().sum()


# In[8]:


# since theres only 2 rows with missing ClosePrice and 1 row with missing PostalCode, drop those rows
df = df.dropna(subset=['ClosePrice', 'PostalCode'])

#filling in some null values
cols = ['LivingArea', 'BedroomsTotal', 'BathroomsTotalInteger', 'LotSizeArea','LotSizeAcres', 'LotSizeSquareFeet', 'MainLevelBedrooms', 'ParkingTotal']
for col in cols:
    median = df[col].median()
    df[col] = df[col].fillna(median)

cols = ['Flooring', 'UnparsedAddress', 'MLSAreaMajor', 'SubdivisionName', 'HighSchoolDistrict', 'Levels', 'City']
for col in cols:
    df[col] = df[col].fillna('Unknown')


df['AssociationFee'] = df['AssociationFee'].fillna(0)
df['GarageSpaces'] = df['GarageSpaces'].fillna(0)

# assume nulls means 1 stories
df['Stories'] = df['Stories'].fillna(1.0)


# In[9]:


# converting categorical fields

binary_cols = ['PoolPrivateYN', 'ViewYN', 'FireplaceYN', 'NewConstructionYN', 'AttachedGarageYN']

for col in binary_cols:
    df[col] = df[col].map({'Y': 1, 'N': 0, True: 1, False: 0})

    # fill remaining null with 0 
    df[col] = df[col].fillna(0).astype(int)


# In[10]:


df['Levels'].value_counts()


# In[11]:


def clean_to_single_level(level_str):
    if pd.isna(level_str):
        return 'Unknown'

    level_str = str(level_str)

    # Prioritize the most significant structural trait
    if 'ThreeOrMore' in level_str:
        return 'ThreeOrMore'
    elif 'Two' in level_str:
        return 'Two'
    elif 'MultiSplit' in level_str:
        return 'MultiSplit'
    elif 'One' in level_str:
        return 'One'
    else:
        return 'Other'

# Apply the cleaning function
df['Levels'] = df['Levels'].apply(clean_to_single_level)

df = pd.get_dummies(df, columns=['Levels'], drop_first=True, dtype=int)


# In[12]:


df.isnull().sum().sort_values(ascending=False)


# In[13]:


# normalizing key numerical fields

num_fields = ['LivingArea', 'LotSizeArea', 'LotSizeSquareFeet', 'LotSizeAcres']
df.loc[:, num_fields] = np.log1p(df[num_fields])


# In[14]:


df.head()


# In[15]:


df.to_csv('Data/cleaned_sold.csv', index=False)


# ### Create Train/Test Split

# In[ ]:


#testing reusable function to see if it works

from utils import create_time_split

df = pd.read_csv('Data/cleaned_sold.csv')
df['CloseDate'] = pd.to_datetime(df['CloseDate'])

max_date = df['CloseDate'].max()
test_start_date = max_date - pd.DateOffset(months=1)

train_df, test_df = create_time_split(df, 'CloseDate', X_months=6, test_start_date=test_start_date, max_date=max_date)

