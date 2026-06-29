#!/usr/bin/env python
# coding: utf-8

# ## 01 - CRMLS Data Exploration

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# #### Combine monthly listings and sold datafiles into 1 dataset

# In[8]:


import glob
import os

def combine_data(input_folder, output_filename):
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if not all_files:
        print(f"No CSV files found in {input_folder}")
        return

    dataframes = []
    for file in all_files:
        df = pd.read_csv(file)
        dataframes.append(df)

    print(f"{'='*60}")
    print(f"PROCESSING DATA: {input_folder}")
    print(f"{'='*60}")

    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    print(f"\n--- Concatenation Summary ---")
    print(f"Combined dataset row count: {len(combined_df)}")

    print(f"\n--- PropertyType Frequency (Before Filtering to Residential and SingleFamilyResidence) ---")
    print(combined_df['PropertyType'].value_counts())

    new_df = combined_df[(combined_df['PropertyType'] == 'Residential') & (combined_df['PropertySubType'] == 'SingleFamilyResidence')].copy()
    print(f"\n--- Filtering Summary ---")
    print(f"Rows after 'Residential' filter: {len(new_df)}")

    print(f"\n--- PropertyType Frequency (After Filter) ---")
    print(new_df['PropertyType'].value_counts())

    new_df.to_csv(output_filename, index=False)
    print(f"\nFile saved as: {output_filename}\n")


# In[12]:


combine_data('Data/CRMLS Sold Files', 'Data/combined_sold.csv')


# #### Data Inspection and Understanding

# In[13]:


df = pd.read_csv('Data/combined_sold.csv')


# In[16]:


df.head(10)


# In[15]:


df.shape


# In[18]:


df.columns


# Notes on Key Columns (From Trestle Property MetaData)
# - `ListingKey`: Unique identifcation key for the proprties
# - `OriginalListPrice`: Original price of the property on the initial agreement between the seller and the seller's broker
# - `ClosePrice`: Final amount of money paid by the purchaser to the seller for the property
# - `LivingArea` : The total livable area within the structure (sq ft)
# - `BedroomsTotal` : The total number of bedrooms in the property
# - `BathroomsTotalInteger` : The total number of bathrooms in the property (including partial bathrooms)

# In[25]:


df.isnull().sum()


# In[28]:


df.isna().any(axis=1).sum()


# #### Key Column Distributions

# In[ ]:


cols = ['ClosePrice', 'LivingArea', 'BedroomsTotal', 'BathroomsTotalInteger']
for col in cols:
    print(f"Distribution for {col}: ")
    print(df[col].describe())

    # Since dataset is so large, log_scale the values to better see the distribution and avoid single bins
    sns.histplot(df[col], kde=True, log_scale=True)
    plt.show()

    sns.boxplot(x=df[col], log_scale=True)
    plt.show()


# ### Final Data Exploration Summary:
# 
# - After combining the monthly sold files and filtering the data, there are 302,332 rows of data that are Single Family Residences
# - 84 columns/variable types
# - There seems to be at least 1 null value in every row of data
# - Avg `ClosePrice` is about $1,295,772
# - Avg `LivingArea` is about 2,033 sq ft
# - Avg number of Bedrooms is about 3 and Bathrooms is 2
# 
# 
