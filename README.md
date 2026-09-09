# IDXExchange-DS-Su26

This project for the IDXExchange Summer 2026 cohort aims to predict closing prices of single-family homes using the CRMLS (California Regional Multiple Listing Service).

## Repository Structure
```
01_exploration.ipynb               - Basic exploratory data analysis
02_preprocessing.ipynb             - Dataset cleaning and preprocessing 
03_baseline_model.ipynb            - Training baseline linear regression model
04_model_comparison.ipynb          - Training decision tree and random forest models
05_feature_engineering.ipynb       - Additional feature engineering and geospatial analysis for retraining models 
06_advanced_models.ipynb           - Training XGBoost models
07_evaluations.ipynb               - Final performance comparison of all predictive models
README.md
app.py                             - Code for StreamLit demo app
house_model.pkl                    - Saved best XGBoost predictive model for demo app 
preprocesser.pkl                   - Saved preprocessing pipeline for demo app 
utils.py                           - Main preprocessing pipeline for training dataset
```

## Dataset Source

The data for this project comes from the CRMLS (California Regional Multiple Listing Service) which is used by real estate professionals to list information about different properties. It includes various features such as living area, number of bedrooms, bathrooms, lot size, and listing and close prices. The data was retrieved from IDXExchange through FileZilla and contains information from 2024-2026, although only the most recent 12 months was used in training the machine learning models. Raw files have not been uploaded to this repository for large size and restrictive policies.

## Preprocessing

Dataset cleaning and the preprocessing pipeline can be seen in `02_preprocessing.ipynb` and `utils.py`. 

The cleaning process includes
- Combining all CRMLS files into a single dataset for analysis
- Filtering data to only keep rows where `PropertyType == "Residential"` and `PropertySubType == "SingleFamilyResidence"`
- Dropping columns/features that have over 80% missing/null values, are identifiers, or have low predictive power
- Dropping rows with invalid numerical values and have a `ClosePrice` outside the 99th percentile (This increased model accuracy by over 10%)
- Filling in missing binary features with 0, numerical features with medians from training dataset, and categorical features with unknown
- Used OneHotEncoder for binary categorical features, log transformed numerical features to prevent dataset skew. 

After cleaning, about 10269150 rows of data was removed, leaving 5161856 rows of data to be used for creating our training/testing sets. Data was split into a training window of 12 months and a testing window consisting of the most recent month.

## Models Tested
With feature enginnering additional columns (`BedBathRatio`, `AgeProperty`) and spatially joining the CA School District Areas 2024-25 boundaries with each property’s coordinates (`SchoolDistrict`), the following models were tested and evaluated on the test set.
| Model | R<sup>2</sup> Score | MAPE | MdAPE |
| --- | --- | --- | --- |
| Baseline Linear Regression | 0.8251  | 0.2154 | 0.1465 |
| Decision Tree | 0.7528 | 0.1748 | 0.1087 |
| Random Forest | 0.8383 | 0.1574 | 0.0993 |
| XGBoost | 0.8799 | 0.1385 | 0.0983 |

## Best Results
The best performance was from an XGBoost model with `n_estimators=2000`,`learning_rate=0.05`,`max_depth=8`, and `eval_metric='rmse'`. It was trained on 122208 rows of data from May 2025 to May 2026, with the testing set being from June 2026. 

## How to Run Code and Launch StreamLit App
To properly rerun the code, data must be downloaded from the California Regional Multiple Listing Service (CRMLS) individually. Without access to dataset, notebooks can only be viewed as it from Github or using Jupyter Notebooks. 
