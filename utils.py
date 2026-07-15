
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def create_time_split(data, date_col, X_months, test_start_date, max_date):
    """
    Splits the data into a training window of X months and a testing window 
    consisting of the most recent month.
    """

    train_start_date = test_start_date - pd.DateOffset(months=X_months)
    
    train_df = data[(data[date_col] >= train_start_date) & (data[date_col] < test_start_date)]
    test_df = data[(data[date_col] >= test_start_date) & (data[date_col] <= max_date)]
    
    print(f"Training Window (X={X_months} months): {train_start_date.date()} to {test_start_date.date()} | Rows: {len(train_df)}")
    print(f"Testing Window (1 month): {test_start_date.date()} to {max_date.date()}")
          
    return train_df, test_df


import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer

def transform_binary_features(X):
    X = X.copy()

    mapping = {
        "Y": 1,
        "N": 0,
        True: 1,
        False: 0
    }

    for col in X.columns:
        X[col] = X[col].map(mapping)

    return X

def clean_levels_column(X):

    def categorize(level):

        if pd.isna(level):
            return np.nan

        level = str(level)

        if "ThreeOrMore" in level:
            return "ThreeOrMore"
        elif "Two" in level:
            return "Two"
        elif "MultiSplit" in level:
            return "MultiSplit"
        elif "One" in level:
            return "One"
        else:
            return "Other"

    return X.iloc[:,0].apply(categorize).to_frame()

binary_pipeline = Pipeline([
    ("binary_map",
        FunctionTransformer(transform_binary_features,
                            validate=False)),
    ("imputer",
        SimpleImputer(
            strategy="most_frequent",
            add_indicator=True
        ))
])

numeric_pipeline = Pipeline([
    ("imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
    ("log",
        FunctionTransformer(
            np.log1p,
            validate=False
        ))
])

levels_pipeline = Pipeline([

    ("clean",
        FunctionTransformer(
            clean_levels_column,
            validate=False
        )),

    ("imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="Unknown"
        )),

    ("onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
])

categorical_pipeline = Pipeline([

    ("imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="Missing"
        )),

    ("encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        ))
])

def get_preprocessing_pipeline(df=None):
    binary_cols = [
        "PoolPrivateYN",
        "ViewYN",
        "FireplaceYN",
        "NewConstructionYN",
        "AttachedGarageYN"
    ]

    numeric_cols = [
        "LivingArea",
        "LotSizeArea",
        "LotSizeSquareFeet",
    ]

    level_cols = ["Levels"]

    if df is not None:

        categorical_cols = [
            c for c in df.select_dtypes(include="object").columns
            if c not in binary_cols + level_cols
        ]

    else:

        categorical_cols = []

    
    preprocessor = ColumnTransformer(

    transformers=[

        ("binary",
         binary_pipeline,
         binary_cols),

        ("numeric",
         numeric_pipeline,
         numeric_cols),

        ("levels",
         levels_pipeline,
         level_cols),

        ("categorical",
         categorical_pipeline,
         categorical_cols)
    ],

    remainder="drop"
)
    return preprocessor

