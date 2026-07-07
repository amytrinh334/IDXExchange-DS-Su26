import pandas as pd

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