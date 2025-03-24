"""
Author: Prathmesh Joshi
Course: Machine Learning
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from train import preprocess_data, feature_engineering

def evaluate_test_data(model, merged_test_df):
    """Applies preprocessing and feature engineering to test data."""
    # Apply preprocessing and feature engineering
    test_clean = preprocess_data(merged_test_df)
    test_fe = feature_engineering(test_clean)  # Will raise error if RFM files missing

    # Prepare features
    drop_cols = ['Id', 'Sales', 'Date', 'LogSales', 'Customers', 
             'StateHoliday_a', 'StoreType_b', 'StoreType_c', 'StoreType_d', 
             'Assortment_b', 'Assortment_c']
    X_test = test_fe.drop(columns=[col for col in drop_cols if col in test_fe.columns], errors='ignore')
    X_test = X_test.select_dtypes(include=[np.number])
    return model.predict(X_test)

# Load and merge external test data
test_df = pd.read_csv('../dataset/test.csv', parse_dates=['Date'])
store = pd.read_csv('../dataset/store.csv')
external_merged = pd.merge(test_df, store, on='Store', how='left')

# Load model and evaluate
model = xgb.XGBRegressor()
model.load_model('./model_weights/xgb_rossmann_model.json')
pred = evaluate_test_data(model, external_merged)

# Create submission file
submission_df = pd.DataFrame({'Id': external_merged['Id'], 'Sales': np.expm1(pred)})
submission_df.to_csv('./results/submission.csv', index=False)

print("Testing Completed. Submission.csv stored in the results folder")