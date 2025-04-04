"""
Author: Prathmesh Joshi
Course: Machine Learning
"""
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# Function: load_data
# Loads train and store data from CSV files.
def load_data(train_path='../dataset/train.csv', store_path='../dataset/store.csv'):
    """
    Loads the Rossmann train and store datasets.
    Args:
      train_path (str): Path to train.csv file.
      store_path (str): Path to store.csv file.
    Returns:
      train (DataFrame): Training dataset.
      store (DataFrame): Store-specific data.
    """
    try:
        train = pd.read_csv(train_path, parse_dates=['Date'])
        store = pd.read_csv(store_path)
        return train, store
    except Exception as e:
        raise(f"Error loading data: {e}")

# Function: merge_data
# Merge train and store data on Store ID.
def merge_data(train, store):
    """
    Merge the train dataset with the store dataset on the 'Store' column.
    Args:
      train (DataFrame): Training dataset.
      store (DataFrame): Store-specific dataset.
    Returns:
      df (DataFrame): Merged dataset.
    """
    df = pd.merge(train, store, on='Store', how='left')
    return df

# Function: preprocess_data
# Basic cleaning: filter out closed stores and fill missing values.
def preprocess_data(df):
    """
    Preprocess the merged dataframe.
      - Remove rows where Sales are 0 (closed stores or irrelevant days).
      - Fill missing values for numeric columns.
    Args:
      df (DataFrame): Merged dataset.
    Returns:
      df (DataFrame): Cleaned dataset.
    """
    # Remove records with zero sales (stores closed or non-operational days)
    if 'Sales' in df.columns:
      df = df[df['Sales'] > 0].copy()

    # Fill missing CompetitionDistance with a large number (indicating far competitor)
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].max(), inplace=True)

    # Fill missing values in competition open columns with the corresponding year or month from Date
    df['CompetitionOpenSinceYear'].fillna(df['Date'].dt.year, inplace=True)
    df['CompetitionOpenSinceMonth'].fillna(df['Date'].dt.month, inplace=True)

    # Fill missing Promo2 related values with 0
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna("None", inplace=True)

    return df

def feature_engineering(df):
    """
    Create new features to enrich the dataset.
      - Extract Year, Month, Day, WeekOfYear, DayOfWeek from Date.
      - Create a binary indicator for weekend.
      - Compute competition open since (combine year and month).
      - Create feature indicating time since last promo2 (if applicable).
      - Encode categorical variables.
      - Compute store-level RFM features (Recency, Frequency, Monetary) based on training data and merge them.
    Args:
      df (DataFrame): Preprocessed dataframe.
    Returns:
      df (DataFrame): DataFrame with new engineered features.
    """
    # Date features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Competition open since: combine year and month into a single numeric value
    df['CompetitionOpenSince'] = df['CompetitionOpenSinceYear'].astype(int)*100 + df['CompetitionOpenSinceMonth'].astype(int)

    # Promo2 duration: months since promo2 started (if promo2 is active)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].astype(int)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].astype(int)
    df['Promo2Active'] = df['Promo2'].apply(lambda x: 1 if x == 1 else 0)
    df['Promo2Since'] = df.apply(lambda row: (row['Year'] - row['Promo2SinceYear']) * 12 + (row['Month'] - (row['Promo2SinceWeek'] // 4 + 1))
                                 if row['Promo2Active'] == 1 and row['Promo2SinceYear'] > 0 else 0, axis=1)

    if 'Sales' in df.columns:
        # Log-transform the target variable "Sales" to stabilize variance
        df['LogSales'] = np.log1p(df['Sales'])

        # RFM Feature Engineering
        ref_date = df['Date'].max()
        rfm = df.groupby('Store').agg(
            LastSaleDate=('Date', 'max'),  # Track last sale date for dynamic adjustment
            Frequency=('Sales', 'count'),
            Monetary=('Sales', 'sum')
        ).reset_index()
        rfm['Recency'] = (ref_date - rfm['LastSaleDate']).dt.days
        rfm = rfm.drop(columns='LastSaleDate')
        # Log-transform
        rfm[['Recency', 'Frequency', 'Monetary']] = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
        # Save RFM and ref_date to disk to enforce training-first dependency
        rfm.to_csv('./model_cache_files/rfm_features.csv', index=False)
        with open('./model_cache_files/ref_date.txt', 'w') as f:
          f.write(str(ref_date))  # Convert Timestamp to string
        # Merge RFM
        df = pd.merge(df, rfm, on='Store', how='left')
    else:
        # For test data: enforce that RFM features and ref_date exist
        try:
            rfm = pd.read_csv('./model_cache_files/rfm_features.csv')
            with open('./model_cache_files/ref_date.txt', 'r') as f:
              ref_date = pd.to_datetime(f.read().strip())  # Read string and convert to Timestamp
        except FileNotFoundError:
            raise RuntimeError("RFM features not found. Train the model first to generate RFM features.")
        # Merge RFM
        df = pd.merge(df, rfm, on='Store', how='left')
        # Adjust Recency dynamically for test dates
        df['Recency'] += (df['Date'] - ref_date).dt.days
        # Reapply log1p to ensure consistency (optional, but recommended)
        df['Recency'] = np.log1p(df['Recency'])

    # Encode categorical features using one-hot encoding:
    df = pd.get_dummies(df, columns=['StateHoliday', 'StoreType', 'Assortment'], drop_first=True)

    return df

# Function: eda
# Generate sensible plots to explore the data.
def eda(df):
    """
    Perform Exploratory Data Analysis:
      - Plot distribution of Sales.
      - Plot time series of Sales for a sample store.
      - Plot correlation heatmap.
    Args:
      df (DataFrame): DataFrame after feature engineering.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Sales'], bins=50, kde=True)
    plt.title("Distribution of Sales")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    plt.show()

    # Time-series plot for a sample store (store 1)
    store1 = df[df['Store'] == 1].sort_values(by='Date')
    plt.figure(figsize=(12, 5))
    plt.plot(store1['Date'], store1['Sales'], marker='o', linestyle='-', markersize=3)
    plt.title("Store 1 Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Correlation heatmap of numerical features
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Heatmap")
    plt.show()

def train_model(X_train, y_train):
    """
    Train an XGBoost regressor using GridSearchCV for hyperparameter tuning.
    Args:
      X_train (DataFrame): Training features.
      y_train (Series): Target variable.
    Returns:
      best_model: Fitted XGBoost model with the best parameters.
    """
    # Define the model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Updated hyperparameter grid tuned for faster convergence on CPU within ~30 minutes.
    param_grid = {
        'n_estimators': [250, 750],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05],            # Focused on mid-range rate
        'subsample': [0.8],                 # Maintained regularization
        'colsample_bytree': [0.9]
    }

    # Use GridSearchCV for hyperparameter tuning
    grid = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                        scoring='neg_root_mean_squared_error', cv=3, verbose=1)
    grid.fit(X_train, y_train)
    print("Best parameters found: ", grid.best_params_)
    print("Best RMSE (negative): ", grid.best_score_)
    best_model = grid.best_estimator_
    return best_model

# Function: evaluate_model
# Predict on test data and evaluate using RMSE, MAE, MAPE. Also, plot predicted vs actual.
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using RMSE, MAE, and MAPE.
    Plot a comparison of predicted vs actual Sales.
    Args:
      model: Trained model.
      X_test (DataFrame): Test features.
      y_test (Series): Test target values.
    """
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    print("Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # Plot actual vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.xlabel("Actual Log Sales")
    plt.ylabel("Predicted Log Sales")
    plt.title("Actual vs. Predicted Log Sales")
    plt.show()

# Main function to run the full pipeline.
def main():
    """
    Main execution function.
    1. Load and merge data.
    2. Preprocess and feature engineer.
    3. Perform EDA (plots).
    4. Prepare data for modeling.
    5. Train XGBoost model with hyperparameter tuning.
    6. Evaluate model performance.
    """
    # 1. Load data (ensure train.csv and store.csv are in your working directory)
    train, store = load_data()
    print("Train and Store data loaded.")

    # 2. Merge train and store data
    df = merge_data(train, store)
    print("Data merged. Shape:", df.shape)

    # 3. Preprocess the data
    df = preprocess_data(df)
    print("Data preprocessed. Shape:", df.shape)

    # 4. Feature engineering
    df = feature_engineering(df)
    print("Feature engineering completed. New shape:", df.shape)

    # 5. Exploratory Data Analysis (EDA)
    print("Starting EDA...")
    eda(df)

    # 6. Prepare data for modeling
    # We use the engineered features and drop columns that are not needed.
    # For modeling, we will predict the log-transformed sales ("LogSales")
    drop_cols = ['Id', 'Sales', 'Date', 'LogSales', 'Customers']  # 'LogSales' is our target; 'Id' is not needed
    # Keep all numeric features; if there are any remaining non-numeric columns, drop them.
    features = df.drop(columns=drop_cols, errors='ignore')

    # Ensure that the features are all numeric
    features = features.select_dtypes(include=[np.number])

    X = features
    y = df['LogSales']  # we predict log sales

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # 7. Train the model
    print("Training the XGBoost model...")
    model = train_model(X_train, y_train)

    # 8. Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    model.save_model('./model_weights/xgb_rossmann_model.json')
    print("Model saved to xgb_rossmann_model.json")

    print("Pipeline completed.")

# Execute main function
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)