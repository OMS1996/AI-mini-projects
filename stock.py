'''
The task requirements step-by-step to ensure the implementation meets the needs. The goal is to train an XGBoost model to predict stock buy signals based on historical data with a high degree of accuracy.

Requirements Recap:
Input Features: ADL, DEMA, RSI, POSC, Month, Sector.
Label: buy_signal (binary label indicating if the 10-day percentage price change is greater than 5%).
Model: XGBoost binary classification.
Performance: Achieve at least 70% accuracy, aim for 90%.
Steps:
Prepare Data: Ensure the features and labels are correctly set up in the training and test DataFrames.
Train Model: Use XGBoost to train a binary classifier.
Predict: Generate predictions on the test data.
Return Predictions: Format and return the predictions in the specified format.

'''

import xgboost as xgb
import pandas as pd

def stock_boost(X_train, y_train, X_test):
    # Convert training data into DMatrix format for XGBoost
    d_train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    
    # Convert test data into DMatrix format for XGBoost
    d_test = xgb.DMatrix(X_test, enable_categorical=True)
    
    # Set up the parameters for the XGBoost model
    params = {
        "objective": "binary:logistic",  # Binary classification objective
        "tree_method": "exact",          # Use the exact greedy algorithm
        "max_cat_to_onehot": 11,         # Maximum category to use one-hot encoding
        "eta": 0.32,                     # Learning rate
        "max_depth": 7                   # Maximum depth of a tree
    }
    
    # Train the XGBoost model
    stock_boost_model = xgb.train(params, d_train)
    
    # Make predictions on the test set
    raw_predictions = stock_boost_model.predict(d_test)
    
    # Apply threshold to get binary predictions
    threshold_predictions = [1 if value > 0.44 else 0 for value in raw_predictions]
    
    # Add predictions to the test DataFrame
    X_test['buy_signal'] = threshold_predictions
    
    # Extract only the necessary columns for the output
    y_test_predictions = X_test[['buy_signal']].copy()
    
    return y_test_predictions

# Example usage
# Assuming X_train, y_train, and X_test are pre-defined Pandas DataFrames
# predictions_df = stock_boost(X_train, y_train, X_test)
# predictions_df.head()
