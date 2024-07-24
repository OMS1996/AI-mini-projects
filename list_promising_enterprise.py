"""
Visa wants to incentivize small and medium businesses (SMBs) to apply for business credit cards. A
product manager within Visa decided to offer business rewards for SMBs when they conduct
transactions with partnered enterprises. Visa has a transaction database which stores transaction
information such as average transaction amount and frequency of transactions between SMBs and
enterprises. Using the provided dataframes, provide the product manager the top 10 enterprises
Visa should partner with to entice SMBs to apply for Visa business credit cards.
You're given two Pandas dataframes. The first dataframe contains the number of monthly
transactions each SMB conducts with each respective enterprise. The rows represent each SMB and
the columns represent each enterprise. This is a sparse dataframe.
The second dataframe contains the average transaction amount each SMB conducts with each
respective enterprise. The rows represent each SMB and the columns represent each enterprise.
This is a sparse dataframe and only has values where the above dataframe has non-zero values.
Use these two dataframes to derive a new dataframe that contains the average total monthly
transaction amount. This resultant dataframe will be used on scikit-learn's TruncatedSVD. The
output of the TruncatedSVD will be an embedding for each enterprise. The top 10 enterprises for
Visa to partner with will have the maximum embedding values.
The list of the top 10 enterprises that you return will be evaluated against the ground truth set using
the Jaccard similarity. The Jaccard similarity between your answer and the ground truth must be
greater than 66%.
"""


import pandas as pd
from sklearn.decomposition import TruncatedSVD

def b2b_svd(avg_trn_amount: pd.DataFrame, num_monthly_trxn: pd.DataFrame) -> list:
    """
    Function to identify the top 10 enterprises Visa should partner with based on transaction data.

    Parameters:
    avg_trn_amount (pd.DataFrame): DataFrame containing the average transaction amount each SMB conducts with each enterprise.
    num_monthly_trxn (pd.DataFrame): DataFrame containing the number of monthly transactions each SMB conducts with each enterprise.

    Returns:
    list: List of the top 10 enterprises.
    """
    
    # Step 1: Calculate the average total monthly transaction amount
    # Multiply the average transaction amount by the number of transactions for each SMB-enterprise pair
    average_monthly_total = avg_trn_amount.mul(num_monthly_trxn)
    
    # Step 2: Transpose the DataFrame to have enterprises as rows and SMBs as columns
    average_monthly_total_T = average_monthly_total.T
    
    # Step 3: Apply TruncatedSVD for dimensionality reduction
    # Initialize the TruncatedSVD model with 10 components and 10 iterations
    enterprise_svd = TruncatedSVD(n_components=10, n_iter=10)
    
    # Fit and transform the transposed DataFrame to get the embeddings for each enterprise
    enterprise_embeddings = enterprise_svd.fit_transform(average_monthly_total_T)
    
    # Step 4: Identify the top enterprises
    # Find the indices of the enterprises with the highest embedding values across each component
    max_enterprise_embeddings = enterprise_embeddings.argmax(axis=0)
    
    # Extract the rows (enterprises) corresponding to these indices
    max_enterprise_rows = average_monthly_total_T.iloc[max_enterprise_embeddings]
    
    # Return the list of top enterprise names
    return list(max_enterprise_rows.index.values)
