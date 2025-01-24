import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class FeatureEngineering:
    """
    A class to perform feature engineering tasks for a given DataFrame.

    Methods
    -------
    create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame
        Creates aggregate features such as total, average, count, and standard deviation of transaction amounts.

    extract_time_features(df: pd.DataFrame) -> pd.DataFrame
        Extracts time-related features from the TransactionStartTime column.

    encode_categorical_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame
        Encodes categorical variables into numerical format using One-Hot or Label Encoding.

    handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame
        Handles missing values by imputation or removal.

    normalize_numerical_features(df: pd.DataFrame, numerical_cols: list, method: str = 'standardize') -> pd.DataFrame
        Normalizes or standardizes numerical features.
    """

    @staticmethod
    def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates aggregate features such as total, average, count, and standard deviation of transaction amounts.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing transaction data.

        Returns
        -------
        pd.DataFrame
            The original DataFrame with added aggregate features for each customer.
        """
        agg_features = df.groupby('CustomerId').agg(
            Total_Transaction_Amount=('Amount', 'sum'),
            Average_Transaction_Amount=('Amount', 'mean'),
            Transaction_Count=('TransactionId', 'count'),
            Std_Transaction_Amount=('Amount', 'std')
        ).reset_index()
        
        # Merge aggregate features with the original DataFrame on CustomerId
        df = df.merge(agg_features, on='CustomerId', how='left')
        return df

    @staticmethod
    def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts time-related features from the TransactionStartTime column.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing transaction data.

        Returns
        -------
        pd.DataFrame
            DataFrame with added time-related features and the original TransactionStartTime removed.
        """
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
        df['Transaction_Day'] = df['TransactionStartTime'].dt.day
        df['Transaction_Month'] = df['TransactionStartTime'].dt.month
        df['Transaction_Year'] = df['TransactionStartTime'].dt.year
        
        return df

