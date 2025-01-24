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

    @staticmethod
    def encode_categorical_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        """
        Encodes categorical variables into numerical format using One-Hot or Label Encoding.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing categorical data.
        categorical_cols : list
            List of categorical columns to encode.

        Returns
        -------
        pd.DataFrame
            DataFrame with encoded categorical features.
        """
        # Label Encoding for all categorical columns
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = label_encoder.fit_transform(df[col].astype(str))  # Ensure values are numeric
        
        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handles missing values by imputation or removal.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame with missing values.
        strategy : str, optional
            The strategy for handling missing values ('mean', 'median', 'mode', 'remove').

        Returns
        -------
        pd.DataFrame
            DataFrame with handled missing values.
        """
        if strategy in ['mean', 'median', 'mode']:
            imputer = SimpleImputer(strategy=strategy)
            df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(df.select_dtypes(include=[np.number]))
        elif strategy == 'remove':
            df.dropna(inplace=True)

        return df
