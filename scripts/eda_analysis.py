import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from IPython.display import display

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Define the CreditRiskEDA class
class CreditRiskAnalysis:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the EDA class with the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to be analyzed.
        """
        self.df = df

    def data_overview(self):
        """Provide an overview of the dataset including shape, data types, and first few rows."""
        
        # Print header
        print("========================================")
        print("            Data Overview           ")
        print("========================================")
        
        # Shape of the DataFrame
        num_rows, num_columns = self.df.shape
        print(f"Number of Rows: {num_rows}")
        print(f"Number of Columns: {num_columns}")
        
        # Data types of columns
        print("\nColumn Data Types:")
        print(self.df.dtypes)
        
        # Display first five rows
        print("\nFirst Five Rows:")
        display(self.df.head())
        
        # Missing values
        missing_values = self.df.isnull().sum()
        print("\nMissing Values Overview:")
        print(missing_values[missing_values > 0])  
        
        # Footer
        print("========================================")
            
    def summary_statistics(self):
        """
        Function to compute summary statistics like mean, median, std, skewness, etc.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the dataset to be analyzed.
        
        Returns:
        --------
        summary_stats : pandas.DataFrame
            DataFrame containing the summary statistics for numeric columns.
        """
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include='number')
        
        # Calculate basic summary statistics
        summary_stats = numeric_df.describe().T
        
        # Add additional statistics
        summary_stats['median'] = numeric_df.median()
        summary_stats['mode'] = numeric_df.mode().iloc[0]
        summary_stats['skewness'] = numeric_df.skew()
        summary_stats['kurtosis'] = numeric_df.kurtosis()
        
        # Calculate additional statistics for dispersion
        summary_stats['range'] = numeric_df.max() - numeric_df.min()
        summary_stats['variance'] = numeric_df.var()
        
        # Calculate interquartile range (IQR) for dispersion
        summary_stats['IQR'] = numeric_df.quantile(0.75) - numeric_df.quantile(0.25)
        
        # Rename index for clarity
        summary_stats.index.name = 'Statistic'
        
        # Print summary statistics
        print("Summary Statistics:\n", summary_stats)
        
        return summary_stats