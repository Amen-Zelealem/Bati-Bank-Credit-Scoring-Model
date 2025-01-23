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
            