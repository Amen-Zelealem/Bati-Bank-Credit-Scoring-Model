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
    
    def plot_numerical_distribution(self, cols):
        """
        Function to plot multiple histograms in a grid layout.

        Parameters:
        -----------
        cols : list
            List of numeric columns to plot.
        """

        # Select numeric columns
        n_cols = len(cols)

        # Automatically determine grid size (square root method)
        n_rows = math.ceil(n_cols**0.5)
        n_cols = math.ceil(n_cols / n_rows)
        
        # Create a color palette
        palette = sns.color_palette("pastel", n_cols)

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), constrained_layout=True)
        axes = axes.flatten()

        for i, col in enumerate(cols):
            sns.histplot(self.df[col], bins=15, kde=True, color=palette[i % n_cols], edgecolor='black', ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=16, fontweight='bold')
            axes[i].set_xlabel(col, fontsize=14)
            axes[i].set_ylabel('Frequency', fontsize=14)
            axes[i].axvline(self.df[col].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
            axes[i].axvline(self.df[col].median(), color='green', linestyle='dashed', linewidth=2, label='Median')
            axes[i].legend(fontsize=12, loc='upper right')

            # Enhance grid and ticks
            axes[i].grid(axis='y', alpha=0.7)
            axes[i].tick_params(axis='both', which='major', labelsize=12)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle('Distribution of Numeric Variables', fontsize=20, fontweight='bold', y=1.02)
        plt.show()