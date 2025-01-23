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

    # Function to plot skewness for each numerical feature
    def plot_skewness(self):
        df = self.df.select_dtypes(include='number')
        skewness = df.skew().sort_values(ascending=False)
        
        plt.figure(figsize=(12, 6))
        
        # Create a bar plot with the correct hue assignment
        sns.barplot(x=skewness.index, y=skewness.values, hue=skewness.index, palette="pastel", edgecolor='black', legend=False)
        
        # Adding gridlines for better readability
        plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Title and labels
        plt.title("Skewness of Numerical Features", fontsize=18, fontweight='bold')
        plt.xlabel("Features", fontsize=14)
        plt.ylabel("Skewness", fontsize=14)
        
        # Customize ticks
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)

        # Adding value labels on top of the bars
        for index, value in enumerate(skewness):
            plt.text(index, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10, color='black')

        plt.tight_layout()
        plt.show()


    def plot_categorical_distribution(self):
        """
        Function to plot the distribution of categorical features in a DataFrame and 
        display the count values on top of each bar.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the dataset to be analyzed.
        """
        # Select categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        cols_with_few_categories = [col for col in categorical_cols if self.df[col].nunique() <= 10]

        # Set up the grid for subplots
        num_cols = len(cols_with_few_categories)
        num_rows = (num_cols + 1) // 2  # Automatically determine the grid size
        
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))
        axes = axes.flatten()

        for i, col in enumerate(cols_with_few_categories):
            ax = sns.countplot(data=self.df, x=col, ax=axes[i], hue=col, legend=False, palette="pastel", edgecolor='black')
            axes[i].set_title(f'Distribution of {col}', fontsize=16, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Frequency', fontsize=14)

            # Add count labels to the top of the bars
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', fontsize=12, color='black', 
                            xytext=(0, 5), textcoords='offset points')

        # Remove any empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.suptitle('Categorical Feature Distributions', fontsize=18, fontweight='bold', y=1.02)
        plt.show()


    def correlation_analysis(self):
        """Prepare and visualize the correlation matrix."""
        corr_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='cool', linewidths=0.5)
        plt.title('Correlation Matrix', fontsize=16)
        plt.show()

    def detect_outliers(self, cols):
            """
            Function to plot boxplots for numerical features to detect outliers.
            
            Parameters:
            -----------
            cols : list
                List of numerical columns to plot.
            """
            # Check if the provided columns exist in the DataFrame
            for col in cols:
                if col not in self.df.columns:
                    print(f"Warning: {col} is not in the DataFrame.")
                    return

            plt.figure(figsize=(15, 10))
            for i, col in enumerate(cols, 1):
                plt.subplot(3, 3, i)
                sns.boxplot(y=self.df[col], color='orange')
                plt.title(f'Boxplot of {col}', fontsize=12)
                plt.ylabel(col)

            plt.tight_layout()
            plt.show()
            print("✅ Boxplots displayed for outlier detection.")