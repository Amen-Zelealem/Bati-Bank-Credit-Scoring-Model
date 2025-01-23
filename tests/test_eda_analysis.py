import unittest
import pandas as pd
from io import StringIO
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add the scripts directory to the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)

from eda_analysis import CreditRiskAnalysis


class TestCreditRiskAnalysis(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            "Age": [25, 30, 35, np.nan],
            "Income": [50000, 60000, 65000, 70000],
            "CreditScore": [700, 720, 680, 690],
        }
        self.df = pd.DataFrame(data)
        self.eda = CreditRiskAnalysis(self.df)

    def test_data_overview_output(self):
        # Capture the printed output
        captured_output = StringIO()
        sys.stdout = captured_output

        # Call the data_overview method
        self.eda.data_overview()

        # Reset redirect.
        sys.stdout = sys.__stdout__

        # Get the printed output
        output = captured_output.getvalue()

        # Check if the output contains expected strings
        self.assertIn("Data Overview", output)
        self.assertIn("Number of Rows: 4", output)
        self.assertIn("Number of Columns: 3", output)
        self.assertIn("Column Data Types:", output)
        self.assertIn("Age            float64", output)  # Adjusted for spacing
        self.assertIn("Income           int64", output)
        self.assertIn("CreditScore      int64", output)
        self.assertIn("Missing Values Overview:", output)
        self.assertIn("Age    1", output)  # Check for missing values

    def test_summary_statistics(self):
        # Generate summary statistics
        summary_stats = self.eda.summary_statistics()

        # Check the shape of the summary statistics
        self.assertEqual(summary_stats.shape[0], 3, "Expected 3 statistics for the numeric columns.")
        self.assertEqual(summary_stats.shape[1], 15, "Expected 15 statistics columns in the summary.")

        # Check specific statistical values for the 'Age' column
        self.assertAlmostEqual(summary_stats.loc['Age', 'mean'], 30.0, places=1, 
                            msg="Mean of 'Age' should be approximately 30.0.")
        self.assertAlmostEqual(summary_stats.loc['Age', 'median'], 30.0, places=1, 
                            msg="Median of 'Age' should be approximately 30.0.")
        self.assertAlmostEqual(summary_stats.loc['Age', 'std'], 5.0, places=1, 
                            msg="Standard deviation of 'Age' should be approximately 5.0.")
        self.assertAlmostEqual(summary_stats.loc['Age', 'skewness'], 0.0, places=1, 
                            msg="Skewness of 'Age' should be close to 0.")

    def test_plot_numerical_distribution(self):
        # Create a sample DataFrame with numeric columns
        data = {
            'A': np.random.normal(loc=0, scale=1, size=100),
            'B': np.random.normal(loc=5, scale=2, size=100),
            'C': np.random.normal(loc=-5, scale=1.5, size=100)
        }
        test_df = pd.DataFrame(data)
        self.eda.df = test_df  # Set the DataFrame for testing

        # List of numeric columns to plot
        numeric_cols = ['A', 'B', 'C']
        print(f"Starting plot for columns: {numeric_cols}")  # Debugging line
        
        # Test if the plot function runs without errors
        try:
            self.eda.plot_numerical_distribution(numeric_cols)
        except Exception as e:
            self.fail(f"plot_numerical_distribution raised {type(e).__name__} unexpectedly: {e}")

        # Ensure that a figure was created
        fig = plt.gcf()
        self.assertIsNotNone(fig, "No figure was created.")
       
if __name__ == '__main__':
    unittest.main()