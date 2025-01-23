import unittest
import pandas as pd
from io import StringIO
import numpy as np
import sys
import os

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
