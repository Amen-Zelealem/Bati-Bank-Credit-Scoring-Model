import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the scripts directory to the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)
from feature_engineering import FeatureEngineering  # Replace with the actual file name if needed


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame for testing."""
        self.df = pd.DataFrame({
            'TransactionId': [1, 2, 3, 4],
            'CustomerId': [101, 101, 102, 103],
            'Amount': [100.0, 200.0, 150.0, np.nan],
            'TransactionStartTime': ['2023-01-01 10:00:00', 
                                     '2023-01-02 12:00:00', 
                                     '2023-01-03 15:00:00', 
                                     '2023-01-04 18:00:00'],
            'Category': ['A', 'B', 'A', 'C']
        })
    
    def test_create_aggregate_features(self):
        """Test aggregate feature creation."""
        df_result = FeatureEngineering.create_aggregate_features(self.df)
        self.assertIn('Total_Transaction_Amount', df_result.columns)
        self.assertIn('Average_Transaction_Amount', df_result.columns)
        self.assertEqual(df_result.loc[df_result['CustomerId'] == 101, 'Transaction_Count'].iloc[0], 2)
    
    def test_extract_time_features(self):
        """Test extraction of time-related features."""
        df_result = FeatureEngineering.extract_time_features(self.df)
        self.assertIn('Transaction_Hour', df_result.columns)
        self.assertIn('Transaction_Day', df_result.columns)
        self.assertIn('Transaction_Month', df_result.columns)
        self.assertEqual(df_result['Transaction_Hour'][0], 10)
    
    def test_encode_categorical_features(self):
        """Test encoding of categorical features."""
        df_result = FeatureEngineering.encode_categorical_features(self.df, ['Category'])
        self.assertTrue(np.issubdtype(df_result['Category'].dtype, np.integer))
    
    def test_handle_missing_values(self):
        """Test handling of missing values."""
        df_result = FeatureEngineering.handle_missing_values(self.df, strategy='mean')
        self.assertFalse(df_result['Amount'].isnull().any())
        self.assertAlmostEqual(df_result['Amount'].iloc[3], 150.0)  # Mean value is imputed
    
    def test_normalize_numerical_features(self):
        """Test normalization of numerical features."""
        df_result = FeatureEngineering.normalize_numerical_features(self.df, ['Amount'], method='normalize')
        self.assertAlmostEqual(df_result['Amount'].min(), 0.0)
        self.assertAlmostEqual(df_result['Amount'].max(), 1.0)

if __name__ == '__main__':
    unittest.main()
