import unittest
import pandas as pd
from unittest.mock import patch, mock_open
import os
import sys

# Add the scripts directory to the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)


from data_loader import load_data
class TestLoadData(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_load_data_success(self, mock_read_csv):
        # Mock the DataFrame returned by read_csv
        mock_df = pd.DataFrame({'Column1': [1, 2], 'Column2': [3, 4]})
        mock_read_csv.return_value = mock_df

        # Call the load_data function
        result = load_data('valid_file.csv')

        # Assert that the DataFrame returned is the same as the mock
        pd.testing.assert_frame_equal(result, mock_df)
        mock_read_csv.assert_called_once_with('valid_file.csv', index_col=0)

    @patch('pandas.read_csv', side_effect=FileNotFoundError("File not found"))
    def test_load_data_file_not_found(self, mock_read_csv):
        with patch('builtins.print') as mocked_print:
            result = load_data('invalid_file.csv')
            self.assertTrue(mocked_print.called)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)

    @patch('pandas.read_csv', side_effect=pd.errors.EmptyDataError("Empty data"))
    def test_load_data_empty_file(self, mock_read_csv):
        with patch('builtins.print') as mocked_print:
            result = load_data('empty_file.csv')
            self.assertTrue(mocked_print.called)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)

    @patch('pandas.read_csv', side_effect=Exception("Some unexpected error"))
    def test_load_data_unexpected_error(self, mock_read_csv):
        with patch('builtins.print') as mocked_print:
            result = load_data('error_file.csv')
            self.assertTrue(mocked_print.called)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)

if __name__ == '__main__':
    unittest.main()