# Import necessary library
import pandas as pd

# Load data function
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file and return a DataFrame.

    Parameters:
    -----------
    file_path : str
        The path to the dataset file.

    Returns:
    --------
    pd.DataFrame
        Loaded dataset in a pandas DataFrame format.
    """
    try:
        df = pd.read_csv(file_path, index_col=0)
        print(f"Data successfully loaded from '{file_path}' with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty or invalid.")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
    return pd.DataFrame()
