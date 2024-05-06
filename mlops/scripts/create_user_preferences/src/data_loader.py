import pandas as pd
from config import DATA_DIR
import os

def load_data(file_name):
    """
    Load data from a specified file within the data directory.
    Only a sample of 2000 rows will be used to load data
    as the original dataset is very large.
    Args:
    file_name (str): The name of the file to load.
    
    Returns:
    DataFrame: A pandas DataFrame containing the loaded data.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data.iloc[:2000, :]
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    data = load_data('million_song_dataset.csv')  # Ensure 'example.csv' exists in the '/data' folder