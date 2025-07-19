import pandas as pd
import os

def load_and_split_data(file_path, train_date_split='20150101', test_date_split='20141231'):
    """
    Loads the stock headlines data, handles missing values, and splits it into
    training and testing sets based on a date.

    Args:
        file_path (str): The absolute path to the 'Stock Headlines.csv' file.
        train_date_split (str): Date string in 'YYYYMMDD' format to split
                                data into training (before) and testing (after).

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Training data (headlines).
            - pd.Series: Training labels.
            - pd.DataFrame: Testing data (headlines).
            - pd.Series: Testing labels.
    """
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None, None

    df.dropna(inplace=True)
    df_copy = df.copy()

    # Splitting the dataset into train and test set
    train = df_copy[df_copy['Date'] < train_date_split]
    test = df_copy[df_copy['Date'] > test_date_split]

    # Splitting the dataset into features (headlines) and labels
    y_train = train['Label']
    train_headlines = train.iloc[:, 2:] # Assuming 'Date' is col 0, 'Label' is col 1, headlines start from col 2

    y_test = test['Label']
    test_headlines = test.iloc[:, 2:] # Assuming 'Date' is col 0, 'Label' is col 1, headlines start from col 2

    return train_headlines, y_train, test_headlines, y_test

if __name__ == '__main__':
    # Example usage:
    # Assuming 'Stock Headlines.csv' is in the 'Data' directory relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file_path = os.path.join(project_root, 'Data', 'Stock Headlines.csv')

    train_df, y_train_series, test_df, y_test_series = load_and_split_data(data_file_path)

    if train_df is not None:
        print("Data loaded and split successfully.")
        print(f"Train headlines shape: {train_df.shape}")
        print(f"Train labels shape: {y_train_series.shape}")
        print(f"Test headlines shape: {test_df.shape}")
        print(f"Test labels shape: {y_test_series.shape}")
    else:
        print("Failed to load data.")
