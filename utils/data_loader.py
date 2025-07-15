# utils/data_loader.py

import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the 1-minute candle + indicator data.

    Args:
        filepath (str): Path to the CSV file containing mock or real options data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with datetime index and necessary columns.
    """
    df = pd.read_csv(filepath)

    # Ensure timestamp is parsed and sorted
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)

    # Optional: drop duplicates or nulls if needed
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    return df


def load_pcr(filepath: str) -> pd.Series:
    """
    Load PCR time series.

    Args:
        filepath (str): Path to the CSV containing PCR data with 'timestamp' and 'pcr' columns.

    Returns:
        pd.Series: Time-indexed PCR data.
    """
    pcr_df = pd.read_csv(filepath)
    pcr_df['timestamp'] = pd.to_datetime(pcr_df['timestamp'])
    pcr_df.set_index('timestamp', inplace=True)
    pcr_df.sort_index(inplace=True)

    return pcr_df['pcr']
