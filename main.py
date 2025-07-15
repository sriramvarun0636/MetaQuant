import pandas as pd
from data.data import get_nifty_last_30min_data
from indicators.macd import calculate_macd
from indicators.vwap import calculate_vwap
from indicators.supertrend import calculate_supertrend
from agent.trade_agent import TradeAgent
import nsepython



# Fetch data
df = get_nifty_last_30min_data()
df.dropna(inplace=True)

# Calculate indicators
df = calculate_macd(df)
df = calculate_vwap(df)
df = calculate_supertrend(df)

# Fetch PCR data (implement your own method to get PCR per minute)
# For example, fetch current PCR and build a time series (dummy here)
import datetime
import numpy as np

# Usage:
option_chain = nsepython.nse_optionchain('NIFTY', 'expiry_date_here')
  # or your PCR source
# Dummy PCR time series indexed by datetime from df
pcr_times = df['Datetime']
pcr_values = np.random.uniform(0.8, 1.2, size=len(pcr_times)) 
