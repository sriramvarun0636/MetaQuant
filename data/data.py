import yfinance as yf
import pandas as pd

# Get data for last trading day only
data = yf.download("RELIANCE.NS", interval="1m", period="1d")

data.reset_index(inplace=True)

# Convert IST 3:00-3:30 PM to UTC 9:30-10:00 AM
from datetime import time

start_utc = time(9, 30)
end_utc = time(10, 0)

filtered_data = data[data['Datetime'].dt.time.between(start_utc, end_utc)]
print(filtered_data)
