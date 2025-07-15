import pandas as pd
import numpy as np
import os

# Create folders if not present
os.makedirs("data", exist_ok=True)

# Configuration
START_TIME = "2023-01-01 09:15"
NUM_MINUTES = 300  # 5 trading hours (09:15 to 14:15)

# Generate timestamps
timestamps = pd.date_range(start=START_TIME, periods=NUM_MINUTES, freq="1min")

# Generate mock price data (simulate up/downtrend)
base_price = 100
price = np.cumsum(np.random.normal(0, 0.3, size=NUM_MINUTES)) + base_price

# Generate mock VWAP around price (a bit noisy)
vwap = price + np.random.normal(0, 0.1, size=NUM_MINUTES)

# Generate mock MACD signals (-1, 0, 1) with some trend persistence
macd_signal = np.zeros(NUM_MINUTES)
for i in range(1, NUM_MINUTES):
    macd_signal[i] = macd_signal[i-1] if np.random.rand() < 0.8 else np.random.choice([-1, 0, 1])

# Generate mock Supertrend signals (-1 or 1) switching every 50 bars roughly
supertrend = np.array([1 if (i // 50) % 2 == 0 else -1 for i in range(NUM_MINUTES)])

# Generate PCR ratio as float >0 (simulate mean around 1)
pcr_ratio = 0.9 + 0.2 * np.sin(np.linspace(0, 10*np.pi, NUM_MINUTES)) + np.random.normal(0, 0.05, NUM_MINUTES)
pcr_ratio = np.clip(pcr_ratio, 0.5, 1.5)

# Calculate PCR flip: 'BULLISH' if PCR ratio crossed above 1 in last 5 mins, 'BEARISH' if crossed below 1, else 'NONE'
pcr_flip = []
window = 5
for i in range(NUM_MINUTES):
    if i < window:
        pcr_flip.append("NONE")
        continue
    recent = pcr_ratio[i-window:i]
    if (recent[-2] > 1 and recent[-1] < 1):
        pcr_flip.append("BEARISH")
    elif (recent[-2] < 1 and recent[-1] > 1):
        pcr_flip.append("BULLISH")
    else:
        pcr_flip.append("NONE")

# Create DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "price": price,
    "vwap": vwap,
    "macd_signal": macd_signal,
    "supertrend": supertrend,
    "pcr_ratio": pcr_ratio,
    "pcr_flip": pcr_flip
})

# Save to CSV
output_path = "data/mock_processed_data.csv"
df.to_csv(output_path, index=False)

print(f"✅ Mock data generated: {output_path}")
