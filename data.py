#list of call options

"""import pandas as pd
from kiteconnect import KiteConnect

api_key = "03ep39d0navol1d9"
access_token = "in3bIGsGgAlCenN9WxC6ObsuqwN4KRzR"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

instruments = kite.instruments(exchange="NFO")
df = pd.DataFrame(instruments)

# Filter CE options for NIFTY expiring on or after 17 July 2025
expiry_filter = pd.to_datetime("2025-07-17")
filtered = df[
    (df['name'].str.contains("NIFTY")) &
    (df['instrument_type'] == "CE") &
    (pd.to_datetime(df['expiry']) >= expiry_filter)
]

print(filtered[['tradingsymbol', 'strike', 'expiry']].head(20))"""


#ATM CALL
"""
import pandas as pd
from kiteconnect import KiteConnect

api_key = "03ep39d0navol1d9"
access_token = "in3bIGsGgAlCenN9WxC6ObsuqwN4KRzR"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Get all NFO instruments (Nifty options)
instruments = kite.instruments(exchange="NFO")
df = pd.DataFrame(instruments)

# Convert expiry column to datetime
df['expiry'] = pd.to_datetime(df['expiry'])

# Filter for exact NIFTY ATM call option
atm_strike = 25250
expiry_date = pd.to_datetime("2025-07-17")

atm_call = df[
    (df['tradingsymbol'].str.startswith("NIFTY")) &  # starts with NIFTY
    (df['strike'] == atm_strike) &
    (df['instrument_type'] == "CE") &
    (df['expiry'] == expiry_date)
]

if atm_call.empty:
    print(f"No ATM Call Option found for strike {atm_strike} and expiry {expiry_date.date()}")
else:
    print(atm_call[['tradingsymbol', 'strike', 'expiry', 'instrument_token']])
"""


"""
import pandas as pd
from kiteconnect import KiteConnect

api_key = "03ep39d0navol1d9"
access_token = "in3bIGsGgAlCenN9WxC6ObsuqwN4KRzR"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Fetch all NFO instruments
instruments = kite.instruments(exchange="NFO")
df = pd.DataFrame(instruments)

expiry_date = "2025-07-17"
strike_price = 25250

# Filter ATM CE and PE for NIFTY expiring on 17 July 2025
atm_options = df[
    (df['name'] == "NIFTY") &
    (df['strike'] == strike_price) &
    (df['expiry'] == expiry_date) &
    (df['instrument_type'].isin(["CE", "PE"]))
]

# Extract instrument tokens for CE and PE
instrument_tokens = atm_options['instrument_token'].astype(str).tolist()
token_str = ",".join(instrument_tokens)   # <-- convert list to CSV string

quotes = kite.quote(token_str)            # <-- pass string, not list


# Collect and print desired data
for idx, row in atm_options.iterrows():
    token_str = str(row['instrument_token'])
    quote_data = quotes.get(token_str, {})
    print(f"Symbol: {row['tradingsymbol']}")
    print(f"Strike: {row['strike']}")
    print(f"Expiry: {row['expiry']}")
    print(f"Option Type: {row['instrument_type']}")
    print(f"LTP: {quote_data.get('last_price', 'N/A')}")
    print(f"Open Interest (OI): {quote_data.get('oi', 'N/A')}")
    print(f"Volume: {quote_data.get('volume', 'N/A')}")
    print("-" * 40)
"""


import pandas as pd
from kiteconnect import KiteConnect

api_key = "03ep39d0navol1d9"
access_token = "in3bIGsGgAlCenN9WxC6ObsuqwN4KRzR"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Fetch all NFO instruments (options)
instruments = kite.instruments(exchange="NFO")
df = pd.DataFrame(instruments)

# Filter for NIFTY options CE and PE with strike 25250 and expiry 2025-07-17
expiry_filter = pd.to_datetime("2025-07-17")
strike_price = 25250

atm_options = df[
    (df['name'] == "NIFTY") &
    (df['strike'] == strike_price) &
    (pd.to_datetime(df['expiry']) == expiry_filter) &
    (df['instrument_type'].isin(["CE", "PE"]))
]

print("ATM Options found:")
print(atm_options[['tradingsymbol', 'instrument_token', 'instrument_type']])

# Extract instrument tokens as strings and join with commas for kite.quote()
instrument_tokens = atm_options['instrument_token'].astype(str).tolist()
token_str = ",".join(instrument_tokens)

# Fetch quotes for these tokens
quotes = kite.quote(token_str)

print("\nOption details with LTP, OI, Volume:\n")
for _, row in atm_options.iterrows():
    token = str(row['instrument_token'])
    tradingsymbol = row['tradingsymbol']
    instrument_type = row['instrument_type']

    if token in quotes:
        quote_data = quotes[token]
        ltp = quote_data.get('last_price', 'NA')
        oi = quote_data.get('oi', 'NA')
        volume = quote_data.get('volume', 'NA')
        print(f"{tradingsymbol} ({instrument_type}): LTP={ltp}, OI={oi}, Volume={volume}")
    else:
        print(f"No quote data found for {tradingsymbol}")