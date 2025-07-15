def calculate_atr(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

def calculate_supertrend(df, period=20, multiplier=2):
    df = calculate_atr(df, period)

    df['UpperBand'] = ((df['High'] + df['Low']) / 2) + (multiplier * df['ATR'])
    df['LowerBand'] = ((df['High'] + df['Low']) / 2) - (multiplier * df['ATR'])

    df['Supertrend'] = True  # True = bullish, False = bearish

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]

        if curr['Close'] > prev['UpperBand']:
            df.at[df.index[i], 'Supertrend'] = True
        elif curr['Close'] < prev['LowerBand']:
            df.at[df.index[i], 'Supertrend'] = False
        else:
            df.at[df.index[i], 'Supertrend'] = df.at[df.index[i-1], 'Supertrend']

            # Adjust bands if trend remains same
            if df.at[df.index[i], 'Supertrend'] and curr['LowerBand'] < prev['LowerBand']:
                df.at[df.index[i], 'LowerBand'] = prev['LowerBand']
            if not df.at[df.index[i], 'Supertrend'] and curr['UpperBand'] > prev['UpperBand']:
                df.at[df.index[i], 'UpperBand'] = prev['UpperBand']

    return df
