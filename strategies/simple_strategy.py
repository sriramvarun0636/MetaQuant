def generate_signal(row):
    """
    Evaluate trading conditions and return trade signal based on the logic:
    BUY CALL if:
        - VWAP > Market Price
        - MACD is bullish
        - Supertrend is bullish
        - PCR flipped from >1 to <1 (within last 5 min)
    BUY PUT if inverse

    Returns:
        str: 'BUY_CALL', 'BUY_PUT', or 'HOLD'
    """
    if (
        row['vwap'] > row['price'] and
        row['macd_signal'] == 1 and
        row['supertrend'] == 1 and
        row['pcr_flip'] == 'BEARISH'
    ):
        return 'BUY_CALL'

    elif (
        row['vwap'] < row['price'] and
        row['macd_signal'] == -1 and
        row['supertrend'] == -1 and
        row['pcr_flip'] == 'BULLISH'
    ):
        return 'BUY_PUT'

    return 'HOLD'
