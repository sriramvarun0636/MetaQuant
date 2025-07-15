from typing import Dict

def generate_signal(indicators: Dict[str, int]) -> str:
    """
    Generate a trading signal based on indicator values.

    Args:
        indicators (Dict[str, int]): Dictionary of indicator names to their signals.
                                     Each signal should be +1 (bullish), -1 (bearish), or 0 (neutral).

    Returns:
        str: 'BUY' if all indicators are bullish (+1),
             'SELL' if all are bearish (-1),
             otherwise 'HOLD'.
    """
    values = list(indicators.values())
    
    if all(v == 1 for v in values):
        return 'BUY'
    elif all(v == -1 for v in values):
        return 'SELL'
    else:
        return 'HOLD'

if __name__ == "__main__":
    sample_indicators = {
        'vwap': 1,
        'supertrend': 1,
        'macd': 1,
        'pcr': 1
    }

    signal = generate_signal(sample_indicators)
    print(f"Generated signal: {signal}")
