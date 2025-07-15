def calculate_position_size(capital, risk_per_trade, entry_price, stop_loss_price):
    """
    Capital: total available
    Risk per trade: percentage (e.g., 0.01 for 1%)
    
    Returns units to buy.
    """
    risk_amount = capital * risk_per_trade
    risk_per_unit = abs(entry_price - stop_loss_price)

    if risk_per_unit == 0:
        return 0

    position_size = risk_amount / risk_per_unit
    return int(position_size)