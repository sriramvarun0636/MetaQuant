def calculate_sl_tp(entry_price: float, supertrend: float, direction: str, buffer: float = 0.05):
    if direction == 'CALL':
        sl = supertrend - buffer
        if sl >= entry_price:
            sl = entry_price - buffer
        tp = entry_price + 3 * (entry_price - sl)
    elif direction == 'PUT':
        sl = supertrend + buffer
        if sl <= entry_price:
            sl = entry_price + buffer
        tp = entry_price - 3 * (sl - entry_price)
    else:
        raise ValueError("Invalid direction: must be 'CALL' or 'PUT'")

    sl = round(sl, 2)
    tp = round(tp, 2)

    # Debug prints (remove or comment out in production)
    print(f"Direction: {direction}, Entry: {entry_price}, SL: {sl}, TP: {tp}")

    return sl, tp
