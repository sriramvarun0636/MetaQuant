import pandas as pd
from strategies.simple_strategy import generate_signal
from riskmanagement.position_sizer import calculate_position_size
from riskmanagement.sl_tp import calculate_sl_tp

CAPITAL = 100000  # total capital
RISK_PER_TRADE = 0.01  # 1% of capital
TRADE_WINDOW_MINUTES = 5

def run_backtest(data: pd.DataFrame):
    trades = []
    i = 0

    while i < len(data) - TRADE_WINDOW_MINUTES:
        window = data.iloc[i:i + TRADE_WINDOW_MINUTES]
        match_found = False

        for j in range(TRADE_WINDOW_MINUTES):
            row = window.iloc[j]
            signal = generate_signal(row)
            if signal in ["BUY_CALL", "BUY_PUT"]:
                entry_price = row["price"]
                direction = signal.replace("BUY_", "")
                stop_loss, take_profit = calculate_sl_tp(entry_price, row["supertrend"], direction)
                size = calculate_position_size(CAPITAL, RISK_PER_TRADE, entry_price, stop_loss)

                trades.append({
                    "timestamp": row["timestamp"],
                    "signal": signal,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "position_size": size,
                })
                match_found = True
                break

        if match_found:
            i += TRADE_WINDOW_MINUTES
        else:
            i += 1

    return trades

if __name__ == "__main__":
    df = pd.read_csv("data/mock_processed_data.csv", parse_dates=["timestamp"])
    trade_logs = run_backtest(df)

    # create outputs folder if not exists
    import os
    os.makedirs("outputs", exist_ok=True)

    output_df = pd.DataFrame(trade_logs)
    output_df.to_csv("outputs/trade_log.csv", index=False)
    print(f"✅ Backtest complete. {len(trade_logs)} trades taken. Saved to outputs/trade_log.csv")
