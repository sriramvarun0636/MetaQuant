import pandas as pd
import numpy as np
import datetime
import pytz
import os
import logging
import matplotlib.pyplot as plt

# ========= ADVANCED CONFIG ==========
API_KEY = "03ep39d0navol1d9"; ACCESS_TOKEN = "w0V8JWMcRxdhgOhWVBZ3NEIJRaWeZfaE"
IST = pytz.timezone('Asia/Kolkata')
SYMBOLS = ['NIFTY', 'BANKNIFTY']
INDEX_DETAILS = {"NIFTY": 50, "BANKNIFTY": 15}
STARTING_CAPITAL = 1_000_000.0
BROKERAGE_PER_TRADE = 40.0
MINIMUM_OPTION_RISK_POINTS = 1.0 

PORTFOLIO_PARAMS = {
    'strategy_allocation': {'AP_Trend': 1.0, 'SMC_OB': 1.0, 'MR_Reversal': 1.0},
    'max_daily_dd_percent': 0.10,
    'max_total_dd_percent': 0.30,
    'risk_per_trade_percent': 0.01,
    'max_trades_per_day': 10,
    'slippage_bps': {'HIGH': 10.0, 'NORMAL': 5.0, 'LOW': 2.0},
    'volatility_risk_modifier': {'HIGH': 0.75, 'NORMAL': 1.0, 'LOW': 1.25},
    'max_lots_per_trade': 100
}

AP_PARAMS = {'adx_threshold': 20, 'sl_atr_mult': 2.5, 'rr_ratio': 3.0}
SMC_PARAMS = {'sl_atr_mult': 2.0, 'rr_ratio': 4.0, 'ob_lookahead': 2, 'ob_impulse_atr_mult': 2.0}
MR_PARAMS = {'bb_window': 20, 'bb_std_dev': 2.0, 'sl_atr_mult': 1.5, 'rr_ratio': 2.0}

ADAPTIVE_PARAMS = {
    'score_risk_multipliers': {1: 0.75, 2: 1.0, 3: 1.25},
    'score_strike_offsets': {1: 0, 2: 0, 3: 1},
    'adx_score_threshold': 35
}

logging.basicConfig(filename='meta_quant_v3.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

def kite_login():
    class MockKiteConnect:
        def __init__(self, api_key): pass
        def instruments(self, segment): return []
        def historical_data(self, token, from_dt, to_dt, interval):
            start, end = pd.to_datetime(from_dt).tz_localize(IST), pd.to_datetime(to_dt).tz_localize(IST)
            freq = 'B' if interval == 'day' else 'min'; index = pd.date_range(start, end, freq=freq, name='date')
            if interval == 'minute':
                valid_times = index.indexer_between_time("09:15", "15:30")
                index = index[valid_times]
            if index.empty: return []
            count = len(index); base_price = 22000 if 'NIFTY' in str(token) else 48000
            if 'OPT' in str(token): base_price = 200
            data = pd.DataFrame(index=index); data['open'] = base_price + np.random.randn(count).cumsum(); data['high'] = data['open'] + np.random.uniform(0, 10, count); data['low'] = data['open'] - np.random.uniform(0, 10, count); data['close'] = data['low'] + np.random.uniform(0, (data['high'] - data['low']), count); data['volume'] = np.random.randint(1000, 10000, count)
            return data.reset_index().to_dict('records')
    return MockKiteConnect(api_key=API_KEY)

def fetch_instruments(kite, segment="NFO", cachefile='nfo_cache.parquet'):
    if os.path.exists(cachefile) and (datetime.date.today() - datetime.date.fromtimestamp(os.path.getmtime(cachefile))).days < 1: return pd.read_parquet(cachefile)
    data = []; today = datetime.date.today()
    for symbol in SYMBOLS:
        for i in range(3): data.append({'instrument_token': f'FUT_{symbol}_{i}', 'name': symbol, 'instrument_type': 'FUT', 'expiry': today + datetime.timedelta(days=(i+1)*30), 'strike': 0})
        step = 50 if symbol == 'NIFTY' else 100; base = 22000 if symbol == 'NIFTY' else 48000
        for i in range(2):
            expiry = today + datetime.timedelta(days=(i+1)*7)
            for offset in range(-15, 16):
                strike = base + offset * step
                data.append({'instrument_token': f'OPT_CE_{symbol}_{expiry}_{strike}', 'name': symbol, 'instrument_type': 'CE', 'expiry': expiry, 'strike': strike}); data.append({'instrument_token': f'OPT_PE_{symbol}_{expiry}_{strike}', 'name': symbol, 'instrument_type': 'PE', 'expiry': expiry, 'strike': strike})
    df = pd.DataFrame(data); df['expiry_date'] = pd.to_datetime(df['expiry']).dt.date; df.to_parquet(cachefile)
    return df

def fetch_ohlc(kite, token, from_dt, to_dt, interval='minute', cache_dir='ohlc_cache'):
    os.makedirs(cache_dir, exist_ok=True); fname = f"{cache_dir}/{token}_{str(from_dt.date())}_{str(to_dt.date())}_{interval}.parquet"
    if os.path.exists(fname) and interval != 'minute': return pd.read_parquet(fname)
    df = pd.DataFrame(kite.historical_data(token, from_dt, to_dt, interval))
    if df.empty: return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date']).dt.tz_convert(IST); df.set_index('date', inplace=True); df.to_parquet(fname)
    return df

def find_fut_token(nfo, symbol, trade_date):
    df = nfo[(nfo['name']==symbol) & (nfo['instrument_type']=='FUT') & (nfo['expiry_date'] > trade_date.date())]
    return None if df.empty else df.sort_values('expiry_date').iloc[0]['instrument_token']

def get_next_expiry(nfo, symbol, date):
    df = nfo[(nfo['name']==symbol) & (nfo['instrument_type'].isin(['CE','PE']))]
    exps = sorted({x for x in df['expiry_date'] if x >= date.date()})
    return exps[0] if exps else None

def find_option_token(nfo, symbol, expiry, strike, otype):
    sub = nfo[(nfo['name']==symbol) & (nfo['expiry_date']==expiry) & (nfo['instrument_type']==otype)]
    if sub.empty: return None, strike
    
    exact_match = sub[sub['strike'] == strike]
    if not exact_match.empty:
        return exact_match.iloc[0]['instrument_token'], strike

    logging.warning(f"Exact strike {strike} not found. Finding nearest available.")
    sub['strike_diff'] = abs(sub['strike'] - strike)
    nearest = sub.sort_values('strike_diff').iloc[0]
    return nearest['instrument_token'], nearest['strike']

def round_to_nearest(x, step): return int(round(x / step) * step)

def get_option_delta_approx(option_type, futures_price, strike_price):
    moneyness = (futures_price - strike_price) / strike_price
    if option_type == 'PE': moneyness *= -1
    if moneyness > 0.015: return 0.8;
    if moneyness > 0.005: return 0.6;
    if moneyness > -0.005: return 0.5;
    if moneyness > -0.015: return 0.3;
    return 0.1

def calculate_atr(df, period=14):
    tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df[f'atr_{period}'] = tr.ewm(span=period, min_periods=period).mean()
    return df

def get_daily_regimes(df_daily_hist):
    if df_daily_hist.empty or len(df_daily_hist) < 50: return "CHOP", "NORMAL"
    df = df_daily_hist.copy()
    df['ema_slow'] = df['close'].ewm(span=50, adjust=False).mean()
    prev_day = df.iloc[-2]
    trend_regime = "BULLISH" if prev_day['close'] > prev_day['ema_slow'] else "BEARISH" if prev_day['close'] < prev_day['ema_slow'] else "CHOP"
    df = calculate_atr(df, period=10)
    df['atr_ma'] = df['atr_10'].rolling(20).mean()
    vol_regime = "HIGH" if prev_day['atr_10'] > prev_day['atr_ma'] * 1.5 else "LOW" if prev_day['atr_10'] < prev_day['atr_ma'] * 0.7 else "NORMAL"
    return trend_regime, vol_regime

def get_signal_score(signal, features_row):
    reason = signal['reason']
    if 'AP_Trend' in reason:
        return 3 if features_row.get('adx_14', 0) > ADAPTIVE_PARAMS['adx_score_threshold'] else 2
    if 'SMC_OB' in reason:
        return 3
    if 'MR_Reversal' in reason:
        bb_std = features_row.get('bb_std', 0)
        if bb_std > 0:
            dist = abs(features_row['close'] - features_row['bb_ma']) / bb_std
            return 3 if dist > 2.5 else 2
        return 2 
    return 1

class StrategyBase:
    def __init__(self, params): self.params = params
    def compute_features(self, df): raise NotImplementedError
    def generate_signals(self, df): raise NotImplementedError

class AlphaPredatorStrategy(StrategyBase):
    def compute_features(self, df):
        if len(df) < 200: return pd.DataFrame()
        df = calculate_atr(df); df['ema_50'] = df['close'].ewm(span=50).mean(); df['ema_200'] = df['close'].ewm(span=200).mean()
        up, down = df['high'].diff(), -df['low'].diff(); plus_dm = np.where((up > down) & (up > 0), up, 0.0); minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr14 = df['atr_14'].rolling(14).sum()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / tr14
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr14
        adx_denominator = (plus_di + minus_di)
        adx_raw = abs(plus_di - minus_di) / adx_denominator.replace(0, np.nan) 
        df['adx_14'] = 100 * adx_raw.ewm(span=14, adjust=False).mean() 
        return df.dropna()

    def generate_signals(self, df, trend_regime):
        features = self.compute_features(df); signals = []
        if features.empty: return signals, features
        for i in range(2, len(features)):
            prev, prev2 = features.iloc[i-1], features.iloc[i-2]
            if trend_regime == 'BULLISH' and prev['ema_50'] > prev['ema_200'] and prev['adx_14'] > self.params['adx_threshold']:
                if prev['close'] > prev['ema_50'] and prev2['close'] <= prev2['ema_50']:
                    signals.append({'type': 'CE', 'reason': 'AP_Trend', 'dt': features.index[i], 'entry_price_fut': features['open'].iloc[i], 'atr': prev['atr_14']})
            elif trend_regime == 'BEARISH' and prev['ema_50'] < prev['ema_200'] and prev['adx_14'] > self.params['adx_threshold']:
                if prev['close'] < prev['ema_50'] and prev2['close'] >= prev2['ema_50']:
                    signals.append({'type': 'PE', 'reason': 'AP_Trend', 'dt': features.index[i], 'entry_price_fut': features['open'].iloc[i], 'atr': prev['atr_14']})
        return signals, features

class SMCPredatorStrategy(StrategyBase):
    def compute_features(self, df):
        if len(df) < 200: return pd.DataFrame()
        df = calculate_atr(df); n = self.params['ob_lookahead']; df['ema_50'] = df['close'].ewm(span=50).mean(); df['ema_200'] = df['close'].ewm(span=200).mean()
        impulse = abs(df['close'].diff(n)) > (df['atr_14'] * self.params['ob_impulse_atr_mult'])
        df['is_bull_ob'] = (df['close'] < df['open']).shift(n) & impulse & (df['close'] > df['open'])
        df['is_bear_ob'] = (df['close'] > df['open']).shift(n) & impulse & (df['close'] < df['open'])
        return df.dropna()

    def generate_signals(self, df, trend_regime):
        features = self.compute_features(df); signals = []
        if features.empty: return signals, features
        for i in range(1, len(features)):
            prev = features.iloc[i-1]
            if trend_regime == 'BULLISH' and prev['ema_50'] > prev['ema_200'] and prev['is_bull_ob']:
                signals.append({'type': 'CE', 'reason': 'SMC_OB', 'dt': features.index[i], 'entry_price_fut': features['open'].iloc[i], 'atr': prev['atr_14']})
            elif trend_regime == 'BEARISH' and prev['ema_50'] < prev['ema_200'] and prev['is_bear_ob']:
                signals.append({'type': 'PE', 'reason': 'SMC_OB', 'dt': features.index[i], 'entry_price_fut': features['open'].iloc[i], 'atr': prev['atr_14']})
        return signals, features

class MeanReversionStrategy(StrategyBase):
    def compute_features(self, df):
        if len(df) < self.params['bb_window']: return pd.DataFrame()
        df = calculate_atr(df); df['bb_ma'] = df['close'].rolling(self.params['bb_window']).mean(); df['bb_std'] = df['close'].rolling(self.params['bb_window']).std()
        df['bb_upper'] = df['bb_ma'] + (self.params['bb_std_dev'] * df['bb_std']); df['bb_lower'] = df['bb_ma'] - (self.params['bb_std_dev'] * df['bb_std'])
        return df.dropna()

    def generate_signals(self, df, trend_regime):
        features = self.compute_features(df); signals = []
        if features.empty or trend_regime != "CHOP": return signals, features
        for i in range(1, len(features)):
            prev = features.iloc[i-1]
            if prev['close'] < prev['bb_lower']:
                signals.append({'type': 'CE', 'reason': 'MR_Reversal', 'dt': features.index[i], 'entry_price_fut': features['open'].iloc[i], 'atr': prev['atr_14']})
            elif prev['close'] > prev['bb_upper']:
                signals.append({'type': 'PE', 'reason': 'MR_Reversal', 'dt': features.index[i], 'entry_price_fut': features['open'].iloc[i], 'atr': prev['atr_14']})
        return signals, features

class PortfolioManager:
    def __init__(self, initial_capital, params):
        self.capital = initial_capital; self.params = params; self.positions = {}; self.daily_pnl = 0.0; self.trades_today = 0; self.trade_log = []; self.missed_trades_log = []
    def reset_daily(self): self.daily_pnl = 0.0; self.trades_today = 0; self.positions = {}; self.missed_trades_log = []
    
    def can_trade_new(self, signal_time):
        if self.trades_today >= self.params['max_trades_per_day']: return False
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= (self.capital * self.params['max_daily_dd_percent']): return False
        if signal_time.time() >= datetime.time(15, 0): return False
        return True

    def calculate_position_size(self, entry_px, sl_px, lot_size, strategy_name, score, vol_modifier):
        risk_per_lot = abs(entry_px - sl_px) * lot_size
        if risk_per_lot <= 0: return 0 
        allocation = self.params['strategy_allocation'].get(strategy_name, 1.0)
        score_modifier = ADAPTIVE_PARAMS['score_risk_multipliers'].get(score, 1.0)
        rupee_risk = (self.capital * self.params['risk_per_trade_percent']) * allocation * score_modifier * vol_modifier
        lots = int(rupee_risk // risk_per_lot)
        return min(lots, self.params.get('max_lots_per_trade', lots))

    def on_fill(self, trade_info): self.trades_today += 1; self.trade_log.append(trade_info)
    def on_exit(self, pnl): self.daily_pnl += pnl; self.capital += pnl
    def on_missed_trade(self, reason): self.missed_trades_log.append(reason)

def run_meta_quant_backtest(start_date, end_date):
    kite = kite_login(); nfo = fetch_instruments(kite)
    if nfo.empty: return pd.DataFrame(), pd.DataFrame()

    strategies = {'AlphaPredator': AlphaPredatorStrategy(AP_PARAMS), 'SMCPredator': SMCPredatorStrategy(SMC_PARAMS), 'MeanReversion': MeanReversionStrategy(MR_PARAMS)}
    portfolio = PortfolioManager(STARTING_CAPITAL, PORTFOLIO_PARAMS)
    all_trades, equity_curve = [], []

    for date in pd.date_range(start_date, end_date, freq='B'):
        logging.info(f"--- Processing {date.date()} ---")
        portfolio.reset_daily()

        for symbol in SYMBOLS:
            fut_token = find_fut_token(nfo, symbol, date)
            if not fut_token: continue

            daily_hist = fetch_ohlc(kite, fut_token, date - datetime.timedelta(days=90), date - datetime.timedelta(days=1), 'day')
            trend_regime, vol_regime = get_daily_regimes(daily_hist)
            logging.info(f"{symbol} Regimes: Trend={trend_regime}, Volatility={vol_regime}")
            vol_risk_modifier = PORTFOLIO_PARAMS['volatility_risk_modifier'].get(vol_regime, 1.0)
            slippage_bps_today = PORTFOLIO_PARAMS['slippage_bps'].get(vol_regime, 5.0)
            slippage_factor = slippage_bps_today / 10000.0

            intraday_df = fetch_ohlc(kite, fut_token, date, date, 'minute')
            if intraday_df.empty: continue
            
            features_dict = {}
            if trend_regime in ["BULLISH", "BEARISH"]:
                features_dict['AlphaPredator'] = strategies['AlphaPredator'].compute_features(intraday_df)
                features_dict['SMCPredator'] = strategies['SMCPredator'].compute_features(intraday_df)
            elif trend_regime == "CHOP":
                features_dict['MeanReversion'] = strategies['MeanReversion'].compute_features(intraday_df)

            signals_with_features = []
            if trend_regime in ["BULLISH", "BEARISH"]:
                ap_sigs, _ = strategies['AlphaPredator'].generate_signals(intraday_df, trend_regime)
                smc_sigs, _ = strategies['SMCPredator'].generate_signals(intraday_df, trend_regime)
                for s in ap_sigs: signals_with_features.append((s, features_dict['AlphaPredator']))
                for s in smc_sigs: signals_with_features.append((s, features_dict['SMCPredator']))
            elif trend_regime == "CHOP":
                mr_sigs, _ = strategies['MeanReversion'].generate_signals(intraday_df, trend_regime)
                for s in mr_sigs: signals_with_features.append((s, features_dict['MeanReversion']))
            
            signals_with_features.sort(key=lambda x: x[0]['dt'])

            for signal, features_df in signals_with_features:
                if not portfolio.can_trade_new(signal['dt']): continue
                
                signal_timestamp = signal['dt']
                available_features = features_df.loc[features_df.index < signal_timestamp]
                if available_features.empty:
                    continue
                feature_row = available_features.iloc[-1]

                lot_size = INDEX_DETAILS[symbol]; step = 50 if symbol == 'NIFTY' else 100
                expiry = get_next_expiry(nfo, symbol, date)
                if not expiry: continue

                signal_score = get_signal_score(signal, feature_row)
                strike_offset = ADAPTIVE_PARAMS['score_strike_offsets'].get(signal_score, 0)
                atm_strike = round_to_nearest(signal['entry_price_fut'], step)
                strike_to_find = atm_strike - (strike_offset * step) if signal['type'] == 'CE' else atm_strike + (strike_offset * step)
                
                opt_token, actual_strike = find_option_token(nfo, symbol, expiry, strike_to_find, signal['type'])
                if not opt_token: continue

                opt_df = fetch_ohlc(kite, opt_token, date, date, 'minute');
                if opt_df.empty: continue
                
                entry_idx = opt_df.index.searchsorted(signal['dt']) + 1
                if entry_idx >= len(opt_df): continue

                entry_row = opt_df.iloc[entry_idx]; entry_px = entry_row['open']; side = 1 if signal['type'] == 'CE' else -1
                option_delta = get_option_delta_approx(signal['type'], signal['entry_price_fut'], actual_strike)
                
                strat_name = signal['reason'].split('_')[0]
                strat_full_name = strat_name + ('Predator' if 'AP' in strat_name or 'SMC' in strat_name else 'Reversion')
                strat_params = strategies[strat_full_name].params
                
                risk_fut = feature_row['atr_14'] * strat_params['sl_atr_mult']
                risk_opt = max(risk_fut * option_delta, MINIMUM_OPTION_RISK_POINTS)
                
                sl_px = entry_px - (risk_opt * side); tp_px = entry_px + (risk_opt * strat_params['rr_ratio'] * side)
                
                lots = portfolio.calculate_position_size(entry_px, sl_px, lot_size, signal['reason'], signal_score, vol_risk_modifier)
                if lots == 0:
                    portfolio.on_missed_trade({'reason': 'ZeroLots', 'signal_time': signal['dt']})
                    continue
                
                portfolio.on_fill({'symbol': symbol, 'entry_time': entry_row.name})
                entry_px_adj = entry_px * (1 + slippage_factor);
                exit_price, exit_time, reason = entry_px_adj, entry_row.name, 'EOD'
                
                trade_df = opt_df[opt_df.index >= entry_row.name]
                
                for k in range(1, len(trade_df)):
                    candle = trade_df.iloc[k]
                    
                    if side == 1:
                        if candle['open'] > candle['close']:
                            if candle['low'] <= sl_px: reason, exit_price = 'SL', sl_px; break
                            if candle['high'] >= tp_px: reason, exit_price = 'TP', tp_px; break
                        else:
                            if candle['high'] >= tp_px: reason, exit_price = 'TP', tp_px; break
                            if candle['low'] <= sl_px: reason, exit_price = 'SL', sl_px; break
                    else:
                        if candle['open'] < candle['close']:
                            if candle['high'] >= sl_px: reason, exit_price = 'SL', sl_px; break
                            if candle['low'] <= tp_px: reason, exit_price = 'TP', tp_px; break
                        else:
                            if candle['low'] <= tp_px: reason, exit_price = 'TP', tp_px; break
                            if candle['high'] >= sl_px: reason, exit_price = 'SL', sl_px; break
                    exit_time = candle.name

                if reason == 'EOD' and not trade_df.empty: exit_price, exit_time = trade_df.iloc[-1]['close'], trade_df.index[-1]
                
                exit_price_adj = exit_price * (1 - slippage_factor if side == 1 else 1 + slippage_factor)
                pnl = (exit_price_adj - entry_px_adj) * side * lot_size * lots; pnl_net = pnl - (BROKERAGE_PER_TRADE * 2 * lots)
                
                portfolio.on_exit(pnl_net)
                all_trades.append({'date':date.date(), 'symbol':symbol, 'strategy':signal['reason'], 'score': signal_score, 'vol_regime': vol_regime, 'pnl':pnl_net, 'exit_reason':reason})
                equity_curve.append({'date': exit_time, 'capital': portfolio.capital})
        
        logging.info(f"Finished processing {date.date()}. Missed Trades: {len(portfolio.missed_trades_log)}")

    trades_df = pd.DataFrame(all_trades); equity_df = pd.DataFrame(equity_curve)
    
    print("\n--- META QUANT BACKTEST COMPLETE (v3.0 - Hardened) ---")
    if not trades_df.empty:
        final_capital = equity_df['capital'].iloc[-1] if not equity_df.empty else STARTING_CAPITAL
        print(f"Total Trades: {len(trades_df)}")
        print(f"Final Capital: ₹{final_capital:,.2f} | Total PnL: ₹{(final_capital - STARTING_CAPITAL):,.2f}")
        print("\n--- Performance by Strategy ---"); print(trades_df.groupby('strategy')['pnl'].agg(['sum', 'count', 'mean']))
        print("\n--- Performance by Signal Score ---"); print(trades_df.groupby('score')['pnl'].agg(['sum', 'count', 'mean']))
        print("\n--- Performance by Volatility Regime ---"); print(trades_df.groupby('vol_regime')['pnl'].agg(['sum', 'count', 'mean']))
        
        if not equity_df.empty:
            equity_df.set_index('date')['capital'].plot(figsize=(12,6), title="Meta Quant Adaptive Equity Curve (Hardened)")
            plt.grid(True); plt.ylabel("Capital (₹)"); plt.xlabel("Date"); plt.show()
    else: print("No trades were generated.")
    return trades_df, equity_df

if __name__ == "__main__":
    start = datetime.date.today() - datetime.timedelta(days=90)
    end = datetime.date.today() - datetime.timedelta(days=1)
    trades, equity = run_meta_quant_backtest(start, end)
    if not trades.empty: trades.to_csv("meta_quant_v3_trades.csv", index=False)
