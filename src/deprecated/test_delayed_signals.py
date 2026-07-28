"""Test delayed signals to eliminate any remaining look-ahead bias."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from strategy_library import STRATEGIES, get_all_strategy_names
import pandas as pd
import sqlite3
import numpy as np


def run_backtest_delayed(signal_delay, holding_period, strategy_name, start='2023-01-01', end='2026-07-17'):
    """Run backtest with delayed signals: T-day uses (T-delay) day factors."""
    
    factors = pd.read_csv('data/processed/factors_all.csv', parse_dates=['date'])
    factors['symbol'] = factors['symbol'].astype(str)
    
    conn = sqlite3.connect('data/processed/etf.sqlite')
    prices = pd.read_sql('SELECT symbol, date, close FROM etf_daily', conn)
    conn.close()
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices[prices['close'] > 0].sort_values(['symbol', 'date'])
    prices['ret'] = prices.groupby('symbol')['close'].pct_change(1)
    
    rets_wide = prices.pivot_table(index='date', columns='symbol', values='ret').sort_index()
    rets_wide = rets_wide.clip(lower=-0.1, upper=0.1).fillna(0)
    
    mask = (rets_wide.index >= start) & (rets_wide.index <= end)
    dates = rets_wide.index[mask]
    
    StrategyClass = STRATEGIES[strategy_name]
    strategy = StrategyClass(factors)
    
    portfolio_returns = []
    current_positions = None
    positions_start_idx = -999
    
    for i, date in enumerate(dates):
        date_str = date.strftime('%Y-%m-%d')
        
        # Rebalance check
        if current_positions is None or (i - positions_start_idx) >= holding_period + 1:
            # Use delayed signal: T-day selects using (T-delay) day factors
            factor_date_idx = i - signal_delay
            if factor_date_idx < 0:
                portfolio_returns.append(0.0)
                positions_start_idx = i
                continue
            
            factor_date = dates[factor_date_idx]
            positions = strategy.get_positions(factor_date, n_hold=20, max_weight=0.05)
            
            if not positions:
                portfolio_returns.append(0.0)
                positions_start_idx = i
                continue
            
            current_positions = positions
            positions_start_idx = i
        
        # Calculate return (T+1 onwards only)
        if date_str in rets_wide.index and current_positions and (i - positions_start_idx) >= 1:
            daily_rets = rets_wide.loc[date_str]
            port_return = sum(
                w * (daily_rets[sym] if sym in daily_rets.index else 0.0)
                for sym, w in current_positions.items()
            )
        else:
            port_return = 0.0
        
        portfolio_returns.append(port_return)
    
    returns = pd.Series(portfolio_returns)
    cum = (1 + returns).cumprod() - 1
    wealth = (1 + returns).cumprod()
    
    ann_ret = wealth.iloc[-1]**(252/len(returns)) - 1 if len(returns) > 0 else 0
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
    dd = ((wealth - wealth.cummax()) / wealth.cummax()).min()
    
    return {'ann_ret': ann_ret, 'sharpe': sharpe, 'max_dd': dd}


def main():
    delays = [0, 1, 5]
    holds = [5, 10]
    strategies = ['S02_Trend_Following', 'S21_LowVol_Defense', 'S22_Tail_Risk', 
                  'S13_Quality', 'S01_CS_Momentum', 'B0_BuyHold_EW']
    
    print('Testing delayed signals (no look-ahead):')
    print('%-25s %-15s %-12s %-8s %-10s' % ('Strategy', 'Delay+Hold', 'Ann%', 'Sharpe', 'MaxDD%'))
    print('-' * 90)
    
    all_results = []
    
    for strat in strategies:
        for delay in delays:
            for hold in holds:
                r = run_backtest_delayed(delay, hold, strat)
                label = 'D%dH%d' % (delay, hold)
                print('%-25s %-15s %-12.2f %-8.4f %-10.2f' % (
                    strat, label, r['ann_ret']*100, r['sharpe'], r['max_dd']*100))
                all_results.append({'strategy': strat, 'delay': delay, 'hold': hold, **r})
    
    pd.DataFrame(all_results).to_csv('data/processed/delayed_signal_test.csv', index=False)
    print('\nResults saved to data/processed/delayed_signal_test.csv')


if __name__ == '__main__':
    main()
