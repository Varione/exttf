"""Test different holding periods to find optimal strategy performance."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from strategy_library import STRATEGIES, get_all_strategy_names
import pandas as pd
import sqlite3
import numpy as np


def run_backtest_holding_period(strategy_name, holding_periods, start='2023-01-01', end='2026-07-17'):
    """Run backtest for a strategy with different holding periods."""
    
    # Load data
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
    
    # Filter dates
    mask = (rets_wide.index >= start) & (rets_wide.index <= end)
    dates = rets_wide.index[mask]
    date_list = [d.strftime('%Y-%m-%d') for d in dates]
    
    StrategyClass = STRATEGIES[strategy_name]
    strategy = StrategyClass(factors)
    
    results = {}
    
    for hold_days in holding_periods:
        portfolio_returns = []
        current_positions = None
        positions_start_idx = -999
        
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')

            # Check if we need to rebalance (no look-ahead: T-1 factor -> T day returns)
            if current_positions is None or (i - positions_start_idx) >= hold_days:
                # Select new positions using (T-1)-day factors, earn from T day
                factor_idx = i - 1
                if factor_idx < 0:
                    portfolio_returns.append(0.0)
                    if current_positions is None:
                        positions_start_idx = i
                    continue

                factor_date = dates[factor_idx]
                positions = strategy.get_positions(factor_date, n_hold=20, max_weight=0.05)

                if not positions or date_str not in rets_wide.index:
                    portfolio_returns.append(0.0)
                    positions_start_idx = i
                    continue

                current_positions = positions
                positions_start_idx = i

            # Calculate return for today using held positions (T day onwards)
            if date_str in rets_wide.index and current_positions and (i - positions_start_idx) >= 0:
                daily_rets = rets_wide.loc[date_str]
                port_return = sum(
                    w * (daily_rets[sym] if sym in daily_rets.index else 0.0)
                    for sym, w in current_positions.items()
                )
            else:
                port_return = 0.0

            portfolio_returns.append(port_return)
        
        # Calculate metrics
        returns = pd.Series(portfolio_returns)
        cum = (1 + returns).cumprod() - 1
        wealth = (1 + returns).cumprod()
        
        ann_ret = wealth.iloc[-1]**(252/len(returns)) - 1 if len(returns) > 0 else 0
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        dd = ((wealth - wealth.cummax()) / wealth.cummax()).min()
        win_rate = (returns > 0).mean() * 100
        
        results[hold_days] = {
            'ann_ret': ann_ret,
            'sharpe': sharpe,
            'max_dd': dd,
            'win_rate': win_rate,
            'total_return': cum.iloc[-1],
        }
    
    return results


def main():
    holding_periods = [1, 3, 5, 10, 20]
    strategies_to_test = get_all_strategy_names()
    
    print('Testing different holding periods...')
    print('%-25s ' % 'Strategy', end='')
    for hp in holding_periods:
        print('%-12s' % ('Hold%d_ann%%' % hp), end='')
    print()
    print('-' * 140)
    
    all_results = {}
    
    for strategy_name in strategies_to_test:
        results = run_backtest_holding_period(strategy_name, holding_periods)
        all_results[strategy_name] = results
        
        row = '%-25s ' % strategy_name
        for hp in holding_periods:
            r = results[hp]
            row += '%-12.2f' % (r['ann_ret'] * 100)
        print(row)
    
    # Save detailed results
    detail_data = []
    for strategy_name, results in all_results.items():
        for hold_days, metrics in results.items():
            detail_data.append({
                'strategy': strategy_name,
                'holding_period': hold_days,
                'ann_ret': metrics['ann_ret'] * 100,
                'sharpe': metrics['sharpe'],
                'max_dd': metrics['max_dd'] * 100,
                'win_rate': metrics['win_rate'],
                'total_return': metrics['total_return'] * 100,
            })
    
    detail_df = pd.DataFrame(detail_data)
    detail_df.to_csv('data/processed/holding_period_test.csv', index=False)
    print('\nDetailed results saved to data/processed/holding_period_test.csv')
    
    # Find best holding period for each strategy
    print('\nBest holding period per strategy:')
    for strategy_name, results in all_results.items():
        best_hp = max(results.keys(), key=lambda hp: results[hp]['sharpe'])
        r = results[best_hp]
        print('  %-25s Hold%d: ann=%.2f%% sharpe=%.4f dd=%.2f%%' % (
            strategy_name, best_hp, r['ann_ret']*100, r['sharpe'], r['max_dd']*100))


if __name__ == '__main__':
    main()
