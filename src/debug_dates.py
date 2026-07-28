import pandas as pd, sqlite3

regime = pd.read_csv('data/processed/regime_predictions.csv', parse_dates=['date'])
print('Regime dates sample:', regime['date'].head(3).tolist())
print('Regime date range:', regime['date'].min(), '~', regime['date'].max())

conn = sqlite3.connect('data/processed/etf.sqlite')
prices = pd.read_sql('SELECT MIN(date) as min_d, MAX(date) as max_d FROM etf_daily', conn)
conn.close()
print('Price date range:', prices['min_d'].iloc[0], '~', prices['max_d'].iloc[0])

regime_dates = set(regime['date'].dt.strftime('%Y-%m-%d'))
print('Regime dates count:', len(regime_dates))

test_date = '2018-01-02'
print(f'Test date {test_date} in regime:', test_date in regime_dates)

# Check if dates are actually matching
conn2 = sqlite3.connect('data/processed/etf.sqlite')
prices_full = pd.read_sql('SELECT DISTINCT date FROM etf_daily LIMIT 10', conn2)
conn2.close()
print('Price dates sample:', prices_full['date'].tolist())
