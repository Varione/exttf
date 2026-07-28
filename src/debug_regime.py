import pandas as pd

regime = pd.read_csv('data/processed/regime_predictions.csv', parse_dates=['date'])
test_period = regime[regime['date'] >= '2018-01-01']

print('Test period regime distribution:')
print(test_period['regime'].value_counts().sort_index())
print()
print(f'Total test days: {len(test_period)}')
for r in [0, 1, 2]:
    n = (test_period['regime'] == r).sum()
    print(f'Regime {r}: {n} days ({n/len(test_period)*100:.1f}%)')

# Check if strategy would be active
print('\nSample dates and regimes:')
print(test_period.head(10)[['date', 'regime']])