import sqlite3, sys
sys.path.insert(0,'src')
conn = sqlite3.connect('data/processed/otf_expanded.sqlite')
dates = ['2022-03-05','2022-03-20','2022-12-31','2023-01-02','2023-05-02','2023-12-31','2024-06-30']
rows = conn.execute('SELECT DISTINCT nav_date FROM otf_fund_nav WHERE nav_date IN (' + ','.join(['?']*len(dates)) + ') ORDER BY nav_date', dates).fetchall()
print('OTF NAV on problem dates:', [r[0] for r in rows])
conn.close()