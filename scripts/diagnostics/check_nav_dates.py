# coding: utf-8
import sqlite3, pandas as pd
conn = sqlite3.connect('data/processed/otf_expanded.sqlite')
dates = ['2022-03-05','2022-03-20','2022-12-31','2023-01-02','2023-05-02','2023-12-31','2024-06-30']
query = 'SELECT DISTINCT nav_date FROM otf_fund_nav WHERE nav_date IN (' + ','.join(['?']*len(dates)) + ') ORDER BY nav_date'
rows = conn.execute(query, dates).fetchall()
print('OTF NAV on problem dates:', [r[0] for r in rows])
conn2 = sqlite3.connect('data/processed/etf.sqlite')
query2 = 'SELECT DISTINCT date FROM etf_daily WHERE date IN (' + ','.join(['?']*len(dates)) + ') ORDER BY date'
rows2 = conn2.execute(query2, dates).fetchall()
print('ETF daily on problem dates:', [r[0] for r in rows2])
conn.close(); conn2.close()