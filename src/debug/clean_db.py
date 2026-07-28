import sqlite3

conn = sqlite3.connect('D:/etf/etf.sqlite')
cursor = conn.cursor()
cursor.execute('DELETE FROM etf_daily_price_modes')
cursor.execute('DROP TABLE IF EXISTS etf_daily_price_modes')
cursor.execute('DROP TABLE IF EXISTS etf_daily_external_reference')
cursor.close()
conn.commit()
conn.close()
