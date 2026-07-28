import sqlite3

conn = sqlite3.connect('D:/etf/etf.sqlite')
cursor = conn.cursor()
cursor.execute('SELECT validation_status, COUNT(*) FROM etf_daily_price_modes GROUP BY validation_status ORDER BY COUNT(*) DESC')
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")

conn.close()
