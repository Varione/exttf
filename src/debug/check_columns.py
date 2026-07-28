import sqlite3

conn = sqlite3.connect('D:/etf/etf.sqlite')
cursor = conn.cursor()
cursor.execute('PRAGMA table_info(etf_daily_price_modes)')
print("etf_daily_price_modes columns:")
for row in cursor.fetchall():
    print(f"  {row[1]}")

cursor.execute('SELECT COUNT(*) FROM etf_daily_price_modes WHERE validation_status IS NULL OR validation_status = ""')
null_count = cursor.fetchone()[0]
print(f"\nRows with NULL/empty validation_status: {null_count}")

conn.close()
