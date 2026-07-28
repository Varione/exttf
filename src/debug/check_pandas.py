import sqlite3
import pandas as pd

conn = sqlite3.connect('D:/etf/etf.sqlite')
cursor = conn.cursor()
cursor.execute('SELECT validation_status, COUNT(*) as count FROM etf_daily_price_modes GROUP BY validation_status ORDER BY count DESC')
modes_df = pd.read_sql_query(cursor)
print("Columns:", modes_df.columns.tolist())
print("Data:")
print(modes_df)

conn.close()
