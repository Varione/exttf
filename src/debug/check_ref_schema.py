import sqlite3
conn = sqlite3.connect("data/processed/etf.sqlite")
cursor = conn.cursor()

# Check reference sources
cursor.execute(
    "SELECT reference_source, source_independent, official_or_exchange, "
    "COUNT(*) as cnt,"
    "SUM(CASE WHEN cumulative_nav IS NOT NULL THEN 1 ELSE 0 END) as has_cum_nav,"
    "SUM(CASE WHEN unit_nav IS NOT NULL THEN 1 ELSE 0 END) as has_unit_nav"
    " FROM etf_daily_external_reference"
    " GROUP BY reference_source, source_independent, official_or_exchange"
    " ORDER BY cnt DESC"
)
print("=== Reference sources ===")
for row in cursor.fetchall():
    print(row)

# Check a sample symbol with both tables
cursor.execute(
    "SELECT pm.symbol, pm.date, pm.total_return_proxy, pm.raw_close, "
    "er.reference_value, er.unit_nav, er.cumulative_nav"
    " FROM etf_daily_price_modes pm"
    " JOIN etf_daily_external_reference er ON pm.symbol=er.symbol AND pm.date=er.date"
    " WHERE pm.symbol = '510300'"
    " ORDER BY pm.date LIMIT 5"
)
print("\n=== Sample join 510300 ===")
for row in cursor.fetchall():
    print(row)

# Count joined rows per symbol
cursor.execute(
    "SELECT pm.symbol, COUNT(*) as cnt"
    " FROM etf_daily_price_modes pm"
    " JOIN etf_daily_external_reference er ON pm.symbol=er.symbol AND pm.date=er.date"
    " GROUP BY pm.symbol ORDER BY cnt DESC"
)
print("\n=== Joined rows per symbol ===")
for row in cursor.fetchall():
    print(row)

# Check cumulative_nav availability per symbol
cursor.execute(
    "SELECT symbol, COUNT(*) as total,"
    "SUM(CASE WHEN cumulative_nav IS NOT NULL THEN 1 ELSE 0 END) as has_cum"
    " FROM etf_daily_external_reference"
    " GROUP BY symbol ORDER BY symbol"
)
print("\n=== cumulative_nav availability ===")
for row in cursor.fetchall():
    print(row)

conn.close()
