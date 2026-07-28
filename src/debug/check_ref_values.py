import sqlite3
conn = sqlite3.connect("data/processed/etf.sqlite")

# Check reference_value vs unit_nav for a symbol with cumulative_nav
rows = conn.execute(
    "SELECT date, reference_value, unit_nav, cumulative_nav"
    " FROM etf_daily_external_reference"
    " WHERE symbol = '510880' ORDER BY date LIMIT 10"
).fetchall()
print("=== 510880 reference data ===")
for r in rows:
    print(r)

# Check if reference_value is normalized or same as unit_nav
rows2 = conn.execute(
    "SELECT symbol, MIN(reference_value), MAX(reference_value), MIN(unit_nav), MAX(unit_nav)"
    " FROM etf_daily_external_reference GROUP BY symbol ORDER BY symbol"
).fetchall()
print("\n=== Value ranges ===")
for r in rows2:
    print(r)

# Check cumulative_nav for 510880
rows3 = conn.execute(
    "SELECT date, reference_value, unit_nav, cumulative_nav"
    " FROM etf_daily_external_reference"
    " WHERE symbol = '510880' AND cumulative_nav IS NOT NULL ORDER BY date LIMIT 10"
).fetchall()
print("\n=== 510880 with cumulative_nav ===")
for r in rows3:
    print(r)

conn.close()
