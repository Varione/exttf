import sqlite3

dates = ['2022-03-05','2022-03-20','2022-12-31','2023-01-02','2023-05-02','2023-12-31','2024-06-30']
placeholders = ','.join(['?']*len(dates))

# Check otf_expanded.sqlite for non-1.0 share_adjustment_factor on holiday dates
conn = sqlite3.connect('data/processed/otf_expanded.sqlite')
query = f"""
    SELECT nav_date, fund_code, share_adjustment_factor, daily_growth_pct, unit_nav
    FROM otf_fund_nav 
    WHERE nav_date IN ({placeholders})
      AND ABS(share_adjustment_factor - 1.0) > 1e-9
    ORDER BY nav_date, fund_code
"""
rows = conn.execute(query, dates).fetchall()
conn.close()

print(f"Holiday dates with share_adjustment_factor != 1.0: {len(rows)} rows")
for r in rows[:20]:
    print(f"  {r[0]} {r[1]} adj={r[2]:.10f} growth={r[3]:.6f}% nav={r[4]:.6f}")

if not rows:
    print("  None found - production data has no non-1.0 adjustment factors on the 7 holiday dates.")