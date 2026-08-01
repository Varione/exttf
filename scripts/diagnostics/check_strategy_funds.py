import sqlite3, json

dates = ['2022-03-05','2022-03-20','2022-12-31','2023-01-02','2023-05-02','2023-12-31','2024-06-30']
placeholders = ','.join(['?']*len(dates))

conn = sqlite3.connect('data/processed/otf_expanded.sqlite')
query = f"""
    SELECT DISTINCT nav_date, fund_code, share_adjustment_factor
    FROM otf_fund_nav 
    WHERE nav_date IN ({placeholders})
      AND ABS(share_adjustment_factor - 1.0) > 1e-9
    ORDER BY nav_date, fund_code
"""
rows = conn.execute(query, dates).fetchall()
conn.close()

# C1 satellite pool and core funds
c1_core = {"160706", "000218", "001512", "260102"}
c1_satellite_pool = {"160706", "000008", "050021", "007466", "110022", "000041", "000311", "001692"}
c2_core = {"160706", "000218", "001512", "260102"}
c2_satellite_pool = {"160706", "000008", "050021", "007466"}

# B2 funds
b2_funds = {"160706", "000218", "001512", "260102"}

c1_used = c1_core | c1_satellite_pool
c2_used = c2_core | c2_satellite_pool

print("Holiday adjustment factors for C1/C2/B2 funds:")
for nav_date, fund_code, adj in rows:
    if fund_code in c1_used or fund_code in c2_used or fund_code in b2_funds:
        print(f"  {nav_date} {fund_code} adj={adj:.10f}")

print("\nTotal holiday adj!=1.0 rows:", len(rows))
c1_count = sum(1 for _, fc, _ in rows if fc in c1_used)
c2_count = sum(1 for _, fc, _ in rows if fc in c2_used)
b2_count = sum(1 for _, fc, _ in rows if fc in b2_funds)
print(f"C1 universe affected: {c1_count}")
print(f"C2 universe affected: {c2_count}")
print(f"B2 universe affected: {b2_count}")