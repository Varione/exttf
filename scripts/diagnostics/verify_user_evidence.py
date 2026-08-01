import sqlite3

dates = ["2022-03-05","2022-03-20","2022-12-31","2023-01-02","2023-05-02","2023-12-31","2024-06-30"]
strategy_funds = {"000008","000071","000148","000218","001512","007466","050021","050025"}

conn = sqlite3.connect("data/processed/otf_expanded.sqlite")
query = f"""
    SELECT nav_date, fund_code, share_adjustment_factor
    FROM otf_fund_nav 
    WHERE nav_date IN ({','.join(['?']*len(dates))}) AND fund_code IN ({','.join(['?']*len(strategy_funds))})
      AND ABS(share_adjustment_factor - 1.0) > 1e-9
    ORDER BY fund_code, nav_date
"""
params = list(dates) + sorted(strategy_funds)
rows = conn.execute(query, params).fetchall()
conn.close()

from collections import defaultdict
by_fund = defaultdict(list)
for d, fc, adj in rows:
    by_fund[fc].append((d, adj))

print("Holiday adj!=1.0 for strategy funds:")
for fc in sorted(by_fund.keys()):
    entries = by_fund[fc]
    max_dev = max(abs(a - 1.0) for _, a in entries)
    print(f"  {fc}: {len(entries)} dates, max|f-1|={max_dev:.2e}")
    for d, a in entries:
        print(f"    {d} adj={a:.10f}")

print(f"\nTotal: {len(rows)} rows across {len(by_fund)} funds")