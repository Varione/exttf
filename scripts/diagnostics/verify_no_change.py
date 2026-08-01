import sqlite3, pandas as pd

dates = ['2022-03-05','2022-03-20','2022-12-31','2023-01-02','2023-05-02','2023-12-31','2024-06-30']
placeholders = ','.join(['?']*len(dates))

conn = sqlite3.connect('data/processed/otf_expanded.sqlite')

# Get all funds with adj!=1.0 on holiday dates
query = f"""
    SELECT nav_date, fund_code, share_adjustment_factor
    FROM otf_fund_nav 
    WHERE nav_date IN ({placeholders}) AND ABS(share_adjustment_factor - 1.0) > 1e-9
    ORDER BY nav_date, fund_code
"""
rows = conn.execute(query, dates).fetchall()

# For each, check if the fund has NAV on the next execution date after that holiday
# Next execution dates: 2022-03-07, 2022-03-21, 2023-01-03, 2023-01-03, 2023-05-03, 2024-01-02, 2024-07-01
next_exec = {
    '2022-03-05': '2022-03-07',
    '2022-03-20': '2022-03-21',
    '2022-12-31': '2023-01-03',
    '2023-01-02': '2023-01-03',
    '2023-05-02': '2023-05-03',
    '2023-12-31': '2024-01-02',
    '2024-06-30': '2024-07-01',
}

print("Checking if funds with holiday adj!=1.0 have NAV on next execution date:")
no_nav_count = 0
for nav_date, fund_code, adj in rows:
    next_date = next_exec.get(nav_date)
    if not next_date:
        continue
    has_nav = conn.execute(
        "SELECT COUNT(*) FROM otf_fund_nav WHERE fund_code=? AND nav_date=?",
        (fund_code, next_date)
    ).fetchone()[0]
    if has_nav == 0:
        print(f"  NO NAV on next exec date {next_date}: {nav_date} {fund_code} adj={adj:.10f}")
        no_nav_count += 1

conn.close()

if no_nav_count == 0:
    print("All funds with holiday adj!=1.0 have NAV on the next execution date.")
    print("Old buggy code would correctly flush pending factors for all of them.")
    print("New searchsorted code produces identical results for production data.")
else:
    print(f"\n{no_nav_count} cases where fund lacks NAV on next execution date.")
    print("Results may differ between old and new aggregation for these funds.")