"""Check how total_return_proxy is constructed and whether it accounts for dividends."""
import sqlite3
import pandas as pd

DB_PATH = "data/processed/etf.sqlite"

with sqlite3.connect(DB_PATH) as conn:
    # Check cumulative_dividend distribution
    print("=== cumulative_dividend statistics ===")
    rows = conn.execute(
        "SELECT symbol, MIN(cumulative_dividend), MAX(cumulative_dividend),"
        " COUNT(CASE WHEN cumulative_dividend > 0 THEN 1 END) as positive_days"
        " FROM etf_daily_price_modes GROUP BY symbol"
        " HAVING MAX(cumulative_dividend) > 0 ORDER BY MAX(cumulative_dividend) DESC LIMIT 30"
    ).fetchall()
    for r in rows:
        print(f"  {r[0]}: min={r[1]:.4f}, max={r[2]:.4f}, positive_days={r[3]}")

    # Check factor_f distribution
    print("\n=== factor_f statistics ===")
    rows2 = conn.execute(
        "SELECT symbol, MIN(factor_f), MAX(factor_f),"
        " COUNT(CASE WHEN factor_f != 1.0 THEN 1 END) as changed_days"
        " FROM etf_daily_price_modes GROUP BY symbol"
        " HAVING MAX(factor_f) > 1.0 ORDER BY MAX(factor_f) DESC LIMIT 30"
    ).fetchall()
    for r in rows2:
        print(f"  {r[0]}: min={r[1]:.4f}, max={r[2]:.4f}, changed_days={r[3]}")

    # Check how total_return_proxy is built for a specific symbol
    print("\n=== 510880 proxy construction ===")
    rows3 = conn.execute(
        "SELECT date, raw_close, cumulative_dividend, split_factor, factor_f,"
        " total_return_proxy, hfq_reference, legacy_proxy"
        " FROM etf_daily_price_modes WHERE symbol='510880'"
        " ORDER BY date LIMIT 20"
    ).fetchall()
    def fmt(v):
        return f"{v:.4f}" if v is not None else "None"
    
    hdr = f"{'date':<12} {'raw_close':>10} {'cum_div':>10} {'split':>8} {'factor_f':>10} {'proxy':>10} {'hfq_ref':>10}"
    print(hdr)
    for r in rows3:
        print(f"{r[0]:<12} {fmt(r[1]):>10} {fmt(r[2]):>10} {fmt(r[3]):>8} {fmt(r[4]):>10} {fmt(r[5]):>10} {fmt(r[6]):>10}")

    # Check a symbol with dividend data
    print("\n=== 510300 proxy construction ===")
    rows4 = conn.execute(
        "SELECT date, raw_close, cumulative_dividend, split_factor, factor_f,"
        " total_return_proxy, hfq_reference"
        " FROM etf_daily_price_modes WHERE symbol='510300'"
        " ORDER BY date LIMIT 20"
    ).fetchall()
    print(hdr)
    for r in rows4:
        print(f"{r[0]:<12} {fmt(r[1]):>10} {fmt(r[2]):>10} {fmt(r[3]):>8} {fmt(r[4]):>10} {fmt(r[5]):>10} {fmt(r[6]):>10}")

    # Check 159518 construction
    print("\n=== 159518 proxy construction ===")
    rows5 = conn.execute(
        "SELECT date, raw_close, cumulative_dividend, split_factor, factor_f,"
        " total_return_proxy, hfq_reference"
        " FROM etf_daily_price_modes WHERE symbol='159518'"
        " ORDER BY date LIMIT 20"
    ).fetchall()
    print(hdr)
    for r in rows5:
        print(f"{r[0]:<12} {fmt(r[1]):>10} {fmt(r[2]):>10} {fmt(r[3]):>8} {fmt(r[4]):>10} {fmt(r[5]):>10} {fmt(r[6]):>10}")

    # Check around April 2025 for 159518
    print("\n=== 159518 around April 2025 ===")
    rows6 = conn.execute(
        "SELECT date, raw_close, cumulative_dividend, split_factor, factor_f,"
        " total_return_proxy, hfq_reference"
        " FROM etf_daily_price_modes WHERE symbol='159518'"
        " AND date BETWEEN '2025-03-25' AND '2025-04-20'"
        " ORDER BY date"
    ).fetchall()
    print(hdr)
    for r in rows6:
        print(f"{r[0]:<12} {fmt(r[1]):>10} {fmt(r[2]):>10} {fmt(r[3]):>8} {fmt(r[4]):>10} {fmt(r[5]):>10} {fmt(r[6]):>10}")

    # Check validation_status distribution
    print("\n=== validation_status distribution ===")
    rows7 = conn.execute(
        "SELECT validation_status, COUNT(*) FROM etf_daily_price_modes GROUP BY validation_status"
    ).fetchall()
    for r in rows7:
        print(f"  {r[0]}: {r[1]}")

    # Check selected_price_mode distribution
    print("\n=== selected_price_mode distribution ===")
    rows8 = conn.execute(
        "SELECT selected_price_mode, COUNT(*) FROM etf_daily_price_modes GROUP BY selected_price_mode"
    ).fetchall()
    for r in rows8:
        print(f"  {r[0]}: {r[1]}")

conn.close()
