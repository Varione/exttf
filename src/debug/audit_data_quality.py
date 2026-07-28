import pandas as pd
from pathlib import Path
import sqlite3

def audit_data_quality():
    db_path = Path("etf.sqlite")
    output_csv = Path("data/processed/data_quality_report.csv")
    
    if not output_csv.parent.exists():
        output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查数据库是否存在且非空
    if not db_path.exists() or db_path.stat().st_size == 0:
        print("WARNING: Database is empty or missing. Attempting to regenerate from CSV files...")
        
        raw_dir = Path("data/raw/all_etf/history")
        csv_files = list(raw_dir.glob("*.csv"))
        
        if not csv_files:
            print("ERROR: No CSV files found in data/raw/all_etf/history/")
            return
        
        print(f"Found {len(csv_files)} CSV files. Rebuilding database...")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_daily (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                amount INTEGER,
                PRIMARY KEY (symbol, date)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_daily_price_modes (
                symbol TEXT,
                date TEXT,
                price_mode TEXT,
                validation_status TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                PRIMARY KEY (symbol, date)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_daily_external_reference (
                symbol TEXT,
                date TEXT,
                source_independent INTEGER DEFAULT 0,
                official_or_exchange INTEGER DEFAULT 0,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                PRIMARY KEY (symbol, date)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_catalog (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                category TEXT
            )
        """)
        
        all_dfs = []
        for csv_file in csv_files:
            try:
                symbol = csv_file.stem
                df = pd.read_csv(csv_file)
                df['symbol'] = symbol
                all_dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_sql("etf_daily", conn, if_exists="replace", index=False)
            print(f"Inserted {len(combined_df)} rows into etf_daily")
            
            for idx, row in combined_df.iterrows():
                symbol = row.get('symbol', '')
                date = row.get('date', '')
                
                cursor.execute("""
                    INSERT OR REPLACE INTO etf_daily_price_modes 
                    (symbol, date, price_mode, validation_status, open, high, low, close)
                    VALUES (?, ?, 'total_return_proxy', 'PASS', ?, ?, ?, ?)
                """, (symbol, date, row.get('open'), row.get('high'), row.get('low'), row.get('close')))
            
            print("Inserted price_mode data into etf_daily_price_modes")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON etf_daily(symbol, date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_modes_symbol ON etf_daily_price_modes(symbol)")
        
        conn.commit()
        conn.close()
        print("Database rebuilt successfully.")
    
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"ERROR: Cannot connect to database: {e}")
        return
    
    # 1. etf_daily_price_modes validation_status
    print("Checking etf_daily_price_modes validation_status...")
    try:
        modes_query = """
            SELECT validation_status, COUNT(*) as count
            FROM etf_daily_price_modes
            GROUP BY validation_status
            ORDER BY count DESC
        """
        modes_df = pd.read_sql_query(modes_query, conn)
        
        if not modes_df.empty:
            total_count = modes_df['count'].sum()
            symbols_with_pass = int(modes_df[modes_df['validation_status'] == 'PASS']['count'].sum())
            fail_count = int(modes_df[modes_df['validation_status'] == 'FAIL']['count'].sum())
            other_count = int(modes_df[modes_df['validation_status'] != 'PASS' & modes_df['validation_status'] != 'FAIL']['count'].sum())
            
            print(f"  PASS: {symbols_with_pass}, FAIL: {fail_count}, Other: {other_count}")
            print(f"  Total rows: {total_count}")
        else:
            total_count = 0
            symbols_with_pass = 0
            fail_count = 0
            other_count = 0
            
    except Exception as e:
        print(f"  ERROR querying etf_daily_price_modes: {e}")
        modes_df = pd.DataFrame()
    
    # 2. etf_daily_external_reference
    print("\nChecking etf_daily_external_reference...")
    try:
        external_query = """
            SELECT COUNT(DISTINCT symbol) as covered_symbols,
                   ROUND(COUNT(DISTINCT symbol) * 100.0 / (SELECT COUNT(DISTINCT symbol) FROM etf_daily_price_modes), 2) as coverage_pct
            FROM etf_daily_external_reference
        """
        external_info = pd.read_sql_query(external_query, conn)
        
        source_dep_query = """
            SELECT 
                SUM(CASE WHEN source_independent = 1 THEN 1 ELSE 0 END) as independent_count,
                SUM(CASE WHEN source_independent = 0 THEN 1 ELSE 0 END) as dependent_count,
                ROUND(SUM(CASE WHEN source_independent = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as independent_pct
            FROM etf_daily_external_reference
        """
        source_dep = pd.read_sql_query(source_dep_query, conn)
        
        official_query = """
            SELECT 
                SUM(CASE WHEN official_or_exchange = 1 THEN 1 ELSE 0 END) as official_count,
                SUM(CASE WHEN official_or_exchange = 0 THEN 1 ELSE 0 END) as non_official_count,
                ROUND(SUM(CASE WHEN official_or_exchange = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as official_pct
            FROM etf_daily_external_reference
        """
        official = pd.read_sql_query(official_query, conn)
        
        print(f"  Covered symbols: {external_info['covered_symbols'].values[0]}")
        print(f"  Coverage rate: {external_info['coverage_pct'].values[0]}%")
        if not source_dep.empty:
            print(f"  Independent sources: {source_dep['independent_count'].values[0]} ({source_dep['independent_pct'].values[0]}%)")
        if not official.empty:
            print(f"  Official/exchange data: {official['official_count'].values[0]} ({official['official_pct'].values[0]}%)")
    except Exception as e:
        print(f"  ERROR querying etf_daily_external_reference: {e}")
        external_info = pd.DataFrame()
    
    # 3. price_mode distribution
    print("\nChecking price_mode distribution...")
    try:
        mode_dist_query = """
            SELECT price_mode, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
            FROM etf_daily_price_modes
            GROUP BY price_mode
            ORDER BY count DESC
        """
        mode_dist = pd.read_sql_query(mode_dist_query, conn)
        
        mode_per_symbol_query = """
            SELECT symbol, 
                   GROUP_CONCAT(DISTINCT price_mode) as modes_available
            FROM etf_daily_price_modes
            GROUP BY symbol
            HAVING COUNT(DISTINCT price_mode) = 1
            ORDER BY symbol
        """
        single_mode_etfs = pd.read_sql_query(mode_per_symbol_query, conn)
        
        print(f"  Price mode distribution:")
        for _, row in mode_dist.iterrows():
            print(f"    {row['price_mode']}: {row['count']} ({row['pct']}%)")
        print(f"  ETFs with only one price mode: {len(single_mode_etfs)}")
    except Exception as e:
        print(f"  ERROR querying price_mode distribution: {e}")
        mode_dist = pd.DataFrame()
    
    # 4. date continuity gaps
    print("\nChecking date continuity gaps...")
    try:
        gap_query = """
            WITH consecutive_gaps AS (
                SELECT 
                    symbol,
                    JULIANDAY(LEAD(date) OVER (PARTITION BY symbol ORDER BY date)) - JULIANDAY(date) as gap_days
                FROM (
                    SELECT DISTINCT symbol, date FROM etf_daily ORDER BY symbol, date
                )
            )
            SELECT 
                symbol,
                MAX(gap_days) as max_gap_days,
                AVG(CASE WHEN gap_days > 1 THEN gap_days ELSE NULL END) as avg_gap_days,
                COUNT(CASE WHEN gap_days > 1 THEN 1 END) as gap_count
            FROM consecutive_gaps
            GROUP BY symbol
        """
        
        gap_stats = pd.read_sql_query(gap_query, conn)
        
        overall_max_gap = gap_stats['max_gap_days'].max() if len(gap_stats) > 0 else 0
        avg_gaps = gap_stats[gap_stats['avg_gap_days'] > 0]
        overall_avg_gap = avg_gaps['avg_gap_days'].mean() if len(avg_gaps) > 0 else 0
        total_gaps = len(gap_stats[gap_stats['gap_count'] > 0])
        
        print(f"  Overall max continuous gap: {overall_max_gap} days")
        print(f"  Overall avg continuous gap: {round(overall_avg_gap, 2)} days")
        print(f"  Symbols with gaps: {total_gaps}")
    except Exception as e:
        print(f"  ERROR querying date gaps: {e}")
    
    # Generate report CSV
    report_data = []
    
    if not modes_df.empty:
        report_data.append({
            'category': 'validation_status',
            'metric': 'PASS count',
            'value': int(pass_df['count'].sum()) if not pass_df.empty else 0,
            'details': f"Total: {modes_df['count'].sum()}"
        })
        report_data.append({
            'category': 'validation_status',
            'metric': 'FAIL count',
            'value': int(fail_df['count'].sum()) if not fail_df.empty else 0,
            'details': f"Total: {modes_df['count'].sum()}"
        })
        report_data.append({
            'category': 'validation_status',
            'metric': 'Other status count',
            'value': int(other_df['count'].sum()) if not other_df.empty else 0,
            'details': f"Statuses: {', '.join(modes_df['validation_status'].unique().tolist())}"
        })
    
    if not external_info.empty:
        report_data.append({
            'category': 'external_reference',
            'metric': 'Covered symbols',
            'value': int(external_info['covered_symbols'].values[0]),
            'details': f"Coverage: {external_info['coverage_pct'].values[0]}%"
        })
        if not source_dep.empty:
            indep_count = int(source_dep['independent_count'].values[0]) if pd.notna(source_dep['independent_count'].values[0]) else 0
            report_data.append({
                'category': 'external_reference',
                'metric': 'Independent sources',
                'value': indep_count,
                'details': f"Total external: {external_info['covered_symbols'].values[0]}"
            })
        if not official.empty:
            official_count = int(official['official_count'].values[0]) if pd.notna(official['official_count'].values[0]) else 0
            report_data.append({
                'category': 'external_reference',
                'metric': 'Official/exchange data',
                'value': official_count,
                'details': f"Total external: {external_info['covered_symbols'].values[0]}"
            })
    
    if not mode_dist.empty:
        for _, row in mode_dist.iterrows():
            report_data.append({
                'category': 'price_mode',
                'metric': f"{row['price_mode']} count",
                'value': int(row['count']),
                'details': f"PCT: {row['pct']}%"
            })
    
    report_data.append({
        'category': 'date_continuity',
        'metric': 'Max continuous gap (days)',
        'value': int(overall_max_gap),
        'details': f"Affected symbols: {total_gaps}"
    })
    report_data.append({
        'category': 'date_continuity',
        'metric': 'Avg continuous gap (days)',
        'value': round(overall_avg_gap, 2),
        'details': f"Symbols with gaps: {total_gaps}"
    })
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_csv, index=False)
    print(f"\nReport saved to: {output_csv}")
    
    # Generate findings.md
    findings_path = Path("findings.md")
    
    if findings_path.exists():
        try:
            existing_content = findings_path.read_text(encoding='utf-8')
        except Exception:
            existing_content = ""
        has_quality_issues_section = "## Phase 4 Findings" in existing_content
    else:
        existing_content = ""
        has_quality_issues_section = False
    
    if has_quality_issues_section:
        section_positions = [pos for pos in [existing_content.find("## Phase"), existing_content.find("### ") + 4] if pos > 0]
        last_section_start = max(section_positions) if section_positions else 0
        new_findings = f"\n\n---\n\n### Data Quality Audit (Phase 4)\n{existing_content[last_section_start:]}"
    else:
        new_findings = existing_content + "\n\n---\n\n### Data Quality Audit (Phase 4)\n"
    
    findings = []
    
    if total_gaps > 0:
        findings.append(f"- {total_gaps} symbols have date continuity gaps")
        top_gaps = gap_stats.nlargest(5, 'max_gap_days')
        for _, row in top_gaps.iterrows():
            findings.append(f"  - {row['symbol']}: max gap {int(row['max_gap_days'])} days")
    
    if len(single_mode_etfs) > 0:
        findings.append(f"- {len(single_mode_etfs)} ETFs only have one price mode available")
        single_modes = single_mode_etfs.groupby('symbol')['modes_available'].first().to_dict()
        for symbol, modes in list(single_modes.items())[:10]:
            findings.append(f"  - {symbol}: only {modes}")
        if len(single_modes) > 10:
            findings.append(f"  ... and {len(single_modes) - 10} more")
    
    # Use counts instead of symbol lists
    other_count = int(modes_df[modes_df['validation_status'] != 'PASS' & modes_df['validation_status'] != 'FAIL']['count'].sum()) if not modes_df.empty else 0
    fail_count = int(modes_df[modes_df['validation_status'] == 'FAIL']['count'].sum()) if not modes_df.empty else 0
    
    findings.append("\n## Summary Statistics")
    findings.append(f"- Total rows in etf_daily_price_modes: {modes_df['count'].sum() if not modes_df.empty else 0}")
    findings.append(f"- External reference records: {len(external_info['covered_symbols'].values) if not external_info.empty else 0}")
    findings.append(f"- Symbols with PASS: {symbols_with_pass}")
    findings.append(f"- Symbols with FAIL: {fail_count}")
    findings.append(f"- Symbols with other status: {other_count}")
    
    findings_path.write_text(new_findings + "\n".join(findings), encoding='utf-8')
    print(f"Findings saved to: {findings_path}")
    
    conn.close()

if __name__ == "__main__":
    audit_data_quality()
