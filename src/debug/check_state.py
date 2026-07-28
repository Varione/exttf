import sqlite3, os, json, hashlib

# Check canonical DB
print("=== Canonical DB: data/processed/etf.sqlite ===")
conn = sqlite3.connect("data/processed/etf.sqlite")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
for t in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM [{t[0]}]").fetchone()[0]
    print(f"  {t[0]}: {count} rows")
conn.close()

# Check root DB
print("\n=== Root DB: etf.sqlite ===")
conn = sqlite3.connect("etf.sqlite")
tables2 = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
for t in tables2:
    count = conn.execute(f"SELECT COUNT(*) FROM [{t[0]}]").fetchone()[0]
    print(f"  {t[0]}: {count} rows")
conn.close()

# Check factor artifact
print("\n=== Factor Artifact ===")
csv_path = "data/processed/factors_all_repaired.csv"
manifest_path = csv_path + ".manifest.json"
print(f"CSV exists: {os.path.exists(csv_path)}")
if os.path.exists(csv_path):
    print(f"CSV size: {os.path.getsize(csv_path)} bytes")
    # Count lines
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        cols = first_line.split(",")
        print(f"Header columns: {len(cols)}")
        line_count = 1
        for _ in f:
            line_count += 1
        print(f"Total lines (incl header): {line_count}")

print(f"Manifest exists: {os.path.exists(manifest_path)}")

# DB fingerprints
for db_path in ["data/processed/etf.sqlite", "etf.sqlite"]:
    h = hashlib.sha256()
    with open(db_path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    print(f"\n{db_path} SHA256: {h.hexdigest()}")
