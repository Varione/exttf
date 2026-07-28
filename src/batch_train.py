"""Batch train across multiple lookback lengths."""

import sys, os, subprocess, time

LOOKBACKS = [30, 40, 60]
PYTHON = r"D:\miniconda\envs\agents\python.exe"

results = []

for lb in LOOKBACKS:
    print(f"\n{'='*60}")
    print(f"LOOKBACK={lb}")
    print(f"{'='*60}\n")

    # Build dataset
    t0 = time.time()
    r1 = subprocess.run(
        [PYTHON, "-u", "src/build_dataset.py", str(lb)],
        capture_output=True, text=True, timeout=300
    )
    print(r1.stdout[-300:] if len(r1.stdout) > 300 else r1.stdout)
    if r1.returncode != 0:
        print(f"Build failed: {r1.stderr[:500]}")
        continue

    # Train all models
    r2 = subprocess.run(
        [PYTHON, "-u", "src/train_classifier.py", str(lb)],
        capture_output=True, text=True, timeout=600
    )
    elapsed = time.time() - t0

    # Extract comparison table
    lines = r2.stdout.split('\n')
    in_table = False
    for line in lines:
        if 'COMPARISON' in line:
            in_table = True
        if in_table:
            print(line)
            if line.startswith('Total time'):
                break

    if r2.returncode != 0:
        err_lines = r2.stderr.split('\n')[-5:]
        print(f"Train error: {' '.join(err_lines)}")

    results.append((lb, elapsed))

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for lb, t in results:
    print(f"  Lookback={lb}: {t:.0f}s")
