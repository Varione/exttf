"""Factor computation engine with parallel processing and correlation analysis."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import hashlib
import json
import sqlite3
import pandas as pd
import numpy as np
import time
from pathlib import Path
from data_loader import load_etf_daily, get_valid_symbols
from factor_definitions import FACTORS, FactorDef


def _compute_db_fingerprint(db_path: str) -> str:
    """Compute a content-based fingerprint of the source database file.

    Hashes raw bytes of the SQLite file so that any data change is detected.
    Falls back to metadata-based hash only if the file cannot be read.
    Limitation: this does not detect changes to WAL/shm files that have not
    been checkpointed into the main database file.
    """
    try:
        h = hashlib.sha256()
        with open(db_path, "rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        try:
            stat = os.stat(db_path)
            return hashlib.sha256(
                f"{stat.st_size}-{stat.st_mtime}-metadata-fallback".encode()
            ).hexdigest()[:16]
        except OSError:
            return "unknown"


def _validate_existing_artifact(
    csv_path: str,
    expected_header_cols: list[str],
    expected_symbols: set[str],
    *,
    expected_symbol_row_counts: dict[str, int] | None = None,
    expected_symbol_date_ranges: dict[str, tuple[str, str]] | None = None,
) -> tuple[set[str], list[str]]:
    """Validate an existing factor CSV artifact for integrity.

    Returns (valid_symbol_set, list_of_errors).
    If errors is non-empty, the artifact must be considered unreliable and
    should NOT be used for checkpoint resume.

    Args:
        csv_path: Path to the CSV artifact.
        expected_header_cols: Expected column names in header order.
        expected_symbols: Set of symbols that must be present.
        expected_symbol_row_counts: Optional dict mapping symbol -> expected row count.
            If provided, each symbol's row count in the artifact must match exactly.
        expected_symbol_date_ranges: Optional dict mapping symbol -> (min_date, max_date).
            If provided, each symbol's date range must match exactly.
    """
    errors: list[str] = []
    valid_symbols: set[str] = set()

    if not os.path.exists(csv_path):
        return valid_symbols, errors

    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            # Validate header
            header_line = fh.readline().strip()
            if not header_line:
                errors.append("ARTIFACT_EMPTY: file exists but is empty")
                return valid_symbols, errors

            actual_cols = [_c.strip() for _c in header_line.split(",")]
            expected_cols_str = expected_header_cols

            if actual_cols != expected_cols_str:
                errors.append(
                    f"HEADER_MISMATCH: expected {len(expected_cols_str)} columns, "
                    f"got {len(actual_cols)}"
                )
                return valid_symbols, errors

            expected_width = len(expected_cols_str)

            # Validate rows
            line_num = 1
            seen_keys: set[tuple[str, str]] = set()
            symbol_date_counts: dict[str, int] = {}
            symbol_dates: dict[str, list[str]] = {}

            for line in fh:
                line_num += 1
                row_fields = line.strip().split(",")

                # Row width check
                if len(row_fields) != expected_width:
                    errors.append(
                        f"ROW_WIDTH_MISMATCH at line {line_num}: "
                        f"expected {expected_width} fields, got {len(row_fields)}"
                    )
                    continue

                symbol = row_fields[0]
                date_val = row_fields[1]

                # Duplicate key check
                key = (symbol, date_val)
                if key in seen_keys:
                    errors.append(
                        f"DUPLICATE_KEY at line {line_num}: ({symbol}, {date_val})"
                    )
                    continue

                seen_keys.add(key)
                valid_symbols.add(symbol)
                symbol_date_counts[symbol] = symbol_date_counts.get(symbol, 0) + 1
                if symbol not in symbol_dates:
                    symbol_dates[symbol] = []
                symbol_dates[symbol].append(date_val)

    except UnicodeDecodeError:
        errors.append("ARTIFACT_CORRUPT: cannot decode file as UTF-8")
    except OSError as exc:
        errors.append(f"ARTIFACT_READ_ERROR: {exc}")

    if errors:
        return valid_symbols, errors

    # Coverage check
    missing_symbols = expected_symbols - valid_symbols
    if missing_symbols:
        errors.append(
            f"INCOMPLETE_COVERAGE: {len(missing_symbols)} symbols missing "
            f"out of {len(expected_symbols)} expected"
        )

    # Row count validation per symbol
    if expected_symbol_row_counts is not None:
        for symbol, expected_count in expected_symbol_row_counts.items():
            actual_count = symbol_date_counts.get(symbol, 0)
            if actual_count != expected_count:
                errors.append(
                    f"ROW_COUNT_MISMATCH: symbol {symbol} has {actual_count} rows "
                    f"but expected {expected_count}"
                )

    # Date range validation per symbol
    if expected_symbol_date_ranges is not None:
        for symbol, (exp_min, exp_max) in expected_symbol_date_ranges.items():
            dates = symbol_dates.get(symbol, [])
            if not dates:
                errors.append(
                    f"DATE_RANGE_MISMATCH: symbol {symbol} has no rows "
                    f"but expected range [{exp_min}, {exp_max}]"
                )
                continue
            actual_min = min(dates)
            actual_max = max(dates)
            if actual_min != exp_min or actual_max != exp_max:
                errors.append(
                    f"DATE_RANGE_MISMATCH: symbol {symbol} has range "
                    f"[{actual_min}, {actual_max}] but expected [{exp_min}, {exp_max}]"
                )

    return valid_symbols, errors


def _write_manifest(
    csv_path: str,
    *,
    completed: bool,
    row_count: int,
    symbol_count: int,
    date_min: str | None,
    date_max: str | None,
    schema_hash: str,
    db_fingerprint: str,
    price_mode: str,
    validation_errors: list[str],
) -> None:
    """Write a machine-readable manifest next to the factor CSV."""
    manifest_path = csv_path + ".manifest.json"
    manifest = {
        "completed": completed,
        "row_count": row_count,
        "symbol_count": symbol_count,
        "date_min": date_min,
        "date_max": date_max,
        "schema_hash": schema_hash,
        "source_db_fingerprint": db_fingerprint,
        "price_mode": price_mode,
        "validation_errors": validation_errors,
    }

    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_manifest, manifest_path)


def compute_factors_for_etf(group: pd.DataFrame, factors: list[FactorDef]) -> pd.DataFrame:
    """Compute all factors for a single ETF."""
    results = {}
    for f in factors:
        try:
            series = f.compute(group)
            results[f.name] = series.values
        except Exception as exc:
            symbol = str(group["symbol"].iloc[0]) if not group.empty else "UNKNOWN"
            raise RuntimeError(f"Factor {f.name} failed for {symbol}") from exc
    return pd.DataFrame(results, index=group.index)


def compute_all_factors(
    db_path: str = "data/processed/etf.sqlite",
    output_path: str = "data/processed/factors_all_repaired.csv",
    symbols: list[str] | None = None,
    price_mode: str = "total_return_proxy",
    checkpoint_path: str = "data/processed/factors_checkpoint_symbols.txt",
    resume_from_checkpoint: bool = True,
) -> pd.DataFrame | None:
    """Compute all factors with artifact integrity guarantees.

    Safety properties:
    - Writes to a temporary sibling file; never modifies the final CSV until
      ALL validation passes (including factor errors, not just post-errors).
    - Validates any existing final artifact before trusting it for resume,
      checking per-symbol row counts and date ranges against source data.
    - A failed or interrupted run preserves the previous final artifact and
      its valid manifest. Only the temporary candidate is removed.
    - Checkpoint resume does NOT silently mark incomplete symbols as complete.
    - Writes a machine-readable manifest next to the final output.
    - On failure, writes a separate failure manifest documenting the cause.
    """
    df = load_etf_daily(db_path, price_mode=price_mode)
    if symbols:
        df = df[df["symbol"].isin(symbols)]

    all_symbols = sorted(df["symbol"].unique())
    total = len(all_symbols)

    # Build expected row counts and date ranges from source data
    expected_symbol_row_counts: dict[str, int] = {}
    expected_symbol_date_ranges: dict[str, tuple[str, str]] = {}
    for sym in all_symbols:
        sym_df = df[df["symbol"] == sym].sort_values("date")
        dates_str = sym_df["date"].dt.strftime("%Y-%m-%d").tolist()
        expected_symbol_row_counts[sym] = len(sym_df)
        if dates_str:
            expected_symbol_date_ranges[sym] = (min(dates_str), max(dates_str))

    metadata_cols = [
        "pit_eligible",
        "pit_observations",
        "pit_median_amount_60d",
        "reference_verified",
        "price_mode",
    ]
    header_cols = ["symbol", "date"] + metadata_cols + [f.name for f in FACTORS]

    # Schema hash for manifest
    schema_hash = hashlib.sha256(
        ",".join(header_cols).encode()
    ).hexdigest()[:16]

    # DB fingerprint for manifest
    db_fingerprint = _compute_db_fingerprint(db_path)

    # --- Integrity-safe resume ---
    # Only trust the existing final CSV if it passes validation against source data.
    completed_symbols: set[str] = set()
    artifact_valid = False

    if os.path.exists(output_path):
        valid_syms, val_errors = _validate_existing_artifact(
            output_path,
            header_cols,
            set(all_symbols),
            expected_symbol_row_counts=expected_symbol_row_counts,
            expected_symbol_date_ranges=expected_symbol_date_ranges,
        )
        if not val_errors:
            completed_symbols.update(valid_syms)
            artifact_valid = True
            print(f"Existing artifact validated: {len(valid_syms)} symbols trusted")
        else:
            print(f"Existing artifact FAILED validation ({len(val_errors)} issues):")
            for err in val_errors[:5]:
                print(f"  - {err}")

    # Checkpoint file is ONLY consulted when the artifact is valid.
    if resume_from_checkpoint and artifact_valid and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as fh:
                cp_symbols = set(line.strip() for line in fh if line.strip())
            # Intersection with validated symbols only
            completed_symbols.update(cp_symbols & set(all_symbols))
            print(f"Checkpoint adds {len(cp_symbols)} symbols (artifact-valid)")
        except Exception as e:
            print(f"Checkpoint read failed ({e}), ignoring checkpoint")

    tmp_path = output_path + ".tmp"
    # Remove stale tmp from interrupted run
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # Always start fresh in tmp file (we never trust a partial tmp)
    out_fh = open(tmp_path, "w", newline="")
    pd.DataFrame(columns=header_cols).to_csv(out_fh, index=False, header=True)

    count_new = 0
    skipped = len(completed_symbols)
    errors: list[tuple[str, str]] = []
    t_start = time.time()

    for symbol in all_symbols:
        if symbol in completed_symbols:
            continue

        group = df[df["symbol"] == symbol].sort_values("date")
        try:
            fac = compute_factors_for_etf(group, FACTORS)
            fac["symbol"] = symbol
            fac["date"] = group["date"].values
            for col in metadata_cols:
                fac[col] = group[col].values
            fac[header_cols].to_csv(out_fh, index=False, header=False)
            out_fh.flush()

            # Update checkpoint (only after successful write)
            completed_symbols.add(symbol)
            with open(checkpoint_path, "a") as fh:
                fh.write(f"{symbol}\n")

        except Exception as exc:
            errors.append((symbol, str(exc)))
            print(f"  ERROR: {symbol} - {exc}")

        count_new += 1
        if count_new % 100 == 0:
            elapsed = time.time() - t_start
            eta = (elapsed / count_new) * max(0, total - len(completed_symbols)) if count_new > 0 else 0
            done = len(completed_symbols)
            print(f"  Progress: {done}/{total} ETFs | "
                  f"{elapsed:.0f}s elapsed, {eta:.0f}s ETA")

    out_fh.close()

    # --- Post-computation validation before atomic replace ---
    n_rows = 0
    output_symbols: set[str] = set()
    symbol_output_counts: dict[str, int] = {}
    symbol_output_dates: dict[str, list[str]] = {}
    seen_post_keys: set[tuple[str, str]] = set()
    date_min: str | None = None
    date_max: str | None = None
    post_errors: list[str] = []

    try:
        with open(tmp_path, "r", encoding="utf-8") as fh:
            header_line = fh.readline().strip()
            actual_cols = [_c.strip() for _c in header_line.split(",")]
            if actual_cols != header_cols:
                post_errors.append("HEADER_MISMATCH after computation")

            for line in fh:
                fields = line.strip().split(",")
                if len(fields) != len(header_cols):
                    post_errors.append(f"ROW_WIDTH at row {n_rows+2}")
                    n_rows += 1
                    continue

                symbol = fields[0]
                d = fields[1]

                # Duplicate key check
                key = (symbol, d)
                if key in seen_post_keys:
                    post_errors.append(
                        f"DUPLICATE_KEY after computation: ({symbol}, {d})"
                    )
                    n_rows += 1
                    continue
                seen_post_keys.add(key)

                output_symbols.add(symbol)
                symbol_output_counts[symbol] = symbol_output_counts.get(symbol, 0) + 1
                if symbol not in symbol_output_dates:
                    symbol_output_dates[symbol] = []
                symbol_output_dates[symbol].append(d)

                if date_min is None or d < date_min:
                    date_min = d
                if date_max is None or d > date_max:
                    date_max = d
                n_rows += 1
    except OSError as exc:
        post_errors.append(f"POST_READ_ERROR: {exc}")

    # Check symbol coverage
    missing_syms = set(all_symbols) - output_symbols
    if missing_syms:
        post_errors.append(
            f"INCOMPLETE_COVERAGE: {len(missing_syms)} symbols not written "
            f"(missing from etf_daily or failed silently)"
        )

    # Check per-symbol row counts against source data
    for symbol in output_symbols:
        expected_count = expected_symbol_row_counts.get(symbol, 0)
        actual_count = symbol_output_counts.get(symbol, 0)
        if actual_count != expected_count:
            post_errors.append(
                f"ROW_COUNT_MISMATCH: symbol {symbol} has {actual_count} rows "
                f"but source has {expected_count}"
            )

    # Check per-symbol date ranges against source data
    for symbol in output_symbols:
        exp_range = expected_symbol_date_ranges.get(symbol)
        if exp_range is None:
            continue
        exp_min, exp_max = exp_range
        actual_dates = symbol_output_dates.get(symbol, [])
        if not actual_dates:
            post_errors.append(
                f"DATE_RANGE_MISMATCH: symbol {symbol} has no rows "
                f"but expected [{exp_min}, {exp_max}]"
            )
            continue
        actual_min = min(actual_dates)
        actual_max = max(actual_dates)
        if actual_min != exp_min or actual_max != exp_max:
            post_errors.append(
                f"DATE_RANGE_MISMATCH: symbol {symbol} has range "
                f"[{actual_min}, {actual_max}] but expected [{exp_min}, {exp_max}]"
            )

    all_validation_errors = post_errors + [e[1] for e in errors]

    # ANY error (factor computation OR post-validation) blocks atomic replace
    if post_errors or errors:
        print(
            f"\nValidation found {len(post_errors)} post-errors and "
            f"{len(errors)} factor errors BEFORE atomic replace."
        )
        for err in post_errors[:5]:
            print(f"  POST: {err}")
        for sym, err in errors[:5]:
            print(f"  FACTOR: {sym}: {err}")
        # Do NOT replace final artifact; preserve existing one and its manifest
        # Clean up only the temporary candidate
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        # Write a failure manifest documenting the candidate failure
        failure_manifest_path = output_path + ".candidate_failure.json"
        with open(failure_manifest_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "status": "CANDIDATE_FAILURE",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "post_errors": post_errors,
                    "factor_errors": [{"symbol": s, "error": e} for s, e in errors],
                    "row_count": n_rows,
                    "symbol_count": len(output_symbols),
                    "date_min": date_min,
                    "date_max": date_max,
                    "schema_hash": schema_hash,
                    "source_db_fingerprint": db_fingerprint,
                    "price_mode": price_mode,
                    "note": (
                        "Previous final artifact and its manifest were preserved. "
                        "This file documents the failed candidate."
                    ),
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Failure manifest written to {failure_manifest_path}")

    else:
        # All checks passed - atomic replace
        os.replace(tmp_path, output_path)
        print(f"\nValidation passed. Atomic replace complete.")

        # Write manifest next to final output (only on success)
        _write_manifest(
            output_path,
            completed=True,
            row_count=n_rows,
            symbol_count=len(output_symbols),
            date_min=date_min,
            date_max=date_max,
            schema_hash=schema_hash,
            db_fingerprint=db_fingerprint,
            price_mode=price_mode,
            validation_errors=[],
        )

        # Clean up checkpoint on success
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    elapsed_total = time.time() - t_start
    print(f"Saved {n_rows} rows ({len(output_symbols)} symbols) to {output_path}")
    print(f"Total time: {elapsed_total:.1f}s | Skipped: {skipped} | Errors: {len(errors)}")
    if errors:
        for sym, err in errors[:5]:
            print(f"  {sym}: {err}")

    return pd.read_csv(output_path, parse_dates=["date"]) if os.environ.get("LOAD_FULL") else None


def compute_correlation_matrix(
    factors_df: pd.DataFrame | None = None,
    csv_path: str = "data/processed/factors_all_repaired.csv",
    top_n: int = 100,
) -> pd.DataFrame:
    """Compute pairwise Spearman correlation between factors using cross-sectional rank."""
    if factors_df is None:
        factors_df = pd.read_csv(csv_path, parse_dates=["date"])

    factor_cols = [factor.name for factor in FACTORS if factor.name in factors_df.columns]
    print(f"Computing {len(factor_cols)}x{len(factor_cols)} correlation matrix...")

    symbol_counts = factors_df["symbol"].value_counts()
    top_symbols = symbol_counts.head(top_n).index
    sampled = factors_df[factors_df["symbol"].isin(top_symbols)].copy()

    corr_df = pd.DataFrame(index=factor_cols, columns=factor_cols)

    factor_data = {}
    for col in factor_cols:
        pvt = sampled.pivot(index="date", columns="symbol", values=col)
        stacked = pvt.stack()
        factor_data[col] = stacked

    all_idx = set(factor_data[factor_cols[0]].index)
    for col in factor_cols[1:]:
        all_idx &= set(factor_data[col].index)
    common_idx = pd.MultiIndex.from_tuples(sorted(all_idx))

    for i, c1 in enumerate(factor_cols):
        if (i + 1) % 20 == 0:
            print(f"  Correlation progress: {i+1}/{len(factor_cols)}")
        s1 = factor_data[c1].loc[common_idx].values
        for j, c2 in enumerate(factor_cols):
            if j < i:
                continue
            s2 = factor_data[c2].loc[common_idx].values
            mask = np.isfinite(s1) & np.isfinite(s2)
            if mask.sum() > 30:
                r = np.corrcoef(s1[mask], s2[mask])[0, 1]
            else:
                r = 0.0
            if np.isnan(r):
                r = 0.0
            corr_df.loc[c1, c2] = r
            corr_df.loc[c2, c1] = r

    return corr_df


def select_uncorrelated_factors(
    corr_df: pd.DataFrame,
    threshold: float = 0.5,
) -> list[str]:
    """Greedy selection of factors with pairwise correlation below threshold."""
    factors = corr_df.index.tolist()
    selected = [factors[0]]

    for f in factors[1:]:
        correlated = False
        for s in selected:
            if abs(corr_df.loc[f, s]) >= threshold:
                correlated = True
                break
        if not correlated:
            selected.append(f)

    return selected


def report_factor_quality(
    factors_df: pd.DataFrame | None = None,
    csv_path: str = "data/processed/factors_all_repaired.csv",
) -> pd.DataFrame:
    """Report factor quality metrics."""
    if factors_df is None:
        factors_df = pd.read_csv(csv_path, parse_dates=["date"])

    factor_cols = [factor.name for factor in FACTORS if factor.name in factors_df.columns]
    reports = []

    for col in factor_cols:
        vals = factors_df[col]
        valid_mask = (vals != 0) & (vals.notna()) & (np.isfinite(vals))
        coverage = valid_mask.mean()
        cs_std = factors_df.loc[valid_mask].groupby("date")[col].std().mean()

        reports.append({
            "factor": col,
            "category": _get_category(col),
            "coverage": f"{coverage:.2%}",
            "cs_std_mean": f"{cs_std:.4f}",
            "mean_abs": f"{vals.loc[valid_mask].abs().mean():.4f}",
        })

    return pd.DataFrame(reports).sort_values("cs_std_mean", ascending=False, key=lambda x: pd.to_numeric(x))


def _get_category(factor_name: str) -> str:
    from factor_definitions import FACTORS
    for f in FACTORS:
        if f.name == factor_name:
            return f.category
    return "unknown"


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "compute"

    if mode == "compute":
        t0 = time.time()
        symbols = get_valid_symbols()
        print(f"Valid symbols: {len(symbols)}")
        print(f"Total factors defined: {len(FACTORS)}")

        compute_all_factors(
            symbols=symbols,
            output_path="data/processed/factors_all_repaired.csv",
            price_mode="total_return_proxy",
        )
        print(f"\nDone in {time.time() - t0:.1f}s")

    elif mode == "analyze":
        t0 = time.time()
        factors_df = pd.read_csv("data/processed/factors_all_repaired.csv", parse_dates=["date"])

        quality = report_factor_quality(factors_df)
        print("=== Factor Quality (top 30 by cross-sectional std) ===")
        print(quality.head(30).to_string(index=False))
        print(f"\nTotal factors: {len(quality)}")

        t1 = time.time()
        corr = compute_correlation_matrix(factors_df, top_n=100)
        corr.to_csv("data/processed/factor_correlation.csv")
        print(f"\nCorrelation computation: {time.time() - t1:.1f}s")

        for thresh in [0.3, 0.5, 0.7]:
            selected = select_uncorrelated_factors(corr, threshold=thresh)
            print(f"\nSelected {len(selected)} factors (threshold={thresh}):")
            cats = {}
            for f in selected:
                cat = _get_category(f)
                cats[cat] = cats.get(cat, 0) + 1
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                print(f"  [{cat}] {count}")
            for f in selected:
                print(f"    {f}")

        print("\n=== Top 30 High Correlation Pairs (>0.8) ===")
        high_corr = []
        factor_cols = corr.index.tolist()
        for i, c1 in enumerate(factor_cols):
            for j, c2 in enumerate(factor_cols):
                if j <= i:
                    continue
                r = abs(corr.loc[c1, c2])
                if r > 0.8:
                    high_corr.append((c1, c2, r))
        high_corr.sort(key=lambda x: x[2], reverse=True)
        for c1, c2, r in high_corr[:30]:
            print(f"  {c1:30s} vs {c2:30s}: {r:.4f}")

        print("\n=== Factors per category ===")
        from factor_definitions import get_all_categories, get_factors_by_category
        for cat in get_all_categories():
            factors = get_factors_by_category(cat)
            print(f"  {cat}: {len(factors)} factors")

        print(f"\nTotal time: {time.time() - t0:.1f}s")
