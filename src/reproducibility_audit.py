"""Deterministic read-only audit gate for reproducibility issues."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

TIMESTAMP_FIELDS = frozenset({"timestamp"})


@dataclass
class Finding:
    file_path: str
    issue_type: str
    severity: str  # P0, P1, or P2
    description: str
    recommended_fix: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Path-aware filtering helpers
# ---------------------------------------------------------------------------

_AUDIT_MODULE_NAME = "reproducibility_audit"


def _is_excluded_path(py_file: Path, root: Path) -> bool:
    """Return True if py_file should be skipped by content-scanning checks.

    Excludes:
      - The audit module itself (it contains detector patterns as test strings)
      - Files under a 'tests/' directory or named test_*.py / *_test.py
      - __init__.py stubs

    Does NOT exclude production/deprecated executable scripts.
    """
    rel = py_file.relative_to(root)
    parts = rel.parts

    # The audit module itself
    if py_file.name == _AUDIT_MODULE_NAME + ".py":
        return True

    # Test directories or test files
    for part in parts[:-1]:
        if part.lower() in ("tests", "test"):
            return True
    base = py_file.stem.lower()
    if base.startswith("test_") or base.endswith("_test"):
        return True

    # __init__.py stubs
    if py_file.name == "__init__.py":
        content = py_file.read_text(encoding="utf-8", errors="replace").strip()
        if not content or all(
            l.strip().startswith(('"""', "'''", "#")) or l.strip() == ""
            for l in content.splitlines()
        ):
            return True

    return False


def _is_line_in_string_or_comment(py_file: Path, line_idx: int) -> bool:
    """Use AST to check if a given line is inside a string literal or comment.

    line_idx is 1-based (matching enumerate(..., 1)).
    Returns True if the line appears only inside a multi-line string that is NOT
    an active expression (i.e., a docstring or standalone string used for testing).
    """
    try:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return False

    lines = source.splitlines()
    if line_idx < 1 or line_idx > len(lines):
        return False

    target_line = lines[line_idx - 1]

    for node in ast.walk(tree):
        # Check multi-line string constants that span the target line
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.lineno <= line_idx <= node.end_lineno:
                # If the string spans multiple lines, the target line is inside a
                # string literal - treat it as not executable code.
                if node.lineno != node.end_lineno:
                    return True
        # Check multi-line comments (triple-quoted strings used as docstrings)
        if isinstance(node, (ast.Expr, ast.FunctionDef, ast.ClassDef, ast.Module)):
            pass  # handled by Constant above

    # Fallback: check if the line is a pure comment
    stripped = target_line.strip()
    if stripped.startswith("#"):
        return True

    return False


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def _is_otc_mode_claimed(content: str) -> bool:
    """Check if content claims OTC mode usage."""
    checks = [
        r'data_mode.*otc',
        r'"data_mode".*="otc"',
        r"'data_mode'.*='otc'",
        r'load_price_series.*data_mode.*=.*"otc"',
        r'load_price_series.*data_mode.*=.*\'otc\'',
    ]
    for pattern in checks:
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
            return True
    return False


def _check_stale_factors_references(root: Path) -> list[Finding]:
    """Scan for stale references to factors_all.csv instead of repaired artifact.

    Checks line-by-line so a file that has both old and new references is not
    skipped entirely. Only the specific lines referencing the stale artifact
    are flagged. Uses path-aware filtering to skip test/audit files.
    """
    findings: list[Finding] = []
    csv_pattern = "factors_all.csv"

    for py_file in root.rglob("*.py"):
        if _is_excluded_path(py_file, root):
            continue

        try:
            lines = py_file.read_text(encoding="utf-8").split("\n")
        except (UnicodeDecodeError, PermissionError):
            continue

        for i, line in enumerate(lines, 1):
            if csv_pattern in line and "factors_all_repaired" not in line:
                stripped = line.strip()
                if not stripped.startswith("#"):
                    if _is_line_in_string_or_comment(py_file, i):
                        continue
                    findings.append(Finding(
                        file_path=str(py_file),
                        issue_type="STALE_FACTORS_REFERENCE",
                        severity="P0",
                        description=f"Line {i}: Direct reference to factors_all.csv instead of factors_all_repaired.csv",
                        recommended_fix=f"Replace 'factors_all.csv' with 'factors_all_repaired.csv' to use repaired data with PIT metadata and corporate action corrections"
                    ))

    return findings


def _check_direct_raw_close_loading(root: Path) -> list[Finding]:
    """Scan for direct raw close loading without using repaired artifacts.

    Uses path-aware filtering to skip test/audit files.
    """
    findings: list[Finding] = []

    for py_file in root.rglob("*.py"):
        if _is_excluded_path(py_file, root):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")

            # Skip files that import data_loader which handles this properly
            if "from data_loader" in content or "import data_loader" in content:
                continue

            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                has_close = '"close"' in line or "'close'" in line
                is_not_price_mode = "price_mode" not in line.lower()
                is_not_raw_close = "raw_close" not in line.lower()
                is_not_load_price_series = "load_price_series" not in line

                if has_close and is_not_price_mode and is_not_raw_close and is_not_load_price_series:
                    if "pd.read_csv" in content or ".csv" in content or "DataFrame" in content:
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue
                        if _is_line_in_string_or_comment(py_file, i):
                            continue
                        findings.append(Finding(
                            file_path=str(py_file),
                            issue_type="DIRECT_RAW_CLOSE_LOADING",
                            severity="P1",
                            description=f"Line {i}: Potential direct close price loading without price_mode conversion",
                            recommended_fix="Use load_price_series() with appropriate price_mode and apply scale factors, or explicitly use raw_close mode with awareness of limitations"
                        ))
        except (UnicodeDecodeError, PermissionError):
            continue

    return findings


def _check_missing_transaction_cost_gross_return(root: Path) -> list[Finding]:
    """Scan for backtest code missing transaction_cost/gross_return usage.

    Uses path-aware filtering to skip test/audit files.
    """
    findings: list[Finding] = []

    for py_file in root.rglob("*.py"):
        if _is_excluded_path(py_file, root):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")

            # Skip files that already handle costs properly
            if "transaction_cost" in content.lower() or "gross_return" in content.lower():
                continue

            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                if _is_line_in_string_or_comment(py_file, i):
                    continue
                has_return_keyword = "return" in line.lower()
                has_assignment = "=" in line and "def " not in line
                if has_return_keyword and (has_assignment or stripped.startswith("return")):
                    findings.append(Finding(
                        file_path=str(py_file),
                        issue_type="MISSING_COST_ACCOUNTING",
                        severity="P1",
                        description=f"Line {i}: Return calculation without explicit transaction_cost or gross_return accounting",
                        recommended_fix="Add transaction_cost deduction: net_return = gross_return - transaction_cost, and track gross_return separately for cost analysis"
                    ))
        except (UnicodeDecodeError, PermissionError):
            continue

    return findings


def _check_missing_pit_price_mode_metadata(root: Path) -> list[Finding]:
    """Scan for factor loading without PIT/price_mode metadata.

    Uses path-aware filtering to skip test/audit files.
    """
    findings: list[Finding] = []

    for py_file in root.rglob("*.py"):
        if _is_excluded_path(py_file, root):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")

            # Check if loading factors without PIT metadata columns
            if "pit_eligible" not in content and "price_mode" not in content.lower():
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    if "read_csv" in line or ".csv" in line:
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue
                        if _is_line_in_string_or_comment(py_file, i):
                            continue
                        findings.append(Finding(
                            file_path=str(py_file),
                            issue_type="MISSING_PIT_METADATA",
                            severity="P2",
                            description=f"Line {i}: Factor loading may lack PIT eligibility and price_mode metadata",
                            recommended_fix="Ensure factors DataFrame includes pit_eligible, pit_observations, pit_median_amount_60d, reference_verified, and price_mode columns from factor_engine.py output"
                        ))
        except (UnicodeDecodeError, PermissionError):
            continue

    return findings


# ---------------------------------------------------------------------------
# DB schema check - accepts both valid schemas
# ---------------------------------------------------------------------------

_SCHEMA_A_REQUIRED = frozenset({"symbol", "date", "price_mode", "validation_status"})
_SCHEMA_B_REQUIRED = frozenset(
    {"symbol", "date", "raw_close", "hfq_reference", "total_return_proxy", "selected_price_mode", "validation_status"}
)


def _check_db_schema_for_metadata(db_path: Path) -> list[Finding]:
    """Check SQLite schema for required PIT and price_mode metadata.

    Accepts two valid schemas for etf_daily_price_modes:
      Schema A: symbol, date, price_mode, validation_status (+ optional extras)
      Schema B: symbol, date, raw_close, hfq_reference, total_return_proxy,
                selected_price_mode, validation_status (+ optional extras)
    """
    findings: list[Finding] = []

    if not db_path.exists():
        return [Finding(
            file_path=str(db_path),
            issue_type="DATABASE_MISSING",
            severity="P1",
            description=f"Database file not found: {db_path}",
            recommended_fix="Ensure data/processed/etf.sqlite exists with etf_daily and etf_daily_price_modes tables"
        )]

    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()

            # Check for required tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            if "etf_daily" not in tables:
                findings.append(Finding(
                    file_path=str(db_path),
                    issue_type="MISSING_ETF_DAILY_TABLE",
                    severity="P0",
                    description="etf_daily table is missing from database",
                    recommended_fix="Run build_dataset.py to create etf_daily table with OHLCV data"
                ))

            if "etf_daily_price_modes" not in tables:
                findings.append(Finding(
                    file_path=str(db_path),
                    issue_type="MISSING_PRICE_MODES_TABLE",
                    severity="P0",
                    description="etf_daily_price_modes table is missing from database",
                    recommended_fix="Run audit_data_quality.py or build_dataset.py to create etf_daily_price_modes table"
                ))

            # Check etf_daily schema for metadata columns.
            if "etf_daily" in tables:
                cursor.execute("PRAGMA table_info(etf_daily)")
                columns = {row[1] for row in cursor.fetchall()}
                # PIT columns are computed in load_etf_daily, not stored in raw DB.

            # Check etf_daily_price_modes schema - accept both valid schemas
            if "etf_daily_price_modes" in tables:
                cursor.execute("PRAGMA table_info(etf_daily_price_modes)")
                columns = frozenset({row[1] for row in cursor.fetchall()})

                has_schema_a = _SCHEMA_A_REQUIRED.issubset(columns)
                has_schema_b = _SCHEMA_B_REQUIRED.issubset(columns)

                if not has_schema_a and not has_schema_b:
                    missing_a = _SCHEMA_A_REQUIRED - columns
                    missing_b = _SCHEMA_B_REQUIRED - columns
                    # Report the smaller set of missing columns
                    if len(missing_a) <= len(missing_b):
                        missing_cols = sorted(missing_a)
                        ref_schema = "Schema A (price_mode)"
                    else:
                        missing_cols = sorted(missing_b)
                        ref_schema = "Schema B (raw_close/selected_price_mode)"
                    findings.append(Finding(
                        file_path=str(db_path),
                        issue_type="MISSING_PRICE_MODES_COLUMNS",
                        severity="P0",
                        description=f"etf_daily_price_modes table does not match either valid schema. "
                                    f"Missing for {ref_schema}: {', '.join(missing_cols)}",
                        recommended_fix="Run audit_data_quality.py to create and populate etf_daily_price_modes table with proper schema"
                    ))

    except sqlite3.Error as e:
        findings.append(Finding(
            file_path=str(db_path),
            issue_type="DATABASE_QUERY_ERROR",
            severity="P2",
            description=f"Failed to query database schema: {e}",
            recommended_fix="Check database file integrity and permissions"
        ))

    return findings


# ---------------------------------------------------------------------------
# DB candidate consistency check
# ---------------------------------------------------------------------------


def _db_fingerprint(db_path: Path) -> str | None:
    """Compute a deterministic fingerprint of the DB schema + sample data."""
    if not db_path.exists():
        return None
    try:
        h = hashlib.sha256()
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            # Schema fingerprint
            cursor.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
            )
            for row in cursor.fetchall():
                h.update(str(row).encode())
            # Data sample fingerprint (first 100 rows of each table)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            for (tbl,) in cursor.fetchall():
                if tbl.startswith("sqlite_"):
                    continue
                try:
                    cursor.execute(f"SELECT * FROM {tbl} LIMIT 100")
                    for row in cursor.fetchall():
                        h.update(str(row).encode())
                except sqlite3.Error:
                    pass
        return h.hexdigest()
    except sqlite3.Error:
        return None


def _check_db_candidate_consistency(db_paths: list[str]) -> list[Finding]:
    """Report when configured DB candidates both exist but differ in fingerprint.

    This is a deterministic consistency finding: if etf.sqlite and
    data/processed/etf.sqlite both exist and have different schema/data, it is
    a reproducibility risk because code may silently read from the wrong one.
    """
    findings: list[Finding] = []
    existing: list[tuple[Path, str]] = []

    for p in db_paths:
        path_obj = Path(p)
        if path_obj.exists():
            fp = _db_fingerprint(path_obj)
            if fp is not None:
                existing.append((path_obj, fp))

    if len(existing) < 2:
        return findings

    # Compare all pairs
    seen_pairs: set[tuple[str, str]] = set()
    for i in range(len(existing)):
        for j in range(i + 1, len(existing)):
            path_a, fp_a = existing[i]
            path_b, fp_b = existing[j]
            pair_key = (str(path_a), str(path_b))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            if fp_a != fp_b:
                findings.append(Finding(
                    file_path=str(path_a),
                    issue_type="DB_CANDIDATE_INCONSISTENCY",
                    severity="P0",
                    description=(
                        f"Configured database candidates differ in schema/data fingerprint: "
                        f"{path_a} (fp={fp_a[:16]}...) vs {path_b} (fp={fp_b[:16]}...). "
                        f"Code may silently read from the wrong database."
                    ),
                    recommended_fix="Ensure all configured DB candidates are identical, or explicitly configure which one to use. Remove stale copies."
                ))

    return findings


# ---------------------------------------------------------------------------
# Factor artifact readiness check
# ---------------------------------------------------------------------------


def _check_factor_artifact_readiness(root: Path) -> list[Finding]:
    """Check that factors_all_repaired.csv has a valid .manifest.json.

    Findings:
      - FACTOR_MANIFEST_MISSING (P0): CSV exists but manifest does not
      - FACTOR_MANIFEST_INCOMPLETE (P0): manifest exists but completed is false
      - FACTOR_MANIFEST_MISMATCH (P1): manifest metadata does not match CSV
    """
    findings: list[Finding] = []
    project_root = root.parent if root.name == "src" else root

    csv_path = project_root / "data" / "processed" / "factors_all_repaired.csv"
    manifest_path = project_root / "data" / "processed" / "factors_all_repaired.csv.manifest.json"

    if not csv_path.exists():
        return findings

    # Manifest missing
    if not manifest_path.exists():
        findings.append(Finding(
            file_path=str(csv_path),
            issue_type="FACTOR_MANIFEST_MISSING",
            severity="P0",
            description=f"Factor artifact {csv_path.name} exists but .manifest.json is missing. "
                        f"Reproducibility cannot be verified without manifest metadata.",
            recommended_fix="Run factor_engine.py to generate factors_all_repaired.csv.manifest.json with row count, symbol list, date range, and completed flag."
        ))
        return findings

    # Parse manifest
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, PermissionError) as e:
        findings.append(Finding(
            file_path=str(manifest_path),
            issue_type="FACTOR_MANIFEST_INCOMPLETE",
            severity="P0",
            description=f"Factor manifest cannot be parsed: {e}",
            recommended_fix="Regenerate factors_all_repaired.csv.manifest.json with valid JSON."
        ))
        return findings

    # Check completed flag
    completed = manifest.get("completed", None)
    if completed is False:
        findings.append(Finding(
            file_path=str(manifest_path),
            issue_type="FACTOR_MANIFEST_INCOMPLETE",
            severity="P0",
            description="Factor manifest exists but 'completed' is false. "
                        "The artifact is not ready for reproducible use.",
            recommended_fix="Re-run factor_engine.py to completion so that completed=true in the manifest."
        ))
    elif completed is None:
        findings.append(Finding(
            file_path=str(manifest_path),
            issue_type="FACTOR_MANIFEST_INCOMPLETE",
            severity="P0",
            description="Factor manifest exists but does not contain a 'completed' field.",
            recommended_fix="Add 'completed': true/false to factors_all_repaired.csv.manifest.json."
        ))

    # Lightweight source check: compare row count and symbols if available in manifest
    if completed is True:
        try:
            import csv as csv_mod

            with open(csv_path, "r", encoding="utf-8") as fh:
                reader = csv_mod.reader(fh)
                headers = next(reader, None)
                row_count = sum(1 for _ in reader)

            if headers:
                symbol_col_idx = None
                for idx, h in enumerate(headers):
                    if h.strip().lower() == "symbol":
                        symbol_col_idx = idx
                        break

                manifest_rows = manifest.get("row_count", manifest.get("rows", None))
                if manifest_rows is not None and str(manifest_rows) != str(row_count):
                    findings.append(Finding(
                        file_path=str(manifest_path),
                        issue_type="FACTOR_MANIFEST_MISMATCH",
                        severity="P1",
                        description=(
                            f"Manifest row_count ({manifest_rows}) does not match "
                            f"actual CSV row count ({row_count})."
                        ),
                        recommended_fix="Regenerate manifest to reflect current CSV state."
                    ))

                manifest_symbols = manifest.get("symbols", manifest.get("symbol_list", None))
                if manifest_symbols is not None and symbol_col_idx is not None:
                    actual_symbols: set[str] = set()
                    with open(csv_path, "r", encoding="utf-8") as fh:
                        reader = csv_mod.reader(fh)
                        next(reader)  # skip header
                        for row in reader:
                            if symbol_col_idx < len(row):
                                actual_symbols.add(row[symbol_col_idx].strip())
                    manifest_set = set(str(s) for s in manifest_symbols)
                    if manifest_set != actual_symbols:
                        findings.append(Finding(
                            file_path=str(manifest_path),
                            issue_type="FACTOR_MANIFEST_MISMATCH",
                            severity="P1",
                            description=(
                                f"Manifest symbols ({len(manifest_set)}) do not match "
                                f"actual CSV symbols ({len(actual_symbols)}). "
                                f"Difference: {len(manifest_set.symmetric_difference(actual_symbols))} symbols."
                            ),
                            recommended_fix="Regenerate manifest to reflect current CSV symbol list."
                        ))
        except Exception:
            pass  # Non-critical; manifest existence and completed flag are the P0 checks

    return findings


# ---------------------------------------------------------------------------
# OTC NAV check (unchanged logic, just path-aware)
# ---------------------------------------------------------------------------


def _check_otc_nav_availability(
    root: Path,
    db_paths: list[str] | None = None,
) -> list[Finding]:
    """Check if OTC NAV tables are available when OTC mode is claimed.

    Uses the configured database candidates (db_paths) instead of scanning
    arbitrary sqlite files under the project tree.
    """
    findings: list[Finding] = []

    # Find all Python files that might claim OTC mode
    otc_files: list[Path] = []
    for py_file in root.rglob("*.py"):
        if _is_excluded_path(py_file, root):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")

            # Check for OTC mode patterns
            if "data_mode" in content.lower() and "otc" in content.lower():
                otc_files.append(py_file)
        except (UnicodeDecodeError, PermissionError):
            continue

    if not otc_files:
        return findings

    # Use configured db_paths instead of scanning all sqlite files
    if db_paths is None:
        db_candidates = list((root.parent).rglob("*.sqlite"))
    else:
        db_candidates = [Path(p) for p in db_paths]

    fund_nav_found = False
    found_db_path: Path | None = None
    for db_path in db_candidates:
        if not db_path.exists():
            continue
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = {row[0] for row in cursor.fetchall()}

                if "fund_nav" in tables:
                    fund_nav_found = True
                    found_db_path = db_path
                    break
        except sqlite3.Error:
            continue

    if not fund_nav_found and db_candidates:
        report_path = str(db_candidates[0]) if db_candidates else "no databases configured"
        findings.append(Finding(
            file_path=report_path,
            issue_type="OTC_NAV_UNAVAILABLE",
            severity="P0",
            description="OTC mode is used in code but fund_nav table is missing from all configured database candidates",
            recommended_fix="Add fund_nav table with symbol/code, date/nav_date, and total_return_nav/adjusted_nav columns for OTC NAV data"
        ))

    return findings


# ---------------------------------------------------------------------------
# Arbitrary return clipping check (path-aware)
# ---------------------------------------------------------------------------


def _check_arbitrary_return_clipping(root: Path) -> list[Finding]:
    """Detect arbitrary return clipping (np.clip or clip around returns).

    Return values should not be silently clipped without explicit documentation
    of the clipping rationale and audit trail. Uses path-aware filtering.
    """
    findings: list[Finding] = []

    clip_patterns = [
        r'np\.clip\s*\(',
        r'\.clip\s*\(',
        r'returns?\s*=\s*.*clip',
        r'clip\s*\(.*return',
    ]

    for py_file in root.rglob("*.py"):
        if _is_excluded_path(py_file, root):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _is_line_in_string_or_comment(py_file, i):
                continue
            for pattern in clip_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(Finding(
                        file_path=str(py_file),
                        issue_type="ARBITRARY_RETURN_CLIPPING",
                        severity="P1",
                        description=f"Line {i}: Return clipping detected (np.clip or similar). "
                                    f"Clipping should be documented with rationale.",
                        recommended_fix="Document the clipping bounds, rationale, and ensure "
                                        "clipped values are tracked for audit purposes."
                    ))
                    break  # One finding per line

    return findings


# ---------------------------------------------------------------------------
# Manifest verification and diff reporting
# ---------------------------------------------------------------------------


def _compare_dicts(
    d1: dict, d2: dict, prefix: str = ""
) -> dict[str, dict[str, Any]]:
    """Recursively compare two dicts, excluding TIMESTAMP_FIELDS."""
    diffs: dict[str, dict[str, Any]] = {}
    all_keys = set(d1.keys()) | set(d2.keys())
    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        if key in TIMESTAMP_FIELDS:
            continue
        v1 = d1.get(key)
        v2 = d2.get(key)
        if v1 is None and v2 is None:
            continue
        if isinstance(v1, dict) and isinstance(v2, dict):
            diffs.update(_compare_dicts(v1, v2, full_key))
        elif v1 != v2:
            diffs[full_key] = {"run1": v1, "run2": v2}
    return diffs


def verify_manifest_reproducibility(
    manifest1_path: str | Path,
    manifest2_path: str | Path,
) -> dict[str, Any]:
    """Compare two experiment manifests for reproducibility.

    Excludes timestamp fields as specified in the experiment protocol.
    Returns a report with match status and per-field differences.
    """
    m1_path = Path(manifest1_path)
    m2_path = Path(manifest2_path)

    if not m1_path.exists():
        return {
            "reproducible": False,
            "error": f"Manifest not found: {m1_path}",
        }
    if not m2_path.exists():
        return {
            "reproducible": False,
            "error": f"Manifest not found: {m2_path}",
        }

    with open(m1_path, "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(m2_path, "r", encoding="utf-8") as f:
        m2 = json.load(f)

    diffs = _compare_dicts(m1, m2)

    # Compare artifact SHA256 values explicitly
    a1 = m1.get("artifacts", {})
    a2 = m2.get("artifacts", {})
    for artifact_name in sorted(set(a1.keys()) | set(a2.keys())):
        sha1 = a1.get(artifact_name, {}).get("sha256")
        sha2 = a2.get(artifact_name, {}).get("sha256")
        if sha1 != sha2:
            diffs[f"artifacts.{artifact_name}.sha256"] = {
                "run1": sha1,
                "run2": sha2,
            }

    # Check required protocol fields exist in both manifests
    REQUIRED_FIELDS = {
        "run_id", "timestamp", "gate_passed", "config",
        "db_info", "factor_manifest", "full_period_results",
        "oos_results", "artifacts", "regime_training_end",
        "python_version", "dependencies", "git_commit",
        "git_status", "code_fingerprint",
    }
    missing_m1 = REQUIRED_FIELDS - set(m1.keys())
    missing_m2 = REQUIRED_FIELDS - set(m2.keys())
    missing_fields: list[str] = []
    if missing_m1:
        missing_fields.append(f"run1 missing: {sorted(missing_m1)}")
    if missing_m2:
        missing_fields.append(f"run2 missing: {sorted(missing_m2)}")

    return {
        "run1_id": m1.get("run_id"),
        "run2_id": m2.get("run_id"),
        "reproducible": len(diffs) == 0 and not missing_fields,
        "num_differences": len(diffs),
        "missing_protocol_fields": missing_fields,
        "differences": diffs,
    }


def verify_summary_reproducibility(
    run1_dir: str | Path,
    run2_dir: str | Path,
) -> dict[str, Any]:
    """Compare summary.csv files from two runs for reproducibility."""
    import pandas as pd

    r1 = Path(run1_dir)
    r2 = Path(run2_dir)

    s1_path = r1 / "summary.csv"
    s2_path = r2 / "summary.csv"

    if not s1_path.exists():
        return {"reproducible": False, "error": f"Summary not found: {s1_path}"}
    if not s2_path.exists():
        return {"reproducible": False, "error": f"Summary not found: {s2_path}"}

    s1 = pd.read_csv(s1_path)
    s2 = pd.read_csv(s2_path)

    cols_match = list(s1.columns) == list(s2.columns)
    shapes_match = s1.shape == s2.shape

    result: dict[str, Any] = {
        "run1_shape": list(s1.shape),
        "run2_shape": list(s2.shape),
        "columns_match": cols_match,
        "shapes_match": shapes_match,
    }

    if not cols_match:
        result["extra_in_run1"] = sorted(set(s1.columns) - set(s2.columns))
        result["extra_in_run2"] = sorted(set(s2.columns) - set(s1.columns))
        result["reproducible"] = False
        return result

    if not s1.equals(s2):
        common = list(s1.columns)
        diff_cols = []
        for col in common:
            c1 = s1[col]
            c2 = s2[col]
            if not c1.equals(c2):
                diff_cols.append(col)
        result["different_columns"] = diff_cols
        result["reproducible"] = False
        return result

    result["reproducible"] = True
    return result


# ---------------------------------------------------------------------------
# Main audit entry point
# ---------------------------------------------------------------------------


def audit_project(
    root: Path | str = "src",
    db_paths: list[str] | None = None,
    output_path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Perform deterministic read-only audit of project for reproducibility issues.

    Args:
        root: Project root directory to scan
        db_paths: List of database paths to check (defaults to common locations)
        output_path: Optional path for JSON output (JSON only when explicitly provided)

    Returns:
        List of finding dictionaries with file_path, issue_type, severity, description, recommended_fix
    """
    root = Path(root)

    if not root.exists():
        raise ValueError(f"Root directory does not exist: {root}")

    # Default database paths
    if db_paths is None:
        db_paths = [
            str(root.parent / "etf.sqlite"),
            str(root.parent / "data" / "processed" / "etf.sqlite"),
        ]

    all_findings: list[Finding] = []

    # Run all audit checks
    all_findings.extend(_check_stale_factors_references(root))
    all_findings.extend(_check_direct_raw_close_loading(root))
    all_findings.extend(_check_missing_transaction_cost_gross_return(root))
    all_findings.extend(_check_missing_pit_price_mode_metadata(root))
    all_findings.extend(_check_arbitrary_return_clipping(root))

    # Check database schemas
    for db_path in db_paths:
        db_path_obj = Path(db_path)
        if db_path_obj.exists():
            all_findings.extend(_check_db_schema_for_metadata(db_path_obj))

    # DB candidate consistency check
    all_findings.extend(_check_db_candidate_consistency(db_paths))

    # Factor artifact readiness check
    all_findings.extend(_check_factor_artifact_readiness(root))

    # Check OTC NAV availability using configured db_paths
    all_findings.extend(_check_otc_nav_availability(root, db_paths=db_paths))

    # Sort findings by severity (P0 first) then by file path
    severity_order = {"P0": 0, "P1": 1, "P2": 2}
    all_findings.sort(key=lambda f: (severity_order.get(f.severity, 3), f.file_path))

    # Output JSON if explicit path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump([f.to_dict() for f in all_findings], fh, indent=2, ensure_ascii=False)
        print(f"Audit complete. {len(all_findings)} findings written to {output_path}")

    # Print summary
    p0_count = sum(1 for f in all_findings if f.severity == "P0")
    p1_count = sum(1 for f in all_findings if f.severity == "P1")
    p2_count = sum(1 for f in all_findings if f.severity == "P2")

    print(f"\nAudit Summary:")
    print(f"  P0 (Critical): {p0_count}")
    print(f"  P1 (High):     {p1_count}")
    print(f"  P2 (Medium):   {p2_count}")
    print(f"  Total:         {len(all_findings)}")

    if all_findings:
        print("\nFindings:")
        for f in all_findings:
            print(f"\n[{f.severity}] {f.issue_type}")
            print(f"  File: {f.file_path}")
            print(f"  Issue: {f.description}")
            print(f"  Fix:   {f.recommended_fix}")

    return [f.to_dict() for f in all_findings]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for reproducibility audit."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deterministic read-only audit gate for reproducibility issues"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Default audit command (positional root)
    audit_parser = subparsers.add_parser("audit", help="Run full project audit")
    audit_parser.add_argument(
        "--root",
        default=".",
        help="Project root directory to scan (default: current directory)"
    )
    audit_parser.add_argument(
        "--output", "-o",
        help="Path for JSON output file (optional, only when explicitly provided)"
    )

    # Manifest verification command
    verify_parser = subparsers.add_parser(
        "verify-manifest",
        help="Compare two experiment manifests for reproducibility"
    )
    verify_parser.add_argument("manifest1", help="Path to first run manifest")
    verify_parser.add_argument("manifest2", help="Path to second run manifest")

    # Summary verification command
    summary_parser = subparsers.add_parser(
        "verify-summary",
        help="Compare two summary.csv files for reproducibility"
    )
    summary_parser.add_argument("run1_dir", help="Directory of first run")
    summary_parser.add_argument("run2_dir", help="Directory of second run")

    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory to scan (default: current directory)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path for JSON output file (optional, only when explicitly provided)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON only (deprecated, use --output instead)"
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "verify-manifest":
            result = verify_manifest_reproducibility(
                args.manifest1, args.manifest2
            )
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            return 0 if result["reproducible"] else 1

        elif args.command == "verify-summary":
            result = verify_summary_reproducibility(args.run1_dir, args.run2_dir)
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            return 0 if result["reproducible"] else 1

        else:
            root = Path(args.root).resolve()
            findings = audit_project(root=root, output_path=args.output)
            return 0 if not findings else 1

    except Exception as e:
        print(f"Audit failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
