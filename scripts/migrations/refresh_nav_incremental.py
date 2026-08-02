"""Incrementally refresh NAV rows in otf_expanded.sqlite.

Keeps the frozen 997-fund pool untouched (no re-sampling, no pool change)
and only appends NAV dates that are missing. Every fund is re-fetched from
the provider and re-derived; rows that already exist are compared against
the provider's current values. Differences in existing rows are reported
as provider revisions (never silently overwritten in bulk); only identical
history + new dates are written back.

Safety rule (audit 2026-08-02): if a fund has CORE revisions (unit_nav,
cumulative_nav, daily_growth_pct or cumulative_nav_imputed differ), no new
rows are appended for that fund in this run. The stored history stays
unchanged and the run exits with code 2 (FORWARD_OBSERVATION_PAUSED) so a
human decides whether to keep old history, rebuild a version, or record a
data revision explicitly.

Persisted outputs (per run, under reports/nav_refresh/):
- refresh_report.json           : full run summary incl. blocked funds
- provider_revision_report.csv  : per-date core revision details

Usage:
    python scripts/migrations/refresh_nav_incremental.py [--workers 6]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

DB_PATH = ROOT / "data" / "processed" / "otf_expanded.sqlite"
REFRESH_REPORT_DIR = ROOT / "reports" / "nav_refresh"
COLS = (
    "fund_code", "nav_date", "unit_nav", "cumulative_nav", "daily_growth_pct",
    "cumulative_nav_imputed", "distribution_per_share", "share_adjustment_factor",
    "total_return_factor",
)


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    frame["cumulative_nav"] = pd.to_numeric(frame["cumulative_nav"], errors="coerce")
    frame["daily_growth_pct"] = pd.to_numeric(frame["daily_growth_pct"], errors="coerce")
    source_return = frame["daily_growth_pct"] / 100.0
    implied_distribution = (
        frame["nav"].shift(1) * (1.0 + source_return) - frame["nav"]
    )
    frame["distribution_per_share"] = implied_distribution.clip(lower=0.0).fillna(0.0)
    frame["share_adjustment_factor"] = (
        frame["nav"].shift(1) * (1.0 + source_return) / frame["nav"]
    ).fillna(1.0)
    if (frame["share_adjustment_factor"] <= 0).any():
        raise ValueError("non-positive share adjustment")
    fallback_return = (
        (frame["nav"] + frame["distribution_per_share"]) / frame["nav"].shift(1) - 1.0
    )
    total_return = source_return.where(source_return.notna(), fallback_return).fillna(0.0)
    if (total_return <= -1.0).any():
        raise ValueError("invalid total return <= -100%")
    frame["total_return_factor"] = (1.0 + total_return).cumprod()
    return frame


def fetch_fund_nav_js(code: str) -> pd.DataFrame:
    """Fetch unit NAV and cumulative NAV from the provider JS data file.

    This is the same request akshare's fund_open_fund_info_em performs for
    the 单位净值走势/累计净值走势 indicators, but the data arrays are parsed
    with the standard library instead of py_mini_racer.  akshare evaluates
    the whole file inside a new V8 isolate per call; concurrent isolates
    crash natively under the thread pool used here, while the arrays are
    plain JSON and parse identically.
    """
    url = f"https://fund.eastmoney.com/pingzhongdata/{code}.js"
    text = requests.get(url, timeout=30).text

    net_match = re.search(r"Data_netWorthTrend = (\[.*?\]);", text, re.S)
    cum_match = re.search(r"Data_ACWorthTrend = (\[.*?\]);", text, re.S)
    if not net_match or not cum_match:
        raise ValueError("Provider NAV data arrays are unavailable")
    net_rows = json.loads(net_match.group(1))
    cum_rows = json.loads(cum_match.group(1))
    if not net_rows or not cum_rows:
        raise ValueError("Unit NAV or cumulative NAV is unavailable")

    unit = pd.DataFrame([{"date": r["x"], "nav": r["y"],
                          "daily_growth_pct": r.get("equityReturn")} for r in net_rows])
    cumulative = pd.DataFrame([{"date": r[0], "cumulative_nav": r[1]}
                               for r in cum_rows])

    for frame in (unit, cumulative):
        frame["date"] = (
            pd.to_datetime(frame["date"], unit="ms", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
        )
    unit["nav"] = pd.to_numeric(unit["nav"], errors="coerce")
    unit["daily_growth_pct"] = pd.to_numeric(
        unit["daily_growth_pct"], errors="coerce"
    )
    cumulative["cumulative_nav"] = pd.to_numeric(
        cumulative["cumulative_nav"], errors="coerce"
    )

    df = unit.merge(cumulative, on="date", how="left", validate="one_to_one")
    df = (
        df.dropna(subset=["date", "nav"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    df["cumulative_nav_imputed"] = df["cumulative_nav"].isna()
    first_cumulative_date = df.loc[
        df["cumulative_nav"].notna(), "date"
    ].min()
    if pd.isna(first_cumulative_date):
        raise ValueError("Cumulative NAV has no valid observations")
    df = df.loc[df["date"] >= first_cumulative_date].reset_index(drop=True)
    cumulative_gap = (df["cumulative_nav"] - df["nav"]).ffill().bfill()
    df["cumulative_nav"] = df["cumulative_nav"].fillna(
        df["nav"] + cumulative_gap
    )
    if df["cumulative_nav"].isna().any():
        raise ValueError("Cumulative NAV reconstruction failed")
    return df


def _fetch(code: str, retries: int = 3) -> dict:
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            frame = _derive(fetch_fund_nav_js(code))
            return {"fund_code": code, "status": "success", "frame": frame}
        except Exception as exc:  # network/provider issues are recorded per fund
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(attempt * 2.0)
    return {"fund_code": code, "status": "error", "error": last_error}


def _round_match(values: tuple, old: tuple) -> bool:
    """Compare stored vs provider values with rounding tolerance.

    The provider returns more decimal places than the original loader
    stored (e.g. 0.9067000000000001 vs 0.9067).  Such pure float artifact
    differences are not provider revisions; compare at 6 decimals.
    """
    if len(values) != len(old):
        return False
    for left, right in zip(values, old):
        if left is None or right is None:
            if left is not None or right is not None:
                return False
            continue
        if isinstance(left, float) and isinstance(right, float):
            if abs(left - right) > 1e-6:
                return False
        elif left != right:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--db",
        default=str(DB_PATH),
        help="target sqlite db with otf_fund_catalog/otf_fund_nav tables "
        "(default otf_expanded.sqlite; use otf_mapped.sqlite for the mapped pool)",
    )
    args = parser.parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"db not found: {db_path}")
        return 2

    with sqlite3.connect(str(db_path)) as conn:
        codes = [
            r[0] for r in conn.execute(
                "SELECT fund_code FROM otf_fund_catalog ORDER BY fund_code"
            ).fetchall()
        ]
    print(f"pool: {len(codes)} funds (db={db_path.name})")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch, code): code for code in codes}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 100 == 0:
                print(f"fetched {index}/{len(codes)}")
    ok = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    print(f"fetched ok={len(ok)} failed={len(failed)}")

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        added, identical = 0, 0
        revised, revised_core = 0, 0
        revised_funds: list[tuple[str, int]] = []
        core_revision_funds: list[tuple[str, int]] = []
        blocked_funds: list[tuple[str, int, int]] = []
        revision_rows_csv: list[dict] = []
        for item in ok:
            frame = item["frame"]
            fund_code = item["fund_code"]
            cur.execute("SELECT nav_date FROM otf_fund_nav WHERE fund_code=?", (fund_code,))
            existing = {r[0] for r in cur.fetchall()}
            new_rows = []
            revision_count = 0
            core_revision_count = 0
            for row in frame.itertuples(index=False):
                date = str(row.date)[:10]
                values = (
                    fund_code, date, float(row.nav), float(row.cumulative_nav),
                    None if pd.isna(row.daily_growth_pct) else float(row.daily_growth_pct),
                    int(bool(row.cumulative_nav_imputed)),
                    float(row.distribution_per_share), float(row.share_adjustment_factor),
                    float(row.total_return_factor),
                )
                if date in existing:
                    cur.execute(
                        "SELECT unit_nav, cumulative_nav, daily_growth_pct, "
                        "cumulative_nav_imputed, distribution_per_share, "
                        "share_adjustment_factor, total_return_factor "
                        "FROM otf_fund_nav WHERE fund_code=? AND nav_date=?",
                        (fund_code, date),
                    )
                    old = cur.fetchone()
                    if old is not None and not _round_match(values[2:], old):
                        revision_count += 1
                        is_core = not _round_match(values[2:6], old[:4])
                        if is_core:
                            core_revision_count += 1
                            revision_rows_csv.append({
                                "fund_code": fund_code,
                                "nav_date": date,
                                "column": "unit_nav|cumulative_nav|daily_growth_pct|"
                                "cumulative_nav_imputed",
                                "provider_value": str(values[2:6]),
                                "stored_value": str(old[:4]),
                                "severity": "CORE_REVISION",
                            })
                    continue
                new_rows.append(values)
            if core_revision_count:
                # Safety rule (audit 2026-08-02): when the provider revised
                # core NAV history, do NOT append new rows for this fund. The
                # derived total_return_factor chain would be recomputed on the
                # revised history and create an artificial discontinuity with
                # stored rows. Keep old history intact, report, and let a
                # human decide: keep old history, rebuild a version, or record
                # a data revision explicitly.
                blocked_funds.append((fund_code, len(new_rows), core_revision_count))
                new_rows = []
            if new_rows:
                cur.executemany(
                    "INSERT OR REPLACE INTO otf_fund_nav VALUES (?,?,?,?,?,?,?,?,?)",
                    new_rows,
                )
                added += len(new_rows)
            if revision_count:
                revised += revision_count
                revised_funds.append((fund_code, revision_count))
            if core_revision_count:
                revised_core += core_revision_count
                core_revision_funds.append((fund_code, core_revision_count))
            if new_rows or revision_count:
                identical += 0
            else:
                identical += 1
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM otf_fund_nav")
        total_rows = cur.fetchone()[0]

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "pool_funds": len(codes),
        "fetch_success": len(ok),
        "fetch_failed": len(failed),
        "fetch_failures": [r["fund_code"] for r in failed[:50]],
        "rows_added": added,
        "rows_identical": "per-fund no-op",
        "any_diff_rows": revised,
        "core_revision_rows": revised_core,
        "core_revision_funds": core_revision_funds[:50],
        "blocked_funds": [
            {"fund_code": code, "rows_not_appended": rows, "core_revisions": cores}
            for code, rows, cores in blocked_funds
        ],
        "core_revision_note": "unit_nav/cumulative_nav/daily_growth_pct/imputed differ; "
        "other diffs are derived-column placeholder artifacts (see build_expanded_pool.py)",
        "safety_rule": "A fund with core NAV revisions is NOT appended this run; "
        "its stored history is preserved unchanged and a human must decide "
        "whether to keep old history, rebuild a version, or record a data revision.",
        "total_nav_rows": total_rows,
        "db": str(db_path),
        "note": "pool unchanged; only missing nav_date rows appended; existing rows compared and never overwritten",
    }

    run_dir = REFRESH_REPORT_DIR / f"nav_refresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "refresh_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    if revision_rows_csv:
        with (run_dir / "provider_revision_report.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "fund_code", "nav_date", "column", "provider_value",
                    "stored_value", "severity",
                ],
            )
            writer.writeheader()
            writer.writerows(revision_rows_csv)
    else:
        (run_dir / "provider_revision_report.csv").write_text(
            "fund_code,nav_date,column,provider_value,stored_value,severity\n",
            encoding="utf-8-sig",
        )
    print(f"persisted reports: {run_dir}")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if blocked_funds:
        print("FORWARD_OBSERVATION_PAUSED: core revisions detected; human decision required")
        return 2
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
